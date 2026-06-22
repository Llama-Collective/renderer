// Pool of meshing workers. RENDERER_PLAN.md §6, §16.
//
// Workers do NOT touch the GPU (meshing needs no GL —). They receive snapshots and
// return transferable build payloads; cancelled jobs return `cancelled`, never partial state.
//
// Two modes behind one interface:
//  - "inline": runs an injected mesh function synchronously on the caller's thread (delivered on a
//    microtask so consumers see the same async contract). The Phase-2 default / test path — the
//    mesher is a closure over the baked-model provider, so the pool stays decoupled from it.
//  - "worker": real Worker threads, fed an opaque `init` payload once. Off the main thread. The
//    worker-side model baking (serializing the resource pack to the worker) lands in Phase 3.
// Both deliver `buildDone` with a `BuildPayload`; the consumer wraps it via `makeBuildOutput`.

import type { SectionSnapshot } from "../world/SnapshotSource";
import { serializeBuildOutput, type SectionBuildOutput } from "./BuildOutput";
import type { WorkerRequest, WorkerResponse } from "./protocol";
import type { SharedMeshQueue } from "./SharedMeshQueue";
import { CancelFlags } from "./CancelFlags";

export type MeshFn = (snapshot: SectionSnapshot) => SectionBuildOutput;

export type WorkerPoolOptions =
  | { mode: "inline"; mesh: MeshFn }
  // WRK-3: `cooperativeCancelA` (default-off) wires a tiny cancel SAB so a superseded Model-A build STOPS
  // meshing instead of fully meshing + cloning a result `deliver()` would drop. Inert when SAB is absent.
  // WRK-2: `workStealing` (default-off) routes Model-A builds through the work-stealing deque (an idle worker
  // pulls the next build) AND arms the addModels model-epoch gate (`modelEpochA` captured at enqueue, shipped
  // on the build, ported to the worker) so a STOLEN build never meshes a new id as AIR. Off ⇒ the exact rr++.
  | { mode: "worker"; init: unknown; size?: number; createWorker: (index: number) => Worker; cooperativeCancelA?: boolean; workStealing?: boolean }
  // Model B: workers drain build jobs from a SHARED ring (zero-copy snapshots). `queue` is the main-thread
  // SharedMeshQueue (its SABs are sent to each worker at init); `fallbackMesh` meshes on the main thread iff
  // the SAB slot pool is momentarily full (so a section always presents — never a silent drop, TRAP 16.A).
  | { mode: "shared"; init: unknown; size?: number; createWorker: (index: number) => Worker; queue: SharedMeshQueue; fallbackMesh: MeshFn };

function defaultSize(): number {
  // Floor at 1: a 1-core host reports hardwareConcurrency === 1, and (1-1)=0 workers would silently drop
  // every build (no reply ever ⇒ section never presents, inFlight leaks). Always keep ≥1 worker.
  return Math.min(Math.max((globalThis.navigator?.hardwareConcurrency ?? 4) - 1, 1), 8);
}

export class WorkerPool {
  private handler: ((res: WorkerResponse) => void) | null = null;
  private readonly cancelled = new Set<number>();

  private readonly inlineMesh: MeshFn | null;
  /** Model B: the shared ring this pool enqueues build jobs into (null for inline / Model A). */
  private readonly queue: SharedMeshQueue | null;
  /** Model B: main-thread mesher used ONLY when the SAB slot pool is full (back-pressure relief). */
  private readonly fallbackMesh: MeshFn | null;
  private readonly workers: Worker[] = [];
  private rr = 0;
  /** ABP-3: when true, Model-A BUILD jobs flow through a work-stealing deque (idle workers pull the next
   *  build) instead of strict round-robin — so a skewed burst doesn't head-of-line-block on one rr victim.
   *  Default OFF ⇒ the exact `rr++` path. `sort` jobs always stay on round-robin (latency-sensitive). */
  workStealing = false;
  /** WRK-2: the Model-A model epoch — the postMessage-channel analogue of Model B's SAB `MODEL_EPOCH` word
   *  (Model A has `queue===null`, so it CANNOT use that word). `post(addModels)` ++this and ships it on the
   *  broadcast; a build CAPTURES the current value at enqueue and ships it too, so the worker that ends up
   *  meshing it (possibly after a STEAL) gates until its `appliedModelEpoch` catches up — never meshing a
   *  newly-minted palette id as AIR. Only stamped when `workStealing` is on; the rr/flag-off path leaves the
   *  build's `epoch` undefined and the worker skips the gate ⇒ byte-identical. Reset in dispose(). */
  private modelEpochA = 0;
  /** ABP-3: per-worker in-flight-build flag (index-aligned with `workers`). At most one build per worker. */
  private readonly busy: boolean[] = [];
  /** ABP-3: builds awaiting an idle worker (Model A + workStealing). Posted to a worker AFTER any pending
   *  `addModels` broadcast, so a deferred build always meshes against the LATEST palette (safe; see post). */
  private readonly pendingBuilds: WorkerRequest[] = [];
  /** WRK-3: the cooperative-cancel SAB (Model A only, default-off). Null when the flag is off OR SAB is
   *  unavailable ⇒ the feature is fully inert and the dispatch/cancel paths are byte-identical to today. */
  private readonly cancelFlags: CancelFlags | null;
  /** WRK-3: which worker INDEX is currently holding each in-flight build jobId, so `cancel()` can write the
   *  CANCEL_REQUEST into the EXACT worker's slot (per-worker scoping — TRAP WRK-3.B; never a global scan).
   *  Populated only when `cancelFlags` is live; entries are cleared on the build's reply. */
  private readonly jobToWorker = new Map<number, number>();

  constructor(opts: WorkerPoolOptions) {
    if (opts.mode === "inline") {
      this.inlineMesh = opts.mesh;
      this.queue = null;
      this.fallbackMesh = null;
      this.cancelFlags = null;
      return;
    }
    // Off-thread meshing (Phase 5). Spawn `size` real workers, hand each the SAME opaque `init` payload
    // (the transferred baked palette models — see MeshInit), and route every worker reply through deliver.
    // Model A: snapshots go out by structured clone per `build` message (sidesteps the detach hazard, TRAP
    // 5.E). Model B ("shared"): jobs flow through the SAB ring (zero-copy snapshots) — the worker gets the
    // ring handles at init and drains it. Results transfer back zero-copy on BOTH.
    this.inlineMesh = null;
    this.queue = opts.mode === "shared" ? opts.queue : null;
    this.fallbackMesh = opts.mode === "shared" ? opts.fallbackMesh : null;
    const sharedHandles = opts.mode === "shared" ? opts.queue.handles : undefined;
    const size = Math.max(1, opts.size ?? defaultSize()); // never 0 — see defaultSize
    // WRK-3: allocate the cooperative-cancel SAB only for Model A with the flag explicitly on AND SAB present
    // (CancelFlags.create returns null when SAB is absent). Off ⇒ null ⇒ no SAB, no extra init field below
    // ⇒ the worker attaches a no-op shim ⇒ byte-identical. (Model B has its own ring-based D_CANCEL.)
    this.cancelFlags = opts.mode === "worker" && opts.cooperativeCancelA ? CancelFlags.create(size) : null;
    // WRK-2: opt into the work-stealing deque + addModels epoch gate (Model A only; default-off ⇒ the exact
    // rr++ path with no epoch attached). The public `workStealing` field stays assignable for the existing
    // ABP-3 tests; this just lets the viewer set it at construction (mirroring `cooperativeCancelA`).
    if (opts.mode === "worker" && opts.workStealing) this.workStealing = true;
    const cancelSab = this.cancelFlags?.buffer;
    for (let i = 0; i < size; i++) {
      const w = opts.createWorker(i); // pass the slot index so the factory can give the worker a unique name
      // Capture the worker INDEX so deliver() can free its work-stealing slot on a build reply (ABP-3).
      w.onmessage = (e: MessageEvent) => this.deliver(e.data as WorkerResponse, i);
      // WRK-3: hand the cancel SAB + this worker's slot index ONLY when the feature is live; otherwise the
      // init message is exactly today's (no `cancelFlags`/`workerIndex` fields) so the worker shims to no-op.
      w.postMessage(
        cancelSab
          ? { type: "init", config: opts.init, shared: sharedHandles, cancelFlags: cancelSab, workerIndex: i }
          : { type: "init", config: opts.init, shared: sharedHandles },
      );
      this.workers.push(w);
      this.busy.push(false);
    }
  }

  onMessage(handler: (res: WorkerResponse) => void): void {
    this.handler = handler;
  }

  /** WRK-1: the worker count used to size the admission gate's per-frame budget (`workers * frameMs *
   *  bytesPerMs`). Inline mode has no real workers but meshes on one thread per microtask ⇒ 1; worker /
   *  shared modes report the spawned worker count. */
  get workerCount(): number {
    return this.workers.length === 0 ? 1 : this.workers.length;
  }

  post(req: WorkerRequest): void {
    if (req.type === "cancel") {
      // Main-side cancel is the LEAK-FREE BACKSTOP and stays unconditional: mark the jobId so deliver()
      // converts its (always-arriving) result to `cancelled`. The worker always replies (buildDone OR a
      // WRK-3 `cancelled`), exactly like the Scheduler's generation stale-drop, which keeps both this set
      // and the workers leak-free (a silently-dropped build would never let deliver clear the set).
      this.cancelled.add(req.jobId);
      // WRK-3 (Model A, default-off): ADDITIONALLY hint the EXACT worker holding this jobId to stop meshing
      // it (per-worker CANCEL_REQUEST, TRAP WRK-3.B — never a global scan). A pure optimization: the worker
      // may have already finished, in which case deliver() still converts the reply. Scoped by jobToWorker,
      // which only ever holds CURRENTLY-in-flight jobIds, so this can't target a worker's NEXT (live) job.
      if (this.cancelFlags) {
        const workerIndex = this.jobToWorker.get(req.jobId);
        if (workerIndex !== undefined) this.cancelFlags.requestCancel(workerIndex, req.jobId);
      }
      return;
    }
    if (req.type === "addModels") {
      // Model B: bump the shared model epoch FIRST, so every build enqueued after this carries the new epoch
      // and a worker won't mesh it before these models land (preserving Model A's per-worker-FIFO ordering
      // across the SAB + postMessage channels). The epoch rides the message; Model A leaves it undefined.
      //
      // WRK-2 (Model A + workStealing): Model A has no SAB MODEL_EPOCH word, so bump the pool-side scalar
      // `modelEpochA` here and ship it on the broadcast — the postMessage-channel analogue of Model B's
      // bump-before-enqueue. A build that enters post() AFTER this captures this same value, so a STOLEN
      // build gates until these models land (never AIR). Flag-off Model A keeps `epoch` undefined ⇒ no gate.
      const epoch = this.queue
        ? this.queue.bumpModelEpoch()
        : this.workStealing
          ? ++this.modelEpochA
          : undefined;
      const msg = epoch !== undefined ? { ...req, epoch } : req;
      for (const w of this.workers) w.postMessage(msg); // every worker grows its own palette map
      return; // inline mode: no-op (the inline mesher uses the live main-thread provider directly)
    }
    if (req.type === "init") return; // sent to workers at construction

    if (this.inlineMesh) {
      if (req.type === "build") {
        const mesh = this.inlineMesh;
        const { snapshot, jobId } = req;
        queueMicrotask(() => {
          const { payload } = serializeBuildOutput(mesh(snapshot));
          this.deliver({ type: "buildDone", jobId, payload });
        });
        return;
      }
      // req.type === "sort": inline mode has no off-thread sorter — the index-only translucent re-sort
      // runs on the main thread in TerrainRenderer. Fail loudly rather than silently drop it (TRAP 16.A).
      throw new Error(`WorkerPool(inline): unsupported request "${req.type}" — translucent re-sorting runs on the main thread, not the pool`);
    }

    if (this.queue && req.type === "build") {
      // Model B: enqueue the build into the shared ring (zero-copy snapshot). If the slot pool is full,
      // mesh THIS one on the main thread so the section still presents — never a silent drop (TRAP 16.A).
      if (!this.queue.enqueue(req.jobId, req.snapshot)) {
        const jobId = req.jobId;
        const { payload } = serializeBuildOutput(this.fallbackMesh!(req.snapshot));
        queueMicrotask(() => this.deliver({ type: "buildDone", jobId, payload }));
      }
      return;
    }

    // Worker dispatch (Model A builds, AND `sort` jobs on BOTH models — the SAB ring carries only builds;
    // the worker's onmessage handles `sort` while it drains, since waitAsync keeps the channel live). The
    // pool always has ≥1 worker (size floored at 1), so an empty pool means worker spawning failed — fail
    // loudly instead of silently dropping (which leaves the section un-presented + leaks inFlight; TRAP 16.A).
    if (this.workers.length === 0) throw new Error("WorkerPool: no workers to dispatch to (spawn failed)");
    // ABP-3: Model-A BUILDs can go through the work-stealing deque (an idle worker pulls the next build, so
    // a skewed burst doesn't head-of-line-block on one round-robin victim). `sort` stays on round-robin
    // (latency-sensitive). Off ⇒ the exact rr++ path for everything. SAFETY: a deferred build is posted to a
    // worker only AFTER any `addModels` broadcast has already gone to every worker, and the palette only ever
    // grows — so a stolen build always meshes against a palette that's a superset of what it needs (never AIR;
    // the same "addModels posted before the build" discipline the rr path relies on, only strengthened).
    if (this.workStealing && req.type === "build") {
      // WRK-2: CAPTURE the current modelEpochA AT ENQUEUE (TRAP WRK-2.C — never the pool's current epoch at
      // pump time). addModels is always broadcast before a build enters post() (syncWorkerModels posts it
      // synchronously, then the next scheduler.tick dispatches the build), so the captured value is exactly
      // the highest addModels this build needs — and no more (a LATER addModels must NOT make it wait). The
      // executing worker gates on this stamped epoch regardless of which worker ends up stealing the build.
      this.pendingBuilds.push({ ...req, epoch: this.modelEpochA });
      this.pumpWorkers();
      return;
    }
    // Round-robin: capture the target index BEFORE the post-increment so cancel() targets the right slot.
    const idx = this.rr++ % this.workers.length;
    if (this.cancelFlags && req.type === "build") this.jobToWorker.set(req.jobId, idx);
    this.workers[idx].postMessage(req);
  }

  /** ABP-3: hand queued builds to idle workers (≤1 in-flight build each). Called on post + on each build
   *  reply, so a fast worker drains the backlog while a slow one stays on its single build. */
  private pumpWorkers(): void {
    for (let i = 0; i < this.workers.length && this.pendingBuilds.length > 0; i++) {
      if (this.busy[i]) continue;
      this.busy[i] = true;
      const req = this.pendingBuilds.shift()!;
      // WRK-3: record which worker stole this build so cancel() targets the exact holding slot (the stolen
      // index is `i`, not a round-robin counter). Only `build` requests reach pendingBuilds.
      if (this.cancelFlags && req.type === "build") this.jobToWorker.set(req.jobId, i);
      this.workers[i].postMessage(req);
    }
  }

  cancel(jobId: number): void {
    this.post({ type: "cancel", jobId });
  }

  /**
   * Model B (INFRA-3): flag a still-queued SAB build as superseded so a worker SKIPS meshing it (reclaiming
   * the snapshot slot + the mesh cycles). No-op for inline / Model A — they have no shared ring, and their
   * stale builds are already dropped main-side by the Scheduler's generation check (the backstop here too).
   */
  cancelSharedJob(jobId: number): void {
    this.queue?.cancelByJobScan(jobId);
  }

  /**
   * WRK-3: supersede a PRIOR in-flight build whose section was re-dirtied (so its generation is provably
   * older than the just-bumped current — TRAP WRK-3.C: only ever the prior, never the current). Routes to
   * Model B's ring cancel AND Model A's cooperative cancel through ONE call, so the Scheduler stays agnostic.
   * Both are HINTS — the generation stale-drop + the main-side `cancelled` set remain the authoritative
   * backstops. No-op for Model A when `cooperativeCancelA` is off (cancelFlags null) ⇒ byte-identical.
   */
  supersede(jobId: number): void {
    this.cancelSharedJob(jobId); // Model B ring cancel (no-op for Model A — no ring)
    if (this.cancelFlags) this.cancel(jobId); // Model A cooperative cancel (no-op for Model B — cancelFlags null)
  }

  dispose(): void {
    for (const w of this.workers) w.terminate();
    this.workers.length = 0;
    this.busy.length = 0; // ABP-3: drop the work-stealing bookkeeping
    this.pendingBuilds.length = 0;
    this.modelEpochA = 0; // WRK-2: a re-created pool starts at epoch 0 (matches a fresh worker's appliedModelEpoch)
    this.jobToWorker.clear(); // WRK-3: drop the cancel-scoping map
    this.cancelled.clear(); // drop any pending main-side cancels (no jobId outlives the pool)
    this.rr = 0; // start a re-created pool's round-robin from worker 0 (no stale offset)
    this.handler = null;
  }

  /** Drop responses for cancelled jobs (converting to `cancelled`) before forwarding. */
  private deliver(res: WorkerResponse, workerIndex?: number): void {
    // ABP-3: a BUILD reply (done or cancelled) frees that worker's single in-flight slot → pump the next
    // queued build onto it. `sort` replies don't occupy a steal slot, so they don't free one. workerIndex is
    // undefined for the inline path (no real worker).
    if (this.workStealing && workerIndex !== undefined && (res.type === "buildDone" || res.type === "cancelled")) {
      this.busy[workerIndex] = false;
      this.pumpWorkers();
    }
    // WRK-3: a build's reply (done OR cancelled) retires its jobToWorker entry, so the map only ever holds
    // CURRENTLY-in-flight jobIds — a later cancel can never write a CANCEL_REQUEST scoped to a finished job's
    // slot (the worker would by then be on its next, live build). Cheap no-op when the feature is off.
    if (this.cancelFlags && (res.type === "buildDone" || res.type === "cancelled")) this.jobToWorker.delete(res.jobId);
    if ((res.type === "buildDone" || res.type === "sortDone") && this.cancelled.has(res.jobId)) {
      this.cancelled.delete(res.jobId);
      this.handler?.({ type: "cancelled", jobId: res.jobId });
      return;
    }
    this.handler?.(res);
  }
}
