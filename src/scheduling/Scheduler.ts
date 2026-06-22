// The frame conductor: coalesce dirty sections → dispatch (mesh) within a CPU budget → upload/commit
// within the upload budget. RENDERER_PLAN.md §3 (frame loop), §16; AUDIT N1, SCHED-1/3, U4.
//
// This is the main-thread DISPATCHER. Today it meshes INLINE (synchronously) during `tick`; Phase 5
// swaps the inline `meshSection` call for a real Web-Worker post + drain, behind this same seam.
//
// It enforces the Stable Presentation Invariant by construction:
//  - N1: a section is queued at most once; re-dirties coalesce + supersede the queued generation, so the
//    budget never runs a doomed stale build (TRAP 1.A).
//  - SCHED-3: the per-frame CPU mesh budget defers a burst (explosion/piston) to later frames — but at
//    least one task always dispatches (no starvation).
//  - The build is always meshed at the section's CURRENT generation; the UploadScheduler then commits
//    generation-matched within its byte/time budget (U4). Old presented geometry stays drawable until a
//    current-gen build is committed (TRAP 4.A/4.B).

import { meshSection, type BakedModelProvider, type MeshOptions } from "../mesh/SectionMesher";
import { isAllAirCore, type SnapshotSource } from "../world/SnapshotSource";
import { Lifecycle, UpdateType, upgradeUpdateType } from "../world/RenderSection";
import type { SectionStore } from "../world/SectionStore";
import type { SectionKey } from "../world/SectionKey";
import { makeBuildOutput, type SectionBuildOutput } from "../workers/BuildOutput";
import { makeSortOutput, type SortOutput } from "../workers/SortOutput";
import { TaskQueue } from "../workers/TaskQueue";
import type { WorkerPool } from "../workers/WorkerPool";
import type { WorkerResponse } from "../workers/protocol";
import type { FluidContext } from "../mesh/FluidMesher";
import { DirtyReason } from "../types";
import type { Generation, Vec3, Vec3i } from "../types";

/** The commit sink the scheduler feeds (the UploadScheduler — typed structurally to avoid a render/ dep). */
export interface CommitSink {
  submitBuild(out: SectionBuildOutput): void;
  /** Stage an off-thread translucent re-sort result (TR-1) — index-only, generation-matched at commit
   *  (TRAP 5.B). Optional so test mocks can omit it. */
  submitSort?(out: SortOutput): void;
  /** `committed` = sections whose GPU data changed; `discardedStale` = generation-stale builds dropped at
   *  `acceptBuild` (optional so test mocks can omit it). Both feed the Phase-P jobs metric. */
  drainAndCommit(currentFrame: number): { committed: number; discardedStale?: number };
}

export interface SchedulerDeps {
  store: SectionStore;
  snapshots: SnapshotSource;
  provider: BakedModelProvider;
  uploads: CommitSink;
  fluids?: FluidContext;
  /** Per-frame CPU mesh budget (ms). Generous by default so a handful of edits mesh same-frame; a big
   *  burst spreads across frames (SCHED-3). 0 / Infinity → no deferral (mesh everything each tick). */
  meshBudgetMs?: number;
  /** ABP-1 (optional): read at the top of each `tick` to size the mesh budget adaptively (e.g. off a
   *  frame-time EMA). Absent ⇒ the fixed `meshBudgetMs` literal is used (flag-off keeps the exact path). */
  meshBudgetOf?: () => number;
  /** WRK-1 (optional): a per-frame QUEUED-DURATION admission gate on the POOL dispatch path. A burst
   *  (explosion ⇒ ~40 sections dirty in one tick) otherwise `buildSnapshot`s + structured-clones every
   *  ~23 KB snapshot and posts every job in one frame (~920 KB clone spike on Model A). With the gate on,
   *  the tick dispatches only while the in-flight bytes estimate stays under `workers * frameMs * bytesPerMs`
   *  and DEFERS the rest into the aging `TaskQueue` (never dropped). Absent ⇒ the EXACT current dispatch
   *  path (byte-identical), mirroring the `meshBudgetOf` precedent. The check sits BEFORE `buildSnapshot`
   *  so it bounds the clone spike, not just the dispatch count; the inline/immediate path is never gated.
   *
   *  - `frameMsOf()` — last-frame time (e.g. a clamped frame-EMA average) for the `workers * frameMs` budget.
   *  - `workers`     — pool worker count (inline ⇒ 1).
   *  - `bytesPerMs`  — a measured constant (NOT a learned regression): snapshot bytes one worker meshes per ms.
   *  - `queuedBytesOf()` — Model B ONLY: the SAB ring's own in-flight byte estimate (`SharedMeshQueue
   *    .queuedBytes()`), used as the single source of truth so we don't double-count. Absent ⇒ the Model-A
   *    main-thread `inFlightBytes` accumulator is used. */
  admissionGate?: {
    frameMsOf: () => number;
    workers: number;
    bytesPerMs: number;
    queuedBytesOf?: () => number;
  };
  /** Injectable clock for deterministic tests. */
  now?: () => number;
  /**
   * If present, meshing runs OFF-THREAD through this pool (Phase 5): `tick` DISPATCHES queued snapshots
   * and the pool's reply drains asynchronously (generation stale-drop → `submitBuild`). Absent ⇒ inline
   * synchronous meshing on the main thread (the default — bit-identical, no mesh→commit latency). The pool
   * owns the mesher: an inline pool wraps `meshSection`; a worker pool meshes off-thread (see MeshInit).
   */
  pool?: WorkerPool;
  /** FS-5 / SL-4: mesh-time toggles (partitionFacing, smoothLighting) applied to inline meshing here AND
   *  shipped to the worker via MeshInit, so the inline + off-thread paths agree. Absent ⇒ the defaults. */
  meshOptions?: MeshOptions;
  /** Render-on-demand: invoked when an OFF-THREAD reply stages a commit (`onWorkerMessage` → submitBuild /
   *  submitSort). The worker reply lands via postMessage with no other wake path, so the dynamic-FPS loop —
   *  which may have idled the instant `inFlight` drained — needs a nudge to run the next `drainAndCommit` and
   *  draw the result. Absent ⇒ no-op (the continuous loop draws every frame anyway). */
  onWork?: () => void;
}

/** Result of one frame's dispatch (for the HUD / Phase-P metrics). */
export interface TickResult {
  meshed: number;
  deferred: number;
  /** Sections whose GPU data changed this frame — the viewer refreshes its draw list iff > 0. */
  committed: number;
  /** Generation-stale builds dropped this frame — off-thread drops (an in-flight build superseded by a
   *  re-dirty) PLUS upload-time `acceptBuild` stale discards. Feeds `instrument.job.discarded` (Bug 3:
   *  without surfacing it the counter stayed pinned at 0 and the Phase-1 DONE-WHEN was unverifiable). */
  discarded: number;
}

const defaultNow = (): number => (typeof performance !== "undefined" ? performance.now() : Date.now());

export class Scheduler {
  readonly queue = new TaskQueue();
  private readonly meshBudgetMs: number;
  private readonly now: () => number;
  private frame = 0;
  /** Cumulative counters (Phase-P jobs/tick): unique sections dispatched, and tasks coalesced away. */
  stats = { meshed: 0, coalesced: 0, deferred: 0 };
  /** Off-thread (Phase 5): the pool meshing runs through, and in-flight build jobs (jobId → section). */
  private readonly pool: WorkerPool | null;
  private nextJobId = 1;
  private readonly inFlight = new Map<number, SectionKey>();
  /** INFRA-3: the LATEST in-flight build jobId per section, so a re-dispatch (which always bumps the
   *  generation ⇒ the prior build is now stale) can flag the prior job for SAB cancellation. */
  private readonly sectionInFlight = new Map<SectionKey, number>();
  /** ABP-2: sections flagged for SAME-FRAME presentation (near, in-frustum edits) → meshed INLINE at the
   *  top of `tick` (never via the async pool) so the same-tick commit swaps `presented`. Maps key → the
   *  dirty reason to snapshot with. Capped per tick (a bulk edit's excess falls back to the normal path). */
  private readonly immediate = new Map<SectionKey, DirtyReason>();
  /** Off-thread stale-drops since the last `tick` (an in-flight build whose section advanced a generation
   *  while the worker meshed). Folded into `TickResult.discarded` and reset each tick (Bug 3). */
  private staleDropped = 0;
  /** WRK-1: the admission gate config (null ⇒ no gate ⇒ the exact current dispatch path). */
  private readonly admission: SchedulerDeps["admissionGate"] | null;
  /** WRK-1 (Model A ONLY): running estimate of snapshot bytes dispatched to the pool but not yet retired.
   *  Added on post (`snap.blocks.byteLength`), subtracted on EVERY terminal path (buildDone / cancelled /
   *  stale-drop) so an idle scene returns to 0 — leaking it would over-throttle the gate permanently. For
   *  Model B the SAB ring's `queuedBytes()` is the source of truth instead (see `admissionGate.queuedBytesOf`),
   *  so this accumulator is NOT touched on the Model-B post (no double-count). */
  private inFlightBytes = 0;
  /** WRK-1 (Model A): per-jobId snapshot bytes, so the right amount is subtracted from `inFlightBytes` when
   *  that job retires — retired in the SAME delete path as `inFlight` (every terminal reply). */
  private readonly jobBytes = new Map<number, number>();

  constructor(private readonly deps: SchedulerDeps) {
    this.meshBudgetMs = deps.meshBudgetMs ?? 8;
    this.now = deps.now ?? defaultNow;
    this.pool = deps.pool ?? null;
    this.admission = deps.admissionGate ?? null;
    if (this.pool) this.pool.onMessage((res) => this.onWorkerMessage(res));
  }

  /**
   * Mark a section dirty and enqueue a coalesced rebuild (N1). A section has AT MOST ONE queued task:
   * `enqueue` coalesces in place via a full OR-join — newest generation, strongest kind, HIGHEST priority,
   * earliest enqueue frame. Routing re-dirties through the SAME `enqueue` (not a generation-only supersede)
   * means a re-dirty carrying a higher priority correctly RAISES the queued task instead of inheriting the
   * first dirty's stale priority — the priority-inversion the old `supersede` branch skipped. Returns the gen.
   */
  markDirty(key: SectionKey, reason: DirtyReason, priority = 0, immediate = false): Generation {
    const gen = this.deps.store.markDirty(key, reason);
    const section = this.deps.store.getOrCreate(key);
    // ABP-2: a near in-frustum edit asks for same-frame presentation — record it for the inline immediate
    // drain at the top of the next tick. It STILL enqueues below (so if the per-tick cap defers it, the
    // normal budgeted/async path is the backstop). Last reason wins on coalesce.
    if (immediate) this.immediate.set(key, reason);
    // LM-2: a Light dirty enqueues a light-only re-bake (cheaper than a full Rebuild but still a re-mesh);
    // OR-joined so a Light+Geometry coalesce keeps the stronger Rebuild. Both dispatch a "build" job — the
    // mesher bakes the new light from the snapshot; the in-place vs full commit is decided at upload time.
    const kind = reason === DirtyReason.Light ? UpdateType.Light : UpdateType.Rebuild;
    section.pendingUpdateType = upgradeUpdateType(section.pendingUpdateType, kind);
    if (section.queued) this.stats.coalesced++; // already queued → enqueue() coalesces in place (SCHED-1)
    else section.queued = true;
    this.queue.enqueue({ sectionKey: key, generation: gen, kind: "build", reason, basePriority: priority, enqueuedFrame: this.frame });
    return gen;
  }

  /**
   * Dispatch queued mesh jobs within the CPU budget (mesh inline, submit to the commit sink), then drain
   * the upload budget. Returns how many sections meshed and how many remain deferred.
   */
  tick(currentFrame: number, cameraPos?: Vec3): TickResult {
    this.frame = currentFrame;
    // SCHED-4: dispatch near-the-camera sections first within the budget (near-first tie-break in pop).
    const cameraSection: Vec3i | undefined = cameraPos
      ? [Math.floor(cameraPos[0]) >> 4, Math.floor(cameraPos[1]) >> 4, Math.floor(cameraPos[2]) >> 4]
      : undefined;
    // ABP-2: mesh near in-frustum edits INLINE first (never via the pool) so the same-tick drainAndCommit
    // presents them THIS frame. Capped, generation-correct, and routed through submitBuild like every other
    // path — only the present TIMING changes, never the bytes.
    let meshed = this.drainImmediate();
    // ABP-1: an adaptive provider (if wired) sizes this frame's CPU mesh budget; absent ⇒ the fixed literal.
    const meshBudgetMs = this.deps.meshBudgetOf ? this.deps.meshBudgetOf() : this.meshBudgetMs;
    // WRK-1: per-frame worker-effort admission budget. Floored at 1 so a pathological config (workers or
    // bytesPerMs misconfigured to 0) can't zero the budget and starve — the `meshed>0` guard below admits
    // ≥1 regardless. Absent gate ⇒ Infinity ⇒ the loop runs exactly as today (byte-identical).
    const budgetBytes = this.admission
      ? Math.max(1, this.admission.workers * this.admission.frameMsOf() * this.admission.bytesPerMs)
      : Infinity;
    // Frame-start in-flight estimate: Model B reads the SAB ring's own counter (single source of truth);
    // Model A uses the main-thread accumulator. Climbs within the frame as we post so the gate trips mid-burst.
    let queuedBytes = this.admission ? (this.admission.queuedBytesOf?.() ?? this.inFlightBytes) : 0;
    const start = this.now();
    while (this.queue.size > 0) {
      // SCHED-3: stop once over budget — but always dispatch at least one (no starvation, TRAP 16.A).
      if (meshed > 0 && this.now() - start >= meshBudgetMs) break;
      const task = this.queue.pop(currentFrame, cameraSection);
      if (!task) break;
      const section = this.deps.store.get(task.sectionKey);
      if (!section || section.lifecycle === Lifecycle.Disposing) continue;
      // WRK-1: POOL-ONLY queued-duration admission gate. This MUST sit BEFORE `buildSnapshot` (so it bounds
      // the ~23 KB clone spike, not just the dispatch count — the whole point) AND before the
      // `section.queued=false` / `pendingUpdateType` clear (so a re-deferred task stays consistently queued).
      // `meshed>0` guarantees we always admit ≥1 this frame (no starvation). The gated task was already
      // popped (deleted from the queue) — RE-ENQUEUE it with its ORIGINAL `enqueuedFrame` so AGE_WEIGHT still
      // ages it (using `currentFrame` would reset its age and let near-camera churn starve it). Never drop it.
      if (this.pool && this.admission && meshed > 0 && queuedBytes >= budgetBytes) {
        this.queue.enqueue(task);
        break;
      }
      section.queued = false;
      section.pendingUpdateType = UpdateType.None; // clear-on-dispatch (N1)
      // Mesh the CURRENT generation (a re-dirty that arrived while queued already bumped it — supersede).
      const gen = section.generation;
      const snap = this.deps.snapshots.buildSnapshot(task.sectionKey, gen, task.reason);
      if (isAllAirCore(snap) && !section.presented) continue; // #15 — empty section, nothing to show
      if (this.pool) {
        // Phase 5: dispatch off-thread. The snapshot carries `generation`; the reply is stale-dropped in
        // onWorkerMessage. submitBuild/commit happen when the result arrives (≤1 frame later for workers).
        const jobId = this.nextJobId++;
        // INFRA-3 / WRK-3: this section is being re-dispatched, so any prior in-flight build for it is
        // superseded (the re-dirty bumped the generation ⇒ the prior is provably an OLDER generation than the
        // just-bumped current — TRAP WRK-3.C: only ever the prior, never the current). `supersede` routes to
        // Model B's ring D_CANCEL AND Model A's cooperative cancel (default-off; no-op when off) through one
        // call. The generation stale-drop in onWorkerMessage remains the authoritative backstop.
        const prior = this.sectionInFlight.get(task.sectionKey);
        if (prior !== undefined) this.pool.supersede(prior);
        this.inFlight.set(jobId, task.sectionKey);
        this.sectionInFlight.set(task.sectionKey, jobId);
        this.pool.post({ type: "build", jobId, snapshot: snap });
        // WRK-1: account the dispatched snapshot bytes so the gate trips mid-burst. The local `queuedBytes`
        // climbs on EVERY model. The persistent `inFlightBytes` accumulator is Model A ONLY — when a Model-B
        // `queuedBytesOf` is wired, the SAB ring's `queuedBytes()` is the single source of truth, so adding
        // here too would double-count (the ring already counts the enqueue) and over-throttle permanently.
        if (this.admission) {
          const bytes = snap.blocks.byteLength;
          queuedBytes += bytes;
          if (!this.admission.queuedBytesOf) {
            this.inFlightBytes += bytes;
            this.jobBytes.set(jobId, bytes);
          }
        }
      } else {
        const out = meshSection(snap, this.deps.provider, this.deps.fluids, this.deps.meshOptions);
        this.deps.uploads.submitBuild(out);
      }
      meshed++;
    }
    const { committed, discardedStale } = this.deps.uploads.drainAndCommit(currentFrame);
    // Discarded = off-thread in-flight stale-drops (onWorkerMessage) + upload-time acceptBuild stale drops.
    const discarded = this.staleDropped + (discardedStale ?? 0);
    this.staleDropped = 0;
    this.stats.meshed += meshed;
    this.stats.deferred = this.queue.size;
    return { meshed, deferred: this.queue.size, committed, discarded };
  }

  /** Render-on-demand: true while any mesh job is queued OR off-thread in-flight. The dynamic-FPS loop reads
   *  this to keep drawing every frame until every pending build commits — an off-thread result lands ≤1
   *  frame later via the worker reply (which has no other path to wake an idle loop). A fully-drained
   *  scheduler reports false, so a settled scene can idle. */
  hasPendingWork(): boolean {
    return this.queue.size > 0 || this.inFlight.size > 0;
  }

  /** ABP-2 cap: at most this many sections mesh INLINE per tick for same-frame presentation, so a bulk
   *  near-edit can't re-introduce the main-thread stall — the excess falls back to the normal path. */
  private static readonly MAX_IMMEDIATE_PER_TICK = 4;

  /**
   * ABP-2: mesh the flagged near in-frustum edits INLINE (synchronously, never via the pool) and submit
   * them so the SAME tick's drainAndCommit presents them this frame. Meshes the CURRENT generation through
   * `submitBuild` → the upload budget, so the stale-drop + Stable Presentation Invariant still hold (only
   * the present timing changes, the bytes are identical to the async path). Removes each from the normal
   * queue to avoid a double-mesh. Capped; any excess simply leaves the immediate set and presents via the
   * normal budgeted/async path. Returns how many actually meshed.
   */
  private drainImmediate(): number {
    if (this.immediate.size === 0) return 0;
    const start = this.now();
    let done = 0;
    for (const [key, reason] of this.immediate) {
      if (done >= Scheduler.MAX_IMMEDIATE_PER_TICK) break; // cap → the rest fall to the normal path
      // P3: ALSO bound by elapsed time — on a heavy frame the remaining immediate edits fall back to the
      // async/deferred path instead of always paying up to MAX inline bakes (this path otherwise bypasses the
      // EMA/admission budget entirely). `done > 0` keeps the same-frame guarantee of at least ONE present.
      if (done > 0 && this.now() - start >= this.meshBudgetMs) break;
      const section = this.deps.store.get(key);
      if (!section || section.lifecycle === Lifecycle.Disposing) continue;
      const gen = section.generation; // mesh the CURRENT generation (a queued re-dirty already bumped it)
      const snap = this.deps.snapshots.buildSnapshot(key, gen, reason);
      if (isAllAirCore(snap) && !section.presented) continue; // empty + nothing shown → nothing to present
      section.queued = false;
      section.pendingUpdateType = UpdateType.None; // clear-on-dispatch (N1)
      this.queue.remove(key); // this section is meshed here — don't let the normal loop mesh it again
      // P1.6: this inline immediate mesh at the current generation will commit THIS tick, so any in-flight
      // WORKER build for the section would be stale-dropped at acceptBuild anyway — supersede it early to save
      // the wasted off-thread mesh (mirrors the normal dispatch path's supersede). No-op without a pool.
      const prior = this.sectionInFlight.get(key);
      if (prior !== undefined) { this.pool?.supersede(prior); this.sectionInFlight.delete(key); }
      const out = meshSection(snap, this.deps.provider, this.deps.fluids, this.deps.meshOptions);
      this.deps.uploads.submitBuild(out); // committed THIS tick by drainAndCommit ⇒ same-frame present
      done++;
    }
    this.immediate.clear(); // processed OR over-cap → both leave the set (leftovers present via the normal path)
    return done;
  }

  /**
   * Off-thread build result (Phase 5). Generation stale-drop (TRAP 4.C / §5): a build is only valid for
   * the section's CURRENT generation — a re-dirty that arrived while the worker meshed bumped it, so the
   * output is stale and its buffers are released. Otherwise it's submitted to the upload budget (U4); the
   * next `drainAndCommit` swaps `presented` generation-matched, preserving the Stable Presentation Invariant.
   */
  private onWorkerMessage(res: WorkerResponse): void {
    if (res.type === "sortDone") {
      // Off-thread re-sort result (TR-1). The payload is self-describing (sectionKey + generation), so it
      // needs no inFlight entry; `acceptSort` generation-matches it (TRAP 5.B) and drops a stale one. A
      // late result just means the old (committed) index showed for an extra frame — no pop.
      this.deps.uploads.submitSort?.(makeSortOutput(res.payload));
      this.deps.onWork?.(); // render-on-demand: wake the loop to drain this staged sort (it has no inFlight entry)
      return;
    }
    // buildDone or cancelled: both retire the in-flight entry for this jobId (and clear the per-section
    // latest-job pointer iff it still points here — a newer dispatch may already have replaced it).
    const key = this.inFlight.get(res.jobId);
    this.inFlight.delete(res.jobId);
    // WRK-1: retire the Model-A in-flight bytes on EVERY terminal path (buildDone, cancelled, AND the
    // stale-drop below all funnel through this delete) — subtracting only on buildDone would leak the
    // estimate upward on a burst of stale-drops/cancels and over-throttle the gate permanently.
    const bytes = this.jobBytes.get(res.jobId);
    if (bytes !== undefined) {
      this.jobBytes.delete(res.jobId);
      this.inFlightBytes -= bytes;
    }
    if (key !== undefined && this.sectionInFlight.get(key) === res.jobId) this.sectionInFlight.delete(key);
    if (res.type !== "buildDone") return; // cancelled (INFRA-3 SAB skip or main-side cancel) — nothing to commit
    if (key === undefined) return;
    const section = this.deps.store.get(key);
    const out = makeBuildOutput(res.payload);
    if (!section || section.lifecycle === Lifecycle.Disposing || out.generation !== section.generation) {
      out.dispose();
      this.staleDropped++; // surfaced via the next tick's TickResult.discarded (Bug 3)
      return;
    }
    this.deps.uploads.submitBuild(out);
    this.deps.onWork?.(); // render-on-demand: this reply emptied inFlight; wake the loop to commit + draw it
  }

  /** Sections still waiting to mesh (deferred across frames). */
  get pending(): number {
    return this.queue.size;
  }

  /** WRK-1 test-only: the Model-A in-flight snapshot-bytes estimate the admission gate throttles on. Exposed
   *  so a test can assert it returns toward 0 on every terminal path (stale-drop / cancel / buildDone) — a
   *  leak here would over-throttle the gate permanently on an idle scene. */
  get inFlightBytesEstimate(): number {
    return this.inFlightBytes;
  }
}
