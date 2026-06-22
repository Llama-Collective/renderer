// Worker entrypoint — off-thread terrain meshing. RENDERER_OPTIMIZATION_PLAN.md Phase 5; RENDERER_PLAN §6, §16.
//
// On `init` it builds a `BakedModelProvider` from the main thread's TRANSFERRED baked palette models
// (see MeshInit — `BakedBlockModel` is pure data, so the worker re-bakes NOTHING and its geometry is
// byte-identical to inline meshing; proven in MeshInit.test). On `build` it meshes the snapshot and posts
// a transferable `BuildPayload` back. No GPU access EVER happens here (meshing needs no GL).
//
// Stale-drop is the MAIN thread's job (every message carries `generation`); the worker also honors a
// `cancel` arriving before it processes the matching `build`. LIMITATIONS (tracked in P5c):
//  - Fluids (water/lava) need their `FluidContext` serialized — not yet in `init` — so this meshes terrain
//    only; fluid sections must stay on the inline path until that lands.
//  - New block ids placed after `init` aren't in the transferred palette (→ null/air) until the main thread
//    pushes incremental model additions.

import { meshSection, MESH_CANCELLED, type MeshOptions } from "../mesh/SectionMesher";
import { WorkerCancelFlags } from "./CancelFlags";
import type { BakedBlockModel } from "../mesh/model/BakedBlockModel";
import type { FluidContext } from "../mesh/FluidMesher";
import { serializeBuildOutput } from "./BuildOutput";
import { fluidsFromSerialized, type MeshInitPayload, type MeshModels } from "./MeshInit";
import type { WorkerRequest } from "./protocol";
import { SharedMeshQueue } from "./SharedMeshQueue";
import { computeSortIndex, deserializeQuads } from "../mesh/SortTrigger";

// Per-section budget for the off-thread DYNAMIC topological re-sort (ms); over budget → the distance-sort
// fallback. Matches the inline budget (TerrainRenderer TOPO_BUDGET_MS) so output is the same off-thread.
const SORT_BUDGET_MS = 2;

interface WorkerScope {
  onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(message: unknown, transfer: Transferable[]): void;
}
const ctx = self as unknown as WorkerScope;

// `Atomics.waitAsync` is Baseline at runtime (everywhere SAB + cross-origin isolation are) but only typed in
// lib es2024+; declare its shape locally rather than bump the whole lib target.
type AtomicsWaitAsync = (ta: Int32Array, index: number, value: number, timeout?: number) =>
  | { async: true; value: Promise<"ok" | "timed-out"> }
  | { async: false; value: "not-equal" | "timed-out" };
const waitAsync = (Atomics as unknown as { waitAsync: AtomicsWaitAsync }).waitAsync;

// Mutable model map so incremental `addModels` (newly-placed block ids) grows the palette in place. The map
// is owned solely by this single-threaded worker, so `addModels` (arriving on the message channel BETWEEN
// drain iterations — waitAsync keeps the channel live) updates it with no race vs the SAB drain loop.
const models = new Map<number, BakedBlockModel | null>();
const provider = { get: (id: number) => models.get(id) ?? null };
let inited = false;
let fluids: FluidContext | undefined;
// WRK-3 (Model A, default-off): this worker's view of its OWN slot in the cooperative-cancel SAB. Null when
// the feature is off / SAB absent (a no-op shim ⇒ the `build` path below is byte-identical to today).
let cancelFlags: WorkerCancelFlags | null = null;
// FS-5 / SL-4: mesh-time toggles from init, applied to every section so off-thread output matches inline.
let meshOptions: MeshOptions | undefined;
// Model B ordering gate: the highest addModels epoch this worker has merged. A SAB job stamped with a higher
// epoch must wait for its addModels postMessage (already in flight — main posts it BEFORE enqueuing the job)
// before meshing, else a newly-minted block id would mesh as AIR (matches Model A's per-worker-FIFO order).
let appliedModelEpoch = 0;

/** Yield a macrotask so a pending `addModels` postMessage (which bumps appliedModelEpoch) gets delivered. */
function nextMacrotask(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Model B drain loop: mesh jobs out of the SHARED ring (zero-copy snapshot views), posting each result back
 * by transfer. Sleeps on the notify futex via `Atomics.waitAsync` (NON-blocking — TRAP 5.D: blocking
 * `Atomics.wait` would freeze the thread AND stall init/addModels delivery + risk self-deadlock). Reading
 * the notify count BEFORE draining closes the lost-wakeup window: a job enqueued during the drain bumps the
 * count, so the subsequent waitAsync returns "not-equal" immediately and we loop again.
 */
async function drainShared(queue: SharedMeshQueue): Promise<void> {
  const notify = queue.notifyView;
  const idx = queue.notifyIndex;
  for (;;) {
    const expected = Atomics.load(notify, idx);
    let job = queue.dequeue();
    while (job !== null) {
      // INFRA-3: a superseded build was flagged before we claimed it — release the slot and report
      // `cancelled` WITHOUT meshing (skip the epoch wait too: we won't use the result). Exactly one
      // release, no `buildDone`; the main side retires the in-flight entry on the `cancelled` reply.
      if (job.cancelled) {
        queue.release(job.slot, job.estimateBytes);
        ctx.postMessage({ type: "cancelled", jobId: job.jobId }, []);
        job = queue.dequeue();
        continue;
      }
      // Ordering gate: don't mesh ahead of this job's models. The addModels postMessage that satisfies
      // `enqueueEpoch` was posted BEFORE this job was enqueued, so it's already queued — a macrotask yield
      // delivers it (bumping appliedModelEpoch). Bounded: guaranteed in flight ⇒ no deadlock.
      while (job.enqueueEpoch > appliedModelEpoch) await nextMacrotask();
      const out = meshSection(job.snapshot, provider, fluids, meshOptions);
      // Release the slot the moment meshing has read it — meshSection retains no reference to the snapshot's
      // (shared) blocks, so the slot is safe to reuse before we serialize/post the (freshly-allocated) result.
      queue.release(job.slot, job.estimateBytes);
      const { payload, transfer } = serializeBuildOutput(out);
      ctx.postMessage({ type: "buildDone", jobId: job.jobId, payload }, transfer);
      job = queue.dequeue();
    }
    const r = waitAsync(notify, idx, expected);
    if (r.async) await r.value; // resolves on the next Atomics.notify (or immediately if count already moved)
  }
}

ctx.onmessage = async (e: MessageEvent<WorkerRequest>): Promise<void> => {
  const req = e.data;
  switch (req.type) {
    case "init": {
      const payload = req.config as MeshInitPayload;
      for (const [id, m] of payload.models) models.set(id, m);
      fluids = payload.fluids ? fluidsFromSerialized(payload.fluids) : undefined;
      meshOptions = payload.meshOptions;
      // WRK-3: attach this worker's slot in the cooperative-cancel SAB (Model A, default-off). Absent
      // ⇒ no-op shim (cancelFlags stays null) ⇒ the `build` case below is byte-identical to today.
      cancelFlags = WorkerCancelFlags.attach(req.cancelFlags, req.workerIndex ?? 0);
      inited = true;
      // Model B: a `shared` payload means jobs arrive via the SAB ring — start draining it. (Model A leaves
      // `shared` undefined and feeds builds through the `build` message branch below.)
      if (req.shared) void drainShared(SharedMeshQueue.attach(req.shared));
      return;
    }
    case "addModels":
      for (const [id, m] of req.models as MeshModels) models.set(id, m);
      if (req.epoch !== undefined && req.epoch > appliedModelEpoch) appliedModelEpoch = req.epoch; // Model B gate
      return;
    case "cancel":
      return; // no-op: the pool never sends cancel to workers — it converts the result main-side (no leak)
    case "build": {
      if (!inited) throw new Error("mesher.worker: received build before init (TRAP 16.A)");
      const jobId = req.jobId;
      // WRK-2 (Model A + work-stealing): the same model-epoch ordering gate the SAB drain loop uses (line 90).
      // `req.epoch` is the pool's `modelEpochA` captured when this build was enqueued — the highest addModels
      // it needs. If a STEAL routed it here before THIS worker merged those models, wait: the satisfying
      // addModels postMessage was broadcast BEFORE the build (syncWorkerModels discipline), so it's already
      // queued ahead — a bounded `nextMacrotask()` yield (NOT a blocking spin — TRAP WRK-2.B, which would
      // freeze the message pump and deadlock) lets it deliver and bump `appliedModelEpoch`. Undefined epoch
      // (rr path / flag-off) ⇒ the loop never runs ⇒ byte-identical synchronous mesh below.
      if (req.epoch !== undefined) {
        while (req.epoch > appliedModelEpoch) await nextMacrotask();
      }
      // WRK-3: when cooperative cancel is wired, publish the jobId this worker is meshing (so a CANCEL_REQUEST
      // matching it can be honored) and poll it per-Y-layer inside meshSection. The poll is scoped to THIS
      // worker's slot (per-worker ACTIVE_JOBID/CANCEL_REQUEST equality), so it can only ever cancel the job
      // this worker holds RIGHT NOW — never a later, live one. Off ⇒ cancelFlags null ⇒ isCancelled undefined
      // ⇒ the exact byte-identical synchronous mesh below.
      if (cancelFlags) {
        const flags = cancelFlags;
        flags.setActive(jobId);
        const out = meshSection(req.snapshot, provider, fluids, meshOptions, () => flags.isCancelled(jobId));
        flags.clearActive(); // ACTIVE_JOBID=0 between jobs ⇒ a stale request matches nothing (monotonic ids)
        if (out === MESH_CANCELLED) {
          // Cancelled mid-mesh: report WITHOUT serializing/transferring (the whole point — skip the clone the
          // result would cost). EXACTLY ONE reply per build — no buildDone follows. The main side retires the
          // in-flight entry on this `cancelled`, identical to Model B's drainShared cancel path.
          ctx.postMessage({ type: "cancelled", jobId }, []);
          return;
        }
        const { payload, transfer } = serializeBuildOutput(out);
        ctx.postMessage({ type: "buildDone", jobId, payload }, transfer);
        return;
      }
      const out = meshSection(req.snapshot, provider, fluids, meshOptions);
      const { payload, transfer } = serializeBuildOutput(out);
      ctx.postMessage({ type: "buildDone", jobId, payload }, transfer);
      return;
    }
    case "sort": {
      // Off-thread translucent re-sort (TR-1): the section's quads + the section-local camera came in the
      // message; compute the new INDEX order (INDICES ONLY — TRAP 12.A) and post it back. The Scheduler
      // generation-matches the result before applying it (TRAP 5.B); a late result just shows the old index.
      const quads = deserializeQuads(req.sortData);
      const indexData = computeSortIndex(quads, req.cameraPos, SORT_BUDGET_MS);
      ctx.postMessage({ type: "sortDone", jobId: req.jobId, payload: { sectionKey: req.sectionKey, generation: req.generation, indexData: indexData.buffer } }, [indexData.buffer]);
      return;
    }
  }
};

export {};
