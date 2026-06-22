// Model B substrate — a SharedArrayBuffer + Atomics meshing queue. RENDERER_OPTIMIZATION_PLAN.md Phase 5
// ("Queue substrate — Model B"); "Multithreading & Concurrency".
//
// This is the large-world tier of off-thread meshing. Model A (WorkerPool "worker" mode) structured-CLONES
// each section snapshot through postMessage; cloning the ~23 KB apron cube per job is the dominant cost when
// streaming thousands of sections. Model B eliminates it: snapshots live in a SHARED data pool that workers
// read in place (`ClonedChunkSection` analog), dispatched over a lock-free ring with a counting
// semaphore (so idle workers sleep and any worker can steal the next job — work-stealing).
//
// Concurrency model (all control state is in ONE Int32 SAB, accessed only via Atomics):
//   - JOB ring: a bounded MPSC-producer / MP-consumer queue of fixed flat descriptors. The main thread is
//     the sole producer; every worker is a consumer. A COUNTING SEMAPHORE (`JOB_COUNT`) makes claiming
//     wait-free and race-free: a consumer that wins `Atomics.sub(JOB_COUNT,1) > 0` owns exactly one job and
//     reads its index from `Atomics.add(JOB_HEAD,1)`. Producer writes the descriptor, THEN bumps the
//     semaphore — the seq-cst store synchronizes-with the consumer's seq-cst load, so the plain descriptor
//     + slot writes are visible to the consumer (JS memory model). FIFO; in-flight ≤ slotCap (bounded by
//     the free-list), and jobCap ≥ slotCap so the ring never overruns.
//   - FREE list: a symmetric semaphore ring of FREE snapshot-slot indices. The main thread ACQUIRES a slot
//     (pop) to write a snapshot into; the worker RELEASES it (push) after meshing. So the slot is never
//     reused while a worker still reads it.
//   - NOTIFY: a futex word the main thread `Atomics.notify`s on each enqueue; workers `Atomics.waitAsync`
//     on it (NOT blocking `wait` — TRAP 5.D: a worker must keep pumping postMessage for init/addModels, and
//     waitAsync sleeps just as efficiently without freezing the thread).
//   - DURATION_SUM: queued work estimate (snapshot bytes), for back-pressure.
//
// Stays main-thread-only (both models): GPU upload, the generation counter + stale-drop, UploadScheduler.
// The worker owns only pure meshing. Generation discipline is UNCHANGED — the descriptor carries it and the
// reply (still a transferred BuildPayload over postMessage) is stale-dropped at drain exactly as in Model A.

import type { SectionSnapshot } from "../world/SnapshotSource";
import { snapshotStride } from "../world/SnapshotSource";
import { sectionKey, type SectionKey } from "../world/SectionKey";
import type { DirtyReason, Generation, Vec3i } from "../types";

// ── Control-word indices (one shared Int32Array) ─────────────────────────────────────────────────────
const JOB_COUNT = 0; // semaphore: jobs available to claim
const JOB_HEAD = 1; // monotonic consumer cursor (mod jobCap)
const JOB_TAIL = 2; // monotonic producer cursor (mod jobCap)
const NOTIFY = 3; // futex: bumped per enqueue; workers waitAsync on it
const FREE_COUNT = 4; // semaphore: free snapshot slots available
const FREE_HEAD = 5; // free-ring consumer cursor (main acquires)
const FREE_TAIL = 6; // free-ring producer cursor (worker releases)
const DURATION_SUM = 7; // queued snapshot bytes (back-pressure estimate)
const MODEL_EPOCH = 8; // bumped on each addModels; stamped onto every enqueued job (ordering gate, below)
const CTRL_LEN = 9;

// ── Job descriptor layout (Int32 per field) ──────────────────────────────────────────────────────────
const D_JOBID = 0;
const D_GEN = 1;
const D_KEYX = 2;
const D_KEYY = 3;
const D_KEYZ = 4;
const D_APRON = 5;
const D_REASON = 6;
const D_SLOT = 7; // snapshot-pool slot index holding this job's bytes
const D_BLOCKS_BYTES = 8;
const D_LIGHT_BYTES = 9;
const D_BIOME_BYTES = 10;
const D_EPOCH = 11; // model-epoch at enqueue time — the worker must apply addModels up to this before meshing
const D_CANCEL = 12; // INFRA-3: set to 1 (main side, by jobId scan) when this queued build was superseded; the
//                      worker SKIPS meshing it at dequeue (releases the slot, posts `cancelled`). Best-effort —
//                      a job already past the dequeue read still meshes and is dropped by the gen stale-drop.
const JOB_FIELDS = 13;

function align4(n: number): number {
  return (n + 3) & ~3;
}

/** Round up to a power of two so ring cursors can index with `& (cap-1)` — which stays correct even after
 *  the monotonic Int32 cursor wraps past 2^31 (the two's-complement low bits are the right ring slot). */
function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/** Transferable SAB handles — structured-cloned by REFERENCE (shared, not copied) into each worker at init. */
export interface SharedMeshQueueHandles {
  ctrl: SharedArrayBuffer;
  jobs: SharedArrayBuffer;
  free: SharedArrayBuffer;
  data: SharedArrayBuffer;
  jobCap: number;
  slotCap: number;
  slotBytes: number;
}

/** A claimed job: a SectionSnapshot whose typed arrays VIEW the shared data slot (zero-copy), plus the slot
 *  index to release once meshing has read it. The snapshot must be fully consumed before `release`. */
export interface ClaimedJob {
  jobId: number;
  slot: number;
  estimateBytes: number;
  /** The model-epoch this job was enqueued at — the worker must have applied addModels through at least this
   *  epoch before meshing, or a newly-minted block id would mesh as AIR (the Model-A per-worker-FIFO
   *  ordering guarantee, reconstructed across the SAB + postMessage channels). */
  enqueueEpoch: number;
  /** INFRA-3: the main thread flagged this build as superseded (a re-dirty advanced the section's
   *  generation) before the worker claimed it. The worker releases the slot and posts `cancelled`
   *  WITHOUT meshing — reclaiming the slot + the mesh cycles `isCancelled()` would. */
  cancelled: boolean;
  snapshot: SectionSnapshot;
}

/** True iff the page can use Model B: SAB exists, the page is cross-origin isolated (TRAP 5.F), AND
 *  `Atomics.waitAsync` (the worker's non-blocking job wait) is present. Otherwise degrade to Model A. */
export function sharedMemoryAvailable(): boolean {
  return (
    typeof SharedArrayBuffer !== "undefined" &&
    (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true &&
    typeof (Atomics as unknown as { waitAsync?: unknown }).waitAsync === "function"
  );
}

/** Bytes one snapshot slot needs for the given apron: blocks (u32) + light (u8) + biomeTint (u32), each
 *  4-aligned so the worker can build typed-array views at the slot offset. */
export function slotBytesFor(apron: number): number {
  const cells = snapshotStride(apron) ** 3;
  return align4(cells * 4) + align4(cells) + align4(cells * 4); // blocks + light + biome
}

export class SharedMeshQueue {
  private readonly ctrl: Int32Array;
  private readonly jobs: Int32Array;
  private readonly free: Int32Array;
  private readonly data: Uint8Array;

  private constructor(
    private readonly h: SharedMeshQueueHandles,
    init: boolean,
  ) {
    this.ctrl = new Int32Array(h.ctrl);
    this.jobs = new Int32Array(h.jobs);
    this.free = new Int32Array(h.free);
    this.data = new Uint8Array(h.data);
    if (init) {
      // Seed the free-list with every slot index; the ring starts FULL (count == slotCap).
      for (let i = 0; i < h.slotCap; i++) this.free[i] = i;
      Atomics.store(this.ctrl, FREE_COUNT, h.slotCap);
      Atomics.store(this.ctrl, FREE_HEAD, 0);
      Atomics.store(this.ctrl, FREE_TAIL, 0);
    }
  }

  /** Allocate the SABs and return a main-thread queue. `apron` sizes each snapshot slot; `slotCap` bounds
   *  in-flight jobs (and `jobCap = slotCap`, so the job ring can never overrun the free-list). */
  static create(opts: { apron: number; slotCap?: number }): SharedMeshQueue {
    const slotCap = nextPow2(Math.max(8, opts.slotCap ?? 64)); // power-of-2 ⇒ wrap-safe `& (cap-1)` indexing
    const jobCap = slotCap; // in-flight ≤ slotCap ⇒ the job ring needs no more entries
    const slotBytes = slotBytesFor(opts.apron);
    const h: SharedMeshQueueHandles = {
      ctrl: new SharedArrayBuffer(CTRL_LEN * 4),
      jobs: new SharedArrayBuffer(jobCap * JOB_FIELDS * 4),
      free: new SharedArrayBuffer(slotCap * 4),
      data: new SharedArrayBuffer(slotCap * slotBytes),
      jobCap,
      slotCap,
      slotBytes,
    };
    return new SharedMeshQueue(h, true);
  }

  /** Worker-side: wrap the SABs transferred at init. Does NOT re-seed (the main side already did). */
  static attach(h: SharedMeshQueueHandles): SharedMeshQueue {
    return new SharedMeshQueue(h, false);
  }

  /** The handles to hand each worker (over postMessage `init`; SABs travel by reference). */
  get handles(): SharedMeshQueueHandles {
    return this.h;
  }

  /** The futex word workers waitAsync on. */
  get notifyView(): Int32Array {
    return this.ctrl;
  }
  get notifyIndex(): number {
    return NOTIFY;
  }

  /** Queued snapshot bytes — getTotalRemainingDuration analog (admission back-pressure). */
  queuedBytes(): number {
    return Atomics.load(this.ctrl, DURATION_SUM);
  }

  /** MAIN: advance the model epoch when posting `addModels`, and return the new value so it can ride the
   *  addModels message — the worker stamps it as "applied" once it has merged those models. Every build
   *  enqueued AFTER this call carries the new epoch, so the worker won't mesh it until the models land. */
  bumpModelEpoch(): number {
    return Atomics.add(this.ctrl, MODEL_EPOCH, 1) + 1;
  }
  /** The current model epoch (the value stamped onto subsequently-enqueued jobs). */
  modelEpoch(): number {
    return Atomics.load(this.ctrl, MODEL_EPOCH);
  }

  /** MAIN: enqueue a build. Acquires a free slot, copies the snapshot's bytes in, writes the descriptor,
   *  then publishes it (semaphore bump + notify). Returns false iff the pool is full (caller falls back). */
  enqueue(jobId: number, snapshot: SectionSnapshot): boolean {
    const slot = this.acquireSlot();
    if (slot < 0) return false; // pool full — caller meshes this one another way (never drop ⇒ no T-PRES break)

    const base = slot * this.h.slotBytes;
    const blocksBytes = snapshot.blocks.byteLength;
    const lightBytes = snapshot.light ? snapshot.light.byteLength : 0;
    const biomeBytes = snapshot.biomeTint ? snapshot.biomeTint.byteLength : 0;
    // Copy into the slot (blocks | light | biome, each 4-aligned — see slotBytesFor).
    this.data.set(new Uint8Array(snapshot.blocks.buffer, snapshot.blocks.byteOffset, blocksBytes), base);
    let off = align4(blocksBytes);
    if (snapshot.light) {
      this.data.set(new Uint8Array(snapshot.light.buffer, snapshot.light.byteOffset, lightBytes), base + off);
    }
    off = align4(off + lightBytes);
    if (snapshot.biomeTint) {
      this.data.set(new Uint8Array(snapshot.biomeTint.buffer, snapshot.biomeTint.byteOffset, biomeBytes), base + off);
    }

    const tail = Atomics.load(this.ctrl, JOB_TAIL) & (this.h.jobCap - 1);
    const d = tail * JOB_FIELDS;
    // S2: the snapshot already carries the section ORIGIN (block coords) — derive the section key coords by
    // `>> 4` instead of splitting the `"sx,sy,sz"` string per enqueued job.
    const kx = snapshot.origin[0] >> 4, ky = snapshot.origin[1] >> 4, kz = snapshot.origin[2] >> 4;
    this.jobs[d + D_JOBID] = jobId;
    this.jobs[d + D_GEN] = snapshot.generation;
    this.jobs[d + D_KEYX] = kx;
    this.jobs[d + D_KEYY] = ky;
    this.jobs[d + D_KEYZ] = kz;
    this.jobs[d + D_APRON] = snapshot.apron;
    this.jobs[d + D_REASON] = snapshot.changedReason;
    this.jobs[d + D_SLOT] = slot;
    this.jobs[d + D_BLOCKS_BYTES] = blocksBytes;
    this.jobs[d + D_LIGHT_BYTES] = lightBytes;
    this.jobs[d + D_BIOME_BYTES] = biomeBytes;
    this.jobs[d + D_EPOCH] = Atomics.load(this.ctrl, MODEL_EPOCH); // the worker gates meshing on this epoch
    this.jobs[d + D_CANCEL] = 0; // INFRA-3: reset — this ring slot may carry a stale cancel flag from a prior cycle

    // PUBLISH: advance the producer cursor, bump the count semaphore (the synchronizes-with edge — the
    // descriptor + slot writes above are now visible to any consumer that observes this), notify workers.
    Atomics.add(this.ctrl, JOB_TAIL, 1);
    Atomics.add(this.ctrl, DURATION_SUM, blocksBytes);
    Atomics.add(this.ctrl, JOB_COUNT, 1);
    Atomics.add(this.ctrl, NOTIFY, 1);
    Atomics.notify(this.ctrl, NOTIFY);
    return true;
  }

  /** WORKER: claim the next job (wait-free), or null if none. Reconstructs the snapshot as VIEWS over the
   *  shared slot — zero-copy. The caller MUST `release(job.slot)` after meshing has finished reading it. */
  dequeue(): ClaimedJob | null {
    // Reserve one job via the counting semaphore. If we over-decremented (no job), undo and bail.
    if (Atomics.sub(this.ctrl, JOB_COUNT, 1) <= 0) {
      Atomics.add(this.ctrl, JOB_COUNT, 1);
      return null;
    }
    const idx = Atomics.add(this.ctrl, JOB_HEAD, 1) & (this.h.jobCap - 1);
    const d = idx * JOB_FIELDS;
    const jobId = this.jobs[d + D_JOBID];
    const generation = this.jobs[d + D_GEN] as Generation;
    const apron = this.jobs[d + D_APRON];
    const slot = this.jobs[d + D_SLOT];
    const blocksBytes = this.jobs[d + D_BLOCKS_BYTES];
    const lightBytes = this.jobs[d + D_LIGHT_BYTES];
    const biomeBytes = this.jobs[d + D_BIOME_BYTES];
    const cancelled = Atomics.load(this.jobs, d + D_CANCEL) !== 0; // INFRA-3 supersede flag (best-effort)
    const key = sectionKey(this.jobs[d + D_KEYX], this.jobs[d + D_KEYY], this.jobs[d + D_KEYZ]);
    const origin: Vec3i = [this.jobs[d + D_KEYX] * 16, this.jobs[d + D_KEYY] * 16, this.jobs[d + D_KEYZ] * 16];

    const base = slot * this.h.slotBytes;
    const blocks = new Uint32Array(this.h.data, base, blocksBytes / 4);
    let off = align4(blocksBytes);
    const light = lightBytes > 0 ? new Uint8Array(this.h.data, base + off, lightBytes) : undefined;
    off = align4(off + lightBytes);
    const biomeTint = biomeBytes > 0 ? new Uint32Array(this.h.data, base + off, biomeBytes / 4) : undefined;

    const snapshot: SectionSnapshot = {
      sectionKey: key,
      generation,
      origin,
      size: 16,
      apron,
      blocks,
      light,
      biomeTint,
      changedReason: this.jobs[d + D_REASON] as DirtyReason,
    };
    return { jobId, slot, estimateBytes: blocksBytes, enqueueEpoch: this.jobs[d + D_EPOCH], cancelled, snapshot };
  }

  /**
   * MAIN (INFRA-3): flag a still-queued build as superseded so the worker skips meshing it. Scans the job
   * descriptors for `jobId` (unique + monotonic, so ≤1 matches) and sets its cancel flag via `Atomics.store`.
   * Returns true iff a matching descriptor was found. Best-effort: if the worker already read the descriptor
   * (or already meshed + released the slot), this no-ops and the main-thread generation stale-drop handles it.
   * `jobCap` is small (≤ slotCap), so the linear scan is trivial and runs only on a re-dirty supersede.
   */
  cancelByJobScan(jobId: number): boolean {
    for (let i = 0; i < this.h.jobCap; i++) {
      const d = i * JOB_FIELDS;
      if (this.jobs[d + D_JOBID] === jobId) {
        Atomics.store(this.jobs, d + D_CANCEL, 1);
        return true;
      }
    }
    return false;
  }

  /** WORKER: return a slot to the free-list after meshing has read it (and drop its back-pressure bytes). */
  release(slot: number, estimateBytes: number): void {
    const t = Atomics.add(this.ctrl, FREE_TAIL, 1) & (this.h.slotCap - 1);
    this.free[t] = slot;
    Atomics.sub(this.ctrl, DURATION_SUM, estimateBytes);
    Atomics.add(this.ctrl, FREE_COUNT, 1);
  }

  /** MAIN: pop a free slot index, or -1 if the pool is full. Symmetric to `release` (a semaphore ring). */
  private acquireSlot(): number {
    if (Atomics.sub(this.ctrl, FREE_COUNT, 1) <= 0) {
      Atomics.add(this.ctrl, FREE_COUNT, 1);
      return -1;
    }
    const idx = Atomics.add(this.ctrl, FREE_HEAD, 1) & (this.h.slotCap - 1);
    return this.free[idx];
  }
}
