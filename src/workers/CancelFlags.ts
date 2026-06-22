// WRK-3: cooperative-cancel flags for Model-A (postMessage) off-thread meshing.
//
// Model B already has cooperative cancel through its SAB ring (D_CANCEL — SharedMeshQueue.cancelByJobScan);
// Model A is the non-cross-origin-isolated postMessage fallback and had NO shared word to poll, so a
// superseded build always fully meshed + serialized + transferred a result that `deliver()` then dropped.
//
// This is a tiny dedicated cancel SAB (NOT the Model-B ring), indexed by WORKER SLOT (not jobId). Two Int32
// words per worker:
//  - ACTIVE_JOBID    : the jobId the worker is currently meshing (0 between jobs).
//  - CANCEL_REQUEST  : the jobId the MAIN thread wants that worker to stop meshing.
// The worker bails mid-mesh only when `Atomics.load(CANCEL_REQUEST) === its own ACTIVE_JOBID`. Per-worker
// indexing makes the match a single equality (no jobId scan) and SCOPES the cancel to the exact worker
// holding that jobId — so a worker that already finished and started the NEXT (live) job is never cancelled
// (TRAP WRK-3.B). jobIds are monotonic (Scheduler.nextJobId++), so a stale request can't match a later job;
// the worker also clears ACTIVE_JOBID=0 between jobs so a request that lands between jobs matches nothing.
//
// Gate: `typeof SharedArrayBuffer !== 'undefined'` ONLY — a plain non-`waitAsync` flag needs neither
// crossOriginIsolated NOR Atomics.waitAsync (TRAP WRK-3.D). If SAB is wholly absent the feature is inert: the
// pool allocates nothing, the init message carries no SAB, the worker attaches a no-op shim, and the whole
// path is byte-identical to today.

/** Two Int32 words per worker slot. */
const WORDS_PER_WORKER = 2;
const ACTIVE_JOBID = 0;
const CANCEL_REQUEST = 1;

/** True iff a plain (non-`waitAsync`) cancel SAB can be constructed. Deliberately does NOT require
 *  `crossOriginIsolated` or `Atomics.waitAsync` (that's the Model-B gate, `sharedMemoryAvailable()`). */
export function cancelFlagsAvailable(): boolean {
  return typeof SharedArrayBuffer !== "undefined";
}

/**
 * Main-thread owner of the cancel SAB. Allocates `size` worker slots, hands the SAB to each worker at init
 * (via the `cancelFlags` field on the init message), and writes a CANCEL_REQUEST when the Scheduler supersedes
 * a prior in-flight job. A pure HINT: the worker may have already finished; `deliver()` still converts the
 * always-arriving reply, and the main-side `cancelled` set + generation stale-drop remain the backstops.
 */
export class CancelFlags {
  private constructor(private readonly flags: Int32Array, readonly buffer: SharedArrayBuffer) {}

  /** Allocate a zeroed cancel SAB for `size` workers. Returns null when SAB is unavailable (feature inert). */
  static create(size: number): CancelFlags | null {
    if (!cancelFlagsAvailable()) return null;
    const sab = new SharedArrayBuffer(size * WORDS_PER_WORKER * Int32Array.BYTES_PER_ELEMENT);
    return new CancelFlags(new Int32Array(sab), sab);
  }

  /** Request the worker at `workerIndex` cancel `jobId` (Atomics.store CANCEL_REQUEST). The worker bails only
   *  if it is STILL meshing that exact jobId (ACTIVE_JOBID === jobId), so this can only ever hit the prior
   *  (superseded) build, never a live one. */
  requestCancel(workerIndex: number, jobId: number): void {
    Atomics.store(this.flags, workerIndex * WORDS_PER_WORKER + CANCEL_REQUEST, jobId);
  }

  // ── Test/worker-equivalent accessors (the worker uses WorkerCancelFlags.attach below) ─────────────
  /** Read a worker slot's CANCEL_REQUEST word (test visibility into what the main thread requested). */
  cancelRequestOf(workerIndex: number): number {
    return Atomics.load(this.flags, workerIndex * WORDS_PER_WORKER + CANCEL_REQUEST);
  }
}

/**
 * The worker's view of ITS OWN slot in the shared cancel SAB. `attach` binds a fixed worker slot index, so
 * `setActive`/`clearActive`/`isCancelled` need no index argument — they always operate on this worker's two
 * words. When the SAB is absent the worker attaches a no-op shim (`null`) and every call is inert.
 */
export class WorkerCancelFlags {
  private readonly active: number;
  private readonly request: number;
  private constructor(private readonly flags: Int32Array, workerIndex: number) {
    this.active = workerIndex * WORDS_PER_WORKER + ACTIVE_JOBID;
    this.request = workerIndex * WORDS_PER_WORKER + CANCEL_REQUEST;
  }

  /** Attach to this worker's slot. Returns null (no-op shim) when no SAB was sent (feature off / inert). */
  static attach(buffer: SharedArrayBuffer | undefined, workerIndex: number): WorkerCancelFlags | null {
    if (!buffer) return null;
    return new WorkerCancelFlags(new Int32Array(buffer), workerIndex);
  }

  /** Mark this worker as meshing `jobId` (so a CANCEL_REQUEST matching it can be honored). */
  setActive(jobId: number): void {
    Atomics.store(this.flags, this.active, jobId);
  }

  /** Clear ACTIVE_JOBID=0 after a build, so a stale CANCEL_REQUEST that arrives between jobs matches nothing.
   *  P3: ALSO clear CANCEL_REQUEST=0 so a slot never carries a stale request value — this drops the safety's
   *  reliance on jobIds being strictly monotonic (a wrapped/reused jobId could otherwise alias a leftover
   *  request). A request for the NEXT job that races this clear is, at worst, missed (the generation stale-drop
   *  at acceptBuild remains the authoritative backstop), never mis-applied. */
  clearActive(): void {
    Atomics.store(this.flags, this.active, 0);
    Atomics.store(this.flags, this.request, 0);
  }

  /** True iff the main thread has requested cancellation of exactly the job this worker is meshing. The
   *  equality (request === jobId, and jobId === ACTIVE_JOBID by construction at the call site) is what scopes
   *  the cancel to THIS worker's current job — a leftover request for a finished job can't match the next. */
  isCancelled(jobId: number): boolean {
    return Atomics.load(this.flags, this.request) === jobId;
  }
}
