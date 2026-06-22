// WRK-3: cooperative-cancel flag SAB. RENDERER FURTHER-OPTIMIZATION PLAN — WRK-3.

import { describe, it, expect } from "vitest";
import { CancelFlags, WorkerCancelFlags, cancelFlagsAvailable } from "./CancelFlags";

describe("CancelFlags (WRK-3)", () => {
  it("create() seeds zeroed words — no job is cancelled before a request", () => {
    const flags = CancelFlags.create(2);
    expect(flags).not.toBeNull();
    const w0 = WorkerCancelFlags.attach(flags!.buffer, 0)!;
    // Worker 0 publishes that it is meshing job 7; nothing has requested a cancel yet.
    w0.setActive(7);
    expect(w0.isCancelled(7)).toBe(false);
    expect(flags!.cancelRequestOf(0)).toBe(0); // request word starts zeroed
  });

  it("requestCancel(slot,jobId) is observed by that worker's isCancelled via Atomics", () => {
    const flags = CancelFlags.create(2)!;
    const w0 = WorkerCancelFlags.attach(flags.buffer, 0)!;
    w0.setActive(9);
    expect(w0.isCancelled(9)).toBe(false);
    flags.requestCancel(0, 9); // main requests cancel of job 9 on worker 0
    expect(w0.isCancelled(9)).toBe(true); // the worker meshing job 9 sees it
    expect(flags.cancelRequestOf(0)).toBe(9);
  });

  it("a request for job A on worker 0 does NOT match worker 1 (per-worker scoping — TRAP WRK-3.B)", () => {
    const flags = CancelFlags.create(2)!;
    const w0 = WorkerCancelFlags.attach(flags.buffer, 0)!;
    const w1 = WorkerCancelFlags.attach(flags.buffer, 1)!;
    w0.setActive(1); // worker 0 meshes job 1
    w1.setActive(2); // worker 1 meshes job 2 (the NEXT, live job)
    flags.requestCancel(0, 1); // cancel ONLY worker 0's job 1
    expect(w0.isCancelled(1)).toBe(true);
    expect(w1.isCancelled(2)).toBe(false); // worker 1's live job is untouched (no cross-cancel)
    expect(flags.cancelRequestOf(1)).toBe(0); // worker 1's request word never written
  });

  it("clearActive clears BOTH the active job AND the stale request, so a leftover request can't match the next job (P3)", () => {
    const flags = CancelFlags.create(1)!;
    const w0 = WorkerCancelFlags.attach(flags.buffer, 0)!;
    w0.setActive(5);
    flags.requestCancel(0, 5); // requested but (say) arrived after the worker finished job 5
    expect(w0.isCancelled(5)).toBe(true);
    w0.clearActive(); // P3: ACTIVE_JOBID=0 AND CANCEL_REQUEST=0 between jobs
    expect(flags.cancelRequestOf(0)).toBe(0); // the stale request VALUE is gone — no reliance on monotonic ids
    // Worker now meshes the NEXT job (id 6). The cleared request can't match it.
    w0.setActive(6);
    expect(w0.isCancelled(6)).toBe(false);
  });

  it("is inert when SharedArrayBuffer is undefined (no-op shim, feature byte-identical)", () => {
    const realSAB = globalThis.SharedArrayBuffer;
    try {
      // Simulate a non-cross-origin-isolated page without SAB.
      (globalThis as { SharedArrayBuffer?: unknown }).SharedArrayBuffer = undefined;
      expect(cancelFlagsAvailable()).toBe(false);
      expect(CancelFlags.create(4)).toBeNull(); // no SAB allocated
      // The worker attaches a no-op shim when no buffer is sent.
      const shim = WorkerCancelFlags.attach(undefined, 0);
      expect(shim).toBeNull();
    } finally {
      (globalThis as { SharedArrayBuffer?: unknown }).SharedArrayBuffer = realSAB;
    }
  });
});
