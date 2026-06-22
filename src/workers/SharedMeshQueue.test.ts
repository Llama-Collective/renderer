// Model B substrate (SharedMeshQueue) — protocol correctness. Node has SharedArrayBuffer + Atomics, so the
// enqueue/dequeue/free-list semaphore logic + the zero-copy snapshot round-trip are fully testable here
// (the multi-WORKER race + memory ordering are exercised by the browser GPU smoke, harness/workers).

import { describe, it, expect } from "vitest";
import { SharedMeshQueue, slotBytesFor } from "./SharedMeshQueue";
import { snapshotStride, type SectionSnapshot } from "../world/SnapshotSource";
import { sectionKey } from "../world/SectionKey";
import { DirtyReason, type Generation, type Vec3i } from "../types";

const APRON = 1;
const CELLS = snapshotStride(APRON) ** 3;

function snap(kx: number, ky: number, kz: number, gen: number, seed: number, withExtras = false): SectionSnapshot {
  const blocks = new Uint32Array(CELLS);
  for (let i = 0; i < CELLS; i++) blocks[i] = (seed * 131 + i) >>> 0;
  const origin: Vec3i = [kx * 16, ky * 16, kz * 16];
  const s: SectionSnapshot = { sectionKey: sectionKey(kx, ky, kz), generation: gen as Generation, origin, size: 16, apron: APRON, blocks, changedReason: DirtyReason.Edit };
  if (withExtras) {
    s.light = new Uint8Array(CELLS).map((_, i) => (seed + i) & 0xff);
    s.biomeTint = new Uint32Array(CELLS).map((_, i) => (seed ^ (i * 7)) >>> 0);
  }
  return s;
}

describe("SharedMeshQueue (Model B substrate)", () => {
  it("round-trips a snapshot zero-copy through the SAB (enqueue on main → dequeue on a worker attach)", () => {
    const q = SharedMeshQueue.create({ apron: APRON, slotCap: 8 });
    const worker = SharedMeshQueue.attach(q.handles); // a separate instance over the SAME shared memory
    const s = snap(2, -3, 4, 7, 42, true);
    expect(q.enqueue(101, s)).toBe(true);

    const job = worker.dequeue();
    expect(job).not.toBeNull();
    expect(job!.jobId).toBe(101);
    expect(job!.snapshot.sectionKey).toBe(sectionKey(2, -3, 4));
    expect(job!.snapshot.generation).toBe(7);
    expect(job!.snapshot.apron).toBe(APRON);
    expect(job!.snapshot.origin).toEqual([32, -48, 64]);
    expect(Array.from(job!.snapshot.blocks)).toEqual(Array.from(s.blocks)); // blocks copied through the slot
    expect(Array.from(job!.snapshot.light!)).toEqual(Array.from(s.light!)); // light + biome too
    expect(Array.from(job!.snapshot.biomeTint!)).toEqual(Array.from(s.biomeTint!));
    // The view is backed by the SHARED data buffer (zero-copy, not a clone).
    expect(job!.snapshot.blocks.buffer).toBeInstanceOf(SharedArrayBuffer);
    expect(worker.dequeue()).toBeNull(); // only one job
  });

  it("dequeues in FIFO order", () => {
    const q = SharedMeshQueue.create({ apron: APRON, slotCap: 8 });
    for (let i = 0; i < 5; i++) q.enqueue(200 + i, snap(i, 0, 0, 1, i));
    const ids = [];
    for (let i = 0; i < 5; i++) { const j = q.dequeue(); ids.push(j!.jobId); q.release(j!.slot, j!.estimateBytes); }
    expect(ids).toEqual([200, 201, 202, 203, 204]);
  });

  it("returns false when the slot pool is full, and recovers after a release (no silent drop)", () => {
    const q = SharedMeshQueue.create({ apron: APRON, slotCap: 8 }); // 8 slots
    const claimed = [];
    for (let i = 0; i < 8; i++) { expect(q.enqueue(i, snap(i, 0, 0, 1, i))).toBe(true); }
    // Pool is full (8 in flight, none released) → the 9th enqueue MUST report full, not drop silently.
    expect(q.enqueue(99, snap(9, 0, 0, 1, 9))).toBe(false);
    // Drain one job → release its slot → a slot frees → enqueue succeeds again.
    const j = q.dequeue();
    q.release(j!.slot, j!.estimateBytes);
    expect(q.enqueue(99, snap(9, 0, 0, 1, 9))).toBe(true);
  });

  it("tracks queued bytes for back-pressure (rises on enqueue, falls on release)", () => {
    const q = SharedMeshQueue.create({ apron: APRON, slotCap: 8 });
    const blockBytes = CELLS * 4;
    expect(q.queuedBytes()).toBe(0);
    q.enqueue(1, snap(0, 0, 0, 1, 1));
    q.enqueue(2, snap(1, 0, 0, 1, 2));
    expect(q.queuedBytes()).toBe(blockBytes * 2);
    const j = q.dequeue();
    q.release(j!.slot, j!.estimateBytes);
    expect(q.queuedBytes()).toBe(blockBytes); // one drained
  });

  it("reuses slots across many enqueue/dequeue cycles (ring wrap, no leak)", () => {
    const q = SharedMeshQueue.create({ apron: APRON, slotCap: 8 });
    // 200 cycles through an 8-slot pool exercises the ring wrap well past capacity.
    for (let i = 0; i < 200; i++) {
      expect(q.enqueue(i, snap(i & 3, 0, 0, 1, i))).toBe(true);
      const j = q.dequeue();
      expect(j!.jobId).toBe(i);
      expect(j!.snapshot.blocks[0]).toBe((i * 131) >>> 0); // the right snapshot landed in the reused slot
      q.release(j!.slot, j!.estimateBytes);
    }
    expect(q.queuedBytes()).toBe(0);
    expect(q.dequeue()).toBeNull();
  });

  it("stamps the current model epoch on each job (the addModels ordering gate)", () => {
    const q = SharedMeshQueue.create({ apron: APRON, slotCap: 8 });
    expect(q.modelEpoch()).toBe(0);
    q.enqueue(1, snap(0, 0, 0, 1, 1));
    expect(q.dequeue()!.enqueueEpoch).toBe(0); // enqueued before any addModels

    const e = q.bumpModelEpoch(); // a new block id's models were pushed
    expect(e).toBe(1);
    expect(q.modelEpoch()).toBe(1);
    q.enqueue(2, snap(0, 0, 0, 1, 2));
    expect(q.dequeue()!.enqueueEpoch).toBe(1); // this build must wait for epoch-1 models before meshing
  });

  it("flags a superseded job so the worker dequeues it cancelled (INFRA-3)", () => {
    const q = SharedMeshQueue.create({ apron: APRON, slotCap: 8 });
    const worker = SharedMeshQueue.attach(q.handles);
    q.enqueue(101, snap(0, 0, 0, 1, 1));
    q.enqueue(102, snap(1, 0, 0, 1, 2));
    expect(q.cancelByJobScan(101)).toBe(true); // job 101 is still queued → flagged
    expect(q.cancelByJobScan(999)).toBe(false); // unknown jobId → nothing flagged

    const a = worker.dequeue();
    expect(a!.jobId).toBe(101);
    expect(a!.cancelled).toBe(true); // worker will release + post `cancelled`, skipping the mesh
    const b = worker.dequeue();
    expect(b!.jobId).toBe(102);
    expect(b!.cancelled).toBe(false); // the non-superseded job still meshes normally
  });

  it("a reused ring slot does NOT carry a stale cancel flag (enqueue resets it)", () => {
    const q = SharedMeshQueue.create({ apron: APRON, slotCap: 8 });
    q.enqueue(1, snap(0, 0, 0, 1, 1));
    q.cancelByJobScan(1); // flag it
    const j1 = q.dequeue();
    expect(j1!.cancelled).toBe(true);
    q.release(j1!.slot, j1!.estimateBytes);
    // Cycle the ring all the way around so the same descriptor index is rewritten.
    for (let i = 0; i < 8; i++) {
      const id = 100 + i;
      expect(q.enqueue(id, snap(i, 0, 0, 1, i))).toBe(true);
      const j = q.dequeue();
      expect(j!.jobId).toBe(id);
      expect(j!.cancelled).toBe(false); // fresh enqueue cleared any prior-cycle cancel
      q.release(j!.slot, j!.estimateBytes);
    }
  });

  it("releasing a cancelled job returns its slot (no leak) — pool refills (INFRA-3)", () => {
    const q = SharedMeshQueue.create({ apron: APRON, slotCap: 8 }); // 8 slots
    for (let i = 0; i < 8; i++) expect(q.enqueue(i, snap(i, 0, 0, 1, i))).toBe(true); // pool full
    q.cancelByJobScan(0); // supersede the first
    expect(q.enqueue(99, snap(9, 0, 0, 1, 9))).toBe(false); // still full until something releases
    // Simulate the worker draining the cancelled job: dequeue → release (no mesh).
    const j = q.dequeue();
    expect(j!.jobId).toBe(0);
    expect(j!.cancelled).toBe(true);
    q.release(j!.slot, j!.estimateBytes); // the cancelled branch's single release
    expect(q.enqueue(99, snap(9, 0, 0, 1, 9))).toBe(true); // slot reclaimed
  });

  it("slotBytesFor accounts for blocks + light + biome (4-aligned)", () => {
    const cells = snapshotStride(APRON) ** 3;
    expect(slotBytesFor(APRON)).toBe(cells * 4 + cells + cells * 4 + ((4 - (cells & 3)) & 3)); // light pad to 4
  });
});
