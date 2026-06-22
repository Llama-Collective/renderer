// BufferArena sub-allocation, free-list reuse, compaction, growth. RENDERER_PLAN §11, §22.

import { describe, it, expect, afterEach } from "vitest";
import { FakeGraphicsDevice } from "./testing/FakeGraphicsDevice";
import { BufferArena, __setArenaAsserts } from "./BufferArena";
import { BufferUsage } from "./GraphicsDevice";

function pattern(byte: number, len = 8): Uint8Array {
  return new Uint8Array(len).fill(byte);
}

function readAlloc(device: FakeGraphicsDevice, arena: BufferArena, id: number): Uint8Array {
  const r = arena.rangeOf(id as never);
  return device.read(arena.gpuBuffer).subarray(r.offsetBytes, r.offsetBytes + r.sizeBytes);
}

describe("BufferArena", () => {
  it("allocates non-overlapping ranges and preserves data", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 64, BufferUsage.Vertex);
    const a = arena.allocate(pattern(0xaa));
    const b = arena.allocate(pattern(0xbb));
    const ra = arena.rangeOf(a);
    const rb = arena.rangeOf(b);
    expect(ra.offsetBytes).not.toBe(rb.offsetBytes);
    expect(readAlloc(device, arena, a)).toEqual(pattern(0xaa));
    expect(readAlloc(device, arena, b)).toEqual(pattern(0xbb));
    expect(arena.stats().usedBytes).toBe(16);
  });

  it("reuses a freed segment for a same-size allocation", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 64, BufferUsage.Index);
    const a = arena.allocate(pattern(0x11));
    const off = arena.rangeOf(a).offsetBytes;
    arena.free(a);
    const c = arena.allocate(pattern(0x22));
    expect(arena.rangeOf(c).offsetBytes).toBe(off); // slot reused
    expect(readAlloc(device, arena, c)).toEqual(pattern(0x22));
  });

  it("merges adjacent free segments", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 24, BufferUsage.Vertex); // exactly 3×8, no free tail
    const a = arena.allocate(pattern(0x1));
    const b = arena.allocate(pattern(0x2));
    arena.allocate(pattern(0x3)); // c stays allocated at the end
    arena.free(a);
    arena.free(b); // adjacent holes a+b coalesce into ONE 16-byte free segment
    expect(arena.stats().freeSegments).toBe(1);
    expect(arena.stats().largestFreeBytes).toBe(16);
  });

  it("compacts fragmented live allocations, preserving ids + data", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 64, BufferUsage.Vertex);
    const a = arena.allocate(pattern(0xa1));
    const b = arena.allocate(pattern(0xb2));
    const c = arena.allocate(pattern(0xc3));
    arena.free(b); // hole between a and c
    arena.compact();
    // a then c packed contiguously; data + ids intact.
    expect(arena.rangeOf(a).offsetBytes).toBe(0);
    expect(arena.rangeOf(c).offsetBytes).toBe(8);
    expect(readAlloc(device, arena, a)).toEqual(pattern(0xa1));
    expect(readAlloc(device, arena, c)).toEqual(pattern(0xc3));
    expect(arena.stats().usedBytes).toBe(16);
  });

  it("grows to fit when over capacity, keeping existing data", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 16, BufferUsage.Vertex);
    const a = arena.allocate(pattern(0xa, 16)); // fills capacity
    const b = arena.allocate(pattern(0xb, 16)); // forces grow
    expect(arena.stats().capacityBytes).toBeGreaterThanOrEqual(32);
    expect(readAlloc(device, arena, a)).toEqual(pattern(0xa, 16));
    expect(readAlloc(device, arena, b)).toEqual(pattern(0xb, 16));
  });

  it("update overwrites in place, rejecting size changes", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 64, BufferUsage.Vertex);
    const a = arena.allocate(pattern(0x00));
    arena.update(a, pattern(0xff));
    expect(readAlloc(device, arena, a)).toEqual(pattern(0xff));
    expect(() => arena.update(a, pattern(0xff, 16))).toThrow();
  });

  it("compact SHRINKS the high-water mark (U8 reclaim), not just defragments", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 16, BufferUsage.Vertex);
    arena.allocate(pattern(0xa, 16));
    const b = arena.allocate(pattern(0xb, 16)); // grow → capacity ≥ 32
    const c = arena.allocate(pattern(0xc, 16)); // grow again → capacity ≥ 64
    const grown = arena.stats().capacityBytes;
    arena.free(b);
    arena.free(c); // only one 16-byte allocation remains, capacity is way oversized
    arena.compact();
    expect(arena.stats().capacityBytes).toBeLessThan(grown); // capacity reclaimed toward the live size
    expect(arena.stats().usedBytes).toBe(16);
  });

  it("shouldReclaim fires only when a lot is reclaimable AND the arena is mostly empty", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 64, BufferUsage.Vertex);
    arena.allocate(pattern(0xa, 16)); // used 16 of 64 → waste 48, used*2 (32) < 64 ⇒ mostly empty
    expect(arena.shouldReclaim(48)).toBe(true); // waste 48 ≥ 48
    expect(arena.shouldReclaim(49)).toBe(false); // waste 48 < 49 ⇒ not enough to bother
    arena.allocate(pattern(0xb, 16)); // used 32 → used*2 (64) NOT < 64 ⇒ no longer "mostly empty"
    expect(arena.shouldReclaim(8)).toBe(false);
  });

  it("layoutRevision bumps on grow + compact (relocation) but NOT on allocate/free/update (F3/TRAP 4.A)", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 64, BufferUsage.Vertex);
    expect(arena.layoutRevision).toBe(0);

    const a = arena.allocate(pattern(0xa, 16)); // fits in initial capacity — no relocation
    const b = arena.allocate(pattern(0xb, 16));
    expect(arena.layoutRevision).toBe(0); // plain allocs don't move existing data

    arena.update(a, pattern(0x1, 16));
    expect(arena.layoutRevision).toBe(0); // in-place overwrite — no move

    arena.free(b);
    expect(arena.layoutRevision).toBe(0); // free just merges free space — existing allocs stay put

    // Force a grow: allocate past the 64-byte capacity.
    arena.allocate(pattern(0xc, 64));
    expect(arena.layoutRevision).toBe(1); // grow relocated everything into a fresh buffer

    // Force a compact.
    arena.free(a);
    arena.compact();
    expect(arena.layoutRevision).toBe(2); // compact relocated into a fresh shrunk buffer
  });

  it("grow relocates all live blocks in ONE submit (INFRA-1), copies==N, bytes intact", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 24, BufferUsage.Vertex); // holds exactly 3×8
    const a = arena.allocate(pattern(0xa1));
    const b = arena.allocate(pattern(0xb2));
    const c = arena.allocate(pattern(0xc3)); // arena now full, 3 live blocks
    const submitsBefore = device.log.submits;
    const copiesBefore = device.log.copies;

    const d = arena.allocate(pattern(0xd4)); // overflows → grow relocates a,b,c
    expect(device.log.submits - submitsBefore).toBe(1); // ONE batched submit, not 3
    expect(device.log.copies - copiesBefore).toBe(3); // still one copy per relocated live block
    // Every id survives the relocation byte-identical (and the new alloc landed).
    expect(readAlloc(device, arena, a)).toEqual(pattern(0xa1));
    expect(readAlloc(device, arena, b)).toEqual(pattern(0xb2));
    expect(readAlloc(device, arena, c)).toEqual(pattern(0xc3));
    expect(readAlloc(device, arena, d)).toEqual(pattern(0xd4));
  });

  it("compact relocates all live blocks in ONE submit (INFRA-1), copies==N", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 64, BufferUsage.Vertex);
    const a = arena.allocate(pattern(0xa1));
    const b = arena.allocate(pattern(0xb2));
    const c = arena.allocate(pattern(0xc3));
    arena.free(b); // hole between a and c → 2 live blocks remain
    const submitsBefore = device.log.submits;
    const copiesBefore = device.log.copies;

    arena.compact();
    expect(device.log.submits - submitsBefore).toBe(1); // ONE submit for the whole compaction
    expect(device.log.copies - copiesBefore).toBe(2); // a and c relocated
    expect(readAlloc(device, arena, a)).toEqual(pattern(0xa1));
    expect(readAlloc(device, arena, c)).toEqual(pattern(0xc3));
  });

  it("INFRA-1 cross-arena: several arenas compacted inside one copy batch collapse to ONE submit", () => {
    const device = new FakeGraphicsDevice();
    // Two arenas (vertex + index) each with a hole, so each compaction relocates live blocks.
    const v = new BufferArena(device, 64, BufferUsage.Vertex);
    v.allocate(pattern(0xa1)); const vb = v.allocate(pattern(0xb2)); const vc = v.allocate(pattern(0xc3));
    v.free(vb); // free the MIDDLE block → a hole that compaction must relocate `vc` past
    const i = new BufferArena(device, 64, BufferUsage.Index);
    const ia = i.allocate(pattern(0x11)); const ib = i.allocate(pattern(0x22)); const ic = i.allocate(pattern(0x33));
    i.free(ib); // same: a hole in the index arena
    const destroyedBefore = device.log.buffersDestroyed;
    const submitsBefore = device.log.submits;

    // Coalesce both compactions into one batch (the uploader's reclaim loop).
    device.beginCopyBatch!();
    v.compact();
    i.compact();
    // Old buffers are NOT destroyed yet — deferred until the batch submit (so copy sources stay alive).
    expect(device.log.buffersDestroyed).toBe(destroyedBefore);
    device.endCopyBatch!();

    expect(device.log.submits - submitsBefore).toBe(1); // ONE submit for BOTH arenas (was 2 before coalescing)
    expect(device.log.buffersDestroyed).toBeGreaterThan(destroyedBefore); // deferred destroys ran after the submit
    // Data survived the coalesced relocation in both arenas.
    expect(readAlloc(device, v, vc)).toEqual(pattern(0xc3));
    expect(readAlloc(device, i, ia)).toEqual(pattern(0x11));
    expect(readAlloc(device, i, ic)).toEqual(pattern(0x33));
  });

  it("best-fit picks the smallest fitting free segment", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 64, BufferUsage.Index);
    const a = arena.allocate(pattern(0x1, 8)); // [0,8)
    const b = arena.allocate(pattern(0x2, 16)); // [8,24)
    arena.allocate(pattern(0x3, 8)); // [24,32) — stays allocated, isolating the tail
    arena.free(a);
    arena.free(b); // adjacent [0,8)+[8,24) coalesce → one 24-byte free block at offset 0; tail [32,64)=32 free
    // An 8-byte alloc best-fits the 24 block (smaller than the 32-byte tail) → offset 0.
    const d = arena.allocate(pattern(0x4, 8));
    expect(arena.rangeOf(d).offsetBytes).toBe(0);
  });
});

// MEM-2: the device-level freed-GPUBuffer-object pool is transparent at the arena layer. These cases drive
// real BufferArena grow/compact churn and assert the pool recycles whole buffer objects (flag on) while
// staying byte-identical in counts (flag off). BufferArena itself is UNCHANGED — the pool lives in the device.
describe("BufferArena + MEM-2 buffer pool", () => {
  it("grow → free-all → compact → grow REUSES a pooled buffer object (flag on): 2nd-grow create is absorbed, poolHits++", () => {
    const device = new FakeGraphicsDevice({ bufferPool: true });
    const arena = new BufferArena(device, 16, BufferUsage.Vertex);
    // Drive the capacity up through several grows so a range of buffer sizes lands in the pool.
    arena.allocate(pattern(0xa, 16)); // fills initial cap=16
    const b = arena.allocate(pattern(0xb, 16)); // grow → cap 32, pools the 16-byte buffer
    const c = arena.allocate(pattern(0xc, 16)); // grow → cap 64, pools the 32-byte buffer
    const d = arena.allocate(pattern(0xd, 16)); // fits in 64 (used 48) — no grow
    // Free everything and compact: compact shrinks to a small fresh buffer and POOLS the big (64-byte) one.
    arena.free(b);
    arena.free(c);
    arena.free(d);
    arena.compact();

    // After compact, the pool holds previously-freed buffer objects (sizes 16 / 64). Snapshot real-alloc + hit
    // counters, then force ONE grow whose requested capacity a pooled buffer satisfies.
    const createdBefore = device.log.buffersCreated;
    const hitsBefore = device.log.poolHits;
    // The arena is now ~20 bytes (one 16-byte live alloc + slack). A 16-byte alloc doesn't fit the 4-byte tail
    // ⇒ grow to ~40 bytes; the pooled 64-byte object best-fits 40 ⇒ recycled, NOT a fresh gpu.createBuffer.
    arena.allocate(pattern(0xe, 16));

    expect(device.log.poolHits).toBe(hitsBefore + 1); // the grow's createBuffer hit the pool exactly once
    expect(device.log.buffersCreated).toBe(createdBefore); // …so NO new real allocation happened
  });

  it("flag OFF is byte-identical: same buffersCreated/buffersDestroyed counts as today, zero pool activity", () => {
    const run = (pool: boolean) => {
      const device = new FakeGraphicsDevice({ bufferPool: pool });
      const arena = new BufferArena(device, 16, BufferUsage.Vertex);
      arena.allocate(pattern(0xa, 16));
      const b = arena.allocate(pattern(0xb, 16)); // grow
      const c = arena.allocate(pattern(0xc, 16)); // grow
      arena.free(b);
      arena.free(c);
      arena.compact();
      arena.dispose();
      return device.log;
    };
    const off = run(false);
    // Flag OFF must run the EXACT pre-MEM-2 path: every create is real, every destroy is real, no pooling.
    expect(off.poolHits).toBe(0);
    expect(off.poolReturns).toBe(0);
    expect(off.poolEvictions).toBe(0);
    // Each grow/compact creates one buffer + destroys the old; dispose destroys the last. Created == destroyed
    // for a fully-disposed arena (the initial + each grow/compact buffer all created, all destroyed).
    expect(off.buffersDestroyed).toBe(off.buffersCreated);
    expect(off.buffersCreated).toBeGreaterThan(0);
  });

  it("flag ON keeps bytes intact across a pooled-buffer grow (reused buffer is overwritten, never read stale)", () => {
    const device = new FakeGraphicsDevice({ bufferPool: true });
    const arena = new BufferArena(device, 16, BufferUsage.Vertex);
    const a = arena.allocate(pattern(0xa1, 16));
    const b = arena.allocate(pattern(0xb2, 16)); // grow
    const c = arena.allocate(pattern(0xc3, 16)); // grow
    arena.free(b);
    arena.compact(); // relocates a + c into a fresh (possibly pooled) buffer
    arena.allocate(pattern(0xd4, 16)); // may reuse a pooled object — its slice is written before any read
    // Every surviving allocation is byte-identical despite buffer-object recycling.
    expect(readAlloc(device, arena, a)).toEqual(pattern(0xa1, 16));
    expect(readAlloc(device, arena, c)).toEqual(pattern(0xc3, 16));
  });
});

describe("BufferArena integrity asserts (INFRA-2)", () => {
  afterEach(() => __setArenaAsserts(false)); // never leak the dev toggle into other suites

  it("a churn sequence stays consistent under auto-asserts", () => {
    __setArenaAsserts(true);
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 24, BufferUsage.Vertex);
    // allocate/free/grow/compact all auto-run checkInvariants — none of this should throw.
    const a = arena.allocate(pattern(0x1));
    const b = arena.allocate(pattern(0x2));
    const c = arena.allocate(pattern(0x3));
    arena.free(b);
    arena.allocate(pattern(0x4)); // reuse the b hole
    arena.allocate(pattern(0x5, 40)); // forces a grow
    arena.free(a);
    arena.free(c);
    arena.compact();
    expect(() => arena.checkInvariants()).not.toThrow();
  });

  it("checkInvariants throws the specific message on a corrupted block", () => {
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 64, BufferUsage.Vertex);
    arena.allocate(pattern(0x1));
    arena.allocate(pattern(0x2));
    // Corrupt the incrementally-tracked total — the audit must catch the drift.
    (arena as unknown as { usedTotal: number }).usedTotal += 4;
    expect(() => arena.checkInvariants()).toThrow(/usedTotal/);
  });

  it("with asserts OFF (production default) the mutators never run the walk — corruption survives an op", () => {
    // __setArenaAsserts left false by afterEach of prior tests / never enabled here.
    const device = new FakeGraphicsDevice();
    const arena = new BufferArena(device, 64, BufferUsage.Vertex);
    arena.allocate(pattern(0x1));
    (arena as unknown as { usedTotal: number }).usedTotal += 4; // would fail checkInvariants
    // The next op's audit() is a dead boolean test in prod — it must NOT throw.
    expect(() => arena.allocate(pattern(0x2))).not.toThrow();
  });
});
