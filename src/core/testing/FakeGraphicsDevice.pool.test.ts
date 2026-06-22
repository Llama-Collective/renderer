// MEM-2: direct device-level freed-GPUBuffer-object pool tests on FakeGraphicsDevice. The GPU-free arena/
// T-PRES suites run on this device, so the pool must mirror WebGPUDevice exactly: (usage, actual-size)
// bucketing, best-fit hits, real-destroy-only liveBuffers accounting, a 'pooled' state distinct from
// 'destroyed', and a bounded (count + bytes) pool that evicts largest/oldest.

import { describe, it, expect } from "vitest";
import { FakeGraphicsDevice } from "./FakeGraphicsDevice";
import { BufferUsage } from "../GraphicsDevice";

describe("FakeGraphicsDevice MEM-2 buffer pool", () => {
  it("flag OFF: destroy really destroys, no pool activity, use-after-destroy still throws", () => {
    const dev = new FakeGraphicsDevice(); // default off
    const h = dev.createBuffer({ sizeBytes: 64, usage: BufferUsage.Vertex });
    expect(dev.liveBufferCount()).toBe(1);
    dev.destroyBuffer(h);
    expect(dev.liveBufferCount()).toBe(0);
    expect(dev.log.buffersDestroyed).toBe(1);
    expect(dev.log.poolReturns).toBe(0);
    expect(dev.log.poolHits).toBe(0);
    // A second create gets a BRAND-NEW buffer (no recycling when off).
    dev.createBuffer({ sizeBytes: 64, usage: BufferUsage.Vertex });
    expect(dev.log.buffersCreated).toBe(2);
    // The destroyed handle is still a use-after-destroy.
    expect(() => dev.read(h)).toThrow(/after destroy/);
  });

  it("flag ON: destroy → create of the same usage+fit RECYCLES the object (poolHit, no new create)", () => {
    const dev = new FakeGraphicsDevice({ bufferPool: true });
    const h1 = dev.createBuffer({ sizeBytes: 64, usage: BufferUsage.Vertex });
    expect(dev.log.buffersCreated).toBe(1);
    dev.destroyBuffer(h1); // returned to the pool, NOT destroyed
    expect(dev.log.buffersDestroyed).toBe(0);
    expect(dev.log.poolReturns).toBe(1);
    expect(dev.liveBufferCount()).toBe(1); // pooled buffer is still live GPU memory

    const h2 = dev.createBuffer({ sizeBytes: 64, usage: BufferUsage.Vertex }); // best-fit hit
    expect(dev.log.poolHits).toBe(1);
    expect(dev.log.buffersCreated).toBe(1); // NO new real allocation
    // The recycled buffer is usable again (un-pooled) and writable.
    dev.writeBuffer(h2, 0, new Uint8Array(64).fill(0x7));
    expect(dev.read(h2)[0]).toBe(0x7);
  });

  it("does NOT reuse across usages: a pooled Vertex buffer never satisfies an Index request (TRAP MEM-2.A)", () => {
    const dev = new FakeGraphicsDevice({ bufferPool: true });
    const v = dev.createBuffer({ sizeBytes: 64, usage: BufferUsage.Vertex });
    dev.destroyBuffer(v); // Vertex buffer now pooled
    expect(dev.log.poolReturns).toBe(1);

    dev.createBuffer({ sizeBytes: 64, usage: BufferUsage.Index }); // same size, DIFFERENT usage
    expect(dev.log.poolHits).toBe(0); // no cross-usage reuse
    expect(dev.log.buffersCreated).toBe(2); // a fresh Index buffer was really created
  });

  it("larger-than-pooled request is a MISS (best-fit on actual size; never hand back too-small — TRAP MEM-2.B)", () => {
    const dev = new FakeGraphicsDevice({ bufferPool: true });
    const small = dev.createBuffer({ sizeBytes: 32, usage: BufferUsage.Vertex });
    dev.destroyBuffer(small); // 32-byte buffer pooled
    expect(dev.log.poolReturns).toBe(1);

    dev.createBuffer({ sizeBytes: 64, usage: BufferUsage.Vertex }); // needs MORE than the pooled 32
    expect(dev.log.poolHits).toBe(0); // miss — the pooled buffer is too small
    expect(dev.log.buffersCreated).toBe(2);
  });

  it("best-fit picks the SMALLEST fitting pooled buffer, keeping its ACTUAL (>= requested) size", () => {
    const dev = new FakeGraphicsDevice({ bufferPool: true });
    const a = dev.createBuffer({ sizeBytes: 128, usage: BufferUsage.Vertex });
    const b = dev.createBuffer({ sizeBytes: 64, usage: BufferUsage.Vertex });
    dev.destroyBuffer(a); // pool: [128]
    dev.destroyBuffer(b); // pool: [128, 64]

    const h = dev.createBuffer({ sizeBytes: 48, usage: BufferUsage.Vertex }); // both fit; 64 is the best fit
    expect(dev.log.poolHits).toBe(1);
    // It KEEPS its actual 64-byte capacity (not shrunk to the 48-byte request) — the over-large tail is safe
    // because the arena only writes/reads its [0, requested) slice.
    expect(dev.read(h).length).toBe(64);
    // The 128 buffer is still pooled (only the best-fit 64 was taken).
    expect(dev.liveBufferCount()).toBe(2); // 128 (pooled) + 64 (now in service)
  });

  it("liveBuffers decrements ONLY on a real destroy/eviction, never on a pool-return (TRAP MEM-2.C)", () => {
    const dev = new FakeGraphicsDevice({ bufferPool: true });
    // Fill the Vertex bucket past its count cap (8) so the 9th return forces an eviction.
    const handles = [];
    for (let i = 0; i < 9; i++) handles.push(dev.createBuffer({ sizeBytes: 16 * (i + 1), usage: BufferUsage.Vertex }));
    expect(dev.liveBufferCount()).toBe(9);

    // Return 8 — all pooled, none destroyed: liveBufferCount unchanged (still 9 alive GPU objects).
    for (let i = 0; i < 8; i++) dev.destroyBuffer(handles[i]);
    expect(dev.log.buffersDestroyed).toBe(0);
    expect(dev.log.poolEvictions).toBe(0);
    expect(dev.liveBufferCount()).toBe(9);

    // Return the 9th (the largest, 144 bytes) — bucket over count cap (8) ⇒ evict the LARGEST/oldest for real.
    dev.destroyBuffer(handles[8]);
    expect(dev.log.poolEvictions).toBe(1);
    expect(dev.log.buffersDestroyed).toBe(1); // the eviction is a REAL destroy
    expect(dev.liveBufferCount()).toBe(8); // …and ONLY now does the live gauge drop
  });

  it("a pooled buffer is in a state DISTINCT from destroyed (use-while-pooled throws its own guard)", () => {
    const dev = new FakeGraphicsDevice({ bufferPool: true });
    const h = dev.createBuffer({ sizeBytes: 64, usage: BufferUsage.Vertex });
    dev.destroyBuffer(h); // pooled, NOT destroyed
    // Touching it via the stale handle while it sits in the pool is rejected — but as 'pooled', not 'destroyed'.
    expect(() => dev.read(h)).toThrow(/while pooled/);
    expect(() => dev.read(h)).not.toThrow(/after destroy/);
  });

  it("T-PRES guard with the pool ON: a returned (post-copy) buffer is never read once recycled", () => {
    // Models grow/compact: write SRC, copy SRC→DST, return SRC to the pool. Reading the returned handle must
    // fail (it is pooled), and recycling it for a new buffer rebinds the handle — the old handle stays invalid.
    const dev = new FakeGraphicsDevice({ bufferPool: true });
    const src = dev.createBuffer({ sizeBytes: 64, usage: BufferUsage.Index });
    const dst = dev.createBuffer({ sizeBytes: 64, usage: BufferUsage.Index });
    dev.writeBuffer(src, 0, new Uint8Array(64).fill(0x9));
    dev.copyBuffer(src, { offsetBytes: 0, sizeBytes: 64 }, dst, 0); // copy completes BEFORE the return
    expect(dev.read(dst)[0]).toBe(0x9); // relocation landed in the live (current) buffer
    dev.destroyBuffer(src); // pool-return of the OLD post-copy buffer
    expect(() => dev.read(src)).toThrow(/while pooled/); // a presented draw could never resolve the old buffer
    // Recycling it for a fresh Index allocation makes the NEW handle valid; the old `src` handle is still pooled-invalid.
    const reused = dev.createBuffer({ sizeBytes: 64, usage: BufferUsage.Index });
    expect(dev.log.poolHits).toBe(1);
    expect(() => dev.read(reused)).not.toThrow();
  });
});
