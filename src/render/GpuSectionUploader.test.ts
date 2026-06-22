// Geometry-hash reuse on rebuild (TRAP 12.D / oldDataMatches). RENDERER_PLAN §12, §22.

import { describe, it, expect } from "vitest";
import { GpuSectionUploader, type GpuCommittedSection } from "./GpuSectionUploader";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { makeBuildOutput } from "../workers/BuildOutput";
import { SortType } from "../mesh/SortTypes";
import { TerrainPass } from "../types";
import { sectionKey } from "../world/SectionKey";
import { packRegionOfSection } from "../world/RegionKey";
import type { TQuad } from "../mesh/TranslucentCollector";

const KEY = sectionKey(0, 0, 0);
const tquad = (): TQuad => ({ extents: new Float32Array([1, 1, 1, 0, 0, 0]), facing: 2, dot: 1, normal: [0, 0, 1], centroid: [0.5, 0.5, 1], positions: [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]] });

function dynamicBuild(quadHash: number, fill = 0) {
  const vertexData = new ArrayBuffer(80);
  if (fill) new Uint8Array(vertexData).fill(fill); // distinct bytes stand in for a pure light-byte change
  return makeBuildOutput({
    sectionKey: KEY,
    generation: 1,
    info: { flags: 0 },
    parts: {},
    translucent: {
      vertexData,
      quadCount: 1,
      sortType: SortType.Dynamic,
      quadHash,
      indexData: new Uint32Array([0, 1, 2, 0, 2, 3]).buffer,
      quads: [tquad()],
    },
  });
}

const STRIDE = 20;
/** A solid-only build of `quads` quads whose vertex bytes are all `fill` (so two builds with the same
 *  geometry shape but different `fill` stand in for a pure light-byte change). */
function solidBuild(fill: number, quads = 2, generation = 1) {
  const vertexData = new ArrayBuffer(quads * 4 * STRIDE);
  new Uint8Array(vertexData).fill(fill);
  return makeBuildOutput({ sectionKey: KEY, generation, info: { flags: 0 }, parts: { [TerrainPass.Solid]: { vertexData, quadCount: quads } } });
}

/** A partitioned solid build: 6 quads (one per axis facing), carrying FS-2 per-facing vertex counts. */
function partitionedBuild(generation = 1) {
  const quads = 6;
  const vertexData = new ArrayBuffer(quads * 4 * STRIDE);
  const facingVertexCounts = Uint32Array.of(4, 4, 4, 4, 4, 4, 0); // one quad per axis, none unassigned
  return makeBuildOutput({ sectionKey: KEY, generation, info: { flags: 0 }, parts: { [TerrainPass.Solid]: { vertexData, quadCount: quads, facingVertexCounts } } });
}

describe("GpuSectionUploader facing partition (FS-3)", () => {
  it("carries per-pass facingVertexCounts into the committed section", () => {
    const up = new GpuSectionUploader(new FakeGraphicsDevice());
    const c = up.upload(partitionedBuild()) as GpuCommittedSection;
    expect(Array.from(c.facingVertexCounts![TerrainPass.Solid]!)).toEqual([4, 4, 4, 4, 4, 4, 0]);
  });

  it("omits facingVertexCounts for an unpartitioned build", () => {
    const up = new GpuSectionUploader(new FakeGraphicsDevice());
    const c = up.upload(solidBuild(0x11)) as GpuCommittedSection;
    expect(c.facingVertexCounts).toBeUndefined();
  });

  it("uploadLight preserves the committed facingVertexCounts (geometry unchanged)", () => {
    const up = new GpuSectionUploader(new FakeGraphicsDevice());
    const c = up.upload(partitionedBuild(1)) as GpuCommittedSection;
    const r = up.uploadLight(c, partitionedBuild(2)) as GpuCommittedSection; // same shape → in-place light
    expect(r).toBe(c);
    expect(Array.from(r.facingVertexCounts![TerrainPass.Solid]!)).toEqual([4, 4, 4, 4, 4, 4, 0]);
  });
});

describe("GpuSectionUploader.uploadLight (P6/LM-2 — in-place light reupload)", () => {
  it("updates vertex bytes in place: same allocation, NO relocation, geometry untouched", () => {
    const dev = new FakeGraphicsDevice();
    const up = new GpuSectionUploader(dev);
    const c1 = up.upload(solidBuild(0x11)) as GpuCommittedSection;
    const alloc = c1.vertexAlloc[TerrainPass.Solid];
    const arena = c1.vertexArena[TerrainPass.Solid]!;
    const rev0 = up.layoutRevision();
    const writes0 = dev.log.writes;
    const created0 = dev.log.buffersCreated;

    const r = up.uploadLight(c1, solidBuild(0x22, 2, 2));
    expect(r).toBe(c1); // same identity — allocations reused
    expect((r as GpuCommittedSection).vertexAlloc[TerrainPass.Solid]).toBe(alloc); // same allocation id
    expect((r as GpuCommittedSection).generation).toBe(2);
    expect(up.layoutRevision()).toBe(rev0); // NO arena grow/compact ⇒ draw (buffer, offset) stays valid
    expect(dev.log.buffersCreated).toBe(created0); // no new device buffers
    expect(dev.log.writes).toBeGreaterThan(writes0); // an in-place writeBuffer happened
    // The slice it updated is the SAME size as the original (byte-level correctness of the update itself is
    // covered by BufferArena.update + SectionMesher.light tests).
    expect(arena.rangeOf(alloc!).sizeBytes).toBe(2 * 4 * STRIDE);
  });

  it("returns 'mismatch' when geometry actually differs (different quad count) → caller full-uploads", () => {
    const dev = new FakeGraphicsDevice();
    const up = new GpuSectionUploader(dev);
    const c1 = up.upload(solidBuild(0x11, 2)) as GpuCommittedSection;
    expect(up.uploadLight(c1, solidBuild(0x22, 3, 2))).toBe("mismatch");
  });

  it("returns null when over the per-frame budget (caller keeps old light, retries)", () => {
    const dev = new FakeGraphicsDevice();
    const up = new GpuSectionUploader(dev, 10_000);
    const c1 = up.upload(solidBuild(0x11, 2)) as GpuCommittedSection;
    up.beginFrame(10); // tiny remaining budget
    expect(up.uploadLight(c1, solidBuild(0x22, 2, 2))).toBeNull();
  });

  it("resyncs the translucent quadHash so a later light-revert rebuild does NOT reuse a stale-light buffer", () => {
    // The light bytes are part of the hashed vertex stream, so the in-place light update changes quadHash.
    // If uploadLight left the cached hash at H(L1), a rebuild that reverts the light to L1 (out hash H(L1))
    // would mis-match-fire the TRAP-12.D reuse, keeping the L2-byte buffer and rendering wrong light.
    const dev = new FakeGraphicsDevice();
    const up = new GpuSectionUploader(dev);
    const HASH_L1 = 0x1111, HASH_L2 = 0x2222;

    const c1 = up.upload(dynamicBuild(HASH_L1)) as GpuCommittedSection;
    const l1Alloc = c1.vertexAlloc[TerrainPass.Translucent];

    // Light L1 → L2 in place: the committed hash must now track the L2 bytes on the GPU.
    const r = up.uploadLight(c1, dynamicBuild(HASH_L2)) as GpuCommittedSection;
    expect(r).toBe(c1);
    expect(r.quadHash).toBe(HASH_L2); // ← the fix: hash follows the in-place byte update

    // A coalesced rebuild that reverts the light back to L1 (geometry unchanged → out hash H(L1)). With the
    // resync, H(L2) !== H(L1) ⇒ NO reuse ⇒ the fresh L1 bytes are uploaded into a NEW allocation.
    const c2 = up.upload(dynamicBuild(HASH_L1), c1) as GpuCommittedSection;
    expect(c2.quadHash).toBe(HASH_L1);
    expect(c2.vertexAlloc[TerrainPass.Translucent]).not.toBe(l1Alloc); // did NOT resurrect the stale-light buffer
  });

  it("PRESERVES the translucent CPU buffer identity on a light update (keeps the DYNAMIC sort-trigger cache — P1.5)", () => {
    const dev = new FakeGraphicsDevice();
    const up = new GpuSectionUploader(dev);
    const c1 = up.upload(dynamicBuild(0x1111, 0x11)) as GpuCommittedSection;
    const cpuBefore = c1.translucentVertexData!; // the buffer DYNAMIC sortState/sortTrigger WeakMaps key off
    expect(new Uint8Array(cpuBefore)[0]).toBe(0x11);

    // A pure-light reupload (same geometry shape, distinct light bytes). Light never moves a vertex, so the
    // sort planes/order are unchanged — the buffer IDENTITY must be preserved so the sort-trigger cache isn't
    // reset into a spurious "first-sight" re-sort, while the BYTES are updated in place (the merger reads new light).
    const r = up.uploadLight(c1, dynamicBuild(0x2222, 0x99)) as GpuCommittedSection;
    expect(r).toBe(c1);
    expect(r.translucentVertexData).toBe(cpuBefore); // identity preserved → no spurious re-sort on a light flicker
    expect(new Uint8Array(r.translucentVertexData!)[0]).toBe(0x99); // bytes updated IN PLACE
    expect(r.quadHash).toBe(0x2222); // hash still follows the new light bytes
  });
});

describe("GpuSectionUploader translucent reuse (TRAP 12.D)", () => {
  it("reuses the translucent allocation when the rebuilt geometry hashes identical", () => {
    const dev = new FakeGraphicsDevice();
    const up = new GpuSectionUploader(dev);

    const c1 = up.upload(dynamicBuild(42)) as GpuCommittedSection;
    const tAlloc = c1.vertexAlloc[TerrainPass.Translucent]; // stable arena allocation id (§11)
    const tArena = c1.vertexArena[TerrainPass.Translucent];
    const tIndexAlloc = c1.indexAlloc[TerrainPass.Translucent]; // stable index-arena allocation id
    const createdAfterFirst = dev.log.buffersCreated;
    expect(tAlloc).toBeDefined();

    // Rebuild with the SAME translucent hash → reuse (no new arena alloc; the vertex AND index allocation
    // ids move across to the new committed section, no new device buffers).
    const c2 = up.upload(dynamicBuild(42), c1) as GpuCommittedSection;
    expect(dev.log.buffersCreated).toBe(createdAfterFirst); // no new device buffers (index reused)
    expect(c2.vertexAlloc[TerrainPass.Translucent]).toBe(tAlloc); // same allocation id
    expect(c2.vertexArena[TerrainPass.Translucent]).toBe(tArena);
    expect(c2.indexAlloc[TerrainPass.Translucent]).toBe(tIndexAlloc);
    expect(c1.vertexAlloc[TerrainPass.Translucent]).toBeUndefined(); // ownership transferred out of prev
    expect(c1.vertexArena[TerrainPass.Translucent]).toBeUndefined();
    expect(c1.indexAlloc[TerrainPass.Translucent]).toBeUndefined();

    // Freeing the old section must NOT release the reused allocation/index (still owned by c2).
    up.free(c1);
    expect(dev.liveBufferCount()).toBeGreaterThan(0);
  });

  it("re-uploads when the rebuilt translucent geometry differs (different hash)", () => {
    const dev = new FakeGraphicsDevice();
    const up = new GpuSectionUploader(dev);
    const c1 = up.upload(dynamicBuild(42)) as GpuCommittedSection;
    const created = dev.log.buffersCreated;
    const c2 = up.upload(dynamicBuild(99), c1) as GpuCommittedSection;
    // Different hash ⇒ fresh vertex AND index allocations (NOT reused). They sub-allocate from the EXISTING
    // arenas, so no new device buffer is created (the §11 win — the count stays flat under churn).
    expect(dev.log.buffersCreated).toBe(created);
    expect(c2.vertexAlloc[TerrainPass.Translucent]).not.toBe(c1.vertexAlloc[TerrainPass.Translucent]);
    expect(c2.indexAlloc[TerrainPass.Translucent]).not.toBe(c1.indexAlloc[TerrainPass.Translucent]);
  });
});

// MEM-2: re-run the uploader's relocation/T-PRES path with the device buffer pool ON. The pool is transparent
// at the uploader/arena layer — the relocation copy still completes before the old buffer is returned, and
// the committed (current) buffer always resolves correct bytes. Recycling whole GPUBuffer objects must NOT
// change which bytes a draw reads.
describe("GpuSectionUploader + MEM-2 buffer pool ON (T-PRES still green)", () => {
  function readSolid(dev: FakeGraphicsDevice, c: GpuCommittedSection): Uint8Array {
    const arena = c.vertexArena[TerrainPass.Solid]!;
    const r = arena.rangeOf(c.vertexAlloc[TerrainPass.Solid]!);
    return dev.read(arena.gpuBuffer).subarray(r.offsetBytes, r.offsetBytes + r.sizeBytes);
  }

  it("repeated growing uploads keep presented bytes correct while the arena grows + pool-returns buffers", () => {
    const dev = new FakeGraphicsDevice({ bufferPool: true });
    const up = new GpuSectionUploader(dev);
    // Many increasingly-large uploads of distinct sections force the solid arena to grow several times; each
    // grow RETURNS its old buffer to the device pool (the relocation copy completes BEFORE the return — T-PRES).
    // Recycling whole GPUBuffer objects must never change which bytes a presented draw resolves.
    const committed: { c: GpuCommittedSection; fill: number }[] = [];
    for (let i = 1; i <= 200; i++) {
      const quads = i; // monotonically larger builds → the arena keeps growing past its initial capacity
      const fill = 0x20 + (i % 200);
      const vertexData = new ArrayBuffer(quads * 4 * STRIDE);
      new Uint8Array(vertexData).fill(fill);
      const build = makeBuildOutput({ sectionKey: sectionKey(i, 0, 0), generation: 1, info: { flags: 0 }, parts: { [TerrainPass.Solid]: { vertexData, quadCount: quads } } });
      committed.push({ c: up.upload(build) as GpuCommittedSection, fill });
    }
    // The arena grew (so old buffers were pool-RETURNED rather than destroyed) — the recycling path is live.
    expect(dev.log.poolReturns).toBeGreaterThan(0);
    expect(dev.log.buffersDestroyed).toBe(0); // pool ON ⇒ a grow's old buffer is retained, never .destroy()'d
    // After all the grows, EVERY surviving section's bytes still resolve correctly off its CURRENT arena buffer
    // (a draw that resolved a returned, pooled buffer would throw the use-while-pooled guard in read()).
    for (const { c, fill } of committed) expect(readSolid(dev, c)[0]).toBe(fill);
  });
});

// MEM-1: per-region arenas (DEFAULT-OFF). The committed section stores the resolved arena REFERENCE per
// pass, so buildSectionDraws resolves (gpuBuffer, offset) IDENTICALLY whether one global arena or a
// per-region arena backs the bytes — no draw-layer change (TRAP MEM-1.A: this does NOT reduce draw calls;
// WebGPU has no multidraw). disposeRegion frees a region's arenas as WHOLE buffers (O(1), zero
// copyBufferBatch). Single-arena mode (the shipped default) must be byte-for-byte unchanged.
describe("GpuSectionUploader regionArenas (MEM-1 — per-region contiguous arenas)", () => {
  const STRIDE2 = 20;
  function solid(sx: number, sy: number, sz: number, fill: number, quads = 2) {
    const vertexData = new ArrayBuffer(quads * 4 * STRIDE2);
    new Uint8Array(vertexData).fill(fill);
    return makeBuildOutput({ sectionKey: sectionKey(sx, sy, sz), generation: 1, info: { flags: 0 }, parts: { [TerrainPass.Solid]: { vertexData, quadCount: quads } } });
  }
  function readSolid(dev: FakeGraphicsDevice, c: GpuCommittedSection): Uint8Array {
    const arena = c.vertexArena[TerrainPass.Solid]!;
    const r = arena.rangeOf(c.vertexAlloc[TerrainPass.Solid]!);
    return dev.read(arena.gpuBuffer).subarray(r.offsetBytes, r.offsetBytes + r.sizeBytes);
  }

  it("ON: a section's draw resolves a valid (gpuBuffer, offset) with the SAME bytes — region-backed", () => {
    const dev = new FakeGraphicsDevice();
    const up = new GpuSectionUploader(dev, Infinity, true);
    const c = up.upload(solid(0, 0, 0, 0xab)) as GpuCommittedSection;
    const arena = c.vertexArena[TerrainPass.Solid]!;
    const alloc = c.vertexAlloc[TerrainPass.Solid]!;
    // The (gpuBuffer, offset) contract is identical to single-arena mode: a real buffer + an O(1) byte range.
    expect(arena.gpuBuffer).toBeDefined();
    expect(typeof arena.rangeOf(alloc).offsetBytes).toBe("number");
    expect(readSolid(dev, c)[0]).toBe(0xab); // the bytes the draw will read are correct
  });

  it("ON: sections in DIFFERENT regions land in DIFFERENT arena instances; same region shares one", () => {
    const dev = new FakeGraphicsDevice();
    const up = new GpuSectionUploader(dev, Infinity, true);
    const a = up.upload(solid(0, 0, 0, 0x01)) as GpuCommittedSection; // region (0,0,0)
    const b = up.upload(solid(1, 0, 0, 0x02)) as GpuCommittedSection; // region (0,0,0) — same (x<8)
    const c = up.upload(solid(8, 0, 0, 0x03)) as GpuCommittedSection; // region (1,0,0) — different
    expect(a.vertexArena[TerrainPass.Solid]).toBe(b.vertexArena[TerrainPass.Solid]); // shared per region
    expect(a.vertexArena[TerrainPass.Solid]).not.toBe(c.vertexArena[TerrainPass.Solid]); // distinct region arena
    // All three resolve their own correct bytes regardless of which arena backs them.
    expect(readSolid(dev, a)[0]).toBe(0x01);
    expect(readSolid(dev, b)[0]).toBe(0x02);
    expect(readSolid(dev, c)[0]).toBe(0x03);
  });

  it("ON: disposeRegion frees a region's pass arenas as WHOLE buffers with ZERO copyBufferBatch", () => {
    const dev = new FakeGraphicsDevice();
    const up = new GpuSectionUploader(dev, Infinity, true);
    up.upload(solid(0, 0, 0, 0x01)); // region (0,0,0)
    up.upload(solid(1, 0, 0, 0x02)); // region (0,0,0), same arena
    up.upload(solid(8, 0, 0, 0x03)); // region (1,0,0), a separate arena that must SURVIVE the dispose
    const liveBefore = dev.liveBufferCount();
    const copiesBefore = dev.log.copies;
    const submitsBefore = dev.log.submits;

    up.disposeRegion(packRegionOfSection(0, 0, 0)); // contiguous reclaim of region (0,0,0)

    expect(dev.log.copies).toBe(copiesBefore); // ZERO relocation copies — a whole-buffer destroy, not a compact
    expect(dev.log.submits).toBe(submitsBefore); // no copy submit either
    expect(dev.log.buffersDestroyed).toBeGreaterThan(0); // the region's arena buffer was destroyed
    expect(dev.liveBufferCount()).toBeLessThan(liveBefore); // GPU memory actually dropped (O(1) reclaim)
    // The OTHER region's section still resolves its bytes — its arena was untouched.
    const c = up.upload(solid(9, 0, 0, 0x04)) as GpuCommittedSection; // region (1,0,0), still alive
    expect(readSolid(dev, c)[0]).toBe(0x04);
  });

  it("OFF (single-arena, the shipped default): disposeRegion is a NO-OP; all sections share one arena", () => {
    const dev = new FakeGraphicsDevice();
    const up = new GpuSectionUploader(dev); // regionArenas defaults false
    const a = up.upload(solid(0, 0, 0, 0x01)) as GpuCommittedSection;
    const c = up.upload(solid(8, 0, 0, 0x02)) as GpuCommittedSection; // different region, but single-arena
    expect(a.vertexArena[TerrainPass.Solid]).toBe(c.vertexArena[TerrainPass.Solid]); // ONE global arena
    const destroyedBefore = dev.log.buffersDestroyed;
    up.disposeRegion(packRegionOfSection(0, 0, 0)); // no-op in single-arena mode
    expect(dev.log.buffersDestroyed).toBe(destroyedBefore); // nothing destroyed — per-section free() owns reclaim
    expect(readSolid(dev, a)[0]).toBe(0x01); // unchanged
    expect(readSolid(dev, c)[0]).toBe(0x02);
  });
});
