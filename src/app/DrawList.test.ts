// F3 — incremental draw-list membership (DrawList) MUST stay byte-equivalent to a full collectDraws()
// across every mutation, especially an arena grow/compaction that RELOCATES unchanged sections (TRAP 4.A:
// the relocation the old once-per-commit full rebuild hid). The incremental cache holds (buffer, offset)
// snapshots; if it failed to re-resolve on relocation, a cached section would draw from a destroyed buffer
// at a stale offset — these tests would catch it as a flatten ≠ collectDraws divergence.

import { describe, it, expect } from "vitest";
import { collectDraws, buildSectionDraws, DrawList, occlusionVisibleSet, filterDrawsByOcclusion, type RegionCullOptions } from "./scene";
import { SectionStore, AcceptResult } from "../world/SectionStore";
import { RegionStore } from "../world/RegionStore";
import { GpuSectionUploader } from "../render/GpuSectionUploader";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { sectionKey, type SectionKey } from "../world/SectionKey";
import { makeBuildOutput } from "../workers/BuildOutput";
import { DirtyReason, TerrainPass, type Vec3 } from "../types";
import { SortType } from "../mesh/SortTypes";
import type { TQuad } from "../mesh/TranslucentCollector";
import type { SectionDraw } from "../render/TerrainRenderer";
import { Camera } from "../camera/Camera";
import { Frustum } from "../camera/Frustum";

const tquad = (): TQuad => ({ extents: new Float32Array([1, 1, 1, 0, 0, 0]), facing: 2, dot: 1, normal: [0, 0, 1], centroid: [0.5, 0.5, 1], positions: [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]] });

function newStore(): SectionStore {
  return new SectionStore(new GpuSectionUploader(new FakeGraphicsDevice()));
}

/** Commit `bytes` of SOLID geometry to a section (markDirty → accept → commit), matching generations. */
function commitSolid(store: SectionStore, key: SectionKey, bytes: number, quads = 1): void {
  const gen = store.markDirty(key, DirtyReason.InitialLoad);
  const build = makeBuildOutput({
    sectionKey: key,
    generation: gen,
    info: { flags: 0 },
    parts: { [TerrainPass.Solid]: { vertexData: new ArrayBuffer(bytes), quadCount: quads } },
  });
  expect(store.acceptBuild(build)).toBe(AcceptResult.Accepted);
  expect(store.commit(store.get(key)!)).toBe(true);
}

/** Commit a DYNAMIC translucent section (allocates an index-arena slice → index buffer + offset). */
function commitTranslucent(store: SectionStore, key: SectionKey, hash: number): void {
  const gen = store.markDirty(key, DirtyReason.InitialLoad);
  const build = makeBuildOutput({
    sectionKey: key,
    generation: gen,
    info: { flags: 0 },
    parts: {},
    translucent: { vertexData: new ArrayBuffer(80), quadCount: 1, sortType: SortType.Dynamic, quadHash: hash, indexData: new Uint32Array([0, 1, 2, 0, 2, 3]).buffer, quads: [tquad()] },
  });
  expect(store.acceptBuild(build)).toBe(AcceptResult.Accepted);
  expect(store.commit(store.get(key)!)).toBe(true);
}

/** Commit an EMPTY build (last block removed) — clears `presented` so the section drops out of the list. */
function commitEmpty(store: SectionStore, key: SectionKey): void {
  const gen = store.markDirty(key, DirtyReason.Edit);
  const build = makeBuildOutput({ sectionKey: key, generation: gen, info: { flags: 0 }, parts: {} });
  expect(store.acceptBuild(build)).toBe(AcceptResult.Accepted);
  store.commit(store.get(key)!);
}

/** Order-independent identity of a draw: origin + the RESOLVED buffer id + byte offset + pass. A stale
 *  cache after a relocation would show a different buffer id / offset than the fresh collectDraws. */
const key1 = (d: SectionDraw): string =>
  JSON.stringify({
    o: d.originBlocks,
    vid: (d.vertex as unknown as { id: number }).id,
    vo: d.vertexOffset ?? 0,
    q: d.quadCount,
    p: d.pass,
    iid: d.index ? (d.index as unknown as { id: number }).id : -1,
    io: d.indexOffset ?? 0,
  });

const norm = (draws: readonly SectionDraw[]): string[] => draws.map(key1).sort();

/** Assert the incremental list equals the full rebuild as a multiset. collectDraws() does NOT drain the
 *  change set, so compute it FIRST; then flatten() drains + reconciles against the same store state. */
function assertEquiv(store: SectionStore, dl: DrawList): void {
  const expected = norm(collectDraws(store).draws);
  const actual = norm(dl.flatten(store));
  expect(actual).toEqual(expected);
}

describe("DrawList incremental membership (F3) — equivalence with collectDraws", () => {
  it("matches after the initial full build", () => {
    const store = newStore();
    commitSolid(store, sectionKey(0, 0, 0), 64);
    commitSolid(store, sectionKey(1, 0, 0), 128);
    commitSolid(store, sectionKey(0, 1, 0), 96);
    assertEquiv(store, new DrawList());
  });

  it("matches after a same-size re-mesh (incremental patch, no relocation)", () => {
    const store = newStore();
    const dl = new DrawList();
    commitSolid(store, sectionKey(0, 0, 0), 64);
    commitSolid(store, sectionKey(1, 0, 0), 64);
    dl.flatten(store); // cache
    commitSolid(store, sectionKey(0, 0, 0), 64); // re-mesh same size
    assertEquiv(store, dl);
  });

  it("matches after a re-mesh that changes size (the section's own alloc relocates)", () => {
    const store = newStore();
    const dl = new DrawList();
    commitSolid(store, sectionKey(0, 0, 0), 64);
    commitSolid(store, sectionKey(1, 0, 0), 64);
    dl.flatten(store);
    commitSolid(store, sectionKey(0, 0, 0), 4096); // bigger → fresh alloc at a new offset
    assertEquiv(store, dl);
  });

  it("matches after an arena GROW relocates already-cached unchanged sections (TRAP 4.A)", () => {
    const store = newStore();
    const dl = new DrawList();
    // Two small sections, cached by the DrawList.
    commitSolid(store, sectionKey(0, 0, 0), 4096);
    commitSolid(store, sectionKey(1, 0, 0), 4096);
    dl.flatten(store);
    const revBefore = store.uploadLayoutRevision();
    // Commit enough large geometry to blow past INITIAL_ARENA_BYTES (256 KiB) → the Solid arena GROWS,
    // relocating sections (0,0,0)/(1,0,0) into a fresh buffer at new offsets.
    for (let i = 0; i < 6; i++) commitSolid(store, sectionKey(2 + i, 0, 0), 64 * 1024);
    expect(store.uploadLayoutRevision()).toBeGreaterThan(revBefore); // a relocation really happened
    assertEquiv(store, dl); // the cached small sections must show their NEW buffer/offset, not stale ones
  });

  it("drops a section emptied to no geometry", () => {
    const store = newStore();
    const dl = new DrawList();
    commitSolid(store, sectionKey(0, 0, 0), 64);
    commitSolid(store, sectionKey(1, 0, 0), 64);
    dl.flatten(store);
    commitEmpty(store, sectionKey(0, 0, 0)); // presented cleared
    assertEquiv(store, dl);
    // The remaining list is exactly section (1,0,0).
    const flat = dl.flatten(store);
    expect(flat.every((d) => d.originBlocks[0] === 16)).toBe(true);
  });

  it("drops a disposed section", () => {
    const store = newStore();
    const dl = new DrawList();
    commitSolid(store, sectionKey(0, 0, 0), 64);
    commitSolid(store, sectionKey(1, 0, 0), 64);
    dl.flatten(store);
    store.dispose(sectionKey(0, 0, 0));
    assertEquiv(store, dl);
  });

  it("matches for translucent sections (index buffer + offset carried)", () => {
    const store = newStore();
    const dl = new DrawList();
    commitTranslucent(store, sectionKey(0, 0, 0), 11);
    commitSolid(store, sectionKey(1, 0, 0), 64);
    commitTranslucent(store, sectionKey(2, 0, 0), 22);
    assertEquiv(store, dl);
    // Spot-check that a translucent draw really carries an index slice.
    const t = dl.flatten(store).find((d) => d.pass === TerrainPass.Translucent);
    expect(t?.index).toBeDefined();
    expect(typeof t?.indexOffset).toBe("number");
  });

  it("REUSES unchanged sections' draw objects on an incremental patch, but RE-RESOLVES them on relocation", () => {
    const store = newStore();
    const dl = new DrawList();
    commitSolid(store, sectionKey(0, 0, 0), 4096);
    commitSolid(store, sectionKey(1, 0, 0), 4096);
    const before = dl.flatten(store);
    const sec1 = before.find((d) => d.originBlocks[0] === 16)!; // section (1,0,0)'s draw object

    // Incrementally patch a DIFFERENT section → section (1,0,0)'s object is reused (same reference).
    commitSolid(store, sectionKey(0, 0, 0), 4096);
    const afterPatch = dl.flatten(store);
    expect(afterPatch.find((d) => d.originBlocks[0] === 16)).toBe(sec1); // O(changed): untouched object reused

    // Force a grow → every cached object is re-resolved (a fresh object with the new offset).
    for (let i = 0; i < 6; i++) commitSolid(store, sectionKey(2 + i, 0, 0), 64 * 1024);
    const afterGrow = dl.flatten(store);
    expect(afterGrow.find((d) => d.originBlocks[0] === 16)).not.toBe(sec1); // relocation → re-resolved
    assertEquiv(store, dl);
  });

  it("a dispose populates presentedChanges (the signal the viewer's draw-list refresh keys on, not tick.committed)", () => {
    // The viewer refreshes this.draws when store.presentedChanges is non-empty. A dispose commits nothing
    // (tick.committed stays 0) but MUST still register so the freed section drops out of the draw list.
    const store = newStore();
    commitSolid(store, sectionKey(0, 0, 0), 64);
    store.clearPresentedChanges(); // drain the commit
    expect(store.presentedChanges.size).toBe(0);
    store.dispose(sectionKey(0, 0, 0));
    expect(store.presentedChanges.has(sectionKey(0, 0, 0))).toBe(true); // dispose is visible to the refresh trigger
  });

  it("buildSectionDraws returns [] for a section with no presented geometry", () => {
    const store = newStore();
    store.markDirty(sectionKey(5, 5, 5), DirtyReason.Edit); // exists but never committed
    expect(buildSectionDraws(store.get(sectionKey(5, 5, 5))!)).toEqual([]);
  });
});

// MEM-1 PARITY GATE (the headline verifiable property). With the region tier ON (per-region arenas + the
// region cull pre-pass) the occlusion-visible SET and the final filtered flat draw list must be EQUAL
// element-for-element to the flags-OFF run for the same camera — proving the region tier changes only WHICH
// arena instance backs bytes and WHICH sections the BFS visits, never the bytes or the visible result (TRAP
// MEM-1.A: no draw-call change — WebGPU has no multidraw; TRAP MEM-1.C: the cull never changes the set).
describe("MEM-1 — region cull + region arenas: byte-identical / set-equal parity with the OFF path", () => {
  /** A SectionStore with the region tier wired (per-region arenas + RegionStore reclaim hook). */
  function regionStore(): SectionStore {
    const dev = new FakeGraphicsDevice();
    const uploader = new GpuSectionUploader(dev, Infinity, /* regionArenas */ true);
    const regions = new RegionStore((rk) => uploader.disposeRegion(rk));
    return new SectionStore(uploader, regions);
  }

  /** Commit the SAME multi-region world into a store: a slab spanning several 8×4×8 regions on x and z. */
  function buildWorld(store: SectionStore): void {
    for (let sx = -10; sx <= 10; sx++)
      for (let sz = -10; sz <= 10; sz++)
        commitSolid(store, sectionKey(sx, 0, sz), 64 + ((sx * 7 + sz) & 7) * 4); // varied byte sizes
  }

  /** The viewer's exact region-cull options: a region's world AABB tested against the same frustum. */
  function regionCullOpts(frustum: Frustum): RegionCullOptions {
    return { regionInFrustum: (lx, ly, lz, hx, hy, hz) => frustum.testAab(lx, ly, lz, hx, hy, hz) };
  }

  /** Visible set + filtered flat draw list for a store at a fixed camera, region cull optionally ON. */
  function runFrame(store: SectionStore, cam: Camera, regionCullOn: boolean): { visible: Set<number>; draws: SectionDraw[] } {
    const vp = cam.viewProjection();
    const frustum = new Frustum().setFromViewProjection(vp);
    const inFrustum = (sx: number, sy: number, sz: number) => frustum.testSection(sx * 16, sy * 16, sz * 16);
    const camPos = cam.position as Vec3;
    const opts = regionCullOn && store.regionStore ? regionCullOpts(frustum) : undefined;
    const visible = occlusionVisibleSet(store, camPos, inFrustum, undefined, undefined, opts);
    const flat = collectDraws(store).draws;
    const filtered = filterDrawsByOcclusion(flat, visible, []);
    return { visible, draws: filtered };
  }

  function camera(): Camera {
    const cam = new Camera();
    cam.target = [0, 0, 0];
    cam.distance = 80;
    cam.yaw = Math.PI * 0.3;
    cam.pitch = Math.PI * 0.15;
    cam.aspect = 1.6;
    return cam;
  }

  it("ON vs OFF: identical occlusion-visible SET for a fixed camera over multi-region geometry", () => {
    const off = newStore(); buildWorld(off);
    const on = regionStore(); buildWorld(on);
    expect(on.regionStore).toBeDefined();
    const cam = camera();
    const a = runFrame(off, cam, false);
    const b = runFrame(on, cam, true);
    expect([...b.visible].sort((x, y) => x - y)).toEqual([...a.visible].sort((x, y) => x - y));
    expect(b.visible.size).toBeGreaterThan(0); // the world is actually visible (a non-trivial parity check)
  });

  it("ON vs OFF: identical FLAT FILTERED draw list (set equality, byte-identical resolved draws)", () => {
    const off = newStore(); buildWorld(off);
    const on = regionStore(); buildWorld(on);
    const cam = camera();
    const a = runFrame(off, cam, false);
    const b = runFrame(on, cam, true);
    // The draws carry resolved (originBlocks, quadCount, pass) — region arenas change the buffer INSTANCE but
    // the per-section geometry (origin/quad count/pass) is identical, so the membership multiset matches. We
    // key on the section-identifying fields (origin + pass + quadCount), not the buffer id (which legitimately
    // differs between a single global arena and a per-region arena — that's the whole point of MEM-1).
    const sig = (d: SectionDraw) => JSON.stringify({ o: d.originBlocks, q: d.quadCount, p: d.pass });
    expect(b.draws.map(sig).sort()).toEqual(a.draws.map(sig).sort());
    // The resolved bytes themselves are identical: each region-backed draw reads the same vertex bytes the
    // single-arena draw would (covered by GpuSectionUploader.test.ts readSolid); here we assert the visible
    // section SET drives an identical draw count → passCount-equivalent (no multidraw, no draw-count change).
    expect(b.draws.length).toBe(a.draws.length);
  });

  it("ON vs OFF: parity holds across several camera orientations (the pre-pass never pops far terrain)", () => {
    const off = newStore(); buildWorld(off);
    const on = regionStore(); buildWorld(on);
    for (const yaw of [0, Math.PI * 0.5, Math.PI, Math.PI * 1.5, Math.PI * 0.27]) {
      const cam = camera();
      cam.yaw = yaw;
      const a = runFrame(off, cam, false);
      const b = runFrame(on, cam, true);
      expect([...b.visible].sort((x, y) => x - y)).toEqual([...a.visible].sort((x, y) => x - y));
    }
  });

  it("disposeRegion via the store: emptying a region frees its bytes with O(1) reclaim, parity preserved", () => {
    const dev = new FakeGraphicsDevice();
    const uploader = new GpuSectionUploader(dev, Infinity, true);
    const regions = new RegionStore((rk) => uploader.disposeRegion(rk));
    const on = new SectionStore(uploader, regions);
    buildWorld(on);
    const off = newStore(); buildWorld(off);

    // Clear an entire region (region (1,0,0): sections sx∈[8,10], sz∈[-10,10] that fall in region x=1)…
    // simplest: dispose every section whose region x-coord is 1 (sx 8..10) and z-region 0 (sz 0..7).
    const copiesBefore = dev.log.copies;
    for (let sx = 8; sx <= 10; sx++)
      for (let sz = 0; sz <= 7; sz++) { on.dispose(sectionKey(sx, 0, sz)); off.dispose(sectionKey(sx, 0, sz)); }
    // The region's whole-buffer reclaim used ZERO relocation copies (O(1) disposeRegion, not a compact).
    expect(dev.log.copies).toBe(copiesBefore);

    // After the clear, ON and OFF still agree element-for-element (no flicker / no stale draws).
    const cam = camera();
    const a = runFrame(off, cam, false);
    const b = runFrame(on, cam, true);
    expect([...b.visible].sort((x, y) => x - y)).toEqual([...a.visible].sort((x, y) => x - y));
    expect(b.draws.length).toBe(a.draws.length);
  });
});
