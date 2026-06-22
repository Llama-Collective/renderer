// OCC-1 — the home-section-keyed reachability cache + zero-alloc reused Set. These exercise the cache
// PRIMITIVES that SchematicViewer.occludedTerrainDrawsCached composes:
//   1. occlusionVisibleSet writes into a caller-owned reused Set (cleared first, same object returned).
//   2. occlusionHomeSection (the cache KEY) is stable across an in-section move and only crosses on a real
//      section-boundary crossing — so the caller re-runs the BFS only on a crossing, never on rotation.
//   3. The BFS-run counter (instrument.job.occBfsRuns — the win metric) increments exactly once per actual
//      occlusionVisibleSet call, and a home-section-stable move that REUSES the cache (gated by the key)
//      runs the BFS zero extra times while returning identical membership.
//
// We drive a real SectionStore (committed geometry) so getByCoords/presented are the live read path the BFS
// uses — no mocks of the graph.

import { describe, it, expect, beforeEach } from "vitest";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { GpuSectionUploader } from "../render/GpuSectionUploader";
import { SectionStore } from "../world/SectionStore";
import { sectionKey } from "../world/SectionKey";
import { DirtyReason, TerrainPass, type Vec3 } from "../types";
import type { SectionBuildOutput } from "../workers/BuildOutput";
import { instrument } from "../core/Instrument";
import { occlusionVisibleSet, occlusionHomeSection } from "./scene";
import { packSectionCoord } from "../world/OcclusionCuller";

function makeBuild(sx: number, sy: number, sz: number, generation = 1, visibility = 0x7fff): SectionBuildOutput {
  const K = sectionKey(sx, sy, sz);
  const bytes = 64;
  return {
    sectionKey: K,
    generation,
    info: { flags: 0, visibilityData: Uint32Array.of(visibility) },
    parts: { [TerrainPass.Solid]: { vertexData: new ArrayBuffer(bytes), quadCount: 4 } },
    approxBytes: bytes,
    dispose() {},
  } as unknown as SectionBuildOutput;
}

/** Commit a drawable section at the given section coords (so it has geometry + presented). */
function commitSection(store: SectionStore, sx: number, sy: number, sz: number): void {
  const K = sectionKey(sx, sy, sz);
  store.markDirty(K, DirtyReason.InitialLoad);
  store.acceptBuild(makeBuild(sx, sy, sz));
  store.commit(store.get(K)!);
}

/** A small populated region: a 3×1×3 slab of drawable sections around the origin. */
function buildStore(): SectionStore {
  const store = new SectionStore(new GpuSectionUploader(new FakeGraphicsDevice()));
  for (let x = -1; x <= 1; x++) for (let z = -1; z <= 1; z++) commitSection(store, x, 0, z);
  return store;
}

/** Camera world position at the CENTER of section (sx,sy,sz). */
function centerOf(sx: number, sy: number, sz: number): Vec3 {
  return [sx * 16 + 8, sy * 16 + 8, sz * 16 + 8];
}

describe("OCC-1 occlusion cache primitives", () => {
  beforeEach(() => {
    instrument.enabled = true;
    instrument.reset();
  });

  it("occlusionVisibleSet writes into a provided reused Set (cleared first, same object returned)", () => {
    const store = buildStore();
    const reuse = new Set<number>([123456, 999999]); // pre-seeded junk that must be cleared
    const cam = centerOf(0, 0, 0);
    const ret = occlusionVisibleSet(store, cam, () => true, reuse);

    expect(ret).toBe(reuse); // same object — zero new Set alloc
    expect(reuse.has(123456)).toBe(false); // cleared
    expect(reuse.has(999999)).toBe(false);
    expect(reuse.size).toBeGreaterThan(0); // populated with the reachable sections
  });

  it("a reachability query (pass-through frustum) yields the same membership regardless of orientation", () => {
    const store = buildStore();
    const cam = centerOf(0, 0, 0);
    const a = new Set<number>();
    const b = new Set<number>();
    // Two different "orientations" — but a reachability query ignores the frustum entirely (always-true).
    occlusionVisibleSet(store, cam, () => true, a);
    occlusionVisibleSet(store, cam, () => true, b);
    expect([...a].sort()).toEqual([...b].sort());
  });

  it("occlusionHomeSection is STABLE across an in-section camera move (no boundary crossing)", () => {
    const store = buildStore();
    // Two positions inside the SAME section (0,0,0): block 1 and block 14 of a 16-wide section.
    const h1 = occlusionHomeSection(store, [1, 8, 1]);
    const h2 = occlusionHomeSection(store, [14, 8, 14]); // still section 0 on every axis
    expect(h1).toEqual([0, 0, 0]);
    expect(h2).toEqual([0, 0, 0]); // unchanged ⇒ the cache key matches ⇒ caller skips the BFS
  });

  it("occlusionHomeSection CROSSES when the camera moves into the next section", () => {
    const store = buildStore();
    const h1 = occlusionHomeSection(store, centerOf(0, 0, 0));
    const h2 = occlusionHomeSection(store, centerOf(1, 0, 0)); // +16 blocks on x ⇒ section +1
    expect(h1).toEqual([0, 0, 0]);
    expect(h2).toEqual([1, 0, 0]); // different ⇒ the cache key differs ⇒ caller re-runs the BFS
  });

  it("occlusionHomeSection CLAMPS a camera far outside the populated region onto the box edge (stable key)", () => {
    const store = buildStore(); // populated x,z ∈ [-1,1]; margin 1 ⇒ box edge at ±2
    // Two far-west camera positions, hundreds of sections apart — both clamp to the SAME box-edge section,
    // so a camera flying outside the world does NOT re-key every frame (TRAP OCC-1: clamp, not raw section).
    const far1 = occlusionHomeSection(store, [-1000 * 16, 8, 0]);
    const far2 = occlusionHomeSection(store, [-2000 * 16, 8, 0]);
    expect(far1).toEqual([-2, 0, 0]); // clamped to lo.x - margin
    expect(far2).toEqual([-2, 0, 0]); // identical ⇒ stable key despite very different raw positions
  });

  it("occlusionHomeSection returns null on an empty world (nothing to seed)", () => {
    const empty = new SectionStore(new GpuSectionUploader(new FakeGraphicsDevice()));
    expect(occlusionHomeSection(empty, centerOf(0, 0, 0))).toBeNull();
  });

  it("instrument.occBfsRuns counts each actual BFS run (the win metric); a key-stable move reuses membership", () => {
    const store = buildStore();
    // Simulate the caller's cache decision: run the BFS once for the initial home section, then on an
    // IN-SECTION move skip it (key unchanged) and on a CROSSING re-run it.
    const reach = new Set<number>();

    // Frame 1: first build ⇒ one BFS run.
    let homePrev = occlusionHomeSection(store, [1, 8, 1])!;
    occlusionVisibleSet(store, [1, 8, 1], () => true, reach);
    const membershipA = [...reach].sort();
    expect(instrument.job.occBfsRuns).toBe(1);
    expect(instrument.job.occVisited).toBeGreaterThan(0);

    // Frame 2: in-section rotation/drift — home section unchanged ⇒ caller skips the BFS (no extra run).
    const home2 = occlusionHomeSection(store, [14, 8, 14])!;
    const stable = home2[0] === homePrev[0] && home2[1] === homePrev[1] && home2[2] === homePrev[2];
    expect(stable).toBe(true);
    // (caller does NOT call occlusionVisibleSet here) — the counter stays at 1.
    expect(instrument.job.occBfsRuns).toBe(1);
    // membership is whatever the cache holds — identical to frame 1.
    expect([...reach].sort()).toEqual(membershipA);

    // Frame 3: cross into section (1,0,0) ⇒ key changes ⇒ caller re-runs the BFS (counter → 2).
    const home3 = occlusionHomeSection(store, centerOf(1, 0, 0))!;
    const crossed = !(home3[0] === homePrev[0] && home3[1] === homePrev[1] && home3[2] === homePrev[2]);
    expect(crossed).toBe(true);
    occlusionVisibleSet(store, centerOf(1, 0, 0), () => true, reach);
    expect(instrument.job.occBfsRuns).toBe(2);
    homePrev = home3;
  });

  // H1 (P0.3): fusing the live frustum INTO BFS traversal dead-ends the search at the frustum boundary and
  // drops geometry reachable only THROUGH an out-of-frustum section — a hole. occludedTerrainDrawsLegacy now
  // traverses with an always-true predicate (the frustum is applied downstream as the renderer's draw filter).
  it("a frustum-FUSED BFS drops a section reachable only THROUGH an out-of-frustum section (H1)", () => {
    const store = new SectionStore(new GpuSectionUploader(new FakeGraphicsDevice()));
    // A corridor of 3 drawable sections in a line on +X: (0,0,0)–(1,0,0)–(2,0,0). The far section is
    // reachable ONLY through the middle one.
    commitSection(store, 0, 0, 0);
    commitSection(store, 1, 0, 0);
    commitSection(store, 2, 0, 0);
    const cam = centerOf(0, 0, 0);

    // Frustum that excludes ONLY the middle section (as if it sat just past a side plane).
    const fused = new Set<number>();
    occlusionVisibleSet(store, cam, (sx) => sx !== 1, fused);
    expect(fused.has(packSectionCoord(0, 0, 0))).toBe(true); // origin always visited
    expect(fused.has(packSectionCoord(2, 0, 0))).toBe(false); // dropped — the H1 hole (frustum-gated traversal)

    // The fix: an always-true predicate (what occludedTerrainDrawsLegacy now passes) keeps the far section
    // reachable; the renderer applies the real frustum afterwards as a per-section draw filter.
    const reachable = new Set<number>();
    occlusionVisibleSet(store, cam, () => true, reachable);
    expect(reachable.has(packSectionCoord(2, 0, 0))).toBe(true);
  });
});
