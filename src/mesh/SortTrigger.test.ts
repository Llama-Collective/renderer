// Off-thread translucent sort (TR-1): the worker computes the re-sort from SERIALIZED quads, so it MUST
// produce a bit-identical index to the inline (render-thread) path — otherwise off-thread glass would paint
// differently than inline. Also covers the GFNI re-sort trigger (crossedAnyPlane) + plane dedup.

import { describe, it, expect } from "vitest";
import {
  serializeQuads, deserializeQuads, computeSortIndex, uniquePlanes, crossedAnyPlane,
  farAngleResortAllowed, newTopoSortState, GATE_QUAD_THRESHOLD, GATE_DIST_EPS, GATE_ANGLE_COS, GATE_CENTER_LOCAL, GATE_REF_DIST,
  type TopoSortState,
} from "./SortTrigger";
import { TFacing, expandQuadOrder, sortIndicesByDistance, type TQuad, type Vec3 } from "./TranslucentCollector";
import { bspSortOrder } from "./TranslucentBsp";
import { topoSortOrder } from "./TranslucentTopoSort";

/** A quad from its 4 corners (axis-aligned-ish), with extents/centroid derived. */
function quad(facing: number, dot: number, normal: Vec3, positions: [Vec3, Vec3, Vec3, Vec3]): TQuad {
  const xs = positions.map((p) => p[0]), ys = positions.map((p) => p[1]), zs = positions.map((p) => p[2]);
  const extents = new Float32Array([Math.max(...xs), Math.max(...ys), Math.max(...zs), Math.min(...xs), Math.min(...ys), Math.min(...zs)]);
  const centroid: Vec3 = [(extents[0] + extents[3]) / 2, (extents[1] + extents[4]) / 2, (extents[2] + extents[5]) / 2];
  return { extents, facing, dot, normal, centroid, positions };
}

/** A few stacked glass panes facing +Z (POS_Z = 2) — distinct dots so the topo sort orders them. */
function fixtureQuads(): TQuad[] {
  const pane = (z: number): TQuad => quad(2, z, [0, 0, 1], [[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]]);
  return [pane(1), pane(4), pane(7), pane(2.5)];
}

/** Parallel non-intersecting panes with a DIAGONAL normal (facing UNALIGNED) at separated offsets — the
 *  BSP precondition (axis-aligned, non-straddling) does NOT hold, so computeSortIndex must use topo. */
function unalignedFixture(): TQuad[] {
  const n: Vec3 = [Math.SQRT1_2, 0, Math.SQRT1_2]; // diagonal in XZ → facing classified UNALIGNED
  const pane = (offset: number): TQuad => {
    const s = offset * Math.SQRT2; // plane x + z = s ⇒ n·x = offset
    const positions: [Vec3, Vec3, Vec3, Vec3] = [[s, 0, 0], [0, 0, s], [0, 1, s], [s, 1, 0]];
    const xs = positions.map((p) => p[0]), ys = positions.map((p) => p[1]), zs = positions.map((p) => p[2]);
    const extents = new Float32Array([Math.max(...xs), Math.max(...ys), Math.max(...zs), Math.min(...xs), Math.min(...ys), Math.min(...zs)]);
    const centroid: Vec3 = [s / 2, 0.5, s / 2];
    const dot = centroid[0] * n[0] + centroid[1] * n[1] + centroid[2] * n[2];
    return { extents, facing: TFacing.Unaligned, dot, normal: n, centroid, positions };
  };
  return [pane(1), pane(3), pane(2)];
}

describe("SortTrigger — off-thread re-sort == inline", () => {
  it("serializeQuads → deserializeQuads round-trips every field the sort reads", () => {
    const quads = fixtureQuads();
    const back = deserializeQuads(serializeQuads(quads));
    expect(back.length).toBe(quads.length);
    for (let i = 0; i < quads.length; i++) {
      expect(Array.from(back[i].extents)).toEqual(Array.from(quads[i].extents));
      expect(back[i].facing).toBe(quads[i].facing);
      expect(back[i].dot).toBe(quads[i].dot);
      expect(back[i].normal).toEqual(quads[i].normal);
      expect(back[i].centroid).toEqual(quads[i].centroid);
      expect(back[i].positions).toEqual(quads[i].positions);
    }
  });

  it("the worker's index (from serialized quads) is BIT-IDENTICAL to the inline index", () => {
    const quads = fixtureQuads();
    const serialized = deserializeQuads(serializeQuads(quads)); // what the worker meshes from
    for (const cam of [[0.5, 0.5, -3], [0.5, 0.5, 10], [0.5, 0.5, 5]] as Vec3[]) {
      const inline = computeSortIndex(quads, cam);
      const offThread = computeSortIndex(serialized, cam);
      expect(Array.from(offThread)).toEqual(Array.from(inline)); // off-thread sort matches inline exactly
    }
    // And the order genuinely depends on the camera (so the test isn't vacuous).
    const front = Array.from(computeSortIndex(quads, [0.5, 0.5, -3]));
    const behind = Array.from(computeSortIndex(quads, [0.5, 0.5, 10]));
    expect(front).not.toEqual(behind);
  });

  it("axis-aligned sections stay on the BSP fast path — byte-identical to bspSortOrder (INFRA-4)", () => {
    const quads = fixtureQuads();
    for (const cam of [[0.5, 0.5, -3], [0.5, 0.5, 10], [2, 2, 2]] as Vec3[]) {
      // The aligned glass/water common case must NOT have moved off the BSP path.
      expect(Array.from(computeSortIndex(quads, cam))).toEqual(Array.from(expandQuadOrder(bspSortOrder(quads, cam))));
    }
  });

  it("UNALIGNED sections route to the topological sort, deterministically + cross-path identical (INFRA-4)", () => {
    const quads = unalignedFixture();
    expect(quads.some((q) => q.facing === TFacing.Unaligned)).toBe(true);
    const serialized = deserializeQuads(serializeQuads(quads)); // what the worker meshes from
    for (const cam of [[-5, 0.5, 0.5], [5, 0.5, 0.5], [0.5, 5, 0.5]] as Vec3[]) {
      const got = Array.from(computeSortIndex(quads, cam));
      const topo = topoSortOrder(quads, cam, false); // unaligned ⇒ topo path, NOT bsp
      expect(topo).not.toBeNull();
      expect(got).toEqual(Array.from(expandQuadOrder(topo!))); // routed to topo, not BSP
      // The off-thread (deserialized) index matches inline — the bit-identity the worker relies on.
      expect(Array.from(computeSortIndex(serialized, cam))).toEqual(got);
    }
  });

  it("MULTI-AXIS aligned sections route to the topological sort (H4) — NOT the BSP order", () => {
    // Two orthogonal walls — parallel panes on the +X axis AND the +Z axis — a multi-facing-axis aligned
    // section. The BSP's centroid-side straddle resolution is no longer provably the correct painter's order
    // here, so computeSortIndex must produce the TOPOLOGICAL order, never the BSP order.
    const xPane = (x: number): TQuad => quad(TFacing.PosX, x, [1, 0, 0], [[x, 0, 0], [x, 0, 1], [x, 1, 1], [x, 1, 0]]);
    const zPane = (z: number): TQuad => quad(TFacing.PosZ, z, [0, 0, 1], [[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]]);
    const quads = [xPane(2), xPane(5), zPane(2), zPane(5)];
    expect(quads.some((q) => q.facing === TFacing.Unaligned)).toBe(false); // all aligned, but on TWO axes
    for (const cam of [[8, 0.5, 8], [-3, 0.5, -3], [8, 0.5, -3]] as Vec3[]) {
      const topo = topoSortOrder(quads, cam, false);
      expect(topo).not.toBeNull();
      expect(Array.from(computeSortIndex(quads, cam))).toEqual(Array.from(expandQuadOrder(topo!))); // topo, not BSP
    }
  });

  it("a huge single-axis parallel wall sorts WITHOUT a stack overflow (H3 — BSP path is capped)", () => {
    // Spatially-sorted parallel +Z panes: the exact arrangement that drove the unbounded BSP recursion to
    // O(n) depth and overflowed the stack (~5000 quads). Above GATE_QUAD_THRESHOLD the BSP path is skipped
    // (distance sort — EXACT for parallel planes), so this must complete and emit a full index buffer.
    const N = 5000;
    const quads: TQuad[] = [];
    for (let z = 0; z < N; z++) quads.push(quad(TFacing.PosZ, z, [0, 0, 1], [[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]]));
    let idx!: Uint32Array;
    expect(() => { idx = computeSortIndex(quads, [0.5, 0.5, -10]); }).not.toThrow();
    expect(idx.length).toBe(N * 6); // 6 indices per quad — a complete buffer, nothing dropped
    let max = 0;
    for (let i = 0; i < idx.length; i++) if (idx[i] > max) max = idx[i];
    expect(max).toBeLessThan(N * 4); // every index references a valid vertex (4 per quad)
  });

  // DynamicTopoData persistent disable (P3): a section that keeps failing / timing out / is over-cap
  // stops re-attempting topo and permanently distance-sorts. Stateless callers (no `state`) are unaffected.
  describe("persistent topo disable (DynamicTopoData)", () => {
    it("an over-cap section disables topo after one attempt, then distance-sorts only", () => {
      const state = newTopoSortState();
      const pane = (z: number): TQuad => quad(TFacing.PosZ, z, [0, 0, 1], [[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]]);
      const huge: TQuad[] = [];
      for (let i = 0; i <= GATE_QUAD_THRESHOLD; i++) huge.push(pane(i)); // > MAX_TOPO_SORT_QUADS active quads
      const cam: Vec3 = [0, 0, GATE_QUAD_THRESHOLD + 100]; // n·cam > dot for every pane ⇒ all front-facing (active > cap)
      expect(state.gfniDisabled).toBe(false);
      const idx = computeSortIndex(huge, cam, 2, state);
      expect(state.gfniDisabled).toBe(true); // over-cap ⇒ permanently disabled
      expect(idx.length).toBe(huge.length * 6); // still a complete, valid distance-fallback index
      // Now disabled, a later call returns the distance order WITHOUT re-attempting topo.
      expect(Array.from(computeSortIndex(huge, cam, 2, state))).toEqual(Array.from(sortIndicesByDistance(huge, cam)));
    });

    it("a successful topo sort keeps gfni enabled and clears the failure counter", () => {
      const state: TopoSortState = { gfniDisabled: false, failures: 3 };
      computeSortIndex(unalignedFixture(), [5, 0.5, 0.5], 2, state); // multi-pane unaligned ⇒ topo path, succeeds
      expect(state.gfniDisabled).toBe(false);
      expect(state.failures).toBe(0); // reset on success (consecutiveTopoSortFailures = 0)
    });

    it("a disabled section ignores otherwise-topo-sortable geometry and uses the distance order", () => {
      // A multi-axis aligned section whose correct (topo) order genuinely differs from the distance order.
      const xPane = (x: number): TQuad => quad(TFacing.PosX, x, [1, 0, 0], [[x, 0, 0], [x, 0, 1], [x, 1, 1], [x, 1, 0]]);
      const zPane = (z: number): TQuad => quad(TFacing.PosZ, z, [0, 0, 1], [[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]]);
      const quads = [xPane(2), xPane(5), zPane(2), zPane(5)];
      const cam: Vec3 = [8, 0.5, 8];
      const enabled = Array.from(computeSortIndex(quads, cam, 2, { gfniDisabled: false, failures: 0 }));
      const disabled = Array.from(computeSortIndex(quads, cam, 2, { gfniDisabled: true, failures: 5 }));
      expect(disabled).toEqual(Array.from(sortIndicesByDistance(quads, cam))); // forced distance
      expect(enabled).not.toEqual(disabled); // the enabled (topo) order genuinely differs (test isn't vacuous)
    });

    it("stateless callers (worker path / bake) are byte-identical with or without the state machine", () => {
      const quads = unalignedFixture();
      const cam: Vec3 = [5, 0.5, 0.5];
      const stateless = Array.from(computeSortIndex(quads, cam, 2)); // no state → legacy behavior
      const enabled = Array.from(computeSortIndex(quads, cam, 2, newTopoSortState())); // fresh state, topo enabled
      expect(enabled).toEqual(stateless); // a fresh state never changes the order, only tracks failures
    });
  });

  it("crossedAnyPlane fires iff the camera crossed a quad plane; uniquePlanes dedups coplanar quads", () => {
    const quads = fixtureQuads();
    const planes = uniquePlanes(quads);
    expect(planes.length / 4).toBe(4); // 4 distinct dots → 4 planes (no dedup here)
    // A coplanar pair collapses to one plane.
    expect(uniquePlanes([quads[0], quads[0]]).length / 4).toBe(1);
    // Moving from z=-3 to z=10 crosses all four panes (z = 1,2.5,4,7).
    expect(crossedAnyPlane(planes, [0.5, 0.5, -3], [0.5, 0.5, 10])).toBe(true);
    // Moving within z<1 crosses none.
    expect(crossedAnyPlane(planes, [0.5, 0.5, -3], [0.5, 0.5, 0.5])).toBe(false);
  });
});

/** TR-1: the huge-section distance/angle re-sort GATE. A SECOND damper layered ON TOP of crossedAnyPlane —
 *  it only changes WHETHER a re-sort runs for a huge (quadCount >= GATE_QUAD_THRESHOLD) section when the gate
 *  flag is on; below the threshold or with the flag off the behavior is byte-for-byte today's. */
describe("SortTrigger — farAngleResortAllowed (huge-section re-sort gate)", () => {
  const C = GATE_CENTER_LOCAL; // section center in local frame = [8,8,8]
  const FWD: Vec3 = [0, 0, -1]; // a reference forward

  // A unit forward rotated `deg` degrees about +Y from FWD=[0,0,-1] ⇒ [-sin, 0, -cos], so dot(FWD,rot)=cos(deg).
  const rotY = (deg: number): Vec3 => {
    const r = (deg * Math.PI) / 180;
    return [-Math.sin(r), 0, -Math.cos(r)];
  };

  it("(a) sub-epsilon translation + sub-epsilon rotation ⇒ false (no re-sort)", () => {
    // Camera one ref-distance from center (R = REF ⇒ effEps = distEps); move a fraction of distEps and rotate
    // well under the angle bound — neither gate trips.
    const last: Vec3 = [C[0], C[1], C[2] - GATE_REF_DIST];
    const cur: Vec3 = [last[0] + GATE_DIST_EPS * 0.25, last[1], last[2]];
    const curFwd = rotY(1); // 1° ≪ ~2.6° eps ⇒ dot ≈ cos(1°) > GATE_ANGLE_COS
    expect(Math.hypot(...cur.map((v, i) => v - last[i]))).toBeLessThan(GATE_DIST_EPS);
    expect(FWD[0] * curFwd[0] + FWD[1] * curFwd[1] + FWD[2] * curFwd[2]).toBeGreaterThan(GATE_ANGLE_COS);
    expect(farAngleResortAllowed(C, last, cur, FWD, curFwd, GATE_DIST_EPS, GATE_ANGLE_COS)).toBe(false);
  });

  it("(b) translation past distEps ALONE (no rotation) ⇒ true", () => {
    const last: Vec3 = [C[0], C[1], C[2] - GATE_REF_DIST]; // R = REF ⇒ effEps = distEps
    const cur: Vec3 = [last[0] + GATE_DIST_EPS * 1.5, last[1], last[2]]; // > effEps
    // No rotation: same forward ⇒ the angle gate cannot be what fired.
    expect(farAngleResortAllowed(C, last, cur, FWD, FWD, GATE_DIST_EPS, GATE_ANGLE_COS)).toBe(true);
  });

  it("(c) rotation past the angle bound ALONE (no translation) ⇒ true", () => {
    const last: Vec3 = [C[0], C[1], C[2] - GATE_REF_DIST];
    const cur: Vec3 = [last[0], last[1], last[2]]; // zero translation ⇒ distance gate cannot fire
    const curFwd = rotY(10); // 10° ≫ ~2.6° eps ⇒ dot = cos(10°) < GATE_ANGLE_COS
    expect(FWD[0] * curFwd[0] + FWD[1] * curFwd[1] + FWD[2] * curFwd[2]).toBeLessThan(GATE_ANGLE_COS);
    expect(farAngleResortAllowed(C, last, cur, FWD, curFwd, GATE_DIST_EPS, GATE_ANGLE_COS)).toBe(true);
  });

  it("(d) distance metric is CENTER-RELATIVE — a far section clears at a smaller world move than a near one", () => {
    // Same world move and same forward (no rotation) for both; only the section distance differs.
    const nearLast: Vec3 = [C[0], C[1], C[2] - GATE_REF_DIST];     // R = 16 (REF) ⇒ effEps = distEps
    const farLast: Vec3 = [C[0], C[1], C[2] - 4 * GATE_REF_DIST];  // R = 64       ⇒ effEps = distEps/4
    // A move strictly between the two effective thresholds: distEps/4 < move < distEps.
    const move = GATE_DIST_EPS * 0.5;
    expect(move).toBeGreaterThan(GATE_DIST_EPS / 4);
    expect(move).toBeLessThan(GATE_DIST_EPS);
    const nearCur: Vec3 = [nearLast[0] + move, nearLast[1], nearLast[2]];
    const farCur: Vec3 = [farLast[0] + move, farLast[1], farLast[2]];
    // FAR clears the gate (smaller effective threshold); NEAR does not — the TR-1 monotonicity.
    expect(farAngleResortAllowed(C, farLast, farCur, FWD, FWD, GATE_DIST_EPS, GATE_ANGLE_COS)).toBe(true);
    expect(farAngleResortAllowed(C, nearLast, nearCur, FWD, FWD, GATE_DIST_EPS, GATE_ANGLE_COS)).toBe(false);
  });

  // The combined inline predicate exactly as the trigger sites apply it (crossedAnyPlane AND the gate clause).
  const combined = (crossed: boolean, gate: boolean, quadCount: number, last: Vec3, cur: Vec3, lastFwd: Vec3, curFwd: Vec3): boolean =>
    crossed && (!gate || quadCount < GATE_QUAD_THRESHOLD ||
      farAngleResortAllowed(C, last, cur, lastFwd, curFwd, GATE_DIST_EPS, GATE_ANGLE_COS));

  it("(e) combined predicate NEVER suppresses below threshold, and NEVER re-sorts when crossedAnyPlane is false", () => {
    const last: Vec3 = [C[0], C[1], C[2] - GATE_REF_DIST];
    const tinyMove: Vec3 = [last[0] + GATE_DIST_EPS * 0.1, last[1], last[2]]; // sub-eps ⇒ gate would suppress
    // Below the quad threshold: identical to today — a plane crossing ALWAYS re-sorts, gate on or off, even
    // for a sub-eps move (the gate must never touch small/normal sections — TRAP TR-1.B).
    expect(combined(true, true, GATE_QUAD_THRESHOLD - 1, last, tinyMove, FWD, FWD)).toBe(true);
    expect(combined(true, false, GATE_QUAD_THRESHOLD - 1, last, tinyMove, FWD, FWD)).toBe(true);
    // Huge section, gate OFF: still today's behavior (a crossing re-sorts even on a sub-eps move).
    expect(combined(true, false, GATE_QUAD_THRESHOLD, last, tinyMove, FWD, FWD)).toBe(true);
    // Huge section, gate ON, sub-eps move ⇒ the gate suppresses the re-sort (the whole point of TR-1).
    expect(combined(true, true, GATE_QUAD_THRESHOLD, last, tinyMove, FWD, FWD)).toBe(false);
    // crossedAnyPlane FALSE ⇒ NEVER a re-sort, regardless of gate/threshold/movement (the gate is a damper
    // layered ON crossedAnyPlane, never a trigger of its own — TRAP TR-1.A).
    const bigMove: Vec3 = [last[0] + 100, last[1], last[2]];
    expect(combined(false, true, GATE_QUAD_THRESHOLD, last, bigMove, FWD, rotY(45))).toBe(false);
    expect(combined(false, false, GATE_QUAD_THRESHOLD, last, bigMove, FWD, rotY(45))).toBe(false);
  });
});
