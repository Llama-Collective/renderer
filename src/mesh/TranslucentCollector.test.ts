// Translucent classification (special cases A/B/C/D) + static-normal sort. RENDERER_PLAN §12, §22.

import { describe, it, expect } from "vitest";
import { TranslucentCollector, sortIndicesStaticNormal, type TQuad } from "./TranslucentCollector";
import { SortType } from "./SortTypes";

type Vec3 = [number, number, number];
type Quad = [Vec3, Vec3, Vec3, Vec3];

// Unit-cube face windings (CCW from outside), matching FaceBakery. Lets tests build real geometry.
const FACE: Record<string, ReadonlyArray<readonly [0 | 1, 0 | 1, 0 | 1]>> = {
  NY: [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]],
  PY: [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
  NZ: [[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]],
  PZ: [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
  NX: [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]],
  PX: [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]],
};
const face = (name: keyof typeof FACE, x: number, y: number, z: number): Quad =>
  FACE[name].map((s) => [x + s[0], y + s[1], z + s[2]] as Vec3) as Quad;

function classify(quads: Quad[]): SortType {
  const c = new TranslucentCollector();
  for (const q of quads) c.add(q[0], q[1], q[2], q[3]);
  return c.classify().sortType;
}

function collect(quads: Quad[]): TQuad[] {
  const c = new TranslucentCollector();
  for (const q of quads) c.add(q[0], q[1], q[2], q[3]);
  return c.classify().quads;
}

describe("TranslucentCollector.classify (special cases)", () => {
  it("no quads → NoTranslucent; one quad → AnyOrder", () => {
    expect(classify([])).toBe(SortType.NoTranslucent);
    expect(classify([face("PZ", 0, 0, 0)])).toBe(SortType.AnyOrder);
  });

  it("a single convex cube (all 6 faces) → AnyOrder (case C: distances match the bounding box)", () => {
    expect(classify([face("PX", 0, 0, 0), face("NX", 0, 0, 0), face("PY", 0, 0, 0), face("NY", 0, 0, 0), face("PZ", 0, 0, 0), face("NZ", 0, 0, 0)])).toBe(SortType.AnyOrder);
  });

  it("two opposing planes, one each → AnyOrder (case B: can't be seen through each other)", () => {
    expect(classify([face("PZ", 0, 0, 0), face("NZ", 0, 0, 0)])).toBe(SortType.AnyOrder);
  });

  it("one normal at multiple planes → StaticNormal (case D)", () => {
    // Two parallel +Z faces at z=1 and z=3 (e.g. clear glass in front of stained, same facing).
    expect(classify([face("PZ", 0, 0, 0), face("PZ", 0, 0, 2)])).toBe(SortType.StaticNormal);
  });

  it("two opposing normals at multiple planes → StaticNormal (case D)", () => {
    expect(classify([face("PZ", 0, 0, 0), face("PZ", 0, 0, 4), face("NZ", 0, 0, 0), face("NZ", 0, 0, 4)])).toBe(SortType.StaticNormal);
  });

  it("a non-convex arrangement (offset +X faces, mixed axes, multiple planes) → topo/dynamic, not AnyOrder", () => {
    // +X faces at two different x AND +Z faces at two different z → fails the cuboid test, multi-axis.
    const t = classify([face("PX", 0, 0, 0), face("PX", 4, 0, 0), face("PZ", 0, 0, 0), face("PZ", 0, 0, 4)]);
    expect(t === SortType.StaticTopo || t === SortType.Dynamic).toBe(true);
    expect(t).not.toBe(SortType.AnyOrder);
  });

  it("computes axis-aligned facing, extents and the normal-relative dot", () => {
    const [q] = collect([face("PZ", 0, 0, 5)]);
    expect(q.facing).toBe(2); // POS_Z
    expect(q.dot).toBeCloseTo(6, 5); // plane at z = 5+1
    expect(q.normal[2]).toBeCloseTo(1, 5);
  });
});

describe("sortIndicesStaticNormal", () => {
  it("orders quads by signed normal distance ascending (far-side first), expanded to vertex indices", () => {
    // Three +Z faces at z = 10, 0, 5 → dots 11, 1, 6 → ascending order q1(z0), q2(z5), q0(z10).
    const quads = collect([face("PZ", 0, 0, 10), face("PZ", 0, 0, 0), face("PZ", 0, 0, 5)]);
    expect(Array.from(sortIndicesStaticNormal(quads))).toEqual([
      4, 5, 6, 4, 6, 7, /* q1 */ 8, 9, 10, 8, 10, 11, /* q2 */ 0, 1, 2, 0, 2, 3, /* q0 */
    ]);
  });
});
