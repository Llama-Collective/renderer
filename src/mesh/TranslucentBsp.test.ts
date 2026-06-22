// BSP translucent sort (TR-3) — bit-identical to the topo sort where the back-to-front order is uniquely
// determined (TRAP 5.C), and a valid permutation otherwise. The topo sort is the established CORRECT order
// (a faithful TopoGraphSorting port); the BSP must reproduce it where there's one right answer.

import { describe, it, expect } from "vitest";
import { bspSortOrder } from "./TranslucentBsp";
import { topoSortOrder } from "./TranslucentTopoSort";
import type { TQuad, Vec3 } from "./TranslucentCollector";

function quad(dot: number, normal: Vec3, positions: [Vec3, Vec3, Vec3, Vec3]): TQuad {
  const xs = positions.map((p) => p[0]), ys = positions.map((p) => p[1]), zs = positions.map((p) => p[2]);
  const extents = new Float32Array([Math.max(...xs), Math.max(...ys), Math.max(...zs), Math.min(...xs), Math.min(...ys), Math.min(...zs)]);
  const centroid: Vec3 = [(extents[0] + extents[3]) / 2, (extents[1] + extents[4]) / 2, (extents[2] + extents[5]) / 2];
  const facing = normal[2] > 0.5 ? 2 : normal[2] < -0.5 ? 5 : normal[0] > 0.5 ? 0 : normal[0] < -0.5 ? 3 : normal[1] > 0.5 ? 1 : 4;
  return { extents, facing, dot, normal, centroid, positions };
}

/** Parallel +Z panes at distinct depths — the back-to-front order is UNIQUELY a distance sort. */
function panes(zs: number[]): TQuad[] {
  return zs.map((z) => quad(z, [0, 0, 1], [[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]]));
}

/** The 6 outward faces of a unit glass box [0,1]³ (a "glass structure" — TRAP 5.C). */
function glassBox(): TQuad[] {
  return [
    quad(1, [1, 0, 0], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]), // +X
    quad(0, [-1, 0, 0], [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]), // -X (dot = -n·p = 0)
    quad(1, [0, 1, 0], [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]), // +Y
    quad(0, [0, -1, 0], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]), // -Y
    quad(1, [0, 0, 1], [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]), // +Z
    quad(0, [0, 0, -1], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]), // -Z
  ];
}

const isPermutation = (order: number[], n: number): boolean =>
  order.length === n && new Set(order).size === n && order.every((i) => i >= 0 && i < n);

describe("TranslucentBsp (TR-3)", () => {
  it("is BIT-IDENTICAL to the topo sort on parallel panes (unique order — TRAP 5.C)", () => {
    const quads = panes([1, 2.5, 4, 7]);
    for (const cam of [[0.5, 0.5, -3], [0.5, 0.5, 10], [0.5, 0.5, 5], [0.5, 0.5, 3]] as Vec3[]) {
      const topo = topoSortOrder(quads, cam, false);
      expect(topo).not.toBeNull();
      expect(bspSortOrder(quads, cam)).toEqual(topo); // same order, exactly
    }
  });

  it("orders FRONT-facing parallel panes farthest-first (definitive back-to-front)", () => {
    const quads = panes([1, 2.5, 4, 7]);
    const cam: Vec3 = [0.5, 0.5, 10]; // +Z side of all panes ⇒ all front-facing; farthest is z=1
    const order = bspSortOrder(quads, cam);
    const depths = order.map((i) => quads[i].dot); // dot == z here; farthest (small z) first
    for (let k = 1; k < depths.length; k++) expect(depths[k]).toBeGreaterThan(depths[k - 1]); // 1,2.5,4,7
  });

  it("returns a valid permutation on a 6-face glass box from many camera angles", () => {
    const quads = glassBox();
    for (const cam of [[2, 0.5, 0.5], [-2, 0.5, 0.5], [0.5, 0.5, 2], [3, 3, 3], [-1, -1, -1], [0.5, 2, 0.5]] as Vec3[]) {
      expect(isPermutation(bspSortOrder(quads, cam), quads.length)).toBe(true);
    }
  });

  it("handles the trivial cases (0, 1 quads)", () => {
    expect(bspSortOrder([], [0, 0, 0])).toEqual([]);
    expect(bspSortOrder(panes([3]), [0, 0, 0])).toEqual([0]);
  });
});
