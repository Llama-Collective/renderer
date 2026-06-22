// Topological translucency sort — correctness vs the visibility relation it sorts. RENDERER_PLAN §12, §22.

import { describe, it, expect } from "vitest";
import { TranslucentCollector, type TQuad, type Vec3 } from "./TranslucentCollector";
import { quadVisibleThrough, topoSortOrder } from "./TranslucentTopoSort";

type V3 = [number, number, number];
type Quad = [V3, V3, V3, V3];

const FACE: Record<string, ReadonlyArray<readonly [0 | 1, 0 | 1, 0 | 1]>> = {
  NY: [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]],
  PY: [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
  NZ: [[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]],
  PZ: [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
  NX: [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]],
  PX: [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]],
};
const NAMES = Object.keys(FACE);
const face = (name: string, x: number, y: number, z: number): Quad =>
  FACE[name].map((s) => [x + s[0], y + s[1], z + s[2]] as V3) as Quad;

function toQuads(list: Quad[]): TQuad[] {
  const c = new TranslucentCollector();
  for (const q of list) c.add(q[0], q[1], q[2], q[3]);
  return c.classify().quads;
}

/**
 * A valid painter's order satisfies every edge among the CAMERA-VISIBLE (front-facing) quads: if b is
 * visible through a, b must precede a. Back-facing quads are emitted first, unsorted (parity),
 * so edges touching them impose no constraint.
 */
function isValidOrder(quads: TQuad[], order: number[], cam: Vec3, failOnIntersection: boolean): boolean {
  const front = (q: TQuad) => q.normal[0] * cam[0] + q.normal[1] * cam[1] + q.normal[2] * cam[2] > q.dot;
  const pos = new Map<number, number>();
  order.forEach((q, i) => pos.set(q, i));
  for (const a of order) {
    if (!front(quads[a])) continue;
    for (const b of order) {
      if (a === b || !front(quads[b])) continue;
      if (quadVisibleThrough(quads[a], quads[b], failOnIntersection) && pos.get(b)! > pos.get(a)!) return false;
    }
  }
  return true;
}

describe("topoSortOrder", () => {
  it("orders parallel overlapping planes far-to-near (clear glass in front of stained)", () => {
    const quads = toQuads([face("PZ", 0, 0, 0), face("PZ", 0, 0, 2)]); // z=1 and z=3
    // Camera on the +Z side: z=1 is farther → drawn first.
    const cam: Vec3 = [0.5, 0.5, 20];
    const order = topoSortOrder(quads, cam, false)!;
    expect(order).toEqual([0, 1]);
    expect(isValidOrder(quads, order, cam, false)).toBe(true);
  });

  it("emits back-facing quads first, then topo-sorts the front-facing ones", () => {
    // A cube viewed from +x,+y,+z: the 3 near faces (PX,PY,PZ) are front; the far 3 are back.
    const quads = toQuads([face("PX", 0, 0, 0), face("NX", 0, 0, 0), face("PY", 0, 0, 0), face("NY", 0, 0, 0), face("PZ", 0, 0, 0), face("NZ", 0, 0, 0)]);
    const order = topoSortOrder(quads, [10, 10, 10] as Vec3, false)!;
    // The back faces (NX,NY,NZ = indices 1,3,5) are emitted before any front face (PX,PY,PZ = 0,2,4).
    const lastBack = Math.max(order.indexOf(1), order.indexOf(3), order.indexOf(5));
    const firstFront = Math.min(order.indexOf(0), order.indexOf(2), order.indexOf(4));
    expect(lastBack).toBeLessThan(firstFront);
  });

  it("produces a valid topological order for many random axis-aligned scenes + cameras", () => {
    let seed = 0x9e3779b9;
    const rand = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 0x100000000);
    const randInt = (n: number) => Math.floor(rand() * n);

    let validated = 0;
    for (let trial = 0; trial < 200; trial++) {
      const n = 2 + randInt(7);
      const list: Quad[] = [];
      const seen = new Set<string>();
      for (let i = 0; i < n; i++) {
        const name = NAMES[randInt(NAMES.length)];
        const x = randInt(4), y = randInt(4), z = randInt(4);
        const key = `${name},${x},${y},${z}`;
        if (seen.has(key)) continue;
        seen.add(key);
        list.push(face(name, x, y, z));
      }
      if (list.length < 2) continue;
      const quads = toQuads(list);
      const cam: Vec3 = [randInt(9) - 2, randInt(9) - 2, randInt(9) - 2];
      const order = topoSortOrder(quads, cam, false);
      if (!order) continue; // cyclic scene → distance fallback (acceptable)
      expect(order.length).toBe(quads.length);
      expect(new Set(order).size).toBe(quads.length); // a permutation
      expect(isValidOrder(quads, order, cam, false)).toBe(true);
      validated++;
    }
    expect(validated).toBeGreaterThan(20); // most random scenes are acyclic + validated
  });
});
