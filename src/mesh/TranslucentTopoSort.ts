// Topological translucency sorting. This is the CORRECT order the BSP
// merely accelerates: it produces the same back-to-front index order, so our render matches 
// without the BSP's internal complexity. RENDERER_PLAN.md §12.
//
// `quadVisibleThrough(a, b)` decides whether b can be seen THROUGH a (⇒ b must be drawn before a).
// A DFS over that implicit visibility graph yields a back-to-front order; a back-edge to a node still
// on the DFS stack means a cycle (no global order exists) → caller falls back to distance sorting.
// With a camera, back-facing quads (camera inside the half space) are emitted first (drawn behind),
// then the front-facing quads are topo-sorted — this is "direction mixing" (no GPU cull).

import { expandQuadOrder, partitionByCamera, TFacing, tFacingOpposite, tFacingSign, type TQuad, type Vec3 } from "./TranslucentCollector";

export { expandQuadOrder };

const HALF_SPACE_EPSILON = 0.001;

/**
 * Safety cap (AUDIT TR-3, `MAX_TOPO_SORT_QUADS`): above this many quads the O(n²) edge-test DFS
 * is abandoned and the caller falls back to the cheap distance sort. The cap is on quad COUNT, so it is
 * deterministic — the mesh-time StaticTopo bake (small sections, no budget) is unaffected and stays
 * bit-identical. The optional time budget is render-path-only (also deterministic-safe: it never runs
 * for the bake because no `budgetMs` is passed there).
 */
const MAX_TOPO_SORT_QUADS = 1000;

const dot3 = (n: Vec3, x: number, y: number, z: number) => n[0] * x + n[1] * y + n[2] * z;

/** extents AABBs overlap with positive volume on every axis (TQuad.extentsIntersect). */
function extentsIntersect(a: Float32Array, b: Float32Array): boolean {
  for (let axis = 0; axis < 3; axis++) {
    const opp = axis + 3;
    if (a[axis] <= b[opp] || b[axis] <= a[opp]) return false;
  }
  return true;
}

/** Whether orthogonal quad B is visible through quad A (the half-space descent test). */
function orthogonalVisibleThrough(a: TQuad, b: TQuad, intersectionsVisible: boolean): boolean {
  const aDir = a.facing;
  const aOpp = tFacingOpposite(a.facing);
  const bDir = b.facing;
  const aSign = tFacingSign(a.facing);
  const bSign = tFacingSign(b.facing);

  const bIntoADescent = aSign * a.extents[aDir] - aSign * b.extents[aOpp];
  const aOutsideBAscent = bSign * a.extents[bDir] - bSign * b.extents[bDir];
  const vis = bIntoADescent > 0 && aOutsideBAscent > 0;

  if (vis && extentsIntersect(a.extents, b.extents)) {
    if (intersectionsVisible) return true;
    return bIntoADescent + aOutsideBAscent > 1;
  }
  return vis;
}

/** Whether `other` is visible through `quad` (an edge `quad → other` in the visibility graph). */
export function quadVisibleThrough(quad: TQuad, other: TQuad, intersectionsVisible: boolean): boolean {
  if (quad === other) return false;

  if (quad.facing !== TFacing.Unaligned && other.facing !== TFacing.Unaligned) {
    // Both aligned.
    if (tFacingOpposite(quad.facing) === other.facing) return false; // opposites never see each other
    if (quad.facing === other.facing) {
      const sign = tFacingSign(quad.facing);
      const dir = quad.facing;
      return sign * quad.extents[dir] > sign * other.extents[dir]; // quad in front of parallel other
    }
    return orthogonalVisibleThrough(quad, other, intersectionsVisible);
  }

  // At least one unaligned: half-space vertex tests on the actual corners.
  const qDot = quad.dot;
  const qN = quad.normal;
  let otherInside = false;
  for (const p of other.positions) {
    if (dot3(qN, p[0], p[1], p[2]) + HALF_SPACE_EPSILON < qDot) { otherInside = true; break; }
  }
  if (!otherInside) return false;

  const oDot = other.dot;
  const oN = other.normal;
  for (const p of quad.positions) {
    if (dot3(oN, p[0], p[1], p[2]) - HALF_SPACE_EPSILON > oDot) return true; // quad not fully inside other
  }
  return false;
}

/** First index ≥ `from` whose `flags` byte is set, or -1. */
function nextSet(flags: Uint8Array, from: number): number {
  for (let i = from; i < flags.length; i++) if (flags[i]) return i;
  return -1;
}

/** Why `topoSortOrder` returned (DynamicTopoData hysteresis input). "ok" ⇒ a valid order; the others
 *  ⇒ null was returned and the caller must distance-fall-back. The persistent-disable treats "timeout"/
 *  "overcap" as immediate (too slow / too big) and "cycle" as a counted failure. Optional out-param — passing
 *  none keeps the legacy 4-arg behavior (the static bake + existing tests ignore it). */
export type TopoStatus = { reason: "ok" | "cycle" | "timeout" | "overcap" };

/**
 * Topologically sort the quads back-to-front. Returns the quad visit order (length = quads.length),
 * or null if the visibility graph has a cycle (no single correct order). `cameraPos` (section-local)
 * enables back-face emission-first; null sorts every quad (static topo). `failOnIntersection` treats
 * intersecting orthogonal quads as mutually visible (used by the static one-time sort). `status`, when
 * supplied, is filled with WHY the function returned (for the persistent-disable heuristic).
 */
export function topoSortOrder(quads: TQuad[], cameraPos: Vec3 | null, failOnIntersection: boolean, budgetMs?: number, status?: TopoStatus): number[] | null {
  if (status) status.reason = "ok";
  const out: number[] = [];

  // Partition by camera: front-facing quads are topo-sorted; back-facing emitted first (drawn behind).
  // Shared with the BSP sorter (partitionByCamera) so the two stay bit-identical (TRAP 5.C).
  const active: number[] = cameraPos
    ? partitionByCamera(quads, cameraPos, out)
    : Array.from({ length: quads.length }, (_, i) => i);

  const m = active.length;
  // AUDIT TR-3: deterministic quad-count cap → bail to the distance fallback for pathological sections.
  if (m > MAX_TOPO_SORT_QUADS) { if (status) status.reason = "overcap"; return null; }
  if (m === 0) return out;
  if (m === 1) { out.push(active[0]); return out; }
  if (m === 2) {
    let a = 0, b = 1;
    if (quadVisibleThrough(quads[active[a]], quads[active[b]], failOnIntersection)) {
      if (failOnIntersection && quadVisibleThrough(quads[active[b]], quads[active[a]], true)) { if (status) status.reason = "cycle"; return null; }
      a = 1; b = 0;
    }
    out.push(active[a], active[b]);
    return out;
  }

  const unvisited = new Uint8Array(m).fill(1);
  const onStack = new Uint8Array(m);
  const stack = new Int32Array(m);
  const nextEdge = new Int32Array(m);
  let visited = 0;
  // AUDIT TR-3 render-path time budget — opt-in (the bake passes no budgetMs, so it's deterministic).
  const tStart = budgetMs !== undefined ? performance.now() : 0;
  let ops = 0;

  while (visited < m) {
    const root = nextSet(unvisited, 0);
    let sp = 0;
    stack[0] = root;
    onStack[root] = 1;
    nextEdge[0] = 0;

    while (sp >= 0) {
      if (budgetMs !== undefined && (++ops & 1023) === 0 && performance.now() - tStart > budgetMs) { if (status) status.reason = "timeout"; return null; }
      const cur = stack[sp];
      const e = nextSet(unvisited, nextEdge[sp]);
      if (e !== -1) {
        if (cur !== e && quadVisibleThrough(quads[active[cur]], quads[active[e]], failOnIntersection)) {
          if (onStack[e]) { if (status) status.reason = "cycle"; return null; } // back-edge → cycle
          onStack[e] = 1;
          nextEdge[sp] = e + 1;
          sp++;
          stack[sp] = e;
          nextEdge[sp] = 0;
          continue;
        }
        const next = e + 1;
        if (next < m) { nextEdge[sp] = next; continue; }
      }
      // no more edges — pop, emit in topological order
      onStack[cur] = 0;
      unvisited[cur] = 0;
      visited++;
      sp--;
      out.push(active[cur]);
    }
  }

  return out;
}
