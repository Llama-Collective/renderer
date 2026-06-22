// Full BSP translucent sort (TR-3) — a plane-partition tree traversed back-to-front. RENDERER_PLAN.md §12.
//
// Painter's-algorithm BSP: partition the quads by a chosen quad's PLANE (front / back / coplanar by signed
// vertex distance), recurse, then traverse so the side AWAY from the camera is emitted FIRST (farthest →
// nearest). This yields a correct back-to-front order — the SAME order the topological sort
// (TopoGraphSorting) produces where the order is uniquely determined (TRAP 5.C: bit-identical to topo on the
// parallel-pane fixture). NON-SPLITTING (TRAP 12.A — a re-sort is INDICES ONLY, it never alters vertices): a
// quad straddling the plane is assigned to the side its centroid falls on. That is exact for axis-aligned
// block faces (a face never straddles another face's plane), and block faces are non-intersecting convex
// polygons, so a BSP traversal is a valid painter's order for them — no cycle/distance fallback needed.

import { partitionByCamera, type TQuad, type Vec3 } from "./TranslucentCollector";

const EPS = 1e-4;

function signedDistance(plane: TQuad, x: number, y: number, z: number): number {
  return plane.normal[0] * x + plane.normal[1] * y + plane.normal[2] * z - plane.dot;
}

/** Back-to-front quad VISIT order for `cameraPos` (section-local) via a BSP traversal: `quads[order[k]]` is
 *  the k-th quad to draw. Length === quads.length (a permutation). Matches the topo sort's camera handling
 *  ("direction mixing"): BACK-facing quads (camera inside the half-space) are emitted FIRST in index
 *  order — they're GPU-culled anyway, so their order is moot — and only the FRONT-facing quads are sorted. */
export function bspSortOrder(quads: TQuad[], cameraPos: Vec3): number[] {
  const out: number[] = [];
  const front = partitionByCamera(quads, cameraPos, out); // back-facing → out first (shared with topoSortOrder)
  partition(quads, front, cameraPos, out);
  return out;
}

function partition(quads: TQuad[], idxs: number[], cam: Vec3, out: number[]): void {
  if (idxs.length === 0) return;
  if (idxs.length === 1) { out.push(idxs[0]); return; }

  const plane = quads[idxs[0]]; // root quad's plane partitions the rest
  const front: number[] = [];
  const back: number[] = [];
  const coplanar: number[] = [idxs[0]];
  for (let i = 1; i < idxs.length; i++) {
    const q = quads[idxs[i]];
    let pos = false, neg = false;
    for (const p of q.positions) {
      const s = signedDistance(plane, p[0], p[1], p[2]);
      if (s > EPS) pos = true;
      else if (s < -EPS) neg = true;
    }
    if (pos && !neg) front.push(idxs[i]);
    else if (neg && !pos) back.push(idxs[i]);
    else if (!pos && !neg) coplanar.push(idxs[i]); // lies on the plane
    else (signedDistance(plane, q.centroid[0], q.centroid[1], q.centroid[2]) >= 0 ? front : back).push(idxs[i]); // straddles → centroid side
  }

  // Emit the side AWAY from the camera first (farthest), then the plane's coplanar quads, then the near side.
  if (signedDistance(plane, cam[0], cam[1], cam[2]) >= 0) {
    partition(quads, back, cam, out);
    for (const c of coplanar) out.push(c);
    partition(quads, front, cam, out);
  } else {
    partition(quads, front, cam, out);
    for (const c of coplanar) out.push(c);
    partition(quads, back, cam, out);
  }
}
