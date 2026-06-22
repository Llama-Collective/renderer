// Collects translucent quads, classifies the section's sort need, and produces sorted indices.
// RENDERER_PLAN.md §12. Classification is a faithful port of 
// `TranslucentGeometryCollector.sortTypeHeuristic` (special cases A/B/C/D) — see §12å
//
// The whole project exists to render transparency correctly while doing as little sorting as
// possible. A section's translucent quads are classified into the cheapest correct sort:
//
//   NoTranslucent    — no translucent geometry.
//   AnyOrder (NONE)  — order can never matter: ≤1 plane (A), two opposing planes (B), or the geometry
//                      is an axis-aligned convex cuboid surface (C). Uses the shared quad EBO.
//   StaticNormal     — one axis (or two opposing normals) at multiple planes (D): sort ONCE by
//                      normal-relative distance, valid for every view. Never re-sorted.
//   StaticTopo       — small sections (STATIC_TOPO; per-normal-count quad limits): one
//                      camera-independent topological sort baked ONCE, valid for every view. Never
//                      re-sorted. Falls back to Dynamic if the visibility graph has a cycle.
//   Dynamic          — anything larger/cyclic (DYNAMIC): the visible order changes as the
//                      camera moves, so the INDEX order is regenerated on a plane-crossing trigger —
//                      the VERTICES are never re-meshed (TRAP 12.A).
//
// TRAP 12.A: re-sorting rewrites INDICES ONLY — never remesh vertices.
// TRAP 12.D: if rebuilt translucent geometry hashes identical, reuse vertices + the sorted index.

import { SortType } from "./SortTypes";
import { writeQuadWinding } from "./VertexFormat";
import { facingFromNormal } from "./ModelQuadFacing"; // FS-1: the single shared facing quantizer

export type Vec3 = readonly [number, number, number];

export interface TranslucentCollectResult {
  sortType: SortType;
  quads: TQuad[];
}

// --- facing model (POS_X=0,POS_Y=1,POS_Z=2,NEG_X=3,NEG_Y=4,NEG_Z=5, UNALIGNED=6) -----
// The geometric quantizer (`facingFromNormal`) now lives in ModelQuadFacing (FS-1, one source of truth).
const UNALIGNED = 6;
const ALIGNED_DIRECTIONS = 6;
const facingSign = (f: number) => (f < 3 ? 1 : -1);
const facingAxis = (f: number) => f % 3;
/** TQuad.QUANTIZE_EPSILON: shrink non-normal extents so coplanar quads don't falsely fill the box. */
const QUANTIZE_EPSILON = 1 / 256;

/** A quad reduced to the metrics the heuristic + sorters + topo sort need (mirrors TQuad). */
export interface TQuad {
  /** [maxX,maxY,maxZ, minX,minY,minZ]. */
  extents: Float32Array;
  /** 0..5 aligned (POS_X..NEG_Z), or 6 = UNALIGNED. */
  facing: number;
  /** Accurate signed distance along the normal (= extents[facing]·sign aligned, centroid·n otherwise). */
  dot: number;
  normal: Vec3;
  centroid: Vec3;
  /** The 4 section-local corners (CCW from outside) — needed by the topo sort for unaligned quads. */
  positions: readonly [Vec3, Vec3, Vec3, Vec3];
}

/** Facing constants (export so the topo sorter + renderer share them). */
export const TFacing = { PosX: 0, PosY: 1, PosZ: 2, NegX: 3, NegY: 4, NegZ: 5, Unaligned: 6 } as const;
export const tFacingSign = (f: number): number => (f < 3 ? 1 : -1);
export const tFacingOpposite = (f: number): number => (f < 3 ? f + 3 : f - 3);

export class TranslucentCollector {
  private readonly quads: TQuad[] = [];

  /** Add a translucent quad by its 4 section-local corners (CCW from outside). */
  add(
    p0: readonly [number, number, number],
    p1: readonly [number, number, number],
    p2: readonly [number, number, number],
    p3: readonly [number, number, number],
  ): void {
    this.quads.push(makeTQuad(p0, p1, p2, p3));
  }

  get quadCount(): number {
    return this.quads.length;
  }

  /** Classify the section's sort need after all quads are added (heuristic). */
  classify(): TranslucentCollectResult {
    const quads = this.quads;
    const result = (sortType: SortType): TranslucentCollectResult => ({ sortType, quads });

    if (quads.length === 0) return result(SortType.NoTranslucent);
    if (quads.length <= 1) return result(SortType.AnyOrder);

    let hasUnaligned = false;
    let alignedFacingBitmap = 0;
    let alignedExtentsMultiple = false;
    const alignedExtremes = [Infinity, Infinity, Infinity, Infinity, Infinity, Infinity].map((_, i) => (i < 3 ? -Infinity : Infinity));
    // Global AABB (only meaningful when no unaligned quads — used by the convex box test C).
    const ext = [-Infinity, -Infinity, -Infinity, Infinity, Infinity, Infinity];
    // Up to two tracked unaligned normals with up to two distances each (special case D, unaligned).
    let uaNormalA: number | null = null, uaDistA1 = NaN, uaDistA2 = NaN, uaNormalAVec: Vec3 | null = null;
    let uaNormalB: number | null = null, uaDistB1 = NaN, uaDistB2 = NaN, uaNormalBVec: Vec3 | null = null;
    let untrackedUnaligned = 0;

    for (const q of quads) {
      if (q.facing !== UNALIGNED) {
        alignedFacingBitmap |= 1 << q.facing;
        if (!hasUnaligned) {
          for (let i = 0; i < 3; i++) ext[i] = Math.max(ext[i], q.extents[i]);
          for (let i = 3; i < 6; i++) ext[i] = Math.min(ext[i], q.extents[i]);
        }
        const existing = alignedExtremes[q.facing];
        if (!alignedExtentsMultiple && Number.isFinite(existing) && existing !== q.dot) alignedExtentsMultiple = true;
        alignedExtremes[q.facing] = facingSign(q.facing) > 0 ? Math.max(existing, q.dot) : Math.min(existing, q.dot);
      } else {
        hasUnaligned = true;
        const key = normalKey(q.normal);
        if (key === uaNormalA) { if (Number.isNaN(uaDistA1)) uaDistA1 = q.dot; else uaDistA2 = q.dot; }
        else if (key === uaNormalB) { if (Number.isNaN(uaDistB1)) uaDistB1 = q.dot; else uaDistB2 = q.dot; }
        else if (uaNormalA === null) { uaNormalA = key; uaDistA1 = q.dot; uaNormalAVec = q.normal; }
        else if (uaNormalB === null) { uaNormalB = key; uaDistB1 = q.dot; uaNormalBVec = q.normal; }
        else untrackedUnaligned++;
      }
    }

    const alignedNormalCount = bitCount(alignedFacingBitmap);
    let unalignedPlaneCount = 0;
    for (const d of [uaDistA1, uaDistA2, uaDistB1, uaDistB2]) if (!Number.isNaN(d)) unalignedPlaneCount++;
    const planeCount = (alignedExtentsMultiple ? 100 : alignedNormalCount) + unalignedPlaneCount;

    let unalignedNormalCount = untrackedUnaligned;
    if (uaNormalA !== null) unalignedNormalCount++;
    if (uaNormalB !== null) unalignedNormalCount++;
    const normalCount = alignedNormalCount + unalignedNormalCount;

    // Special case A: ≤1 plane → any order.
    if (planeCount <= 1) return result(SortType.AnyOrder);

    if (!hasUnaligned) {
      const opposing = bitmapIsOpposingAligned(alignedFacingBitmap);
      // B: two opposing face planes can't be seen through each other.
      if (planeCount === 2 && opposing) return result(SortType.AnyOrder);
      // C: the convex axis-aligned cuboid surface test — distances line up with the bounding box.
      if (!alignedExtentsMultiple) {
        let passes = true;
        for (let dir = 0; dir < ALIGNED_DIRECTIONS; dir++) {
          const extreme = alignedExtremes[dir];
          if (!Number.isFinite(extreme)) continue;
          const sign = dir < 3 ? 1 : -1;
          if (sign * extreme !== ext[dir]) { passes = false; break; }
        }
        if (passes) return result(SortType.AnyOrder);
      }
      // D: opposing aligned normals, or a single aligned normal → one static order suffices.
      if (opposing || alignedNormalCount === 1) return result(SortType.StaticNormal);
    } else if (alignedNormalCount === 0) {
      // D for unaligned: one normal, or two opposing unaligned normals.
      if (unalignedNormalCount === 1 || (unalignedNormalCount === 2 && uaNormalAVec && uaNormalBVec && areOpposite(uaNormalAVec, uaNormalBVec))) {
        return result(SortType.StaticNormal);
      }
    } else if (planeCount === 2) {
      // D with one aligned + one unaligned opposing normal.
      const dir = Math.log2(alignedFacingBitmap & -alignedFacingBitmap) | 0;
      const alignedNormal = ALIGNED_NORMAL_VECS[dir];
      if (uaNormalAVec && areOpposite(uaNormalAVec, alignedNormal)) return result(SortType.StaticNormal);
    }

    // Attempt a one-time STATIC_TOPO sort for small sections (per-normal-count quad limits);
    // otherwise the section is fully DYNAMIC and re-sorts on a plane-crossing trigger.
    const limitIdx = Math.min(Math.max(normalCount, 2), STATIC_TOPO_LIMITS.length - 1);
    if (quads.length <= STATIC_TOPO_LIMITS[limitIdx]) return result(SortType.StaticTopo);
    return result(SortType.Dynamic);
  }
}

/**
 * Build the sort metrics (`TQuad`) for one quad from its 4 corners (CCW from outside). The single
 * source of truth shared by the section mesher (`TranslucentCollector.add`) and the transient
 * moving-transparent merger (`render/TransientTranslucent`), so both classify/sort identically.
 */
export function makeTQuad(
  p0: readonly [number, number, number],
  p1: readonly [number, number, number],
  p2: readonly [number, number, number],
  p3: readonly [number, number, number],
): TQuad {
  // Geometric normal = (p2−p0)×(p3−p1) normalized (FaceBakery convention).
  const ax = p2[0] - p0[0], ay = p2[1] - p0[1], az = p2[2] - p0[2];
  const bx = p3[0] - p1[0], by = p3[1] - p1[1], bz = p3[2] - p1[2];
  let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
  const len = Math.hypot(nx, ny, nz) || 1;
  nx /= len;
  ny /= len;
  nz /= len;
  const facing = facingFromNormal(nx, ny, nz);

  // Extents (AABB of the 4 corners), with non-normal axes shrunk by QUANTIZE_EPSILON.
  let maxX = Math.max(p0[0], p1[0], p2[0], p3[0]), minX = Math.min(p0[0], p1[0], p2[0], p3[0]);
  let maxY = Math.max(p0[1], p1[1], p2[1], p3[1]), minY = Math.min(p0[1], p1[1], p2[1], p3[1]);
  let maxZ = Math.max(p0[2], p1[2], p2[2], p3[2]), minZ = Math.min(p0[2], p1[2], p2[2], p3[2]);
  const axis = facing === UNALIGNED ? -1 : facingAxis(facing);
  if (axis !== 0) { maxX -= QUANTIZE_EPSILON; minX += QUANTIZE_EPSILON; if (minX > maxX) minX = maxX; }
  if (axis !== 1) { maxY -= QUANTIZE_EPSILON; minY += QUANTIZE_EPSILON; if (minY > maxY) minY = maxY; }
  if (axis !== 2) { maxZ -= QUANTIZE_EPSILON; minZ += QUANTIZE_EPSILON; if (minZ > maxZ) minZ = maxZ; }
  const extents = new Float32Array([maxX, maxY, maxZ, minX, minY, minZ]);

  const centroid: Vec3 = [(p0[0] + p1[0] + p2[0] + p3[0]) / 4, (p0[1] + p1[1] + p2[1] + p3[1]) / 4, (p0[2] + p1[2] + p2[2] + p3[2]) / 4];
  const normal: Vec3 = [nx, ny, nz];
  const dot = facing === UNALIGNED ? centroid[0] * nx + centroid[1] * ny + centroid[2] * nz : extents[facing] * facingSign(facing);

  return { extents, facing, dot, normal, centroid, positions: [p0, p1, p2, p3] };
}

/** Max quads (indexed by unique-normal count, clamped 2..5) for which a static topo sort is attempted. */
const STATIC_TOPO_LIMITS = [-1, -1, 250, 100, 50, 30];

const ALIGNED_NORMAL_VECS: readonly (readonly [number, number, number])[] = [
  [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],
];

function bitCount(n: number): number {
  let c = 0;
  while (n) { n &= n - 1; c++; }
  return c;
}

/** True iff exactly two facings are set and they are an opposing axis pair (dir, dir+3). */
function bitmapIsOpposingAligned(bitmap: number): boolean {
  if (bitCount(bitmap) !== 2) return false;
  for (let dir = 0; dir < 3; dir++) if (bitmap === ((1 << dir) | (1 << (dir + 3)))) return true;
  return false;
}

/**
 * Pack a normal into ONE int key (AUDIT TR-8) — replaces the per-quad `"x,y,z"` string + `===` compare.
 * Each component is normalized (∈[-1,1]) → ×256 → [-256,256] → +512 → [256,768] (< 1024), so it fits in
 * 10 bits; three components pack into 30 bits. Same `Math.round(n·256)` quantization as the old string
 * key, so classification is identical — just compared as a number.
 */
function normalKey(n: readonly [number, number, number]): number {
  return (((Math.round(n[0] * 256) + 512) | ((Math.round(n[1] * 256) + 512) << 10) | ((Math.round(n[2] * 256) + 512) << 20)) >>> 0);
}

function areOpposite(a: readonly [number, number, number], b: readonly [number, number, number]): boolean {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] < -1 + 1e-3;
}

/**
 * Static back-to-front order (STATIC_NORMAL_RELATIVE): sort quads by signed distance along
 * their own normal (`dot`), ascending = far-side first. Valid for every view, computed once. The
 * interleaving between opposing facings is irrelevant (they can't be seen through each other).
 */
export function sortIndicesStaticNormal(quads: TQuad[]): Uint32Array {
  return expandQuadOrder(quadOrder(quads.length, (a, b) => quads[a].dot - quads[b].dot));
}

/**
 * Distance fallback (last resort when topo sorting fails): farthest centroid first.
 * Indices only — the vertices never change (TRAP 12.A).
 */
export function sortIndicesByDistance(quads: TQuad[], cameraPos: Vec3): Uint32Array {
  // Decorate ONCE (AUDIT TR-4): precompute each quad's squared distance so the comparator is a plain
  // key lookup, not an O(n log n)× centroid recompute.
  const n = quads.length;
  const key = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const c = quads[i].centroid;
    const dx = c[0] - cameraPos[0], dy = c[1] - cameraPos[1], dz = c[2] - cameraPos[2];
    key[i] = dx * dx + dy * dy + dz * dz;
  }
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => key[b] - key[a]); // farthest centroid first
  return expandQuadOrder(order);
}

/** Indices 0..count-1 sorted by `cmp`. */
function quadOrder(count: number, cmp: (a: number, b: number) => number): Uint32Array {
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[i] = i;
  return order.sort(cmp);
}

/**
 * Split quads by camera facing (the SHARED first step of both the BSP and topological sorters, so their
 * camera handling can't diverge — TRAP 5.C bit-identity): a quad is FRONT-facing when `normal·cam > dot`.
 * BACK-facing quads are pushed onto `out` FIRST, in index order (they're GPU back-face-culled anyway, so
 * their relative order is moot); the FRONT-facing indices are returned for the caller to sort.
 */
export function partitionByCamera(quads: TQuad[], cam: Vec3, out: number[]): number[] {
  const front: number[] = [];
  for (let i = 0; i < quads.length; i++) {
    const q = quads[i];
    if (q.normal[0] * cam[0] + q.normal[1] * cam[1] + q.normal[2] * cam[2] > q.dot) front.push(i);
    else out.push(i);
  }
  return front;
}

/** Expand a quad visit order into the 6 vertex indices per quad (0,1,2, 0,2,3 within each quad). */
export function expandQuadOrder(order: ArrayLike<number>): Uint32Array {
  const out = new Uint32Array(order.length * 6);
  for (let k = 0; k < order.length; k++) writeQuadWinding(out, k * 6, order[k] * 4);
  return out;
}
