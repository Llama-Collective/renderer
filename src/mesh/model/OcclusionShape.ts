// Per-block-state occlusion + per-quad face culling. RENDERER_PLAN §24.6.
//
// Vanilla/cull a quad by its `cullface`: it's hidden iff the neighbor fully occludes that
// side (or self+neighbor are the same skip-render group — clear glass / panes / same fluid).
//
// We derive a block's occlusion from its BAKED geometry: a quad that is OPAQUE and flush+full on a
// side occludes that side (a full cube → all 6). Partial flush faces (slabs/stairs) record a rect;
// coverage is the single-rect-contains test (sufficient for flush boxes — a documented Phase-2
// simplification of vanilla's full VoxelShape subtraction). The full-cube case is the fast path.

import { Direction, opposite } from "../../types";
import type { BakedQuad } from "./FaceBakery";
import type { Vec3 } from "./bakeMath";

/** Minimal quad shape the culler needs (satisfied by BakedQuad and BakedRenderQuad). */
export interface CullableQuad {
  cullface: Direction | null;
  positions: readonly [Vec3, Vec3, Vec3, Vec3];
}

const EPS = 1e-6;

interface Rect2 {
  a0: number;
  b0: number;
  a1: number;
  b1: number;
}
interface FaceOcclusion {
  /** Side fully covered by opaque flush geometry. */
  full: boolean;
  /** Opaque flush rects on this side in the face's in-plane [0,1]² coords. */
  rects: Rect2[];
}
export interface OcclusionShape {
  byFace: Record<Direction, FaceOcclusion>;
  /** True if at least one side occludes (a quick "is this block relevant to culling" check). */
  occludesAny: boolean;
}

const FACE_AXIS: Record<Direction, 0 | 1 | 2> = {
  [Direction.West]: 0,
  [Direction.East]: 0,
  [Direction.Down]: 1,
  [Direction.Up]: 1,
  [Direction.North]: 2,
  [Direction.South]: 2,
};
const FACE_BOUNDARY: Record<Direction, 0 | 1> = {
  [Direction.Down]: 0,
  [Direction.Up]: 1,
  [Direction.North]: 0,
  [Direction.South]: 1,
  [Direction.West]: 0,
  [Direction.East]: 1,
};

/**
 * Does this block fully occlude sight on ALL six faces (a solid opaque cube)? This is the conservative
 * sight-blocker predicate for the section visibility graph (OCC-1, `computeSectionVisibility`): anything
 * less than a full cube — slabs, stairs, glass, fences, air — must NOT count as a blocker, or the
 * occlusion BFS would over-cull and hide geometry that is actually visible.
 */
export function isFullOpaqueCube(occ: OcclusionShape): boolean {
  const f = occ.byFace;
  return (
    f[Direction.Down].full &&
    f[Direction.Up].full &&
    f[Direction.North].full &&
    f[Direction.South].full &&
    f[Direction.West].full &&
    f[Direction.East].full
  );
}

/** SL-1: does this block contribute to ambient-occlusion darkening? Smooth AO uses `occludesAny` (ANY
 *  opaque flush face), NOT `isFullOpaqueCube`, so a slab/stair/partial block still darkens the corners it
 *  touches (`AoNeighborInfo` uses the block's opacity, not a full-cube test). */
export function occludesAo(occ: OcclusionShape): boolean {
  return occ.occludesAny;
}

/** The two in-plane axes for a face (same for a face and its opposite, so footprints compare). */
function inPlane(face: Direction, p: readonly [number, number, number]): [number, number] {
  const axis = FACE_AXIS[face];
  if (axis === 1) return [p[0], p[2]]; // up/down → (x,z)
  if (axis === 2) return [p[0], p[1]]; // north/south → (x,y)
  return [p[2], p[1]]; // east/west → (z,y)
}

function footprint(quad: CullableQuad, face: Direction): Rect2 {
  let a0 = Infinity, b0 = Infinity, a1 = -Infinity, b1 = -Infinity;
  for (const p of quad.positions) {
    const [a, b] = inPlane(face, p);
    a0 = Math.min(a0, a);
    b0 = Math.min(b0, b);
    a1 = Math.max(a1, a);
    b1 = Math.max(b1, b);
  }
  return { a0, b0, a1, b1 };
}

function isFull(r: Rect2): boolean {
  return r.a0 <= EPS && r.b0 <= EPS && r.a1 >= 1 - EPS && r.b1 >= 1 - EPS;
}
function contains(outer: Rect2, inner: Rect2): boolean {
  return outer.a0 <= inner.a0 + EPS && outer.b0 <= inner.b0 + EPS && outer.a1 >= inner.a1 - EPS && outer.b1 >= inner.b1 - EPS;
}

function emptyFace(): FaceOcclusion {
  return { full: false, rects: [] };
}

/**
 * Compute a block's occlusion from its baked quads. `isOpaque(sprite)` reports whether a sprite is
 * a full opaque (solid-layer) texture — only opaque flush geometry occludes (glass does not).
 */
export function computeOcclusion(quads: BakedQuad[], isOpaque: (sprite: string) => boolean): OcclusionShape {
  const byFace: Record<Direction, FaceOcclusion> = {
    [Direction.Down]: emptyFace(),
    [Direction.Up]: emptyFace(),
    [Direction.North]: emptyFace(),
    [Direction.South]: emptyFace(),
    [Direction.West]: emptyFace(),
    [Direction.East]: emptyFace(),
  };
  let occludesAny = false;

  for (const quad of quads) {
    if (!isOpaque(quad.sprite)) continue;
    const face = quad.normal;
    const axis = FACE_AXIS[face];
    const boundary = FACE_BOUNDARY[face];
    if (!quad.positions.every((p) => Math.abs(p[axis] - boundary) <= EPS)) continue; // not flush
    const rect = footprint(quad, face);
    byFace[face].rects.push(rect);
    if (isFull(rect)) {
      byFace[face].full = true;
      occludesAny = true;
    } else {
      occludesAny = true;
    }
  }
  return { byFace, occludesAny };
}

/**
 * Should a quad with a `cullface` be drawn given the neighbor across that face?
 *  - no cullface → always drawn.
 *  - same non-null skip-render group (clear glass / panes / fluid) → culled.
 *  - neighbor fully/partially occludes the quad's footprint → culled.
 */
export function shouldDrawQuad(
  quad: CullableQuad,
  neighbor: OcclusionShape | null,
  selfSkipGroup: string | null,
  neighborSkipGroup: string | null,
): boolean {
  if (quad.cullface === null) return true;
  if (neighbor === null) return true; // air / non-occluding neighbor
  if (selfSkipGroup !== null && selfSkipGroup === neighborSkipGroup) return false;
  const occ = neighbor.byFace[opposite(quad.cullface)];
  if (occ.full) return false;
  if (occ.rects.length === 0) return true;
  const fp = footprint(quad, quad.cullface);
  return !occ.rects.some((r) => contains(r, fp));
}
