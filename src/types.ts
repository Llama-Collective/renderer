// Shared pure-data primitives. No GPU types, no three.js, no DOM-renderer state.
// Safe to import ANYWHERE, including inside workers. Keep this file dependency-free.
// See RENDERER_PLAN.md §3 (module boundaries) and §23 (maintainability).

export type Vec3 = readonly [number, number, number];
export type Vec3i = readonly [number, number, number];

/** Section side length in blocks. RENDERER_PLAN §4. */
export const SECTION_SIZE = 16;

/**
 * Monotonic per-section edit counter. Bumped on every dirtying edit (and the border
 * neighbor's, for face edits). A build/sort output is only accepted if its generation
 * MATCHES the section's current generation — not merely "newer in time". RENDERER_PLAN
 * §4 (TRAP 4.C).
 */
export type Generation = number;

/** Coarse terrain render passes. There are only three. RENDERER_PLAN §12. */
export enum TerrainPass {
  Solid = 0,
  Cutout = 1,
  Translucent = 2,
}
export const TERRAIN_PASSES: readonly TerrainPass[] = [
  TerrainPass.Solid,
  TerrainPass.Cutout,
  TerrainPass.Translucent,
];

/**
 * Six face directions. Index order is also used for section neighbor links and for the
 * per-face cull-result cache. RENDERER_PLAN §8.
 */
export enum Direction {
  Down = 0,
  Up = 1,
  North = 2,
  South = 3,
  West = 4,
  East = 5,
}
export const DIRECTION_COUNT = 6;

/** Unit block-space step per Direction (index == enum value). The single source of truth for the
 *  neighbour-offset / face-normal table that the mesher, face bakery, visibility flood, and entity
 *  emit all need — kept here (GPU-free, importable anywhere) so the six copies can't drift. */
export const DIRECTION_OFFSET: readonly Vec3i[] = [
  [0, -1, 0], // Down
  [0, 1, 0], // Up
  [0, 0, -1], // North
  [0, 0, 1], // South
  [-1, 0, 0], // West
  [1, 0, 0], // East
];

/** Opposite face direction (Down↔Up, North↔South, West↔East). The enum pairs each axis ADJACENTLY
 *  (Down=0/Up=1, …), so `dir ^ 1` flips it — depends on that layout (see the Direction enum order). */
export function opposite(dir: Direction): Direction {
  return (dir ^ 1) as Direction;
}

/** Clamp `x` to [0, 1]. */
export function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** Clamp `v` to [lo, hi]. */
export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Squared Euclidean distance between two points (no sqrt — for comparisons/sorting). */
export function dist2(a: Vec3, b: Vec3): number {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

/** A byte sub-range inside a GPU buffer arena. RENDERER_PLAN §11. */
export interface ByteRange {
  offsetBytes: number;
  sizeBytes: number;
}

/**
 * Normalized [0,1] atlas rect for a sprite. Pure data (the atlas TEXTURE lives in core/), so the
 * GPU-free mesher can bake UVs into vertices. RENDERER_PLAN §7.
 */
export interface SpriteUv {
  u: number;
  v: number;
  width: number;
  height: number;
}

/**
 * Logical id for a sub-allocation inside a BufferArena. A plain branded number (NOT a GPU
 * object), so world/ and workers/ may hold these without importing core GPU types.
 * RENDERER_PLAN §11, §23.
 */
declare const AllocBrand: unique symbol;
export type AllocationId = number & { readonly [AllocBrand]: true };

/** Reason a section was dirtied — drives scheduling priority. RENDERER_PLAN §16. */
export enum DirtyReason {
  Edit = 0,
  Simulation = 1,
  NeighborBorder = 2,
  InitialLoad = 3,
  /** A connectivity-NEUTRAL light change (P6/LM-2): same geometry, only the lightmap bytes differ. Routes
   *  to the light-only invalidation path — an in-place light-byte reupload (no realloc/relocation) that does
   *  NOT re-run the occlusion BFS. Global brightness is even cheaper (a free LUT rewrite, no dirty at all). */
  Light = 4,
}
