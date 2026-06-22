// MEM-1 — RenderRegion keying (8×4×8 = 256-section RenderRegion).
//
// groups sections into 8×4×8 RenderRegions, each with its own GL buffer arena, and culls at
// region granularity before the per-section pass. This is the pure CPU-side keying half of that tier:
// the section→region coordinate map + a packed region key.
//
// PURE MODULE — nothing imports it on the default path. The region tier is default-off (the `regions?`
// injection on SectionStore + the `regionArenas`/`viewerRegionCullEnabled` flags); these helpers are
// only reached when a flag is on, so adding the file changes no shipped behavior.
//
// `packRegionCoord` reuses the SAME bias/span scheme as `OcclusionCuller.packSectionCoord` so a region
// key composes with the rest of the packed-int keying. Region coords are >>3/>>2 of section coords —
// strictly smaller magnitude — so they sit well inside the existing ±16384 bias headroom (no overflow).

import type { Vec3i } from "../types";

/** log2 of the region span on each axis (RenderRegion: 8×4×8 sections). */
export const REGION_SHIFT_X = 3; // 8 sections
export const REGION_SHIFT_Y = 2; // 4 sections
export const REGION_SHIFT_Z = 3; // 8 sections

/** The region (in region coords) that contains section `(sx,sy,sz)`. Arithmetic shift = floor-divide,
 *  correct for negatives (−1 >> 3 === −1) so a section at a negative coord maps to the region below it. */
export function regionOf(sx: number, sy: number, sz: number): Vec3i {
  return [sx >> REGION_SHIFT_X, sy >> REGION_SHIFT_Y, sz >> REGION_SHIFT_Z];
}

// Same bias scheme as OcclusionCuller.packSectionCoord. Section coords pack against BIAS=1<<14; region
// coords are smaller by 2^3 / 2^2 / 2^3, so the same bias gives ample headroom (the packed cube stays
// far under Number.MAX_SAFE_INTEGER). Kept LOCAL (not imported) so this module is dependency-free and the
// "smaller magnitude ⇒ no overflow" invariant is asserted by the round-trip test, not by a shared constant.
const BIAS = 1 << 14; // ±16384 region coords of headroom (parity with packSectionCoord)
const SPAN = 1 << 15;

/** Pack a REGION coordinate (region units) into a single non-negative int — the per-region map key. */
export function packRegionCoord(rx: number, ry: number, rz: number): number {
  return ((rx + BIAS) * SPAN + (ry + BIAS)) * SPAN + (rz + BIAS);
}

/** Pack the region key for a SECTION coordinate — the common path (region-of then pack), allocation-free. */
export function packRegionOfSection(sx: number, sy: number, sz: number): number {
  return packRegionCoord(sx >> REGION_SHIFT_X, sy >> REGION_SHIFT_Y, sz >> REGION_SHIFT_Z);
}

/** Inverse of `packRegionCoord` (region coords). Test/diagnostic only — the hot path never unpacks. */
export function unpackRegionCoord(packed: number): Vec3i {
  const rz = (packed % SPAN) - BIAS;
  const ry = (Math.floor(packed / SPAN) % SPAN) - BIAS;
  const rx = Math.floor(packed / (SPAN * SPAN)) - BIAS;
  return [rx, ry, rz];
}
