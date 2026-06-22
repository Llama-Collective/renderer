// 7-slice quad facing model + the SINGLE geometric facing quantizer (FS-1). `ModelQuadFacing`.
//
// A quad is bucketed by its outward normal into one of 7 slices: the six axis facings plus UNASSIGNED for
// anything not axis-aligned. This is the ONE quantizer shared by translucent sort classification
// (TranslucentCollector) AND the opaque/cutout facing partition (FS-2), so the two can never drift.
//
// Facing is derived from the GEOMETRIC normal of the 4 corners — NOT a precomputed shade/normal field — so
// a NO_SHADE axis face (which carries no shade-based normal) still buckets to its axis. The classifier is a
// faithful port of the translucent collector's old private `facingFromNormal` (identical thresholds), now
// the single source of truth.

/** POS_X=0, POS_Y=1, POS_Z=2, NEG_X=3, NEG_Y=4, NEG_Z=5, UNASSIGNED=6 (not axis-aligned). */
export const ModelQuadFacing = { PosX: 0, PosY: 1, PosZ: 2, NegX: 3, NegY: 4, NegZ: 5, Unassigned: 6 } as const;
/** Number of facing slices (6 axis + 1 unassigned). */
export const FACING_COUNT = 7;

const ALIGN_EPSILON = 1e-4;

/** Quantize a (unit-ish) normal to one of the 7 facings: an axis facing if it's axis-aligned within
 *  `ALIGN_EPSILON`, else UNASSIGNED. Same thresholds the translucent classifier has always used. */
export function facingFromNormal(nx: number, ny: number, nz: number): number {
  const ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz);
  if (ax > 1 - ALIGN_EPSILON && ay < ALIGN_EPSILON && az < ALIGN_EPSILON) return nx > 0 ? ModelQuadFacing.PosX : ModelQuadFacing.NegX;
  if (ay > 1 - ALIGN_EPSILON && ax < ALIGN_EPSILON && az < ALIGN_EPSILON) return ny > 0 ? ModelQuadFacing.PosY : ModelQuadFacing.NegY;
  if (az > 1 - ALIGN_EPSILON && ax < ALIGN_EPSILON && ay < ALIGN_EPSILON) return nz > 0 ? ModelQuadFacing.PosZ : ModelQuadFacing.NegZ;
  return ModelQuadFacing.Unassigned;
}

/** Facing of a quad from its 4 corners (CCW from outside), via the GEOMETRIC normal `(p2−p0)×(p3−p1)` —
 *  the same convention as `makeTQuad`/`FaceBakery`. Geometry-only ⇒ a NO_SHADE axis face still buckets. */
export function quadFacing(
  p0: readonly [number, number, number],
  p1: readonly [number, number, number],
  p2: readonly [number, number, number],
  p3: readonly [number, number, number],
): number {
  const ax = p2[0] - p0[0], ay = p2[1] - p0[1], az = p2[2] - p0[2];
  const bx = p3[0] - p1[0], by = p3[1] - p1[1], bz = p3[2] - p1[2];
  let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
  const len = Math.hypot(nx, ny, nz) || 1;
  return facingFromNormal(nx / len, ny / len, nz / len);
}
