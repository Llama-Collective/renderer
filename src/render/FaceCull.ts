// Per-section visible-face mask (FS-4) — `getVisibleFaces`. RENDERER_OPTIMIZATION_PLAN.
//
// A section's opaque/cutout geometry is partitioned into 7 facing slices (FS-2). Before recording draws we
// drop the slices that face AWAY from the camera — the GPU already back-face-culls those triangles, so the
// result is pixel-identical while ~halving submitted opaque triangles. This is the conservative slice test:
// a facing is KEPT whenever the camera could be on its front side, with a ±APRON-block bias so a face flush
// with (or just behind) the camera plane — including block-model overhang — is never wrongly dropped (which
// would punch a hole). UNASSIGNED (unaligned) faces can point any direction, so they are ALWAYS kept.

import { ModelQuadFacing } from "../mesh/ModelQuadFacing";

/** camera bias: keep a facing for cameras up to this many blocks behind its plane (overhang/flush). */
export const FACE_CULL_APRON = 3;
const SECTION = 16;
const { PosX, PosY, PosZ, NegX, NegY, NegZ, Unassigned } = ModelQuadFacing;

/**
 * Bitmask (1<<facing) of the facing slices potentially visible from the camera for a section at
 * `origin` (block coords). Branch-free; UNASSIGNED is always set. `camX/Y/Z` are camera block coords.
 */
export function getVisibleFaces(
  camX: number, camY: number, camZ: number,
  originX: number, originY: number, originZ: number,
): number {
  let mask = 1 << Unassigned; // unaligned slice: never culled (can face any way)
  // STRICT `>` / `<`, exactly `DefaultChunkRenderer.getVisibleFaces` (`BitwiseMath.greaterThan(cam,
  // min-3)` / `lessThan(cam, max+3)`). +face kept while the camera is strictly in front of the section's min
  // plane minus APRON; −face kept while strictly in front of the max plane plus APRON. At the exact apron
  // edge the face is dropped (does the same): any +X overhang quad is then either edge-on (zero area)
  // or back-facing to the camera, so the GPU back-face-cull makes the drop pixel-identical (never a hole).
  if (camX > originX - FACE_CULL_APRON) mask |= 1 << PosX;
  if (camX < originX + SECTION + FACE_CULL_APRON) mask |= 1 << NegX;
  if (camY > originY - FACE_CULL_APRON) mask |= 1 << PosY;
  if (camY < originY + SECTION + FACE_CULL_APRON) mask |= 1 << NegY;
  if (camZ > originZ - FACE_CULL_APRON) mask |= 1 << PosZ;
  if (camZ < originZ + SECTION + FACE_CULL_APRON) mask |= 1 << NegZ;
  return mask;
}
