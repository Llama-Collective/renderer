// Vanilla item diffuse lighting (`com.mojang.blaze3d.platform.Lighting`). RENDERER_PLAN.md §18.
// Items are NOT shaded by the terrain per-face table — they get two-directional diffuse lighting that
// depends on the (model-transformed) normal, so a block item's faces relight as it rotates and an
// inventory block shows the iconic "top brightest, then the two lower faces" gradient. The shader does:
//   shade = min(1, (max(0,dot(L0,n)) + max(0,dot(L1,n)))·0.6 + 0.4)        // include/light.glsl
// where n is the world-space normal. We feed the light pair in that same post-model space:
//   • world entities → the level lights directly (DIFFUSE_LIGHT_0/1), so they rotate with the block;
//   • GUI slots → the lights pre-transformed by vanilla's FULL ITEMS_3D pose (scaling(1,-1,1) + two
//     rotateYXZ terms), so an inventory block gets its iconic top-brightest / graded-side shading.
// light0.w carries the mode flag: > 0.5 ⇒ use diffuse lighting (else the shader keeps the axis shade).

import { mul, rotationX, rotationY, scaling, transformDir } from "../../mesh/entity/mat4";

type Vec4 = readonly [number, number, number, number];

function norm3(x: number, y: number, z: number): [number, number, number] {
  const l = Math.hypot(x, y, z);
  return [x / l, y / l, z / l];
}

const L0 = norm3(0.2, 1.0, -0.7); // DIFFUSE_LIGHT_0
const L1 = norm3(-0.2, 1.0, 0.7); // DIFFUSE_LIGHT_1

/** World/level item lights (block-item entities): rotate with the model. light0.w = 1 ⇒ diffuse mode. */
export const WORLD_ITEM_LIGHT0: Vec4 = [L0[0], L0[1], L0[2], 1];
export const WORLD_ITEM_LIGHT1: Vec4 = [L1[0], L1[1], L1[2], 0];

// GUI block-item lights: the FULL vanilla `Lighting` ITEMS_3D pose, not just its Y-flip (AUDIT M7). Vanilla
// (Lighting.java:37) pre-transforms DIFFUSE_LIGHT_0/1 by
//   scaling(1,-1,1) · rotateYXZ(1.0821041, 3.2375858, 0) · rotateYXZ(-0.3926991, 2.3561945, 0)
// and the renderer's gui slot matrix supplies the screen-space basis the shader's `dot(L, model·n)` lights
// against — so the two missing rotateYXZ terms are what give a GUI block its iconic top-brightest /
// graded-side look (the scaling(1,-1,1) alone was the old approximation). rotateYXZ(y,x,z) = Ry·Rx·Rz.
const ITEMS_3D = mul(
  scaling(1, -1, 1),
  rotationY(1.0821041), rotationX(3.2375858),
  rotationY(-0.3926991), rotationX(2.3561945),
);
const G0 = transformDir(ITEMS_3D, L0[0], L0[1], L0[2]); // unit-length (orthonormal pose)
const G1 = transformDir(ITEMS_3D, L1[0], L1[1], L1[2]);

/** GUI item lights (inventory slots): DIFFUSE pre-transformed by the full vanilla ITEMS_3D pose. */
export const GUI_ITEM_LIGHT0: Vec4 = [G0[0], G0[1], G0[2], 1];
export const GUI_ITEM_LIGHT1: Vec4 = [G1[0], G1[1], G1[2], 0];

// Vanilla's LEVEL diffuse lights (DIFFUSE_LIGHT_0/1) light EVERY world entity — mobs, minecarts, dropped
// items — not just items. Box-model entities pass these so a spinning minecart/sheep shades smoothly via
// diffuse, instead of the terrain axis-shade table that snaps between values at the 45° axis boundaries.
// light1.w = 1 turns on PER-FACE lighting (vanilla `ENTITY_CUTOUT`'s `PER_FACE_LIGHTING`): each visible
// face is lit by ITS viewer-side normal, so an open box's (minecart) inner walls don't darken when their
// outward normal faces away. Items keep light1.w = 0 (per-vertex; their solid cubes hide inner faces, and
// the GUI slot's Y-flip would invert front-facing).
export const WORLD_ENTITY_LIGHT0 = WORLD_ITEM_LIGHT0;
export const WORLD_ENTITY_LIGHT1: Vec4 = [L1[0], L1[1], L1[2], 1];
