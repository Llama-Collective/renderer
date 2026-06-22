// Bed (vanilla BedRenderer + BedRenderer.HeadModel/FootModel). RENDERER_PLAN.md §18.
// Each bed BLOCK (head or foot) is its own BlockEntityRecord. The model is laid flat by the transform,
// matching vanilla BedRenderer.modelTransform's op order (M = T·Rx·Rzc, applied as M·v):
//   translate(0, 0.5625, 0) → rotateX(90°) → rotateZ(180 + toYRot(facing)) about block centre.
// (AUDIT H4: rotateX and the centred rotateZ were previously swapped — JOML's .rotate()/.rotateAround()
// post-multiply, so Rx(90) must precede Rzc — which floated the bed upright for 3 of 4 facings.)
// The dye colour is baked into entity/bed/<color>, so NO tint.

import type { ModelPartDef } from "../../../mesh/entity/ModelPart";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { boxBE, strip, type BlockEntityDef } from "./registry";
import { DEG, mul, rotationX, rotationZ, toYRot, translation, type Mat4 } from "./transforms";
import { DYE_NAMES } from "./dyes";

const HEAD: ModelPartDef[] = [
  { name: "main", pivot: [0, 0, 0], cubes: [{ origin: [0, 0, 0], size: [16, 16, 6], uv: [0, 0] }] },
  { name: "left_leg", pivot: [0, 0, 0], rotation: [Math.PI / 2, 0, Math.PI / 2], cubes: [{ origin: [0, 6, 0], size: [3, 3, 3], uv: [50, 6] }] },
  { name: "right_leg", pivot: [0, 0, 0], rotation: [Math.PI / 2, 0, Math.PI], cubes: [{ origin: [-16, 6, 0], size: [3, 3, 3], uv: [50, 18] }] },
];

const FOOT: ModelPartDef[] = [
  { name: "main", pivot: [0, 0, 0], cubes: [{ origin: [0, 0, 0], size: [16, 16, 6], uv: [0, 22] }] },
  { name: "left_leg", pivot: [0, 0, 0], rotation: [Math.PI / 2, 0, 0], cubes: [{ origin: [0, 6, -16], size: [3, 3, 3], uv: [50, 0] }] },
  { name: "right_leg", pivot: [0, 0, 0], rotation: [Math.PI / 2, 0, (3 * Math.PI) / 2], cubes: [{ origin: [-16, 6, -16], size: [3, 3, 3], uv: [50, 12] }] },
];

/** Rotate `deg` about the Z axis through the block centre (0.5, 0.5, 0.5). */
function rotAroundZcentre(deg: number): Mat4 {
  return mul(translation(0.5, 0.5, 0.5), rotationZ(deg * DEG), translation(-0.5, -0.5, -0.5));
}

/** Dye colour from "<color>_bed" → "<color>" (default white). */
function bedColor(rec: BlockEntityRecord): string {
  return strip(rec.type).replace(/_bed$/, "");
}

export const BED: BlockEntityDef = boxBE({
  types: DYE_NAMES.map((c) => `${c}_bed`),
  texWidth: 64,
  texHeight: 64,
  parts: (rec) => (rec.props.part === "head" ? HEAD : FOOT),
  textures: DYE_NAMES.map((c) => `entity/bed/${c}`),
  texture: (rec) => `entity/bed/${bedColor(rec)}`,
  transform: (rec) =>
    mul(translation(0, 0.5625, 0), rotationX(Math.PI / 2), rotAroundZcentre(180 + toYRot(rec.props.facing))),
});
