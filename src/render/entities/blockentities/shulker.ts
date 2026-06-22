// Shulker box (per-dye, all 6 facings, lid progress). RENDERER_PLAN.md §18, Phase 4.5c.
// Vanilla ShulkerModel + ShulkerBoxRenderer: lid + base; transform
//   T(.5,.5,.5)·scale(.9995)·dirRotation(facing)·scale(1,-1,-1)·translate(0,-1,0).
// Lid: y drops by progress·8px, yRot = 270°·progress (ShulkerBoxRenderer.setupAnim).

import type { ModelPartDef } from "../../../mesh/entity/ModelPart";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { DYE_NAMES } from "./dyes";
import { boxBE, strip, type BlockEntityDef } from "./registry";
import { DEG, mul, rotationX, rotationY, rotationZ, scaling, translation, identity, type Mat4 } from "./transforms";

const PARTS: ModelPartDef[] = [
  { name: "lid", pivot: [0, 24, 0], cubes: [{ origin: [-8, -16, -8], size: [16, 12, 16], uv: [0, 0] }] },
  { name: "base", pivot: [0, 24, 0], cubes: [{ origin: [-8, -8, -8], size: [16, 8, 16], uv: [0, 28] }] },
];

// Direction.getRotation() as a matrix. Vanilla builds it with `Quaternionf.rotationXYZ(x,y,z)`, which is
// `rotationX(x).rotateY(y).rotateZ(z)` — JOML post-multiplies, so the matrix is `Rx·Ry·Rz` (applied as
// M·v). The two-axis facings are therefore `Rx·Rz`, NOT `Rz·Rx` (AUDIT H5: the order was reversed, which
// flipped local +Y to the WRONG axis for north/west/east — 4 of 6 facings mis-oriented). It rotates
// local +Y onto the facing's unit normal (up→+Y, down→−Y, north→−Z, south→+Z, west→−X, east→+X).
export function dirRotation(facing: string | undefined): Mat4 {
  switch (facing) {
    case "down": return rotationX(Math.PI); // rotationX(π)
    case "north": return mul(rotationX(Math.PI / 2), rotationZ(Math.PI)); // rotationXYZ(π/2, 0, π)
    case "south": return rotationX(Math.PI / 2); // rotationX(π/2)
    case "west": return mul(rotationX(Math.PI / 2), rotationZ(Math.PI / 2)); // rotationXYZ(π/2, 0, π/2)
    case "east": return mul(rotationX(Math.PI / 2), rotationZ(-Math.PI / 2)); // rotationXYZ(π/2, 0, −π/2)
    case "up":
    default: return identity();
  }
}

function shulkerTransform(rec: BlockEntityRecord): Mat4 {
  return mul(
    translation(0.5, 0.5, 0.5),
    scaling(0.9995, 0.9995, 0.9995),
    dirRotation(rec.props.facing),
    scaling(1, -1, -1),
    translation(0, -1, 0),
  );
}

function shulkerTexture(rec: BlockEntityRecord): string {
  // The dye is encoded in the block NAME (`red_shulker_box`), NOT a blockstate property — `rec.props.color`
  // is never populated for shulkers (cf. bed.ts, which derives its dye the same way). Strip the suffix.
  const c = strip(rec.type).replace(/_shulker_box$/, "");
  return DYE_NAMES.includes(c) ? `entity/shulker/shulker_${c}` : "entity/shulker/shulker";
}

export const SHULKER: BlockEntityDef = boxBE({
  types: ["shulker_box", ...DYE_NAMES.map((c) => `${c}_shulker_box`)],
  texWidth: 64,
  texHeight: 64,
  parts: PARTS,
  textures: ["entity/shulker/shulker", ...DYE_NAMES.map((c) => `entity/shulker/shulker_${c}`)],
  texture: shulkerTexture,
  transform: shulkerTransform,
  // OPENABLE: vanilla ShulkerBoxRenderer.setupAnim — the lid drops by progress·8px and yaws 270°·progress.
  // `progress` is the container's lid openness (open-viewer count), integrated to a smooth value by the
  // baker; a closed box (progress 0) stays in the BBB-static section bake, an opening/open one draws per-frame.
  openable: true,
  animate: (_rec, _clock, ctx) => {
    const p = ctx.openness ?? 0;
    return { lid: { offsetPixels: [0, -p * 8, 0], rotation: [0, 270 * p * DEG, 0] } };
  },
});
