// Enchanting-table / lectern book (vanilla BookModel + EnchantTableRenderer / LecternRenderer).
// BookModel: leftLid + rightLid (thin covers) + seam (spine) + leftPages/rightPages (page stacks) +
// flipPage1. Here we bake an OPEN book (openness o = 1.1) straight into the part rotations so no
// runtime animate() is needed. Placement per host: enchanting_table hovers + tilts the book ~80°;
// lectern lays it on the slope (67.5°) facing its FACING (LecternRenderer.render).

import type { ModelPartDef } from "../../../mesh/entity/ModelPart";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { boxBE, type BlockEntityDef } from "./registry";
import { DEG, mul, rotationY, rotationZ, toYRot, translation, type Mat4 } from "./transforms";

const O = 1.1; // openness

const PARTS: ModelPartDef[] = [
  { name: "left_lid", pivot: [0, 0, -1], rotation: [0, Math.PI + O, 0], cubes: [{ origin: [-6, -5, 0], size: [6, 10, 0.01], uv: [0, 0] }] },
  { name: "right_lid", pivot: [0, 0, 1], rotation: [0, -O, 0], cubes: [{ origin: [0, -5, 0], size: [6, 10, 0.01], uv: [16, 0] }] },
  { name: "seam", pivot: [0, 0, 0], rotation: [0, Math.PI / 2, 0], cubes: [{ origin: [-1, -5, 0], size: [2, 10, 0.01], uv: [12, 0] }] },
  { name: "left_pages", pivot: [0, 0, 0], rotation: [0, O, 0], cubes: [{ origin: [0, -4, -0.99], size: [5, 8, 1], uv: [0, 10] }] },
  { name: "right_pages", pivot: [0, 0, 0], rotation: [0, -O, 0], cubes: [{ origin: [0, -4, -0.01], size: [5, 8, 1], uv: [12, 10] }] },
  { name: "flip_page1", pivot: [0, 0, 0], rotation: [0, O * 0.1, 0], cubes: [{ origin: [0, -4, 0], size: [5, 8, 0.01], uv: [24, 10] }] },
];

function bookTransform(rec: BlockEntityRecord): Mat4 {
  if (rec.type === "lectern") {
    return mul(
      translation(0.5, 1.0625, 0.5),
      rotationY(-(toYRot(rec.props.facing) + 90) * DEG),
      rotationZ(67.5 * DEG),
      translation(0, -0.125, 0),
    );
  }
  return mul(translation(0.5, 0.78, 0.5), rotationZ(80 * DEG));
}

export const BOOK: BlockEntityDef = boxBE({
  types: ["enchanting_table", "lectern"],
  texWidth: 64,
  texHeight: 32,
  parts: PARTS,
  textures: ["entity/enchanting_table_book"],
  texture: () => "entity/enchanting_table_book",
  transform: bookTransform,
  // HYBRID: the lectern stand + the enchanting-table cube come from the TERRAIN model; this BE only adds
  // the open book on top. Keep terrain (else the stand/table vanishes, leaving a floating book).
  hybridTerrain: true,
});
