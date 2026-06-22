// Bell (vanilla BellModel + BellRenderer). RENDERER_PLAN.md §18.
// BellModel: bell_body cube (pivot 8,12,8) with a bell_base child (pivot -8,-12,-8) on a 32×32 sheet.
// BellRenderer swings the body about X when rung; here we loop a small preview swing off the render clock.

import type { ModelPartDef } from "../../../mesh/entity/ModelPart";
import { boxBE, type BlockEntityDef } from "./registry";
import { identity } from "./transforms";

const PARTS: ModelPartDef[] = [
  {
    name: "bell_body",
    pivot: [8, 12, 8],
    cubes: [{ origin: [-3, -6, -3], size: [6, 7, 6], uv: [0, 0] }],
    children: [
      { name: "bell_base", pivot: [-8, -12, -8], cubes: [{ origin: [4, 4, 4], size: [8, 2, 8], uv: [0, 13] }] },
    ],
  },
];

export const BELL: BlockEntityDef = boxBE({
  types: ["bell"],
  texWidth: 32,
  texHeight: 32,
  parts: PARTS,
  textures: ["entity/bell/bell_body"],
  texture: (_rec) => "entity/bell/bell_body",
  transform: (_rec) => identity(),
  animate: (_rec, clock) => ({ bell_body: { rotation: [Math.sin(clock * 3) * 0.25, 0, 0] } }),
  animated: true,
  // HYBRID: the bell's posts/bars come from the TERRAIN model (bell_floor/wall/ceiling/between_walls);
  // this BE only hangs the swinging bell body. Keep terrain (else the frame/support vanishes).
  hybridTerrain: true,
});
