// Copper Golem Statue. RENDERER_PLAN.md §18.
// Vanilla CopperGolemStatueBlockRenderer + CopperGolemModel (STANDING pose only for v1):
// body (with head + right_arm + left_arm children) + right_leg + left_leg. Texture varies by
// oxidation level (waxed_ stripped). Placement = mob-style flip + facing rotation + lift so feet
// sit near y=0.

import type { ModelPartDef } from "../../../mesh/entity/ModelPart";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { boxBE, strip, type BlockEntityDef } from "./registry";
import { DEG, mul, oppositeFacing, rotationY, scaling, toYRot, translation } from "./transforms";

/** STANDING pose, exact vanilla pixels. Roots = [body, right_leg, left_leg]. */
const PARTS: ModelPartDef[] = [
  {
    name: "body",
    pivot: [0, -5, 0],
    cubes: [{ origin: [-4, -6, -3], size: [8, 6, 6], uv: [0, 15] }],
    children: [
      {
        name: "head",
        pivot: [0, -6, 0],
        cubes: [
          { origin: [-4, -5, -5], size: [8, 5, 10], uv: [0, 0], inflate: 0.015 },
          { origin: [-1, -2, -6], size: [2, 3, 2], uv: [56, 0] },
          { origin: [-1, -9, -1], size: [2, 4, 2], uv: [37, 8], inflate: -0.015 },
          { origin: [-2, -13, -2], size: [4, 4, 4], uv: [37, 0], inflate: -0.015 },
        ],
      },
      {
        name: "right_arm",
        pivot: [-4, -6, 0],
        cubes: [{ origin: [-3, -1, -2], size: [3, 10, 4], uv: [36, 16] }],
      },
      {
        name: "left_arm",
        pivot: [4, -6, 0],
        cubes: [{ origin: [0, -1, -2], size: [3, 10, 4], uv: [50, 16] }],
      },
    ],
  },
  {
    name: "right_leg",
    pivot: [0, -5, 0],
    cubes: [{ origin: [-4, 0, -2], size: [4, 5, 4], uv: [0, 27] }],
  },
  {
    name: "left_leg",
    pivot: [0, -5, 0],
    cubes: [{ origin: [0, 0, -2], size: [4, 5, 4], uv: [16, 27] }],
  },
];

const OXIDATION: Record<string, string> = {
  copper_golem_statue: "copper_golem",
  exposed_copper_golem_statue: "copper_golem_exposed",
  weathered_copper_golem_statue: "copper_golem_weathered",
  oxidized_copper_golem_statue: "copper_golem_oxidized",
};

function statueTexture(rec: BlockEntityRecord): string {
  const t = strip(rec.type).replace(/^waxed_/, "");
  return `entity/copper_golem/${OXIDATION[t] ?? "copper_golem"}`;
}

export const COPPER_GOLEM_STATUE: BlockEntityDef = boxBE({
  types: [
    "copper_golem_statue", "exposed_copper_golem_statue", "weathered_copper_golem_statue", "oxidized_copper_golem_statue",
    "waxed_copper_golem_statue", "waxed_exposed_copper_golem_statue", "waxed_weathered_copper_golem_statue", "waxed_oxidized_copper_golem_statue",
  ],
  texWidth: 64,
  texHeight: 64,
  parts: PARTS,
  textures: [
    "entity/copper_golem/copper_golem",
    "entity/copper_golem/copper_golem_exposed",
    "entity/copper_golem/copper_golem_weathered",
    "entity/copper_golem/copper_golem_oxidized",
  ],
  texture: statueTexture,
  // Vanilla CopperGolemStatueBlockRenderer: translate(0.5,0,0.5)·rotateY(-opposite.toYRot). The entity
  // model is built Y-down + centred at the origin, so a plain scale(-1,-1,1) flips it Y-up with its feet
  // at y=0 (model spans y≈-1.5..0). NO extra lift or off-centre translate (those made it float).
  transform: (rec) =>
    mul(
      translation(0.5, 0, 0.5),
      rotationY(-toYRot(oppositeFacing(rec.props.facing ?? "north", "north")) * DEG),
      scaling(-1, -1, 1),
    ),
});
