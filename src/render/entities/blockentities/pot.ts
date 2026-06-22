// Decorated pot. RENDERER_PLAN.md §18, Phase 4.5c. Vanilla DecoratedPotRenderer: a base layer (neck +
// top/bottom discs, 32² texture) + four flat side planes (16² sherd/side texture). Multi-texture, so it
// hand-builds two bake groups instead of `boxBE`. Sherd-per-side selection is a follow-up; v1 uses the
// default side sprite. Wobble animation deferred (static).

import { bakeModel, type ModelPartDef } from "../../../mesh/entity/ModelPart";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { Direction, TerrainPass } from "../../../types";
import type { BEBakeContext, BlockEntityDef } from "./registry";
import { facingRot, sectionLocal } from "./transforms";

const SIDE = (uv: [number, number] = [1, 0]) => ({ origin: [0, 0, 0] as [number, number, number], size: [14, 16, 0] as [number, number, number], uv, faces: [Direction.North] });

const BASE_PARTS: ModelPartDef[] = [
  { name: "neck", pivot: [0, 37, 16], rotation: [Math.PI, 0, 0], cubes: [
    { origin: [4, 17, 4], size: [8, 3, 8], uv: [0, 0], inflate: -0.1 },
    { origin: [5, 20, 5], size: [6, 1, 6], uv: [0, 5], inflate: 0.2 },
  ] },
  { name: "top", pivot: [1, 16, 1], cubes: [{ origin: [0, 0, 0], size: [14, 0, 14], uv: [-14, 13], faces: [Direction.Up] }] },
  { name: "bottom", pivot: [1, 0, 1], cubes: [{ origin: [0, 0, 0], size: [14, 0, 14], uv: [-14, 13], faces: [Direction.Down] }] },
];
const SIDE_PARTS: ModelPartDef[] = [
  { name: "back", pivot: [15, 16, 1], rotation: [0, 0, Math.PI], cubes: [SIDE()] },
  { name: "left", pivot: [1, 16, 1], rotation: [0, -Math.PI / 2, Math.PI], cubes: [SIDE()] },
  { name: "right", pivot: [15, 16, 15], rotation: [0, Math.PI / 2, Math.PI], cubes: [SIDE()] },
  { name: "front", pivot: [1, 16, 15], rotation: [Math.PI, 0, 0], cubes: [SIDE()] },
];

const BASE_TEX = "entity/decorated_pot/decorated_pot_base";
const SIDE_TEX = "entity/decorated_pot/decorated_pot_side";

export const DECORATED_POT: BlockEntityDef = {
  types: ["decorated_pot"],
  textures: [BASE_TEX, SIDE_TEX],
  animated: false,
  bake(rec: BlockEntityRecord, ctx: BEBakeContext) {
    const o = ctx.sectionOrigin;
    const base = sectionLocal(rec, o, facingRot(rec.props.facing, 180));
    const out = [];
    const baseRect = ctx.uvFor(BASE_TEX);
    if (baseRect) out.push(...bakeModel(BASE_PARTS, { texWidth: 32, texHeight: 32, uvRect: baseRect, base, pass: TerrainPass.Cutout, shade: true }));
    const sideRect = ctx.uvFor(SIDE_TEX);
    if (sideRect) out.push(...bakeModel(SIDE_PARTS, { texWidth: 16, texHeight: 16, uvRect: sideRect, base, pass: TerrainPass.Cutout, shade: true }));
    return out.length ? out : null;
  },
};
