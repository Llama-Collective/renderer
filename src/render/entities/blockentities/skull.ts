// Skulls / mob heads. RENDERER_PLAN.md §18. Vanilla SkullModel + SkullBlockRenderer:
//   - mob heads (skeleton/wither_skeleton/zombie/creeper) → createMobHeadLayer: head only, **texHeight 32**;
//   - player → createHumanoidHeadLayer: head + inflated hat, **texHeight 64**;
//   - piglin → 64 (no hat here, approximated); dragon → special model (approximated by the head box).
// Using the wrong texHeight makes the box-unwrap sample the wrong (transparent) region, so faces get
// alpha-discarded — hence per-type texHeight. Floor heads rotate by 22.5°·rotation; wall heads offset
// toward FACING; the base flips the model (scale(-1,-1,1)) into the lower half of the cell.

import { bakeModel, type ModelPartDef } from "../../../mesh/entity/ModelPart";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { TerrainPass } from "../../../types";
import { strip, type BEBakeContext, type BlockEntityDef } from "./registry";
import { DEG, mul, oppositeFacing, rotationY, scaling, sectionLocal, toYRot, translation } from "./transforms";

const HEAD: ModelPartDef = { name: "head", pivot: [0, 0, 0], cubes: [{ origin: [-4, -8, -4], size: [8, 8, 8], uv: [0, 0] }] };
const HAT: ModelPartDef = { name: "hat", pivot: [0, 0, 0], cubes: [{ origin: [-4, -8, -4], size: [8, 8, 8], uv: [32, 0], inflate: 0.25 }] };

// texH = the actual texture PNG height (the box-unwrap normalizes pixel UVs by it). Skeleton/wither/
// creeper skull textures are 64×32; zombie/player/piglin are 64×64. Using the wrong height makes faces
// sample transparent regions → alpha-discarded ("missing faces").
const META: Record<string, { texH: number; hat: boolean; tex: string }> = {
  skeleton: { texH: 32, hat: false, tex: "entity/skeleton/skeleton" },
  wither_skeleton: { texH: 32, hat: false, tex: "entity/skeleton/wither_skeleton" },
  zombie: { texH: 64, hat: false, tex: "entity/zombie/zombie" },
  creeper: { texH: 32, hat: false, tex: "entity/creeper/creeper" },
  player: { texH: 64, hat: true, tex: "entity/player/wide/steve" },
  piglin: { texH: 64, hat: false, tex: "entity/piglin/piglin" },
  dragon: { texH: 32, hat: false, tex: "entity/enderdragon/dragon" },
};

function baseType(rec: BlockEntityRecord): string {
  return strip(rec.type).replace(/_(wall_)?(skull|head)$/, "");
}

function skullTransform(rec: BlockEntityRecord) {
  if (rec.type.includes("_wall_")) {
    const facing = rec.props.facing ?? "north";
    const stepX = facing === "west" ? -1 : facing === "east" ? 1 : 0;
    const stepZ = facing === "north" ? -1 : facing === "south" ? 1 : 0;
    // oppositeFacing(facing, facing) reproduces the old local: cardinal flip, non-cardinal passthrough.
    return mul(translation(0.5 - stepX * 0.25, 0.25, 0.5 - stepZ * 0.25), rotationY(-toYRot(oppositeFacing(facing, facing)) * DEG), scaling(-1, -1, 1));
  }
  // Vanilla SkullBlockRenderer floor: translation(0.5,0,0.5) · rotateY(-segment·22.5°) · scale(-1,-1,1).
  // (No extra translate — the head model is centred at X/Z origin, so it spins in place and sits centred.)
  const seg = parseInt(rec.props.rotation ?? "0", 10) || 0;
  return mul(translation(0.5, 0, 0.5), rotationY(-(seg * 22.5) * DEG), scaling(-1, -1, 1));
}

const TYPES = ["skeleton", "wither_skeleton", "zombie", "creeper", "player", "piglin", "dragon"]
  .flatMap((b) => [`${b}_skull`, `${b}_wall_skull`, `${b}_head`, `${b}_wall_head`]);

export const SKULL: BlockEntityDef = {
  types: TYPES,
  textures: Object.values(META).map((m) => m.tex),
  animated: false,
  bake(rec: BlockEntityRecord, ctx: BEBakeContext) {
    const meta = META[baseType(rec)] ?? META.player;
    const rect = ctx.uvFor(meta.tex);
    if (!rect) return null;
    const o = ctx.sectionOrigin;
    const base = sectionLocal(rec, o, skullTransform(rec));
    const parts = meta.hat ? [HEAD, HAT] : [HEAD];
    return bakeModel(parts, { texWidth: 64, texHeight: meta.texH, uvRect: rect, base, pass: TerrainPass.Cutout, shade: true });
  },
};
