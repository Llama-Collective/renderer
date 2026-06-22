// Banners (base dye + ≤16 pattern×dye layers + idle wave). RENDERER_PLAN.md §18, Phase 4.5d.
// Vanilla BannerRenderer: pole + bar (cloth `entity/banner/banner_base`), then the flag is composited as
// stacked coplanar layers — the white cloth, a base-dye layer (`entity/banner/base`), then up to 16
// pattern layers (`entity/banner/<pattern>`), each tinted by its own dye (DyeColor.getTextureDiffuseColor).
// Multi-texture + per-layer tint, so it hand-builds bake groups instead of `boxBE`. Coplanar layers are
// separated in depth by a tiny per-layer `inflate` (our cutout pass depth-tests `Less`, where vanilla's
// banner render type relies on draw order + `LEQUAL`).

import { bakeModel, type ModelPartDef, type PartPose } from "../../../mesh/entity/ModelPart";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { TerrainPass } from "../../../types";
import { dyeRgba } from "./dyes";
import { strip, type BEBakeContext, type BlockEntityDef } from "./registry";
import { facingRot, mul, rotAroundY, scaling, sectionLocal, translation, type Mat4 } from "./transforms";
import { DYE_NAMES } from "./dyes";

// Every banner pattern asset id (vanilla `entity/banner/*`); all stitched so any `patterns` prop resolves.
const PATTERNS = [
  "base", "border", "bricks", "circle", "creeper", "cross", "curly_border", "diagonal_left", "diagonal_right",
  "diagonal_up_left", "diagonal_up_right", "flow", "flower", "globe", "gradient", "gradient_up", "guster",
  "half_horizontal", "half_horizontal_bottom", "half_vertical", "half_vertical_right", "mojang", "piglin",
  "rhombus", "skull", "small_stripes", "square_bottom_left", "square_bottom_right", "square_top_left",
  "square_top_right", "straight_cross", "stripe_bottom", "stripe_center", "stripe_downleft", "stripe_downright",
  "stripe_left", "stripe_middle", "stripe_right", "stripe_top", "triangle_bottom", "triangle_top",
  "triangles_bottom", "triangles_top",
];

const CLOTH_TEX = "entity/banner/banner_base";
const BASE_TEX = "entity/banner/base";
const patTex = (p: string) => `entity/banner/${p}`;

const POLE: ModelPartDef = { name: "pole", pivot: [0, 0, 0], cubes: [{ origin: [-1, -42, -1], size: [2, 42, 2], uv: [44, 0] }] };
const BAR: ModelPartDef = { name: "bar", pivot: [0, 0, 0], cubes: [{ origin: [-10, -44, -1], size: [20, 2, 2], uv: [0, 42] }] };
const FLAG: ModelPartDef[] = [{ name: "flag", pivot: [0, -44, 0], cubes: [{ origin: [-10, 0, -1], size: [20, 40, 1], uv: [0, 0] }] }];

// Per-layer depth separation (model px) so coplanar flag layers don't z-fight; UVs ignore inflate.
const INFLATE_STEP = 0.1;

function color(rec: BlockEntityRecord): string {
  const name = strip(rec.type);
  return DYE_NAMES.find((d) => name.startsWith(`${d}_`)) ?? "white";
}
const isWall = (rec: BlockEntityRecord) => rec.type.includes("wall");

/** Parse `patterns="stripe_bottom:red,cross:blue"` → ordered {pattern,color} layers (≤16). */
function parsePatterns(s: string | undefined): { pattern: string; color: string }[] {
  return (s ?? "")
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean)
    .map((t) => {
      const [pattern, c] = t.split(":");
      return { pattern: pattern.trim(), color: (c ?? "white").trim() };
    })
    .slice(0, 16);
}

function bannerTransform(rec: BlockEntityRecord): Mat4 {
  const flip = mul(translation(0.5, 0, 0.5), scaling(2 / 3, -2 / 3, -2 / 3));
  if (isWall(rec)) return mul(facingRot(rec.props.facing, 0), flip);
  const seg = Number.parseInt(rec.props.rotation ?? "0", 10) || 0;
  return mul(rotAroundY(-seg * 22.5), flip);
}

/** The flag's idle wave (vanilla BannerFlagModel xRot), applied to every flag layer when animating. */
function wavePose(clock: number): Record<string, PartPose> {
  const phase = (clock * 0.2) % 1;
  const x = (-0.0125 + 0.01 * Math.cos(2 * Math.PI * phase)) * Math.PI;
  return { flag: { rotation: [x, 0, 0] } };
}

export const BANNER: BlockEntityDef = {
  types: DYE_NAMES.flatMap((d) => [`${d}_banner`, `${d}_wall_banner`]),
  textures: [CLOTH_TEX, ...PATTERNS.map(patTex)],
  animated: true,
  idleLoop: true, // the flag wave is a continuous idle loop vanilla always shows (BBE §5) — never bake frozen
  bake(rec: BlockEntityRecord, ctx: BEBakeContext, clock: number, animating: boolean) {
    const o = ctx.sectionOrigin;
    const base = sectionLocal(rec, o, bannerTransform(rec));
    const poses = animating ? wavePose(clock) : undefined;
    const out = [];

    // Pole + bar (cloth, untinted). Wall banners have no pole.
    const clothRect = ctx.uvFor(CLOTH_TEX);
    if (clothRect) {
      const structural = isWall(rec) ? [BAR] : [POLE, BAR];
      out.push(...bakeModel(structural, { texWidth: 64, texHeight: 64, uvRect: clothRect, base, pass: TerrainPass.Cutout, shade: true }));
    }

    // Flag layers, bottom→top: white cloth, the base-dye flag (`entity/banner/base`), then each pattern in
    // its own dye. Each is pushed out slightly so the coplanar layers don't z-fight.
    const layers: { tex: string; color?: number }[] = [
      { tex: CLOTH_TEX }, // white knit cloth
      { tex: BASE_TEX, color: dyeRgba(color(rec)) }, // solid base-dye flag
      ...parsePatterns(rec.props.patterns).map((p) => ({ tex: patTex(p.pattern), color: dyeRgba(p.color) })),
    ];
    layers.forEach((layer, i) => {
      const rect = ctx.uvFor(layer.tex);
      if (!rect) return; // texture not stitched (unknown pattern) → skip that layer
      const flag = FLAG.map((f) => ({ ...f, cubes: f.cubes!.map((c) => ({ ...c, inflate: (c.inflate ?? 0) + i * INFLATE_STEP })) }));
      out.push(...bakeModel(flag, { texWidth: 64, texHeight: 64, uvRect: rect, base, poses, color: layer.color, pass: TerrainPass.Cutout, shade: true }));
    });

    return out.length ? out : null;
  },
};
