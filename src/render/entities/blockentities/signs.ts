// Signs (standing / wall / hanging) + bitmap-font text. RENDERER_PLAN.md §18, Phase 4.5d.
// Vanilla StandingSignRenderer / HangingSignRenderer board geometry + AbstractSignRenderer text: up to 4
// lines, each centred (x = −Font.width/2) and stacked by LINE_HEIGHT, the block centred on the board. The
// text is rendered at scale 0.010416667 blocks/px vs the board's 0.6666667/16 (≈ 0.25× the board pixel
// scale). Multi-texture (wood board + `font/ascii`), so it hand-builds bake groups instead of `boxBE`.

import { bakeModel, type ModelPartDef } from "../../../mesh/entity/ModelPart";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { Direction, TerrainPass } from "../../../types";
import { strip, WOOD_TYPES, type BEBakeContext, type BlockEntityDef } from "./registry";
import { DEG, mul, rotAroundY, rotationX, scaling, sectionLocal, translation, facingRot, type Mat4 } from "./transforms";
import { FONT_SIZE, FONT_TEX, signTextParts } from "./font";

const STANDING: ModelPartDef[] = [
  { name: "board", pivot: [0, 0, 0], cubes: [{ origin: [-12, -14, -1], size: [24, 12, 2], uv: [0, 0] }] },
  { name: "stick", pivot: [0, 0, 0], cubes: [{ origin: [-1, -2, -1], size: [2, 14, 2], uv: [0, 14] }] },
];
const WALL: ModelPartDef[] = [{ name: "board", pivot: [0, 0, 0], cubes: [{ origin: [-12, -14, -1], size: [24, 12, 2], uv: [0, 0] }] }];
// Hanging sign (vanilla HangingSignRenderer.createHangingSignLayer, texW/H 64×32). Board + per-attachment
// connectors: WALL → a top plank + the 4 X-chains; CEILING → the 4 X-chains; CEILING_MIDDLE → a V-bar.
const HANGING_BOARD: ModelPartDef = { name: "board", pivot: [0, 0, 0], cubes: [{ origin: [-7, 0, -1], size: [14, 10, 2], uv: [0, 12] }] };
const HANGING_PLANK: ModelPartDef = { name: "plank", pivot: [0, 0, 0], cubes: [{ origin: [-8, -6, -2], size: [16, 2, 4], uv: [0, 0] }] };
const HANGING_V_CHAINS: ModelPartDef = { name: "vChains", pivot: [0, 0, 0], cubes: [{ origin: [-6, -6, 0], size: [12, 6, 0], uv: [14, 6] }] };
// Each chain segment: a 3×6 flat plane pivoted at x=±5, y=−6, rotated ±45° about Y (two per side → an X).
const chain = (x: number, ry: number, uv: [number, number]): ModelPartDef => ({ pivot: [x, -6, 0], rotation: [0, ry, 0], cubes: [{ origin: [-1.5, 0, 0], size: [3, 6, 0], uv }] });
const HANGING_CHAINS: ModelPartDef[] = [
  chain(-5, -Math.PI / 4, [0, 6]), chain(-5, Math.PI / 4, [6, 6]),
  chain(5, -Math.PI / 4, [0, 6]), chain(5, Math.PI / 4, [6, 6]),
];

// Text placement on the board, in board-MODEL-block space (the sign transform then flips + scales + places
// it). 1 font-px = TEXT_SCALE board-px. The viewer-facing board side after the model's rotationX(π) is the
// −Z (North) face, so glyphs use the North face + mirrored X to read correctly. Per board kind.
const TEXT: Record<"standing" | "hanging", { scale: number; y: number; z: number }> = {
  // signTransform's board scale (2/3 standing, 1 hanging) multiplies TEXT.scale, so TEXT.scale·boardScale/16
  // must equal vanilla's text px-scale: standing 0.010416667 → 0.25·(2/3); hanging 0.0140625 → 0.225·1.
  standing: { scale: 0.25, y: -0.5, z: -0.085 }, // text centred on the 24px board (24/16·2/3)
  hanging: { scale: 0.225, y: 0.6325, z: -0.073 }, // vanilla HangingSignRenderer: 0.0140625·16; TEXT_OFFSET + baseT
};

function wood(rec: BlockEntityRecord): string {
  const name = strip(rec.type);
  return WOOD_TYPES.find((w) => name.startsWith(`${w}_`)) ?? "oak";
}
const isHanging = (rec: BlockEntityRecord) => rec.type.includes("hanging");
const isWall = (rec: BlockEntityRecord) => rec.type.includes("wall");

function signTexture(rec: BlockEntityRecord): string {
  const w = wood(rec);
  return isHanging(rec) ? `entity/signs/hanging/${w}` : `entity/signs/${w}`;
}
function signParts(rec: BlockEntityRecord): ModelPartDef[] {
  if (isHanging(rec)) {
    if (isWall(rec)) return [HANGING_BOARD, HANGING_PLANK, ...HANGING_CHAINS];
    // Ceiling: `attached=true` hangs by a V-bar directly under a block; else by the X-chains.
    return rec.props.attached === "true" ? [HANGING_BOARD, HANGING_V_CHAINS] : [HANGING_BOARD, ...HANGING_CHAINS];
  }
  return isWall(rec) ? WALL : STANDING;
}

function signTransform(rec: BlockEntityRecord): Mat4 {
  const hanging = isHanging(rec);
  const scale = hanging ? 1 : 2 / 3;
  const flip = mul(translation(0.5, hanging ? 0.9375 : 0.5, 0.5), rotationX(Math.PI), scaling(scale, scale, scale));
  if (isWall(rec)) {
    // StandingSignRenderer.baseTransformation (WALL case, SignRenderer): after the facing Y-rotation push
    // the board back against the wall + down a touch — vanilla translate(0, -0.3125, -0.4375) — BEFORE the
    // flip/scale. Without it a wall sign renders floating at the block centre (the standing position minus
    // its post) instead of flush on the wall. Hanging wall signs keep the hanging base (their board hangs
    // from a plank, no wall-flush offset), so the offset is non-hanging-only.
    return hanging
      ? mul(facingRot(rec.props.facing, 0), flip)
      : mul(facingRot(rec.props.facing, 0), translation(0, -0.3125, -0.4375), flip);
  }
  const seg = Number.parseInt(rec.props.rotation ?? "0", 10) || 0;
  return mul(rotAroundY(-seg * 22.5), flip);
}

/** Parse the `text` prop into ≤4 lines (split on literal/real newlines or `|`). */
function textLines(rec: BlockEntityRecord): string[] {
  return (rec.props.text ?? "").split(/\\n|\n|\|/).slice(0, 4);
}

export const SIGN: BlockEntityDef = {
  types: WOOD_TYPES.flatMap((w) => [`${w}_sign`, `${w}_wall_sign`, `${w}_hanging_sign`, `${w}_wall_hanging_sign`]),
  textures: [...WOOD_TYPES.flatMap((w) => [`entity/signs/${w}`, `entity/signs/hanging/${w}`]), FONT_TEX],
  animated: false,
  bake(rec: BlockEntityRecord, ctx: BEBakeContext) {
    const o = ctx.sectionOrigin;
    const base = sectionLocal(rec, o, signTransform(rec));
    const out = [];

    // Wood board (+ stick / hanging planks).
    const boardRect = ctx.uvFor(signTexture(rec));
    if (boardRect) out.push(...bakeModel(signParts(rec), { texWidth: 64, texHeight: 32, uvRect: boardRect, base, pass: TerrainPass.Cutout, shade: true }));

    // Text glyphs, composited on the board front from the bitmap font.
    const lines = textLines(rec).filter((l) => l.trim().length > 0);
    const fontRect = lines.length ? ctx.uvFor(FONT_TEX) : undefined;
    if (fontRect && lines.length) {
      const t = TEXT[isHanging(rec) ? "hanging" : "standing"];
      const glyphBase = mul(base, translation(0, t.y, t.z), scaling(t.scale, t.scale, t.scale));
      const parts = signTextParts(textLines(rec), false, Direction.North);
      out.push(...bakeModel(parts, { texWidth: FONT_SIZE, texHeight: FONT_SIZE, uvRect: fontRect, base: glyphBase, color: 0x000000ff, pass: TerrainPass.Cutout, shade: false }));
    }

    return out.length ? out : null;
  },
};

void DEG;
