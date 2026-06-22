// The light look-up table (P6/LM-1). RENDERER_PLAN.md Phase 6.
//
// A 16×16 RGBA8 texture indexed by (blockLevel, skyLevel) — x = block light 0..15, y = sky light 0..15.
// Each per-vertex lightmap (mesh/Lighting.ts) stores those two levels as texel-center coordinates; the
// shader samples this table and MULTIPLIES the texel color by it. Because the light is a LUT *level*,
// not a baked color, global brightness / day-night / night-vision is a single 1 KiB `writeTexture` — NO
// remesh (LM-1, the only genuinely free win — TRAP 6.A). The table is stored as plain linear multipliers
// (Rgba8Unorm, NOT -srgb): the shader uses the sampled value directly, with no second sRGB decode.
//
// DEFAULT calibration (`brightness 1`, `ambient 0`): (block 0, sky 15) → white (255,255,255). So a scene
// with the editor's full-bright default lightmap (sky 15 everywhere) renders byte-identically to the
// pre-lighting renderer — the golden-image safety the encoding decision hinges on (TRAP 6.B).

import { clamp01 } from "../types";

export const LIGHT_LUT_SIZE = 16;

export interface LightLutParams {
  /** Global brightness multiplier (day-night / gamma knob). 1 = full daylight; 0 = pitch black. */
  brightness?: number;
  /** Minimum lit fraction even at light 0 (vanilla's faint ambient / "minimum light" + night-vision floor). */
  ambient?: number;
  /** Sky-light scale — how much the sky channel contributes (drop toward 0 to simulate night). 1 = full day. */
  skyScale?: number;
  /**
   * Per-level falloff curve (SL-5). `'linear'` (DEFAULT) keeps the byte-identical golden ramp. `'vanilla'`
   * applies Minecraft's `LightTexture` curve `l/(4−3l)` to the combined level, so mid-range levels read
   * darker (the linear ramp reads too bright vs. vanilla). LUT-only ⇒ a recalibration is a free 1 KiB
   * re-upload, no remesh. Endpoints are exact (`falloff(0)=0`, `falloff(1)=1`) so full-bright still maps to
   * white (TRAP 6.B). */
  curve?: "linear" | "vanilla";
  /** Optional gamma applied to the final lit fraction (default 1 = identity). Endpoints (0,1) are preserved
   *  for any gamma > 0; >1 darkens mid-tones, <1 brightens them. */
  gamma?: number;
}

/** Vanilla `LightTexture` falloff: l ∈ [0,1] → l/(4−3l). Exact at the endpoints (0→0, 1→1). */
function vanillaFalloff(l: number): number {
  return l / (4 - 3 * l);
}

/**
 * Build the 16×16 RGBA8 light table (256 texels, row-major: index = (sky*16 + block)*4). v1 combines block
 * and sky light by taking the brighter contribution (vanilla's lightmap is a max-blend of the two channels),
 * lifts it off the `ambient` floor, and scales by global `brightness`. Returned grey (R=G=B) — a future pass
 * can tint sky vs. block (warm torchlight / cool moonlight) without touching the vertex encoding.
 */
export function buildLightLut(params: LightLutParams = {}): Uint8Array {
  const brightness = params.brightness ?? 1;
  const ambient = clamp01(params.ambient ?? 0);
  const skyScale = params.skyScale ?? 1;
  const curve = params.curve ?? "linear"; // SL-5: default linear keeps the existing golden table
  const gamma = params.gamma ?? 1;
  const N = LIGHT_LUT_SIZE;
  const out = new Uint8Array(N * N * 4);
  for (let sky = 0; sky < N; sky++) {
    for (let block = 0; block < N; block++) {
      const skyF = (sky / (N - 1)) * skyScale;
      const blockF = block / (N - 1);
      const level = Math.max(skyF, blockF); // brighter of the two channels wins (vanilla max-blend)
      // SL-5: bend the level through the vanilla falloff (LUT-only). Endpoints exact ⇒ full-bright stays
      // white; default 'linear' leaves `level` untouched (byte-identical to the pre-SL-5 table).
      const shaped = curve === "vanilla" ? vanillaFalloff(level) : level;
      let lit = clamp01((ambient + (1 - ambient) * shaped) * brightness);
      if (gamma !== 1) lit = Math.pow(lit, gamma); // pow preserves the 0 and 1 endpoints for gamma > 0
      const byte = Math.round(lit * 255);
      const o = (sky * N + block) * 4;
      out[o] = byte;
      out[o + 1] = byte;
      out[o + 2] = byte;
      out[o + 3] = 255;
    }
  }
  return out;
}

/** The default identity table — full daylight, no ambient lift. Sampling it for a full-bright vertex
 *  (block 0, sky 15) returns white, so the default render is unchanged from before lighting landed. */
export function defaultLightLut(): Uint8Array {
  return buildLightLut();
}
