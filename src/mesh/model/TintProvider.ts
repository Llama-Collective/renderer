// Resolve a face's tintindex → an RGB color. RENDERER_PLAN §24.8.
//
// Biome-CAPABLE (climate-sampled from the grass/foliage colormaps) but defaults to NO biome
// (fixed plains climate), since schematics carry no biome yet. The tint is baked into vertex color
// at mesh time (one color path, TRAP 7.D). Pure data, GPU-free.

import { clamp01 } from "../../types";
import { stripNamespace } from "./ModelTypes";

export interface Colormap {
  width: number;
  height: number;
  /** RGBA8, length width*height*4. */
  rgba: Uint8Array;
}

/** Biome climate point (0..1). The colormap is a triangular gradient indexed by it. */
export interface Climate {
  temperature: number;
  downfall: number;
}

/** Default climate when a schematic carries no biome (vanilla plains). */
export const PLAINS: Climate = { temperature: 0.8, downfall: 0.4 };

const WHITE = 0xffffff;
const DEFAULT_GRASS = 0x91bd59;
const DEFAULT_FOLIAGE = 0x77ab2f;
const DEFAULT_WATER = 0x3f76e4;
const SPRUCE_FOLIAGE = 0x619961;
const BIRCH_FOLIAGE = 0x80a755;

/** Biome/state tint families — a closed set, so the `switch` in `tintFor` stays exhaustive. */
enum TintType {
  Grass = "grass",
  Foliage = "foliage",
  Water = "water",
  Redstone = "redstone",
  None = "none",
}

// Vanilla BlockColors.createDefault() registers tint sources by an EXPLICIT block list, NOT a name
// heuristic. Only these leaves get the biome FOLIAGE tint; spruce/birch leaves use fixed constants
// (handled in tintFor); cherry / azalea / flowering_azalea / pale_oak leaves are NOT tinted — the old
// `includes("leaves")` heuristic wrongly greened them (AUDIT M4: cherry rendered green, not pink).
const FOLIAGE_BLOCKS = new Set(["oak_leaves", "jungle_leaves", "acacia_leaves", "dark_oak_leaves", "mangrove_leaves", "vine"]);
const GRASS_BLOCKS = new Set(["grass_block", "grass", "short_grass", "tall_grass", "fern", "large_fern", "sugar_cane"]);

function tintTypeOf(block: string): TintType {
  const n = stripNamespace(block);
  if (n === "redstone_wire") return TintType.Redstone;
  if (n === "spruce_leaves" || n === "birch_leaves" || FOLIAGE_BLOCKS.has(n)) return TintType.Foliage;
  if (GRASS_BLOCKS.has(n)) return TintType.Grass;
  if (n.includes("water")) return TintType.Water;
  return TintType.None;
}

export class TintProvider {
  constructor(private readonly colormaps: { grass?: Colormap; foliage?: Colormap } = {}) {}

  /**
   * Tint color (0xRRGGBB) for `block`'s `tintindex` at `climate` (default plains). Returns white
   * (no tint) for tintindex < 0 or untinted blocks. `props` carries blockstate properties for
   * tints that depend on state (redstone wire's `power` ramp).
   */
  tintFor(block: string, tintindex: number, climate: Climate = PLAINS, props?: Record<string, string>): number {
    if (tintindex < 0) return WHITE;
    switch (tintTypeOf(block)) {
      case TintType.Redstone:
        return redstoneTint(props?.power);
      case TintType.Grass:
        return sample(this.colormaps.grass, climate, DEFAULT_GRASS);
      case TintType.Foliage: {
        const n = stripNamespace(block);
        if (n === "spruce_leaves") return SPRUCE_FOLIAGE;
        if (n === "birch_leaves") return BIRCH_FOLIAGE;
        return sample(this.colormaps.foliage, climate, DEFAULT_FOLIAGE);
      }
      case TintType.Water:
        return DEFAULT_WATER;
      default:
        return WHITE;
    }
  }
}

/** Redstone-dust color (0xRRGGBB) by power 0..15 — vanilla RedStoneWireBlock.getColorForPower:
 *  r = f*0.6 + (f>0 ? 0.4 : 0.3); g = clamp(f²*0.7 − 0.5, 0, 1); b = clamp(f²*0.6 − 0.7, 0, 1). */
function redstoneTint(power: string | undefined): number {
  const p = Math.max(0, Math.min(15, Number.parseInt(power ?? "0", 10) || 0));
  const f = p / 15;
  const r = f * 0.6 + (f > 0 ? 0.4 : 0.3);
  const g = clamp01(f * f * 0.7 - 0.5);
  const b = clamp01(f * f * 0.6 - 0.7);
  return (Math.round(r * 255) << 16) | (Math.round(g * 255) << 8) | Math.round(b * 255);
}

/** Sample a 256-wide colormap at the climate point (vanilla GrassColor/FoliageColor math). */
function sample(cm: Colormap | undefined, climate: Climate, fallback: number): number {
  if (!cm) return fallback;
  const t = clamp01(climate.temperature);
  const d = clamp01(climate.downfall) * t;
  const i = Math.floor((1 - t) * (cm.width - 1));
  const j = Math.floor((1 - d) * (cm.height - 1));
  const o = (j * cm.width + i) * 4;
  if (o < 0 || o + 2 >= cm.rgba.length) return fallback;
  return (cm.rgba[o] << 16) | (cm.rgba[o + 1] << 8) | cm.rgba[o + 2];
}

