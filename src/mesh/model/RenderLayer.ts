// Classify a quad's render layer (TerrainPass) from its sprite's alpha. RENDERER_PLAN §24.7.
//
// Vanilla (26.1.2): the layer is derived per-quad from the sprite, not a per-block table — any
// translucent (partial-alpha) pixel → TRANSLUCENT; any fully-transparent pixel → CUTOUT; else
// SOLID. `force_translucent` (glass pane object texture) overrides to TRANSLUCENT.

import { TerrainPass } from "../../types";

export interface SpriteOpacity {
  /** Has alpha==0 pixels (holes) → cutout. */
  hasTransparent: boolean;
  /** Has 0<alpha<255 pixels → translucent. */
  hasTranslucent: boolean;
}

export function classifyLayer(opacity: SpriteOpacity | undefined, forceTranslucent: boolean): TerrainPass {
  if (forceTranslucent || opacity?.hasTranslucent) return TerrainPass.Translucent;
  if (opacity?.hasTransparent) return TerrainPass.Cutout;
  return TerrainPass.Solid;
}

/** Scan an RGBA sprite (all frames) for transparent / translucent texels. */
export function scanOpacity(rgba: Uint8Array): SpriteOpacity {
  let hasTransparent = false;
  let hasTranslucent = false;
  for (let i = 3; i < rgba.length; i += 4) {
    const a = rgba[i];
    if (a === 0) hasTransparent = true;
    else if (a < 255) hasTranslucent = true;
    if (hasTransparent && hasTranslucent) break;
  }
  return { hasTransparent, hasTranslucent };
}

/** A sprite is a solid occluder iff it has no transparent/translucent pixels. */
export function isOpaqueSprite(opacity: SpriteOpacity | undefined): boolean {
  return !opacity || (!opacity.hasTransparent && !opacity.hasTranslucent);
}
