// Packed per-vertex light. RENDERER_PLAN.md §9, Phase 6 (LM-1/LM-4).
//
// THE ENCODING DECISION (TRAP 6.B): light is baked as a LUT-sampled *LEVEL*, never a pre-multiplied
// vertex color. The packed word carries the (block, sky) light LEVELS (0..15) as texture COORDINATES
// into a small light-LUT (render/LightLut.ts) — so a global brightness / day-night change is a free
// LUT rewrite (no remesh), and a per-block light change is a partial light-byte reupload (LM-2), NOT a
// full re-stitch of vertex colors. Were light pre-multiplied into `colorRGBA`, every global light
// change would force a remesh of every section (defeating LM-1).
//
// Layout — the 4 reserved vertex bytes 16..19 (location 3 = `uint8x4`, see mesh/VertexFormat.ts):
//   byte 16  blockCoord  = blockLevel*16 + 8   (LUT x texel center; 0..15 → 8..248, the plan's clamp)
//   byte 17  skyCoord    = skyLevel*16 + 8     (LUT y texel center)
//   byte 18  ao          = ambient-occlusion multiplier × 255 (255 = unoccluded; v1 bakes 255, the
//                          slot is the durable home for smooth AO later — SmoothLightPipeline)
//   byte 19  flags       = bit 0 emissive/full-bright (reserved; v1 bakes 0)
// TRAP 9.A: this is precomputed at BAKE time into the snapshot/vertex stream — never a per-vertex
// dynamic lookup at draw time. Emissive/full-bright is an explicit bit, NOT inferred from transparency
// (TRAP 7.D).

/** All light fields packed into one 32-bit word (block coord, sky coord, ao, flags). LITTLE-ENDIAN:
 *  byte0 = block, byte1 = sky, byte2 = ao, byte3 = flags — matches the `setUint32(.., true)` write and
 *  the WGSL `uint8x4` → `vec4<u32>` decode (x=block, y=sky, z=ao, w=flags). */
export type PackedLight = number;

/** Number of discrete light levels (Minecraft 0..15). */
export const LIGHT_LEVELS = 16;
/** AO byte meaning "fully unoccluded" (multiplier 1.0). v1 bakes this for every vertex. */
export const AO_NONE = 255;

/** Map a 0..15 light level to its LUT texel-center coordinate byte (level*16 + 8 → 8..248). The half-texel
 *  offset (+8) lands exactly on the texel center of a 16-wide LUT so `coord/256` samples that level cleanly
 *  (no neighbor bleed), and clamps so an out-of-range level can never read off the LUT edge. */
export function lightCoord(level: number): number {
  const l = level < 0 ? 0 : level > LIGHT_LEVELS - 1 ? LIGHT_LEVELS - 1 : level | 0;
  return l * 16 + 8;
}

/** Recover the 0..15 level from a coordinate byte (inverse of `lightCoord`). */
export function lightLevelOf(coord: number): number {
  return (coord - 8) >> 4;
}

/**
 * Pack a vertex light word. `blockLight`/`skyLight` are 0..15 levels; `ao` is a 0..255 occlusion
 * multiplier (255 = unoccluded); `emissive` forces full-bright (sets the flag bit AND lifts block light
 * to 15 so the block reads bright through the LUT even in darkness — vanilla light_emission analog).
 */
export function packLight(blockLight: number, skyLight: number, ao: number, emissive: boolean): PackedLight {
  const block = emissive ? lightCoord(LIGHT_LEVELS - 1) : lightCoord(blockLight);
  const sky = lightCoord(skyLight);
  const a = ao < 0 ? 0 : ao > 255 ? 255 : ao | 0;
  const flags = emissive ? 1 : 0;
  return (block | (sky << 8) | (a << 16) | (flags << 24)) >>> 0;
}

export function unpackLight(word: PackedLight): {
  blockLight: number;
  skyLight: number;
  ao: number;
  emissive: boolean;
} {
  return {
    blockLight: lightLevelOf(word & 0xff),
    skyLight: lightLevelOf((word >>> 8) & 0xff),
    ao: (word >>> 16) & 0xff,
    emissive: ((word >>> 24) & 1) === 1,
  };
}

/** Full-bright vertex light (block 0, sky 15, unoccluded) — the editor's default when no propagated light
 *  data is supplied. The default LUT maps (block 0, sky 15) to white, so a full-bright scene renders
 *  byte-identically to the pre-lighting renderer (TRAP 6.B golden-image safety). */
export const FULLBRIGHT_LIGHT: PackedLight = packLight(0, LIGHT_LEVELS - 1, AO_NONE, false);
