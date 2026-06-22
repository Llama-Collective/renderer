// Parse a texture `.png.mcmeta`. RENDERER_PLAN §24.9.
//
// Fixes cubane's bug: a sprite is animated ONLY if the mcmeta has an `animation` key. Many 26.x
// block mcmeta are `{"texture":{"mipmap_strategy":...}}` (mipmap hints, NOT animation) — those must
// NOT be treated as animated. Pure data, GPU-free.

export interface AnimationMeta {
  /** Ticks per frame (default 1; 1 tick = 50ms). */
  frameTime: number;
  /** Explicit frame order (indices into the vertical strip); undefined → sequential. */
  frames?: number[];
  interpolate: boolean;
  /** Sub-frame size in px, if the strip isn't square frames. */
  width?: number;
  height?: number;
}

/** Returns the animation metadata, or null when the mcmeta has no `animation` block. */
export function parseAnimationMeta(json: unknown): AnimationMeta | null {
  if (typeof json !== "object" || json === null) return null;
  const anim = (json as Record<string, unknown>).animation;
  if (typeof anim !== "object" || anim === null) return null;
  const a = anim as Record<string, unknown>;

  const frames = Array.isArray(a.frames)
    ? a.frames.map((f) => (typeof f === "object" && f !== null ? Number((f as Record<string, unknown>).index) : Number(f)))
    : undefined;

  return {
    frameTime: typeof a.frametime === "number" ? a.frametime : 1,
    frames,
    interpolate: a.interpolate === true,
    width: typeof a.width === "number" ? a.width : undefined,
    height: typeof a.height === "number" ? a.height : undefined,
  };
}
