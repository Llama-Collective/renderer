// Block texture atlas + animated-sprite ticking. RENDERER_PLAN.md §7.
//
// Stitches decoded RGBA sprites into ONE `-srgb` texture (samples decode to linear — §17.2) and hands
// out normalized UV rects. Resource packs MIX sprite resolutions (vanilla block textures are 16², but
// shelves / many others are 32²). Like vanilla's `Stitcher`, we pack each sprite at its NATIVE resolution into an arbitrary-rectangle layout — NOT a
// uniform tile grid, and NEVER upscaling. (We don't mip — terrain uses nearest, §7 — so a 1px gutter +
// half-texel UV inset stands in for vanilla's mip padding to stop bleeding.)
//
// CRITICAL (TRAP 7.A / 13.C): animated textures (water/lava/fire) live IN the atlas at a SINGLE, STABLE
// cell — the UVs baked into vertices never change. `tick()` advances each animated sprite's frame and
// re-uploads the atlas with the new frame blitted into its cell (vanilla's "update pixels, not UVs").

import { TextureFormat, type GpuTextureHandle, type GraphicsDevice } from "./GraphicsDevice";
import { packEntityAtlas, type EntitySprite } from "./EntityAtlasManager";
import { blitRect, padCellEdges } from "./atlasBlit";
import type { SpriteUv } from "../types";

export type { SpriteUv } from "../types";

/** A decoded sprite to stitch. Animated sprites stack `frameCount` square tiles (mcmeta convention). */
export interface SpriteSource {
  name: string;
  /** Native tile size in px (square). Sprites of DIFFERENT sizes may share one atlas (packed natively). */
  size: number;
  /** RGBA8 pixels, length `size*size*4` per frame, frames concatenated. */
  rgba: Uint8Array;
  frameCount?: number;
  frameTimeTicks?: number;
  /**
   * Explicit playback order (mcmeta `frames`): each entry is a strip-frame index. Honors vanilla's
   * custom sequences — e.g. lava_still ping-pongs `[0..19, 18..1]` for a seamless loop. Omitted →
   * sequential `0..frameCount-1`.
   */
  frameOrder?: number[];
  /** mcmeta `interpolate`: cross-fade between consecutive frames each tick (magma, prismarine, …). */
  interpolate?: boolean;
}

/** Per-animated-sprite playback state + the source frames to blit into its atlas cell. */
interface Animation {
  frames: Uint8Array; // the strip: all distinct frames concatenated (tile×tile rgba each)
  tile: number; // this sprite's native size (frames are tile×tile)
  cellX: number;
  cellY: number;
  /** Playback sequence of strip-frame indices (may revisit frames — e.g. lava's ping-pong). */
  order: number[];
  frameTimeTicks: number;
  /** Position WITHIN `order` (not the strip-frame index). */
  current: number;
  clockTicks: number;
  /** Cross-fade current→next frame by `clockTicks/frameTimeTicks` each tick (vanilla `interpolate`). */
  interpolate: boolean;
}

interface AtlasEntry {
  /** Stable normalized rect of this sprite's cell (frame-independent). */
  rect: SpriteUv;
  anim?: Animation;
}

/** ATL-1: rows reserved BELOW the packed content for incremental `append` (new block types placed at
 *  runtime). The atlas height is fixed at build to `content + headroom`, so existing sprites' normalized
 *  UVs NEVER move when a sprite is appended (TRAP 1.C) — the texture handle is stable too (appends are
 *  sub-rect uploads into the reserved strip). When the strip fills, the caller falls back to a full rebuild. */
const APPEND_HEADROOM_PX = 256;

export class AtlasManager {
  private tex: GpuTextureHandle | null = null;
  private widthPx = 0;
  private heightPx = 0;
  /** CPU mirror of the atlas pixels, so `tick()` can re-blit a frame and re-upload. */
  private pixels: Uint8Array | null = null;
  /** Reused contiguous staging buffer for per-cell sub-rect uploads (AUDIT #4). Grown to the largest tile. */
  private staging: Uint8Array | null = null;
  private readonly entries = new Map<string, AtlasEntry>();
  /** Shelf-packer cursor over the reserved headroom strip [contentH, heightPx) for `append` (ATL-1). */
  private appendX = 0;
  private appendY = 0;
  private appendShelfH = 0;

  constructor(private readonly device: GraphicsDevice) {}

  /**
   * Stitch `sources` into one atlas texture and record per-sprite UV rects. Replaces any previous atlas.
   * Mixed resolutions are packed at native size (vanilla Stitcher). Each sprite gets ONE cell — its first
   * playback frame is blitted now; `tick()` swaps in later frames. Synchronous (callers pre-decode).
   */
  build(sources: SpriteSource[]): void {
    if (sources.length === 0) throw new Error("AtlasManager.build: no sprites");
    // Arbitrary-rectangle pack at NATIVE resolution — vanilla-faithful, no upscaling.
    const pack = packEntityAtlas(sources.map((s): EntitySprite => ({ name: s.name, width: s.size, height: s.size, rgba: s.rgba })));
    const W = (this.widthPx = pack.width);
    const contentH = pack.height;
    const H = (this.heightPx = contentH + APPEND_HEADROOM_PX); // reserve headroom below content for ATL-1 appends
    // Start the append shelf-packer at the top of the reserved strip.
    this.appendX = 0;
    this.appendY = contentH;
    this.appendShelfH = 0;

    const atlas = new Uint8Array(W * H * 4);
    const byName = new Map(sources.map((s) => [s.name, s]));
    this.entries.clear();

    for (const p of pack.placements) {
      const s = byName.get(p.name)!;
      const frameCount = s.frameCount ?? 1;
      // Edge-to-edge UV rect (texel boundaries), NOT a half-texel CENTRE inset: explicit/partial face UVs
      // (a trapdoor's 3px side strip, a composter's rim) must land on the exact texel they name. The 1px
      // gutter mirror below keeps the far (u=1/v=1) edge clean. (TRAP 7.B — was a ½-texel inset that bled.)
      const rect: SpriteUv = { u: p.x / W, v: p.y / H, width: s.size / W, height: s.size / H };
      const entry: AtlasEntry = { rect };
      let firstFrame = 0;
      if (frameCount > 1) {
        const order = s.frameOrder && s.frameOrder.length > 0 ? s.frameOrder : Array.from({ length: frameCount }, (_, k) => k);
        firstFrame = order[0] ?? 0;
        entry.anim = { frames: s.rgba, tile: s.size, cellX: p.x, cellY: p.y, order, frameTimeTicks: s.frameTimeTicks ?? 1, current: 0, clockTicks: 0, interpolate: s.interpolate ?? false };
      }
      blit(s.rgba, firstFrame, s.size, atlas, W, p.x, p.y);
      padCellEdges(atlas, W, H, p.x, p.y, s.size); // mirror border → gutter (whole atlas is uploaded below)
      this.entries.set(s.name, entry);
    }

    this.pixels = atlas;
    if (this.tex) this.device.destroyTexture(this.tex);
    this.tex = this.device.createTexture({ width: W, height: H, format: TextureFormat.Rgba8Srgb, label: "block-atlas" });
    this.device.writeTexture(this.tex, atlas);
  }

  /**
   * ATL-1: incrementally add STATIC sprites into the reserved headroom WITHOUT re-packing — existing
   * sprites' cells + the atlas (W,H) are untouched, so every previously-baked UV stays valid (TRAP 1.C),
   * and the GPU texture handle is stable (sub-rect uploads only). Returns false (placing nothing more) if a
   * sprite is animated or the headroom is exhausted — the caller then falls back to a full `build`. Sprites
   * already present are skipped. Idempotent-safe: a false return leaves a consistent (partially-grown) atlas.
   */
  append(sources: SpriteSource[]): boolean {
    if (!this.pixels || !this.tex) return false;
    const W = this.widthPx;
    const H = this.heightPx;
    for (const s of sources) {
      if (this.entries.has(s.name)) continue; // already atlased
      if ((s.frameCount ?? 1) > 1) return false; // animated → full rebuild handles it correctly
      const size = s.size;
      if (size > W) return false; // can't fit the atlas width
      // Shelf-pack: wrap to a new shelf when the current row can't fit; fail when the strip is exhausted.
      if (this.appendX + size > W) {
        this.appendY += this.appendShelfH;
        this.appendX = 0;
        this.appendShelfH = 0;
      }
      if (this.appendY + size > H) return false; // headroom exhausted
      const x = this.appendX;
      const y = this.appendY;
      blit(s.rgba, 0, size, this.pixels, W, x, y); // mirror (frame 0 — static)
      padCellEdges(this.pixels, W, H, x, y, size); // mirror border → gutter (matches build; edge-to-edge UVs)
      const rect: SpriteUv = { u: x / W, v: y / H, width: size / W, height: size / H };
      this.entries.set(s.name, { rect });
      // Upload the cell PLUS its 1px right/bottom gutter from the strided CPU mirror so the mirrored edge
      // reaches the GPU (clamped to the atlas bounds). Reuses the per-cell staging buffer.
      const uw = Math.min(size + 1, W - x);
      const uh = Math.min(size + 1, H - y);
      const need = uw * uh * 4;
      if (!this.staging || this.staging.length < need) this.staging = new Uint8Array(need);
      for (let row = 0; row < uh; row++) {
        const src = ((y + row) * W + x) * 4;
        this.staging.set(this.pixels.subarray(src, src + uw * 4), row * uw * 4);
      }
      this.device.writeTexture(this.tex, this.staging.subarray(0, need), 0, { x, y, width: uw, height: uh });
      this.appendX += size;
      if (size > this.appendShelfH) this.appendShelfH = size;
    }
    return true;
  }

  /** Whether a sprite is already in the atlas (so a caller can decide append vs full rebuild). */
  has(spritePath: string): boolean {
    return this.entries.has(spritePath);
  }

  get texture(): GpuTextureHandle {
    if (!this.tex) throw new Error("AtlasManager: build() first");
    return this.tex;
  }

  /** Release the GPU atlas texture + CPU mirrors (§8 single-owner dispose). Call from the scene teardown
   *  that owns this manager. Safe to call before build() (no texture yet). */
  dispose(): void {
    if (this.tex) this.device.destroyTexture(this.tex);
    this.tex = null;
    this.pixels = null;
    this.staging = null;
    this.entries.clear();
  }

  /** Atlas width in px (atlas is no longer guaranteed square — see `size`). */
  get sizePixels(): number {
    return this.widthPx;
  }

  /** Atlas dimensions [width, height] in px. */
  get size(): readonly [number, number] {
    return [this.widthPx, this.heightPx];
  }

  /** Stable UV rect for a sprite (frame-independent — the cell, not the current frame). */
  uvFor(spritePath: string): SpriteUv {
    const e = this.entries.get(spritePath);
    if (!e) throw new Error(`AtlasManager: unknown sprite "${spritePath}"`);
    return e.rect;
  }

  isAnimated(spritePath: string): boolean {
    return !!this.entries.get(spritePath)?.anim;
  }

  /** True if ANY sprite in the atlas animates (water/lava/fire/…). Render-on-demand reads this to keep the
   *  loop drawing while textures advance; a fully-static atlas lets the loop idle. The animated set is fixed
   *  per scene (the palette determines it), so this is only consulted on the idle-decision path. */
  hasAnimated(): boolean {
    for (const e of this.entries.values()) if (e.anim) return true;
    return false;
  }

  /**
   * Advance animation clocks by `dtTicks`; when any animated sprite's frame changes, blit it into the
   * CPU atlas and re-upload. Static sprites are no-ops.
   */
  tick(dtTicks: number): void {
    if (!this.pixels || !this.tex) return;
    for (const e of this.entries.values()) {
      const a = e.anim;
      if (!a) continue;
      a.clockTicks += dtTicks;
      let stepped = false;
      while (a.clockTicks >= a.frameTimeTicks) {
        a.clockTicks -= a.frameTimeTicks;
        a.current = (a.current + 1) % a.order.length; // step through the playback ORDER
        stepped = true;
      }
      if (a.interpolate) {
        // Vanilla SpriteContents.Ticker: cross-fade current→next by the sub-frame fraction EVERY tick,
        // so slow animations (magma, prismarine, sea lantern, sculk) blend instead of stepping.
        const next = (a.current + 1) % a.order.length;
        blitBlend(a.frames, a.order[a.current], a.order[next], a.clockTicks / a.frameTimeTicks, a.tile, this.pixels, this.widthPx, a.cellX, a.cellY);
        this.uploadCell(a); // the blended pixels shift continuously
      } else if (stepped) {
        blit(a.frames, a.order[a.current], a.tile, this.pixels, this.widthPx, a.cellX, a.cellY);
        this.uploadCell(a);
      }
    }
  }

  /**
   * Re-upload ONE animated sprite's cell (AUDIT #4): copy its `tile×tile` cell out of the strided CPU
   * atlas mirror into a contiguous staging buffer and write just that sub-rect (~1 KB) — instead of
   * re-uploading the whole multi-MB atlas every time any sprite's frame advances.
   */
  private uploadCell(a: Animation): void {
    if (!this.pixels || !this.tex) return;
    const tile = a.tile;
    const rowBytes = tile * 4;
    const need = tile * rowBytes;
    if (!this.staging || this.staging.length < need) this.staging = new Uint8Array(need);
    const stg = this.staging;
    const px = this.pixels;
    const stride = this.widthPx * 4;
    for (let row = 0; row < tile; row++) {
      const src = (a.cellY + row) * stride + a.cellX * 4;
      stg.set(px.subarray(src, src + rowBytes), row * rowBytes);
    }
    this.device.writeTexture(this.tex, stg.subarray(0, need), 0, { x: a.cellX, y: a.cellY, width: tile, height: tile });
  }
}

/** Blit a per-pixel cross-fade `lerp(frameA, frameB, t)` of a sprite into the atlas at (dx,dy). */
function blitBlend(src: Uint8Array, frameA: number, frameB: number, t: number, tile: number, dst: Uint8Array, dstStridePx: number, dx: number, dy: number): void {
  const srcRowBytes = tile * 4;
  const offA = frameA * tile * srcRowBytes;
  const offB = frameB * tile * srcRowBytes;
  const inv = 1 - t;
  for (let row = 0; row < tile; row++) {
    let sA = offA + row * srcRowBytes;
    let sB = offB + row * srcRowBytes;
    let d = ((dy + row) * dstStridePx + dx) * 4;
    for (let i = 0; i < srcRowBytes; i++, sA++, sB++, d++) {
      dst[d] = (src[sA] * inv + src[sB] * t + 0.5) | 0;
    }
  }
}

/** Copy one frame (the `frame`-th `tile×tile` block) of a sprite into the atlas at (dx,dy). */
function blit(src: Uint8Array, frame: number, tile: number, dst: Uint8Array, dstStridePx: number, dx: number, dy: number): void {
  blitRect(src, tile, tile, frame * tile * tile * 4, dst, dstStridePx, dx, dy);
}
