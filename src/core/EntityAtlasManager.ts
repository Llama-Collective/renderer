// Arbitrary-rect texture atlas for entity / block-entity skins. RENDERER_PLAN.md §18 (Phase 4.5b).
//
// The block `AtlasManager` packs EQUAL tiles; entity/BE textures are non-uniform (chest 64×64, sheep
// 64×32, signs 64×32, banner sheets…), so they need their own arbitrary-rect atlas (the §18
// `EntityAtlasManager`). Linear-color like blocks (`-srgb` target → samples decode to linear). A simple
// shelf packer with a 1px gutter (no bleed under Nearest filtering) — the packing is pure + unit-tested;
// this class only stitches + uploads. Box-unwrap UVs are normalized by the model's logical texture size,
// then remapped into the returned rect (bake-once + UV remap, §18).

import { TextureFormat, type GpuTextureHandle, type GraphicsDevice } from "./GraphicsDevice";
import { blitRect } from "./atlasBlit";
import type { SpriteUv } from "../types";

export interface EntitySprite {
  name: string;
  width: number;
  height: number;
  /** RGBA8, length width*height*4. */
  rgba: Uint8Array;
}

export interface Placement {
  name: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PackResult {
  width: number;
  height: number;
  placements: Placement[];
}

const GUTTER = 1; // px between sprites — avoids edge bleed

/**
 * Shelf-pack arbitrary-size sprites into a power-of-two atlas. Sprites are placed tallest-first into
 * left-to-right rows (shelves), wrapping when the row overflows `maxWidth`. Pure + deterministic.
 */
export function packEntityAtlas(sprites: readonly EntitySprite[], maxWidth?: number): PackResult {
  if (sprites.length === 0) throw new Error("packEntityAtlas: no sprites");
  const sorted = [...sprites].sort((a, b) => b.height - a.height || b.width - a.width || a.name.localeCompare(b.name));
  const widest = Math.max(...sprites.map((s) => s.width + GUTTER));
  const area = sprites.reduce((s, sp) => s + (sp.width + GUTTER) * (sp.height + GUTTER), 0);
  const width = nextPow2(Math.max(widest, maxWidth ?? Math.ceil(Math.sqrt(area))));

  const placements: Placement[] = [];
  let x = GUTTER;
  let y = GUTTER;
  let shelfH = 0;
  for (const s of sorted) {
    if (x + s.width + GUTTER > width) {
      // New shelf.
      y += shelfH + GUTTER;
      x = GUTTER;
      shelfH = 0;
    }
    placements.push({ name: s.name, x, y, width: s.width, height: s.height });
    x += s.width + GUTTER;
    shelfH = Math.max(shelfH, s.height);
  }
  const height = nextPow2(y + shelfH + GUTTER);
  return { width, height, placements };
}

export class EntityAtlasManager {
  private tex: GpuTextureHandle | null = null;
  private widthPx = 0;
  private heightPx = 0;
  private readonly rects = new Map<string, SpriteUv>();

  constructor(private readonly device: GraphicsDevice) {}

  /** Pack + stitch + upload. Replaces any previous atlas. Synchronous (sprites pre-decoded). */
  build(sprites: readonly EntitySprite[]): void {
    const pack = packEntityAtlas(sprites);
    this.widthPx = pack.width;
    this.heightPx = pack.height;
    const byName = new Map(sprites.map((s) => [s.name, s]));
    const atlas = new Uint8Array(pack.width * pack.height * 4);
    this.rects.clear();

    for (const p of pack.placements) {
      const src = byName.get(p.name)!;
      blit(src.rgba, src.width, src.height, atlas, pack.width, p.x, p.y);
      this.rects.set(p.name, {
        u: p.x / pack.width,
        v: p.y / pack.height,
        width: p.width / pack.width,
        height: p.height / pack.height,
      });
    }

    if (this.tex) this.device.destroyTexture(this.tex);
    this.tex = this.device.createTexture({ width: pack.width, height: pack.height, format: TextureFormat.Rgba8Srgb, label: "entity-atlas" });
    this.device.writeTexture(this.tex, atlas);
  }

  get texture(): GpuTextureHandle {
    if (!this.tex) throw new Error("EntityAtlasManager: build() first");
    return this.tex;
  }

  /** Release the GPU atlas texture (§8 single-owner dispose). Call from the scene teardown that owns it. */
  dispose(): void {
    if (this.tex) this.device.destroyTexture(this.tex);
    this.tex = null;
    this.rects.clear();
  }

  get size(): readonly [number, number] {
    return [this.widthPx, this.heightPx];
  }

  has(name: string): boolean {
    return this.rects.has(name);
  }

  uvFor(name: string): SpriteUv {
    const r = this.rects.get(name);
    if (!r) throw new Error(`EntityAtlasManager: unknown sprite "${name}"`);
    return r;
  }
}

function blit(src: Uint8Array, sw: number, sh: number, dst: Uint8Array, dstStridePx: number, dx: number, dy: number): void {
  blitRect(src, sw, sh, 0, dst, dstStridePx, dx, dy);
}

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}
