// Load a Minecraft resource-pack zip and expose the data the renderer needs. §24.2, §24.9.
//
// Fetches `pack.zip` (or reads it from the IndexedDB cache), then loads ONLY what's asked for:
// `preloadBlocks` walks the blockstates + model parent chains for a set of blocks (a few dozen
// files, not the whole 4400-file pack), and sprites/colormaps decode on demand. Browser-only
// orchestration (JSZip + fetch + canvas); the pure resolution/bake/tint logic it feeds is unit-
// tested separately, and this is verified live in Chrome.

import JSZip from "jszip";
import { stripNamespace, type RawBlockModel, type RawBlockState, type RawVariant } from "../mesh/model/ModelTypes";
import type { RawModelProvider } from "../mesh/model/ModelResolver";
import { scanOpacity, type SpriteOpacity } from "../mesh/model/RenderLayer";
import type { Colormap } from "../mesh/model/TintProvider";
import type { SpriteSource } from "../core/AtlasManager";
import { parseAnimationMeta } from "./textureMeta";
import { decodePng } from "./TextureDecoder";
import { cacheGet, cachePut } from "./AssetCache";

const ROOT = "assets/minecraft/";

export interface LoadedSprites {
  sprites: SpriteSource[];
  opacity: Map<string, SpriteOpacity>;
}

export class ResourcePack {
  private readonly blockstates = new Map<string, RawBlockState>();
  private readonly models = new Map<string, RawBlockModel>();

  private constructor(private readonly zip: JSZip) {}

  /** Fetch (or read from IndexedDB) and open the pack zip. Parses nothing until asked. */
  static async load(url: string): Promise<ResourcePack> {
    const cacheKey = `zip:${url}`;
    let bytes = await cacheGet(cacheKey);
    if (!bytes) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`ResourcePack: failed to fetch ${url} (${res.status})`);
      bytes = await res.arrayBuffer();
      await cachePut(cacheKey, bytes);
    }
    return new ResourcePack(await JSZip.loadAsync(bytes));
  }

  /** Open a pack from already-fetched zip bytes (the app loads the pack as an ArrayBuffer). */
  static async fromBytes(bytes: ArrayBuffer | Uint8Array): Promise<ResourcePack> {
    return new ResourcePack(await JSZip.loadAsync(bytes));
  }

  /** Count of texture PNGs the pack contains — a meaningful "pack loaded" figure (the renderer only
   *  decodes the sprites a world actually uses, so this is the pack's capacity, not the atlased count). */
  textureCount(): number {
    let n = 0;
    for (const path of Object.keys(this.zip.files)) {
      if (path.endsWith(".png") && path.includes("/textures/")) n++;
    }
    return n;
  }

  /** Parse the blockstates for `blockNames` + every model they (transitively) reference. */
  async preloadBlocks(blockNames: Iterable<string>): Promise<void> {
    const modelIds = new Set<string>();
    for (const name of new Set(blockNames)) {
      // Strip the namespace for BOTH the file path and the map key, so callers may pass "minecraft:stone"
      // or "stone" interchangeably (getBlockState/getModel also key by the stripped name).
      const bare = stripNamespace(name);
      const bs = await this.readJson<RawBlockState>(`blockstates/${bare}.json`);
      if (!bs) continue;
      this.blockstates.set(bare, bs);
      collectModelIds(bs, modelIds);
    }
    const queue = [...modelIds].map(stripNamespace);
    while (queue.length) {
      const id = queue.shift()!;
      if (this.models.has(id)) continue;
      const model = await this.readJson<RawBlockModel>(`models/${id}.json`);
      if (!model) continue;
      this.models.set(id, model);
      if (model.parent) queue.push(stripNamespace(model.parent));
    }
  }

  getBlockState(name: string): RawBlockState | undefined {
    return this.blockstates.get(stripNamespace(name));
  }
  getModel(id: string): RawBlockModel | undefined {
    return this.models.get(stripNamespace(id));
  }
  modelProvider(): RawModelProvider {
    return { getModel: (id) => this.getModel(id) };
  }

  /** Read an item model JSON (`models/item/<name>.json`) — for item rendering (generated vs block). */
  async readItemModel(name: string): Promise<RawBlockModel | undefined> {
    return this.readJson<RawBlockModel>(`models/item/${stripNamespace(name)}.json`);
  }

  /** Decode sprite ids ("block/oak_planks") into atlas SpriteSources + opacity info. */
  async loadSprites(spriteIds: Iterable<string>): Promise<LoadedSprites> {
    const sprites: SpriteSource[] = [];
    const opacity = new Map<string, SpriteOpacity>();
    await Promise.all(
      [...new Set(spriteIds)].map(async (id) => {
        const png = this.zip.file(`${ROOT}textures/${id}.png`);
        if (!png) return; // missing texture → mesher falls back to white/no-uv
        const img = await decodePng(new Uint8Array(await png.async("arraybuffer")));
        const meta = this.zip.file(`${ROOT}textures/${id}.png.mcmeta`);
        const anim = meta ? parseAnimationMeta(JSON.parse(await meta.async("string"))) : null;
        sprites.push({
          name: id,
          size: img.width,
          rgba: img.rgba,
          frameCount: anim ? Math.max(1, Math.floor(img.height / img.width)) : 1,
          frameTimeTicks: anim?.frameTime ?? 1,
          // Honor an explicit playback order (e.g. lava_still's ping-pong) for a seamless loop.
          frameOrder: anim?.frames,
          interpolate: anim?.interpolate ?? false, // cross-fade frames (magma, prismarine, …)
        });
        opacity.set(id, scanOpacity(img.rgba));
      }),
    );
    return { sprites, opacity };
  }

  /**
   * Decode arbitrary-size textures (entity / block-entity skins) by full path id, e.g.
   * "entity/sheep/sheep". Unlike `loadSprites` these aren't square-tile/animated sprites — they feed the
   * `EntityAtlasManager` (RENDERER_PLAN §18, Phase 4.5b). Missing ids are skipped (logged by the caller).
   */
  async loadRawTextures(ids: Iterable<string>): Promise<{ name: string; width: number; height: number; rgba: Uint8Array }[]> {
    const out: { name: string; width: number; height: number; rgba: Uint8Array }[] = [];
    await Promise.all(
      [...new Set(ids)].map(async (id) => {
        const png = this.zip.file(`${ROOT}textures/${id}.png`);
        if (!png) return;
        const img = await decodePng(new Uint8Array(await png.async("arraybuffer")));
        out.push({ name: id, width: img.width, height: img.height, rgba: img.rgba });
      }),
    );
    return out;
  }

  /** Decode the grass + foliage colormaps for the TintProvider (if present). */
  async loadColormaps(): Promise<{ grass?: Colormap; foliage?: Colormap }> {
    const load = async (name: string): Promise<Colormap | undefined> => {
      const f = this.zip.file(`${ROOT}textures/colormap/${name}.png`);
      if (!f) return undefined;
      const img = await decodePng(new Uint8Array(await f.async("arraybuffer")));
      return { width: img.width, height: img.height, rgba: img.rgba };
    };
    return { grass: await load("grass"), foliage: await load("foliage") };
  }

  private async readJson<T>(path: string): Promise<T | undefined> {
    const f = this.zip.file(ROOT + path);
    return f ? (JSON.parse(await f.async("string")) as T) : undefined;
  }
}

/** Collect model ids referenced by a blockstate's variants + multipart. */
function collectModelIds(bs: RawBlockState, out: Set<string>): void {
  const add = (v: RawVariant | RawVariant[]): void => {
    if (Array.isArray(v)) v.forEach((x) => out.add(x.model));
    else out.add(v.model);
  };
  if (bs.variants) for (const k of Object.keys(bs.variants)) add(bs.variants[k]);
  if (bs.multipart) for (const c of bs.multipart) add(c.apply);
}
