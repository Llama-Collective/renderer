// resources/ — load a Minecraft resource-pack zip (JSZip) with an IndexedDB cache, decode
// textures/mcmeta, and feed the GPU-free baker. Browser-only I/O. RENDERER_PLAN.md §24.2, §24.9.

export { ResourcePack } from "./ResourcePack";
export type { LoadedSprites } from "./ResourcePack";
export { decodePng } from "./TextureDecoder";
export type { DecodedImage } from "./TextureDecoder";
export { parseAnimationMeta } from "./textureMeta";
export type { AnimationMeta } from "./textureMeta";
export { cacheGet, cachePut } from "./AssetCache";
