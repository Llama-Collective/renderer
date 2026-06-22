// core/ — graphics backend abstraction + GPU resource management.
// The ONLY module allowed to touch real WebGL/WebGPU objects. RENDERER_PLAN.md §3, §23.

export * from "./GraphicsDevice";
export { WebGL2Device } from "./webgl2";
export { WebGPUDevice } from "./webgpu";
export { BufferArena } from "./BufferArena";
export type { ArenaStats } from "./BufferArena";
export { AtlasManager } from "./AtlasManager";
export type { SpriteSource, SpriteUv } from "./AtlasManager";
export { EntityAtlasManager, packEntityAtlas } from "./EntityAtlasManager";
export type { EntitySprite, Placement, PackResult } from "./EntityAtlasManager";
export { PipelineCache } from "./PipelineCache";
export { PASS_STATE } from "./RenderPassState";
export { FakeGraphicsDevice } from "./testing/FakeGraphicsDevice";
