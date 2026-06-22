// render/ — terrain passes, upload/commit, transients, effects, entities, debug.
// Touches the GraphicsDevice (the only module besides core/ that may). RENDERER_PLAN §3.

export { GpuSectionUploader } from "./GpuSectionUploader";
export type { GpuCommittedSection } from "./GpuSectionUploader";
export { TerrainRenderer } from "./TerrainRenderer";
export type { SectionDraw } from "./TerrainRenderer";
export { TERRAIN_WGSL } from "./terrainShader";
export { UploadScheduler, DEFAULT_UPLOAD_BUDGET } from "./UploadScheduler";
export type { UploadBudget, UploadStats, UploadSchedulerOptions } from "./UploadScheduler";
export { PistonTransients } from "./transient/PistonTransients";
export type { MovingPistonInput } from "./transient/PistonTransients";
export { ExplosionEffects } from "./effects/ExplosionEffects";
export { EntityRenderer } from "./entities/EntityRenderer";
export type { EntityDraw } from "./entities/EntityRenderer";
export { EntityScene } from "./entities/EntityScene";
export type { EntitySnapshot, RenderEntity } from "./entities/EntityScene";
export { EntityModelFactory } from "./entities/EntityModelFactory";
export type { BoxModelSource, BoxModelLayer, EntityFactoryDeps } from "./entities/EntityModelFactory";
export { createBoxModelSource } from "./entities/boxModels";
export type { AtlasRects, EntityModelDef } from "./entities/boxModels";
export { BlockEntitySceneBaker } from "./entities/BlockEntitySceneBaker";
export { bakeBlockEntity, splitSectionBEs, hasBlockEntityModel, blockEntitySpecials, allBETextures, registeredBETypes, getBEDef, hasBEDef, boxBE } from "./entities/blockentities";
export type { BlockEntityDef, BoxBEConfig, BEBakeContext, BESpecialDraw } from "./entities/blockentities";
export { EntityWorld } from "./entities/EntityWorld";
export type { EntityWorldDeps } from "./entities/EntityWorld";
export { ENTITY_WGSL } from "./entities/entityShader";
export { DebugOverlays } from "./debug/DebugOverlays";
