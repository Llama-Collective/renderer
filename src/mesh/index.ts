// mesh/ — model baking, culling, lighting, meshing, translucent sorting. Data-only.
// No GPU types. Runs (mostly) inside workers. RENDERER_PLAN.md §3, §6–§13, §24.

export { SortType } from "./SortTypes";
export * from "./VertexFormat";
export * from "./Lighting";
export { meshSection } from "./SectionMesher";
export type { BakedModelProvider } from "./SectionMesher";
export { meshFluidCell } from "./FluidMesher";
export { TranslucentCollector, sortIndicesStaticNormal, sortIndicesByDistance, expandQuadOrder } from "./TranslucentCollector";
export type { TQuad, TranslucentCollectResult } from "./TranslucentCollector";
export { topoSortOrder, quadVisibleThrough } from "./TranslucentTopoSort";

// --- model baker (§24) ---
export * from "./model/ModelTypes";
export { resolveModel, resolveFaceTexture } from "./model/ModelResolver";
export type { RawModelProvider } from "./model/ModelResolver";
export { resolveBlockState } from "./model/BlockStateResolver";
export type { BlockProps } from "./model/BlockStateResolver";
export { bakeModelPart, bakeParts } from "./model/FaceBakery";
export type { BakedQuad, Vec2 } from "./model/FaceBakery";
export { computeOcclusion, shouldDrawQuad } from "./model/OcclusionShape";
export type { OcclusionShape, CullableQuad } from "./model/OcclusionShape";
export { opposite } from "../types";
export { classifyLayer, scanOpacity, isOpaqueSprite } from "./model/RenderLayer";
export type { SpriteOpacity } from "./model/RenderLayer";
export { TintProvider, PLAINS } from "./model/TintProvider";
export type { Colormap, Climate } from "./model/TintProvider";
export { ModelBaker } from "./model/BakedBlockModel";
export type { BakedBlockModel, BakedRenderQuad, BlockEntry, BakerDeps } from "./model/BakedBlockModel";
// Entity / block-entity box-model engine (namespaced — `bakeModelPart` here differs from FaceBakery's).
export * as entityModel from "./entity/ModelPart";
export { packVertices } from "./VertexFormat";
