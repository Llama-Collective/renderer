// App-facing layer: the single SchematicViewer façade + the world model + the scene pipeline.
export { SchematicViewer } from "./SchematicViewer";
export type { RendererSimulationDiff, RendererBlockChange, RendererEntityChange, ApplyDiffOptions } from "./SchematicViewer";
export { WorldModel, loadWorld, AIR, hasBlockEntityData, blockStateKey } from "./WorldModel";
export type { LoadedWorld, BlockEntityCandidate, SetBlockResult } from "./WorldModel";
export { pickWorld, entityBoundsSize } from "./pick";
export type { Ray, OutlineBox, PickInputs } from "./pick";
export * as placement from "./placement";
export * from "./scene";
