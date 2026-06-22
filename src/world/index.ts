// world/ — section store, dirty tracking, generations, snapshots.
// No GPU types here (holds logical AllocationIds only). RENDERER_PLAN.md §3, §4, §5, §6.

export * from "./SectionKey";
export * from "./RenderSection";
export type { SectionUploader, CommittedSection } from "./SectionUploader";
export { SectionStore } from "./SectionStore";
export type { AcceptResult } from "./SectionStore";
export { DirtyTracking } from "./DirtyTracking";
export type { BlockEdit } from "./DirtyTracking";
export { SnapshotSource, APRON, snapshotIndex, snapshotStride } from "./SnapshotSource";
export type { SectionSnapshot, BlockSource } from "./SnapshotSource";
export { BlockEntityIndex } from "./BlockEntityIndex";
export type { BlockEntityRecord } from "./BlockEntityIndex";
