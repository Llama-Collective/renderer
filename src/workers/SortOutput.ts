// Translucent sort output — INDEX DATA ONLY. RENDERER_PLAN.md §12 (TRAP 12.A).
//
// A sort never touches vertex data and never clears geometry (Invariant rule 5). Like
// build outputs, it owns its transferable buffer and must be disposed when stale.

import type { Generation } from "../types";
import type { SectionKey } from "../world/SectionKey";

export interface SortOutput {
  sectionKey: SectionKey;
  generation: Generation;
  /** New translucent index order for the current camera. */
  indexData: ArrayBuffer;
  dispose(): void;
}

/** Wrap an off-thread `sortDone` payload (sectionKey + generation + transferred index bytes) as a SortOutput
 *  the store can stage/commit (TR-1). `dispose` drops the buffer ref so a stale-dropped sort releases it. */
export function makeSortOutput(payload: { sectionKey: SectionKey; generation: Generation; indexData: ArrayBuffer }): SortOutput {
  const out: SortOutput = {
    sectionKey: payload.sectionKey,
    generation: payload.generation,
    indexData: payload.indexData,
    dispose() {
      out.indexData = new ArrayBuffer(0);
    },
  };
  return out;
}
