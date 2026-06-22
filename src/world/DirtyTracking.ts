// Translates world/simulation diffs into dirty sections + border neighbors.
// RENDERER_PLAN.md §6 (TRAP 6.C), §16.

import type { DirtyReason } from "../types";
import { sectionOfBlock } from "./SectionKey";
import type { SectionKey } from "./SectionKey";

export interface BlockEdit {
  x: number;
  y: number;
  z: number;
  reason: DirtyReason;
}

function localCoord(v: number): number {
  return ((v % 16) + 16) % 16;
}

export class DirtyTracking {
  /**
   * Every section that must rebuild for a block edit: the containing section AND any
   * section sharing a face the edit touches (TRAP 6.C). A block on a section boundary
   * dirties the neighbor across that face, per axis — so a corner edit dirties up to three
   * neighbors. Border edits MUST dirty both sides or cross-section culling goes stale.
   */
  sectionsAffectedBy(edit: BlockEdit): SectionKey[] {
    const { x, y, z } = edit;
    const set = new Set<SectionKey>();
    set.add(sectionOfBlock(x, y, z));

    if (localCoord(x) === 0) set.add(sectionOfBlock(x - 1, y, z));
    if (localCoord(x) === 15) set.add(sectionOfBlock(x + 1, y, z));
    if (localCoord(y) === 0) set.add(sectionOfBlock(x, y - 1, z));
    if (localCoord(y) === 15) set.add(sectionOfBlock(x, y + 1, z));
    if (localCoord(z) === 0) set.add(sectionOfBlock(x, y, z - 1));
    if (localCoord(z) === 15) set.add(sectionOfBlock(x, y, z + 1));

    return [...set];
  }
}
