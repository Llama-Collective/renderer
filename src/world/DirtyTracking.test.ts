// Border-edit dirtying — GATE 4.1(d) / TRAP 6.C. RENDERER_PLAN.md §6.

import { describe, it, expect } from "vitest";
import { DirtyTracking } from "./DirtyTracking";
import { SectionStore } from "./SectionStore";
import { sectionKey } from "./SectionKey";
import { DirtyReason } from "../types";
import type { SectionUploader, CommittedSection } from "./SectionUploader";

// SectionStore needs an uploader; these tests never commit, so a no-op uploader is fine.
const noopUploader: SectionUploader = {
  upload: () => null,
  uploadSort: (prev: CommittedSection) => prev,
  free: () => {},
};

describe("DirtyTracking border edits", () => {
  it("a block on a section face affects both adjacent sections", () => {
    const dt = new DirtyTracking();
    // x=15 is the +X face of section (0,0,0); the neighbor across it is (1,0,0).
    const keys = dt.sectionsAffectedBy({ x: 15, y: 5, z: 5, reason: DirtyReason.Edit });
    expect(keys).toContain(sectionKey(0, 0, 0));
    expect(keys).toContain(sectionKey(1, 0, 0));
  });

  it("an interior block affects only its own section", () => {
    const dt = new DirtyTracking();
    const keys = dt.sectionsAffectedBy({ x: 5, y: 5, z: 5, reason: DirtyReason.Edit });
    expect(keys).toEqual([sectionKey(0, 0, 0)]);
  });

  it("markDirty bumps the generation of every affected section (both sides)", () => {
    const store = new SectionStore(noopUploader);
    const dt = new DirtyTracking();
    const keys = dt.sectionsAffectedBy({ x: 15, y: 5, z: 5, reason: DirtyReason.Edit });
    keys.forEach((k) => store.markDirty(k, DirtyReason.Edit));

    expect(store.get(sectionKey(0, 0, 0))!.generation).toBe(1);
    expect(store.get(sectionKey(1, 0, 0))!.generation).toBe(1);
  });
});
