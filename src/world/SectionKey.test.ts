import { describe, it, expect } from "vitest";
import { sectionKey, parseSectionKey, sectionOfBlock } from "./SectionKey";

describe("SectionKey", () => {
  it("round-trips section coordinates incl. negatives", () => {
    expect(parseSectionKey(sectionKey(1, -2, 3))).toEqual([1, -2, 3]);
    expect(parseSectionKey(sectionKey(0, 0, 0))).toEqual([0, 0, 0]);
  });

  it("floor-divides block coords by 16, correctly for negatives", () => {
    expect(sectionOfBlock(0, 0, 0)).toBe(sectionKey(0, 0, 0));
    expect(sectionOfBlock(15, 15, 15)).toBe(sectionKey(0, 0, 0));
    expect(sectionOfBlock(16, 0, 0)).toBe(sectionKey(1, 0, 0));
    expect(sectionOfBlock(-1, 0, 0)).toBe(sectionKey(-1, 0, 0));
    expect(sectionOfBlock(-16, 0, 0)).toBe(sectionKey(-1, 0, 0));
    expect(sectionOfBlock(-17, 0, 0)).toBe(sectionKey(-2, 0, 0));
  });
});
