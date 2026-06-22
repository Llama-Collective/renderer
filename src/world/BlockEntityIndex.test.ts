import { describe, expect, it } from "vitest";
import { BlockEntityIndex } from "./BlockEntityIndex";
import { sectionOfBlock } from "./SectionKey";

const be = (x: number, y: number, z: number, type = "minecraft:chest", animating = false) => ({ x, y, z, type, props: {}, animating });

describe("BlockEntityIndex", () => {
  it("keys BEs by their section and returns them via inSection", () => {
    const idx = new BlockEntityIndex();
    const sk = idx.set(be(3, 4, 5)); // section 0,0,0
    expect(sk).toBe(sectionOfBlock(3, 4, 5));
    expect(idx.inSection(sk).map((r) => [r.x, r.y, r.z])).toEqual([[3, 4, 5]]);
    expect(idx.size).toBe(1);
  });

  it("groups multiple BEs in the same section and separates across sections", () => {
    const idx = new BlockEntityIndex();
    idx.set(be(1, 1, 1));
    idx.set(be(15, 15, 15)); // still section 0,0,0
    idx.set(be(20, 0, 0)); // section 1,0,0
    expect(idx.inSection(sectionOfBlock(1, 1, 1)).length).toBe(2);
    expect(idx.inSection(sectionOfBlock(20, 0, 0)).length).toBe(1);
  });

  it("handles negative coordinates (floor-divide sectioning)", () => {
    const idx = new BlockEntityIndex();
    const sk = idx.set(be(-1, -1, -1)); // section -1,-1,-1
    expect(sk).toBe(sectionOfBlock(-1, -1, -1));
    expect(idx.inSection(sk).length).toBe(1);
  });

  it("remove unlinks the BE and prunes the empty section bucket", () => {
    const idx = new BlockEntityIndex();
    idx.set(be(2, 2, 2));
    const sk = idx.remove(2, 2, 2);
    expect(sk).toBe(sectionOfBlock(2, 2, 2));
    expect(idx.inSection(sk!)).toEqual([]);
    expect(idx.occupiedSections()).toEqual([]);
    expect(idx.remove(2, 2, 2)).toBeNull(); // already gone
  });

  it("set replaces an existing BE at the same position", () => {
    const idx = new BlockEntityIndex();
    idx.set(be(0, 0, 0, "minecraft:chest"));
    idx.set(be(0, 0, 0, "minecraft:trapped_chest"));
    expect(idx.size).toBe(1);
    expect(idx.get(0, 0, 0)!.type).toBe("minecraft:trapped_chest");
  });

  it("setAnimating returns the section to re-mesh only when the flag actually changes", () => {
    const idx = new BlockEntityIndex();
    idx.set(be(5, 5, 5, "minecraft:bell", false));
    expect(idx.setAnimating(5, 5, 5, true)).toBe(sectionOfBlock(5, 5, 5));
    expect(idx.get(5, 5, 5)!.animating).toBe(true);
    expect(idx.setAnimating(5, 5, 5, true)).toBeNull(); // unchanged → no re-mesh
    expect(idx.setAnimating(9, 9, 9, true)).toBeNull(); // no BE there
  });
});
