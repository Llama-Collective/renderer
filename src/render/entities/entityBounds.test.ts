// Entity bounding-box dimensions + axis-aligned construction. Mirrors the sim's EntityDimensions and
// guards the "never tilted with the model" requirement (carts on slopes keep an axis-aligned AABB).

import { describe, it, expect } from "vitest";
import { entityAabbSize, entityAabb } from "./entityBounds";

describe("entityAabbSize", () => {
  it("matches the simulation EntityDimensions per type", () => {
    expect(entityAabbSize("minecraft:item")).toEqual({ width: 0.25, height: 0.25 });
    expect(entityAabbSize("minecraft:tnt")).toEqual({ width: 0.98, height: 0.98 });
    expect(entityAabbSize("minecraft:falling_block")).toEqual({ width: 0.98, height: 0.98 });
    expect(entityAabbSize("minecraft:sheep")).toEqual({ width: 0.9, height: 1.3 });
    expect(entityAabbSize("minecraft:pig")).toEqual({ width: 0.9, height: 0.9 });
    expect(entityAabbSize("minecraft:cow")).toEqual({ width: 0.9, height: 1.4 });
    expect(entityAabbSize("minecraft:minecart")).toEqual({ width: 0.98, height: 0.7 });
    expect(entityAabbSize("minecraft:arrow")).toEqual({ width: 0.5, height: 0.5 });
  });

  it("treats the renderer's primed_tnt alias as tnt (sim type is minecraft:tnt)", () => {
    expect(entityAabbSize("minecraft:primed_tnt")).toEqual({ width: 0.98, height: 0.98 });
    expect(entityAabbSize("minecraft:tnt")).toEqual({ width: 0.98, height: 0.98 });
  });

  it("strips the minecraft: prefix and matches variant suffixes", () => {
    expect(entityAabbSize("minecart")).toEqual({ width: 0.98, height: 0.7 });
    expect(entityAabbSize("minecraft:chest_minecart")).toEqual({ width: 0.98, height: 0.7 });
    expect(entityAabbSize("minecraft:tnt_minecart")).toEqual({ width: 0.98, height: 0.7 });
    expect(entityAabbSize("minecraft:oak_boat")).toEqual({ width: 1.375, height: 0.5625 });
    expect(entityAabbSize("minecraft:oak_chest_boat")).toEqual({ width: 1.375, height: 0.5625 });
    expect(entityAabbSize("minecraft:bamboo_raft")).toEqual({ width: 1.375, height: 0.5625 });
    expect(entityAabbSize("minecraft:bamboo_chest_raft")).toEqual({ width: 1.375, height: 0.5625 });
    expect(entityAabbSize("minecraft:spectral_arrow")).toEqual({ width: 0.5, height: 0.5 });
  });

  it("falls back to the generic mob footprint for unknown types", () => {
    expect(entityAabbSize("minecraft:zombie")).toEqual({ width: 0.6, height: 1.8 });
  });
});

describe("entityAabb", () => {
  it("centres x/z on the position with the feet at y (sim makeBoundingBox convention)", () => {
    const { min, max } = entityAabb("minecraft:sheep", [10, 64, -3]);
    // width 0.9 → ±0.45 around (x,z); y is the feet, top = y + 1.3.
    expect(min).toEqual([10 - 0.45, 64, -3 - 0.45]);
    expect(max).toEqual([10 + 0.45, 64 + 1.3, -3 + 0.45]);
  });

  it("is axis-aligned and rotation-independent — a minecart box never tilts (takes no rotation input)", () => {
    // Same position, regardless of any yRot/xRot the sim might report: the AABB is purely a function of
    // type + feet position, so a cart on a slope keeps an upright, axis-aligned box.
    const a = entityAabb("minecraft:minecart", [0.5, 70, 0.5]);
    expect(a.min).toEqual([0.5 - 0.49, 70, 0.5 - 0.49]); // 0.98/2
    expect(a.max).toEqual([0.5 + 0.49, 70 + 0.7, 0.5 + 0.49]);
    // The box edges are world-axis-aligned: min < max on every axis, and the footprint is square.
    expect(a.max[0] - a.min[0]).toBeCloseTo(0.98);
    expect(a.max[2] - a.min[2]).toBeCloseTo(0.98);
    expect(a.max[1] - a.min[1]).toBeCloseTo(0.7);
  });

  // ItemFrame.createBoundingBox (ItemFrame.java): centred on the position (NOT feet), thin (0.0625) on the
  // facing axis, 0.75 square on the other two (1.0 for a framed map). Oriented by `facing`, never tilted.
  it("item frame: centred + thin on the facing axis (0.75 square otherwise)", () => {
    const south = entityAabb("minecraft:item_frame", [5.5, 5.5, 5], { facing: "south" });
    expect(south.max[2] - south.min[2]).toBeCloseTo(0.0625); // thin on Z (north/south facing)
    expect(south.max[0] - south.min[0]).toBeCloseTo(0.75);
    expect(south.max[1] - south.min[1]).toBeCloseTo(0.75);
    expect(south.min[1]).toBeCloseTo(5.5 - 0.375); // centred on y, not feet-based

    const east = entityAabb("minecraft:glow_item_frame", [5, 5.5, 5.5], { facing: "east" });
    expect(east.max[0] - east.min[0]).toBeCloseTo(0.0625); // thin on X (east/west facing)
    expect(east.max[1] - east.min[1]).toBeCloseTo(0.75);

    const up = entityAabb("minecraft:item_frame", [5.5, 5, 5.5], { facing: "up" });
    expect(up.max[1] - up.min[1]).toBeCloseTo(0.0625); // thin on Y (up/down facing)
    expect(up.max[0] - up.min[0]).toBeCloseTo(0.75);
  });

  it("item frame: a framed map widens the square faces to 1.0", () => {
    const map = entityAabb("minecraft:item_frame", [5.5, 5.5, 5], { facing: "south", map: "true" });
    expect(map.max[0] - map.min[0]).toBeCloseTo(1.0);
    expect(map.max[1] - map.min[1]).toBeCloseTo(1.0);
    expect(map.max[2] - map.min[2]).toBeCloseTo(0.0625); // still thin on the facing axis
  });
});
