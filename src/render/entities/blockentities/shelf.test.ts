// SHELF surface items: filled slots render at their OWN index (empty slots are kept as gaps, not
// collapsed) — the renderer reads the sim's comma-joined, slot-indexed `items` prop.

import { describe, it, expect } from "vitest";
import { SHELF, CAMPFIRE } from "./special";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";

const rec = (items: string): BlockEntityRecord => ({
  x: 0, y: 0, z: 0, type: "minecraft:oak_shelf", props: { facing: "south", items }, animating: false,
});

// facing "south" → yaw 0 → item draw's world x (model[12]) = 0.5 + (slot - 1) * 0.3125 - 0.125.
const slotX = (slot: number): number => 0.5 + (slot - 1) * 0.3125 - 0.125;

describe("SHELF special items", () => {
  it("keeps each item at its true slot index (slot 1 empty)", () => {
    const draws = SHELF.special!(rec("minecraft:diamond,,minecraft:apple"), 0);
    expect(draws).toHaveLength(2);
    const x = Object.fromEntries(draws.map((d) => [d.kind === "item" ? d.item : "?", d.kind === "item" ? d.model[12] : NaN]));
    expect(x["minecraft:diamond"]).toBeCloseTo(slotX(0), 5);
    expect(x["minecraft:apple"]).toBeCloseTo(slotX(2), 5); // slot 2, NOT collapsed to slot 1 (slotX(1)=0.375)
  });

  it("renders all three slots when full", () => {
    expect(SHELF.special!(rec("minecraft:a,minecraft:b,minecraft:c"), 0)).toHaveLength(3);
  });

  it("renders nothing for an empty/absent shelf", () => {
    expect(SHELF.special!(rec(",,"), 0)).toHaveLength(0);
    expect(SHELF.special!(rec(""), 0)).toHaveLength(0);
    expect(SHELF.special!({ x: 0, y: 0, z: 0, type: "minecraft:oak_shelf", props: { facing: "south" }, animating: false }, 0)).toHaveLength(0);
  });
});

describe("CAMPFIRE special items", () => {
  const cfRec = (items: string): BlockEntityRecord => ({
    x: 0, y: 0, z: 0, type: "minecraft:campfire", props: { facing: "north", items }, animating: false,
  });

  it("renders only filled cooking slots (empties skipped, 4 slots max)", () => {
    const draws = CAMPFIRE.special!(cfRec("minecraft:porkchop,,minecraft:potato,"), 0);
    expect(draws).toHaveLength(2);
    expect(draws.map((d) => (d.kind === "item" ? d.item : "?"))).toEqual(["minecraft:porkchop", "minecraft:potato"]);
  });

  it("renders nothing for an empty campfire", () => {
    expect(CAMPFIRE.special!(cfRec(",,,"), 0)).toHaveLength(0);
    expect(CAMPFIRE.special!(cfRec(""), 0)).toHaveLength(0);
  });
});
