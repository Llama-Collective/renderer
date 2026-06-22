// Skip-render group derivation + its wiring through the baker. §8/§24.6, TRAP 8.A/8.B. AUDIT H2.

import { describe, it, expect } from "vitest";
import { deriveSkipGroup } from "./SkipGroup";
import { ModelBaker, type BakerDeps } from "./BakedBlockModel";
import { fixtureBlockstates, fixtureProvider } from "./__fixtures__/load";
import { TintProvider } from "./TintProvider";
import { shouldDrawQuad } from "./OcclusionShape";
import { Direction } from "../../types";

describe("deriveSkipGroup", () => {
  it("groups the HalfTransparentBlock + IronBars families by block identity", () => {
    const exact: [string, string][] = [
      ["glass", "glass"], ["tinted_glass", "tinted_glass"],
      ["ice", "ice"], ["frosted_ice", "frosted_ice"], ["blue_ice", "blue_ice"],
      ["honey_block", "honey_block"], ["slime_block", "slime_block"],
      ["glass_pane", "glass_pane"], ["iron_bars", "iron_bars"],
      ["white_stained_glass", "white_stained_glass"], ["red_stained_glass_pane", "red_stained_glass_pane"],
      ["copper_bars", "copper_bars"], ["waxed_oxidized_copper_bars", "waxed_oxidized_copper_bars"],
    ];
    for (const [name, group] of exact) expect(deriveSkipGroup(name), name).toBe(group);
    expect(deriveSkipGroup("minecraft:glass")).toBe("glass"); // namespace stripped
  });

  it("gives DIFFERENT blocks DIFFERENT groups so their boundary renders (TRAP 8.B)", () => {
    expect(deriveSkipGroup("white_stained_glass")).not.toBe(deriveSkipGroup("red_stained_glass"));
    // a stained-glass BLOCK vs the same-colour PANE are different blocks → different groups (vanilla:
    // HalfTransparentBlock vs IronBarsBlock never skip against each other).
    expect(deriveSkipGroup("white_stained_glass")).not.toBe(deriveSkipGroup("white_stained_glass_pane"));
  });

  it("returns null for opaque / ordinary blocks (occlusion handles those)", () => {
    for (const n of ["packed_ice", "stone", "dirt", "oak_planks", "oak_leaves", "white_concrete", "glowstone"]) {
      expect(deriveSkipGroup(n), n).toBeNull();
    }
  });
});

describe("ModelBaker wires the derived skip group (AUDIT H2)", () => {
  const deps: BakerDeps = {
    blockstate: (name) => fixtureBlockstates[name],
    models: fixtureProvider,
    uvFor: () => ({ u: 0, v: 0, width: 1, height: 1 }),
    opacityOf: (sprite) => (sprite.includes("glass") ? { hasTransparent: false, hasTranslucent: true } : { hasTransparent: false, hasTranslucent: false }),
    tint: new TintProvider(),
  };

  it("real glass / stained-glass / pane bake with a derived group and NO explicit cullGroup", () => {
    const baker = new ModelBaker(deps);
    expect(baker.bake({ name: "glass", props: {} }).skipGroup).toBe("glass");
    expect(baker.bake({ name: "white_stained_glass", props: {} }).skipGroup).toBe("white_stained_glass");
    expect(baker.bake({ name: "glass_pane", props: {} }).skipGroup).toBe("glass_pane");
    expect(baker.bake({ name: "stone", props: {} }).skipGroup).toBeNull();
  });

  it("adjacent same glass cull their shared face; glass↔stained-glass do not (the H2 fix end-to-end)", () => {
    const baker = new ModelBaker(deps);
    const glass = baker.bake({ name: "glass", props: {} });
    const stained = baker.bake({ name: "white_stained_glass", props: {} });
    const eastQuad = glass.quads.find((q) => q.cullface === Direction.East)!;
    expect(eastQuad).toBeDefined();
    // glass ↔ glass: shared face culled by the derived group (was rendered before H2).
    expect(shouldDrawQuad(eastQuad, glass.occlusion, glass.skipGroup, glass.skipGroup)).toBe(false);
    // glass ↔ different-colour stained glass: boundary rendered (TRAP 8.B).
    expect(shouldDrawQuad(eastQuad, stained.occlusion, glass.skipGroup, stained.skipGroup)).toBe(true);
  });

  it("an explicit cullGroup still overrides the derivation", () => {
    expect(new ModelBaker(deps).bake({ name: "glass", props: {}, cullGroup: "custom" }).skipGroup).toBe("custom");
    expect(new ModelBaker(deps).bake({ name: "stone", props: {}, cullGroup: "x" }).skipGroup).toBe("x");
  });
});
