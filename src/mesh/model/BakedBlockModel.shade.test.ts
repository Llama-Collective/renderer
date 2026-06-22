// SL-1 — per-block shade-brightness + emissive data, and the occludesAo predicate (AO uses occludesAny,
// NOT isFullOpaqueCube, so a partial block still darkens corners).

import { describe, it, expect } from "vitest";
import { ModelBaker, type BakerDeps, type BlockEntry } from "./BakedBlockModel";
import { occludesAo, isFullOpaqueCube } from "./OcclusionShape";
import { TintProvider } from "./TintProvider";
import { fixtureBlockstates, fixtureProvider } from "./__fixtures__/load";

function baker(extra?: Partial<BakerDeps>): ModelBaker {
  return new ModelBaker({
    blockstate: (name) => fixtureBlockstates[name],
    models: fixtureProvider,
    uvFor: () => ({ u: 0, v: 0, width: 1, height: 1 }),
    opacityOf: (sprite) => (sprite.includes("glass") ? { hasTransparent: false, hasTranslucent: true } : { hasTransparent: false, hasTranslucent: false }),
    tint: new TintProvider(),
    ...extra,
  });
}

const STONE: BlockEntry = { name: "stone", props: {} };
const GLASS: BlockEntry = { name: "glass", props: {} };

describe("BakedBlockModel shade-brightness + emissive (SL-1)", () => {
  it("defaults: shadeBrightness 1.0, not emissive", () => {
    const m = baker().bake(STONE);
    expect(m.shadeBrightness).toBe(1.0);
    expect(m.emissive).toBe(false);
  });

  it("honors shadeBrightnessOf / emissiveOf when provided", () => {
    const m = baker({ shadeBrightnessOf: (e) => (e.name === "stone" ? 0.8 : 1), emissiveOf: (e) => e.name === "stone" }).bake(STONE);
    expect(m.shadeBrightness).toBe(0.8);
    expect(m.emissive).toBe(true);
  });

  it("occludesAo follows occludesAny (NOT isFullOpaqueCube): stone occludes, glass does not", () => {
    const stone = baker().bake(STONE);
    const glass = baker().bake(GLASS);
    expect(occludesAo(stone.occlusion)).toBe(true); // a solid opaque cube darkens AO
    expect(isFullOpaqueCube(stone.occlusion)).toBe(true);
    expect(occludesAo(glass.occlusion)).toBe(false); // translucent glass never darkens AO
  });
});
