// Occlusion derivation + per-quad cull + render-layer classification. §24.6–§24.7, §22.

import { describe, it, expect } from "vitest";
import { computeOcclusion, shouldDrawQuad } from "./OcclusionShape";
import { classifyLayer, scanOpacity, isOpaqueSprite } from "./RenderLayer";
import { bakeModelPart, type BakedQuad } from "./FaceBakery";
import { Direction, opposite, TerrainPass } from "../../types";
import type { ResolvedModel, VariantPart } from "./ModelTypes";

const ID: VariantPart = { model: "", x: 0, y: 0, z: 0, uvlock: false };
const OPAQUE = () => true;

function box(toY: number, sprite = "block/stone"): BakedQuad[] {
  const f = (cf: string) => ({ texture: "#all", cullface: cf });
  const model: ResolvedModel = {
    ambientocclusion: true,
    textures: { all: { sprite, forceTranslucent: false } },
    elements: [{ from: [0, 0, 0], to: [16, toY, 16], faces: { down: f("down"), up: f("up"), north: f("north"), south: f("south"), west: f("west"), east: f("east") } }],
  };
  return bakeModelPart(model, ID);
}

const cube = () => box(16);
const slab = () => box(8); // bottom slab

describe("computeOcclusion", () => {
  it("a full opaque cube occludes all six faces", () => {
    const occ = computeOcclusion(cube(), OPAQUE);
    for (const d of [Direction.Down, Direction.Up, Direction.North, Direction.South, Direction.West, Direction.East]) {
      expect(occ.byFace[d].full).toBe(true);
    }
  });

  it("a bottom slab occludes only its Down face fully", () => {
    const occ = computeOcclusion(slab(), OPAQUE);
    expect(occ.byFace[Direction.Down].full).toBe(true);
    expect(occ.byFace[Direction.Up].full).toBe(false); // top is at y=0.5, not flush
    expect(occ.byFace[Direction.North].full).toBe(false); // half-height side
    expect(occ.byFace[Direction.North].rects.length).toBe(1); // but a partial flush rect exists
  });

  it("glass (non-opaque) occludes nothing", () => {
    const occ = computeOcclusion(box(16, "block/glass"), () => false);
    expect(occ.occludesAny).toBe(false);
  });
});

describe("shouldDrawQuad", () => {
  const upQuad = cube().find((q) => q.cullface === Direction.Up)!;
  const cubeOcc = computeOcclusion(cube(), OPAQUE);

  it("culls a cullface quad against a full-cube neighbor", () => {
    expect(shouldDrawQuad(upQuad, cubeOcc, null, null)).toBe(false);
  });
  it("draws against air (no neighbor)", () => {
    expect(shouldDrawQuad(upQuad, null, null, null)).toBe(true);
  });
  it("draws against a non-occluding (glass) neighbor", () => {
    const glassOcc = computeOcclusion(box(16, "block/glass"), () => false);
    expect(shouldDrawQuad(upQuad, glassOcc, null, null)).toBe(true);
  });
  it("culls between same skip-render group (clear glass), draws between different groups", () => {
    const glassOcc = computeOcclusion(box(16, "block/glass"), () => false);
    expect(shouldDrawQuad(upQuad, glassOcc, "glass", "glass")).toBe(false);
    expect(shouldDrawQuad(upQuad, glassOcc, "glass", "white_glass")).toBe(true);
  });
  it("a quad with no cullface always draws", () => {
    const free = { ...upQuad, cullface: null };
    expect(shouldDrawQuad(free, cubeOcc, null, null)).toBe(true);
  });
  it("opposite() pairs faces", () => {
    expect(opposite(Direction.Up)).toBe(Direction.Down);
    expect(opposite(Direction.East)).toBe(Direction.West);
  });
});

describe("RenderLayer classification", () => {
  it("maps opacity → pass", () => {
    expect(classifyLayer(undefined, false)).toBe(TerrainPass.Solid);
    expect(classifyLayer({ hasTransparent: true, hasTranslucent: false }, false)).toBe(TerrainPass.Cutout);
    expect(classifyLayer({ hasTransparent: false, hasTranslucent: true }, false)).toBe(TerrainPass.Translucent);
    expect(classifyLayer(undefined, true)).toBe(TerrainPass.Translucent); // force_translucent
  });

  it("scanOpacity detects holes vs partial alpha", () => {
    const opaque = new Uint8Array([10, 20, 30, 255, 40, 50, 60, 255]);
    const cutout = new Uint8Array([10, 20, 30, 255, 0, 0, 0, 0]);
    const translucent = new Uint8Array([10, 20, 30, 128]);
    expect(isOpaqueSprite(scanOpacity(opaque))).toBe(true);
    expect(scanOpacity(cutout)).toMatchObject({ hasTransparent: true });
    expect(scanOpacity(translucent)).toMatchObject({ hasTranslucent: true });
  });
});
