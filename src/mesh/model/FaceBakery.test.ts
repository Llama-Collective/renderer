// Bake math + element→quad geometry, against synthetic + real (pack.zip) models. §24.5, §22.

import { describe, it, expect } from "vitest";
import { bakeModelPart, bakeParts, type BakedQuad } from "./FaceBakery";
import { resolveModel } from "./ModelResolver";
import { resolveBlockState } from "./BlockStateResolver";
import { fixtureBlockstates, fixtureProvider } from "./__fixtures__/load";
import { blockstateMatrix, elementMatrix } from "./bakeMath";
import { Direction } from "../../types";
import type { ResolvedModel, VariantPart } from "./ModelTypes";

const ID: VariantPart = { model: "", x: 0, y: 0, z: 0, uvlock: false };

function fullCube(uvlock = false, x = 0, y = 0): { model: ResolvedModel; part: VariantPart } {
  const f = (cf: string) => ({ texture: "#all", cullface: cf });
  return {
    model: {
      ambientocclusion: true,
      textures: { all: { sprite: "block/stone", forceTranslucent: false } },
      elements: [{ from: [0, 0, 0], to: [16, 16, 16], faces: { down: f("down"), up: f("up"), north: f("north"), south: f("south"), west: f("west"), east: f("east") } }],
    },
    part: { ...ID, x, y, uvlock },
  };
}

const allCoordsIn01 = (q: BakedQuad) => q.positions.every((p) => p.every((c) => Math.abs(c) < 1e-9 || Math.abs(c - 1) < 1e-9));

describe("bakeMath", () => {
  it("elementMatrix applies the 1/cos rescale (45° → ×√2 on perpendicular axes)", () => {
    const m = elementMatrix("y", 45, true);
    // r·diag(√2,1,√2) ≈ [1,0,1, 0,1,0, -1,0,1]
    expect(m.map((v) => Math.round(v))).toEqual([1, 0, 1, 0, 1, 0, -1, 0, 1]);
  });
  it("blockstate y=90 is an exact integer rotation (vanilla NEGATES → R_y(-90))", () => {
    // BlockModelRotation builds the matrix from -y, so blockstate y=90 is R_y(-90), not R_y(+90). This is
    // what makes directional blocks face the correct cardinal (east↔west not swapped). §24.5.
    expect(blockstateMatrix(0, 90, 0).map((v) => Math.round(v * 1000) / 1000)).toEqual([0, 0, -1, 0, 1, 0, 1, 0, 0]);
  });
});

describe("FaceBakery — full cube", () => {
  it("emits 6 axis-aligned quads with distinct normals and {0,1} corners", () => {
    const { model, part } = fullCube();
    const quads = bakeModelPart(model, part);
    expect(quads).toHaveLength(6);
    expect(new Set(quads.map((q) => q.normal)).size).toBe(6);
    expect(quads.every(allCoordsIn01)).toBe(true);
    expect(quads.every((q) => q.cullface === q.normal)).toBe(true); // full cube: cullface == facing
  });

  it("up face gets the default full-sprite UV", () => {
    const up = bakeModelPart(...Object.values(fullCube()) as [ResolvedModel, VariantPart]).find((q) => q.normal === Direction.Up)!;
    expect(up.uvs).toEqual([[0, 0], [0, 1], [1, 1], [1, 0]]);
  });

  it("blockstate x=90 permutes facings but keeps unit-cube corners", () => {
    const { model, part } = fullCube(false, 90, 0);
    const quads = bakeModelPart(model, part);
    expect(new Set(quads.map((q) => q.normal)).size).toBe(6);
    expect(quads.every(allCoordsIn01)).toBe(true);
  });

  it("uvlock rotates the up-face UV by vanilla's INVERSE face transform under a y rotation (AUDIT H1)", () => {
    const locked = bakeModelPart(...Object.values(fullCube(true, 0, 90)) as [ResolvedModel, VariantPart]).find((q) => q.normal === Direction.Up)!;
    const plain = bakeModelPart(...Object.values(fullCube(false, 0, 90)) as [ResolvedModel, VariantPart]).find((q) => q.normal === Direction.Up)!;
    // Plain (no uvlock) keeps the model UV regardless of blockstate rotation.
    expect(plain.uvs).toEqual([[0, 0], [0, 1], [1, 1], [1, 0]]);
    // uvlock applies vanilla `inverseFaceTransformation` (FaceBakery.bakeVertex) using the same blockRot as
    // the geometry. With vanilla's NEGATED rotation (R_y(-90)) the world-fixed up-face UV is this order;
    // the FORWARD-transform bug would instead give [[0,1],[1,1],[1,0],[0,0]] (its transpose) — guarded here.
    expect(locked.uvs).toEqual([[1, 0], [0, 0], [0, 1], [1, 1]]);
  });
});

describe("FaceBakery — flat (zero-thickness) elements", () => {
  const flatModel = (faces: Record<string, { texture: string }>): { model: ResolvedModel; part: VariantPart } => ({
    model: {
      ambientocclusion: true,
      textures: { fire: { sprite: "block/fire_0", forceTranslucent: false } },
      // A flat-Z plane (campfire-fire shape): from.z == to.z.
      elements: [{ from: [0.8, 1, 8], to: [15.2, 17, 8], faces }],
    },
    part: ID,
  });

  it("a flat plane with BOTH opposite faces keeps both (back-face culling shows one per side, vanilla)", () => {
    // Coplanar opposite-wound quads don't z-fight under back-face culling — exactly one is camera-facing
    // from any viewpoint — so both are kept (a cross/campfire plane is visible from both sides). The old
    // CullMode.None renderer dropped the negative face; with culling that would make it one-sided.
    const quads = bakeModelPart(...Object.values(flatModel({ north: { texture: "#fire" }, south: { texture: "#fire" } })) as [ResolvedModel, VariantPart]);
    expect(quads.length).toBe(2);
    expect(new Set(quads.map((q) => q.normal))).toEqual(new Set([Direction.North, Direction.South]));
  });

  it("a flat plane with only the NEGATIVE face still emits it (one-sided decorations survive)", () => {
    const quads = bakeModelPart(...Object.values(flatModel({ north: { texture: "#fire" } })) as [ResolvedModel, VariantPart]);
    expect(quads.length).toBe(1);
    expect(quads[0].normal).toBe(Direction.North);
  });
});

describe("FaceBakery — real models", () => {
  it("oak_stairs bakes the slab+step with a mid-block (y=0.5) edge and cullfaces", () => {
    const parts = resolveBlockState(fixtureBlockstates.oak_stairs, { facing: "east", half: "bottom", shape: "straight" });
    const quads = bakeParts(parts, fixtureProvider);
    expect(quads.length).toBeGreaterThanOrEqual(10); // 6 (slab) + step faces
    const hasMidEdge = quads.some((q) => q.positions.some((p) => Math.abs(p[1] - 0.5) < 1e-6));
    expect(hasMidEdge).toBe(true);
    expect(quads.some((q) => q.cullface === Direction.Down)).toBe(true);
  });

  it("oak_log axis=x stays a 6-face unit cube after the x90/y90 rotation", () => {
    const quads = bakeParts(resolveBlockState(fixtureBlockstates.oak_log, { axis: "x" }), fixtureProvider);
    expect(quads).toHaveLength(6);
    expect(new Set(quads.map((q) => q.normal)).size).toBe(6);
    expect(quads.every(allCoordsIn01)).toBe(true);
  });

  it("grass_block tints the top + 4 side-overlay faces (tintindex 0)", () => {
    const quads = bakeParts(resolveBlockState(fixtureBlockstates.grass_block, { snowy: "false" }), fixtureProvider);
    expect(quads).toHaveLength(10); // cube(6) + overlay(4)
    expect(quads.filter((q) => q.tintindex === 0)).toHaveLength(5); // up + 4 overlay sides
  });

  it("glass_pane_side carries forceTranslucent through to its quads", () => {
    const quads = bakeModelPart(resolveModel("block/glass_pane_side", fixtureProvider), ID);
    expect(quads.length).toBeGreaterThan(0);
    expect(quads.some((q) => q.forceTranslucent)).toBe(true);
  });
});
