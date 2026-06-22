// Fluid corner heights, flow→UV rotation, and face culling. RENDERER_PLAN.md §13, §22.

import { describe, it, expect } from "vitest";
import { meshFluidCell, FluidType, type FluidAppearance, type FluidQuad, type FluidSampler, type FluidState } from "./FluidMesher";
import { Direction } from "../types";

/** Collect a fluid cell's emitted faces into an array (the tests assert over a `FluidQuad[]`; meshFluidCell
 *  now pushes through a sink — A2). */
function collectFluid(state: FluidState, lx: number, ly: number, lz: number, app: FluidAppearance, s: FluidSampler): FluidQuad[] {
  const out: FluidQuad[] = [];
  meshFluidCell(state, lx, ly, lz, app, s, (positions, atlasUV, normal, colorRGBA) => out.push({ positions, atlasUV, normal, colorRGBA }));
  return out;
}

// Encode fluid state into an id so a plain Record can back the sampler. 0 = air, 1 = solid, 2 = a
// non-occluding non-air block (e.g. glass — renders a fluid bottom over it).
const FLUID_BASE = 1000;
const WATER = (level = 0) => FLUID_BASE + FluidType.Water * 100 + level;
const LAVA = (level = 0) => FLUID_BASE + FluidType.Lava * 100 + level;
const SOLID = 1;
const GLASS = 2;

function sampler(world: Record<string, number>, occluders: Set<string> = new Set()): FluidSampler {
  return {
    idAt: (x, y, z) => world[`${x},${y},${z}`] ?? 0,
    fluidOf: (id) => (id >= FLUID_BASE ? { type: Math.floor((id - FLUID_BASE) / 100), level: (id - FLUID_BASE) % 100 } : null),
    // A SOLID block fully occludes (vanilla stone); explicit `occluders` model any other full occluder.
    occludesFace: (x, y, z) => occluders.has(`${x},${y},${z}`) || world[`${x},${y},${z}`] === SOLID,
  };
}

const APP: FluidAppearance = { sprite: { u: 0, v: 0, width: 1, height: 1 }, colorRGBA: 0x3f76e4cc, translucent: true, shade: true };
const SOURCE = { type: FluidType.Water, level: 0 };

describe("meshFluidCell", () => {
  it("a lone source (air around + below) emits top + 4 sides + bottom; corners droop toward air", () => {
    const quads = collectFluid(SOURCE, 0, 0, 0, APP, sampler({ "0,0,0": WATER(0) }));
    expect(quads).toHaveLength(6);
    expect(quads[0].normal).toBe(Direction.Up); // top is emitted first
    // Corner height = weighted avg of self (8/9, ×10) with two air neighbors (0, ×1 each) = 8.889/12.
    expect(quads[0].positions[0][1]).toBeCloseTo((8 / 9 * 10) / 12, 5);
  });

  it("a source on solid (occluding) ground drops the bottom face (5 quads)", () => {
    const quads = collectFluid(SOURCE, 0, 1, 0, APP, sampler({ "0,1,0": WATER(0), "0,0,0": SOLID }));
    expect(quads).toHaveLength(5);
    expect(quads.some((q) => q.normal === Direction.Down)).toBe(false);
  });

  it("keeps the bottom face over a non-occluding non-air block, e.g. glass (AUDIT M1)", () => {
    // Water resting on glass/leaves/fences/slabs: the block below isn't air, but doesn't fully occlude
    // upward, so vanilla still renders the fluid underside. The old air-only test would have dropped it.
    const quads = collectFluid(SOURCE, 0, 1, 0, APP, sampler({ "0,1,0": WATER(0), "0,0,0": GLASS }));
    expect(quads).toHaveLength(6);
    expect(quads.some((q) => q.normal === Direction.Down)).toBe(true);
  });

  it("culls the shared side between two same-type fluids (regardless of level)", () => {
    const world = { "0,1,0": WATER(0), "1,1,0": WATER(4), "0,0,0": SOLID, "1,0,0": SOLID };
    const quads = collectFluid(SOURCE, 0, 1, 0, APP, sampler(world));
    // top + N + S + W = 4 (east side culled against the adjacent water). No bottom (solid below).
    expect(quads).toHaveLength(4);
    expect(quads.some((q) => q.normal === Direction.East)).toBe(false);
  });

  it("same fluid stacked above → no top face, full-height (1.0) sides", () => {
    const world = { "0,1,0": WATER(0), "0,2,0": WATER(0), "0,0,0": SOLID };
    const quads = collectFluid(SOURCE, 0, 1, 0, APP, sampler(world));
    expect(quads.some((q) => q.normal === Direction.Up)).toBe(false); // interior — top culled
    expect(quads).toHaveLength(4); // 4 sides, no top, no bottom
    // A side's top corner reaches y = 2 (ly + full height 1).
    expect(Math.max(...quads.flatMap((q) => q.positions.map((p) => p[1])))).toBeCloseTo(2, 6);
  });

  it("downhill neighbor sets the flow direction → top UVs are rotated (not the still layout)", () => {
    // East neighbor is lower water (level 4) → flow points +x. With back-face cull, the same-fluid
    // east side is culled, so quads[0] is the rotated top.
    const world = { "0,1,0": WATER(0), "1,1,0": WATER(4), "0,0,0": SOLID, "1,0,0": SOLID };
    const quads = collectFluid(SOURCE, 0, 1, 0, APP, sampler(world));
    const top = quads[0];
    expect(top.normal).toBe(Direction.Up);
    // angle = atan2(0,1) - π/2 = -π/2 ⇒ s=-0.25, c=0 ⇒ uNW = (0.5+0.25, 0.5-0.25) = (0.75, 0.25).
    expect(top.atlasUV[0][0]).toBeCloseTo(0.75, 5);
    expect(top.atlasUV[0][1]).toBeCloseTo(0.25, 5);
  });

  it("a still pool (no height gradient) keeps the un-rotated top UVs", () => {
    // Four same-level neighbors → no flow → canonical still UVs (NW corner at sprite 0,0).
    const world: Record<string, number> = { "0,1,0": WATER(0) };
    for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) world[`${dx},1,${dz}`] = WATER(0);
    for (const [dx, dz] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) world[`${dx},0,${dz}`] = SOLID;
    const top = collectFluid(SOURCE, 0, 1, 0, APP, sampler(world))[0];
    expect(top.normal).toBe(Direction.Up);
    expect(top.atlasUV[0]).toEqual([0, 0]);
  });

  it("culls fluid faces against full opaque occluders (DefaultFluidRenderer)", () => {
    // Source on solid ground, walled by occluders on all 4 sides + above → only ... nothing visible.
    const world = { "0,1,0": WATER(0), "0,0,0": SOLID };
    const occ = new Set(["1,1,0", "-1,1,0", "0,1,1", "0,1,-1", "0,2,0"]); // E,W,S,N neighbors + above
    expect(collectFluid(SOURCE, 0, 1, 0, APP, sampler(world, occ))).toHaveLength(0);
    // Open one side (remove the east occluder) → exactly that one side face is emitted.
    occ.delete("1,1,0");
    const quads = collectFluid(SOURCE, 0, 1, 0, APP, sampler(world, occ));
    expect(quads).toHaveLength(1);
    expect(quads[0].normal).toBe(Direction.East);
  });

  it("lava appearance flags it opaque + full-bright (no directional shade)", () => {
    const lavaApp: FluidAppearance = { sprite: { u: 0, v: 0, width: 1, height: 1 }, colorRGBA: 0xffffffff, translucent: false, shade: false };
    const quads = collectFluid({ type: FluidType.Lava, level: 0 }, 0, 1, 0, lavaApp, sampler({ "0,1,0": LAVA(0), "0,0,0": SOLID }));
    // shade:false ⇒ every face uses NO_SHADE (6), never a directional index.
    expect(quads.every((q) => q.normal === 6)).toBe(true);
  });
});
