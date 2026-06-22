// The per-face shade table must equal vanilla AdjacencyInfo and stay shared across shaders. AUDIT H3.

import { describe, it, expect } from "vitest";
import { DIRECTION_SHADE, wgslShadeArray } from "./faceShade";
import { Direction } from "../types";
import { NO_SHADE } from "../mesh/VertexFormat";
import { TERRAIN_WGSL } from "./terrainShader";
import { ENTITY_WGSL } from "./entities/entityShader";

describe("DIRECTION_SHADE (vanilla BlockModelLighter.AdjacencyInfo shadeWeights)", () => {
  it("matches vanilla per-face brightness — West/East = 0.6, NOT 0.65", () => {
    expect(DIRECTION_SHADE[Direction.Down]).toBe(0.5);
    expect(DIRECTION_SHADE[Direction.Up]).toBe(1.0);
    expect(DIRECTION_SHADE[Direction.North]).toBe(0.8);
    expect(DIRECTION_SHADE[Direction.South]).toBe(0.8);
    expect(DIRECTION_SHADE[Direction.West]).toBe(0.6);
    expect(DIRECTION_SHADE[Direction.East]).toBe(0.6);
    expect(DIRECTION_SHADE[NO_SHADE]).toBe(1.0); // index 6 = full bright sentinel
    expect(DIRECTION_SHADE).toHaveLength(7);
  });

  it("terrain + entity shaders build SHADE from the shared table and never the old 0.65", () => {
    const lit = wgslShadeArray();
    expect(lit).toBe("array<f32, 7>(0.5, 1.0, 0.8, 0.8, 0.6, 0.6, 1.0)");
    for (const wgsl of [TERRAIN_WGSL, ENTITY_WGSL]) {
      expect(wgsl).toContain(lit);
      expect(wgsl).not.toContain("0.65"); // the pre-fix West/East value must be gone
    }
  });
});
