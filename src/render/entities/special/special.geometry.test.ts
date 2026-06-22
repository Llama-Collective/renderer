// CPU geometry for the special block-entity effects (end portal/gateway, beacon beam, wireframe box).
// AUDIT P1.7 (portal shouldRenderFace + per-block layer count) / P2.1 (special/ had no tests).

import { describe, it, expect } from "vitest";
import { portalCube, PORTAL_FLOATS_PER_VERTEX } from "./PortalEffect";
import { beamQuads, BEAM_FLOATS_PER_VERTEX } from "./BeamEffect";
import { boxEdges, BOX_FLOATS_PER_VERTEX } from "./BoxEffect";
import type { Vec3 } from "../../../types";

const F = PORTAL_FLOATS_PER_VERTEX;

/** Min/max world-Y and the per-vertex layer count from a packed portal vertex list (x,y,z,layers). */
function portalExtent(out: number[]): { minY: number; maxY: number; layers: number } {
  let minY = Infinity, maxY = -Infinity, layers = 0;
  for (let i = 0; i < out.length; i += F) {
    minY = Math.min(minY, out[i + 1]);
    maxY = Math.max(maxY, out[i + 1]);
    layers = out[i + 3];
  }
  return { minY, maxY, layers };
}

describe("end portal/gateway geometry (vanilla TheEndPortalRenderer.shouldRenderFace)", () => {
  const pos: Vec3 = [10, 64, -3];

  it("a non-gateway end-portal block renders TOP+BOTTOM only (2 faces), 15 layers, y 0.375→0.75", () => {
    const out: number[] = [];
    portalCube(out, pos, false);
    expect(out.length / F / 4).toBe(2); // 2 quads
    const { minY, maxY, layers } = portalExtent(out);
    expect(minY).toBeCloseTo(64 + 0.375, 6);
    expect(maxY).toBeCloseTo(64 + 0.75, 6);
    expect(layers).toBe(15);
  });

  it("an end gateway renders ALL 6 faces, 16 layers, and fills the whole block (y 0→1)", () => {
    const out: number[] = [];
    portalCube(out, pos, true);
    expect(out.length / F / 4).toBe(6); // 6 quads
    const { minY, maxY, layers } = portalExtent(out);
    expect(minY).toBeCloseTo(64, 6);
    expect(maxY).toBeCloseTo(65, 6);
    expect(layers).toBe(16);
  });
});

describe("beacon beam geometry", () => {
  it("emits a 4-quad core tube + a 4-quad glow tube, both spanning [y0, y0+height]", () => {
    const core: number[] = [], glow: number[] = [];
    const pos: Vec3 = [0, 60, 0];
    beamQuads(core, glow, pos, 0xff0000, 8, 0);
    const G = BEAM_FLOATS_PER_VERTEX;
    expect(core.length / (4 * G)).toBe(4); // 4 side quads
    expect(glow.length / (4 * G)).toBe(4);
    for (const arr of [core, glow]) {
      for (let i = 0; i < arr.length; i += G) {
        expect(arr[i + 1]).toBeGreaterThanOrEqual(60 - 1e-6);
        expect(arr[i + 1]).toBeLessThanOrEqual(68 + 1e-6);
      }
    }
  });
});

describe("wireframe box geometry", () => {
  it("emits 12 edges = 24 line-list vertices, all within the box bounds", () => {
    const out: number[] = [];
    const min: Vec3 = [0, 0, 0], max: Vec3 = [1, 2, 3];
    boxEdges(out, min, max, 0x00ff00);
    const B = BOX_FLOATS_PER_VERTEX;
    expect(out.length / B).toBe(24); // 12 edges × 2 endpoints
    for (let i = 0; i < out.length; i += B) {
      expect(out[i]).toBeGreaterThanOrEqual(0);
      expect(out[i]).toBeLessThanOrEqual(1);
      expect(out[i + 1]).toBeGreaterThanOrEqual(0);
      expect(out[i + 1]).toBeLessThanOrEqual(2);
      expect(out[i + 2]).toBeGreaterThanOrEqual(0);
      expect(out[i + 2]).toBeLessThanOrEqual(3);
    }
  });
});
