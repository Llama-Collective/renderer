// SL-2 — smooth lighting / per-vertex AO. The subtle traps: the corner→vertex permutation must match the
// geometry winding (no 90° AO rotation), an OPEN corner must read exactly AO_NONE (so open faces stay
// byte-identical to flat), and an occluder must darken the RIGHT corner.

import { describe, it, expect } from "vitest";
import { smoothFaceLight, isAlignedFullFace, AO_OPEN, AO_FLOOR } from "./SmoothLight";
import { unpackLight } from "./Lighting";
import { Direction } from "../types";
import { snapshotIndex, snapshotStride } from "../world/SnapshotSource";
import { FACE_CORNERS } from "./model/FaceBakery";

const APRON = 2;
const CELLS = snapshotStride(APRON) ** 3;

/** A uniform light array: sky 15, block 0 everywhere (the editor's default full-bright). */
function uniformLight(): Uint8Array {
  return new Uint8Array(CELLS).fill(15 << 4);
}

describe("smoothFaceLight (SL-2)", () => {
  it("an OPEN face (no occluders) reads AO_NONE on every corner and uniform light (== flat)", () => {
    const ao = new Uint8Array(CELLS); // no occluders
    const words = smoothFaceLight(Direction.Up, 5, 5, 5, uniformLight(), ao, APRON, 1.0);
    for (const w of words) {
      const u = unpackLight(w);
      expect(u.ao).toBe(AO_OPEN); // 255 — open corners are byte-identical to AO_NONE
      expect(u.blockLight).toBe(0);
      expect(u.skyLight).toBe(15);
    }
  });

  it("an occluder darkens the corner it touches, not the opposite corner", () => {
    const ao = new Uint8Array(CELLS);
    // Face Up of cell (5,5,5): the sampling plane is centred at (5,6,5). Corner 0 = FACE_CORNERS[Up][0]
    // = [0,1,0] → samples toward −x,−z. Put an occluder at the −x edge cell (4,6,5).
    ao[snapshotIndex(4, 6, 5, APRON)] = 1;
    const words = smoothFaceLight(Direction.Up, 5, 5, 5, uniformLight(), ao, APRON, 1.0);
    const c0 = unpackLight(words[0]).ao; // corner [0,1,0] — touches the occluder
    const c2 = unpackLight(words[2]).ao; // corner [1,1,1] — opposite side, untouched
    expect(c0).toBeLessThan(255); // darkened
    expect(c2).toBe(255); // still open
  });

  it("both edges occluding → the corner floors at AO_FLOOR, NOT pure black (H2)", () => {
    const ao = new Uint8Array(CELLS);
    ao[snapshotIndex(4, 6, 5, APRON)] = 1; // −x edge
    ao[snapshotIndex(5, 6, 4, APRON)] = 1; // −z edge
    const c0 = unpackLight(smoothFaceLight(Direction.Up, 5, 5, 5, uniformLight(), ao, APRON, 1.0)[0]).ao;
    // AO level 0 (both sides) now maps to AO_FLOOR (~0.2), matching vanilla/— never multiplier 0.
    expect(c0).toBe(Math.round(AO_FLOOR * 255)); // 51, not 0
    expect(c0).toBeGreaterThan(0);
  });

  it("shade-brightness < 1 dims the AO even on an open face (own byte, not premultiplied)", () => {
    const ao = new Uint8Array(CELLS);
    const dim = unpackLight(smoothFaceLight(Direction.Up, 5, 5, 5, uniformLight(), ao, APRON, 0.5)[0]).ao;
    expect(dim).toBe(Math.round(0.5 * 255)); // open AO 1.0 × shade 0.5
  });

  it("isAlignedFullFace matches FACE_CORNERS exactly and rejects insets / null", () => {
    for (const dir of [Direction.Down, Direction.Up, Direction.North, Direction.South, Direction.West, Direction.East]) {
      expect(isAlignedFullFace(dir, FACE_CORNERS[dir])).toBe(true); // a full cube face
    }
    // An inset top face (y = 0.9375, a grass-path-like cut) is NOT a full face.
    const inset = FACE_CORNERS[Direction.Up].map(([x, , z]) => [x, 0.9375, z] as [number, number, number]);
    expect(isAlignedFullFace(Direction.Up, inset)).toBe(false);
    expect(isAlignedFullFace(null, FACE_CORNERS[Direction.Up])).toBe(false);
  });
});
