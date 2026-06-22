// FS-4 — getVisibleFaces mask correctness. A wrong apron would drop a VISIBLE front face → holes, so the
// edge cases (apron boundary, octant, always-on UNASSIGNED) are pinned exactly.

import { describe, it, expect } from "vitest";
import { getVisibleFaces, FACE_CULL_APRON } from "./FaceCull";
import { ModelQuadFacing } from "../mesh/ModelQuadFacing";

const F = ModelQuadFacing;
const has = (mask: number, f: number) => (mask & (1 << f)) !== 0;

describe("getVisibleFaces (FS-4)", () => {
  it("always keeps the UNASSIGNED slice (unaligned faces point any way)", () => {
    expect(has(getVisibleFaces(100, 0, 0, 0, 0, 0), F.Unassigned)).toBe(true);
    expect(has(getVisibleFaces(-100, -100, -100, 0, 0, 0), F.Unassigned)).toBe(true);
  });

  it("a camera far +X keeps +X, drops -X (and keeps both centred Y/Z faces)", () => {
    const m = getVisibleFaces(100, 8, 8, 0, 0, 0); // section [0,16]³, camera far east
    expect(has(m, F.PosX)).toBe(true);
    expect(has(m, F.NegX)).toBe(false);
    expect(has(m, F.PosY)).toBe(true);
    expect(has(m, F.NegY)).toBe(true);
  });

  it("a camera far -X keeps -X, drops +X", () => {
    const m = getVisibleFaces(-100, 8, 8, 0, 0, 0);
    expect(has(m, F.NegX)).toBe(true);
    expect(has(m, F.PosX)).toBe(false);
  });

  it("a camera inside the section keeps all six axis faces (GPU back-face-culls per-tri)", () => {
    const m = getVisibleFaces(8, 8, 8, 0, 0, 0);
    for (const f of [F.PosX, F.PosY, F.PosZ, F.NegX, F.NegY, F.NegZ]) expect(has(m, f)).toBe(true);
  });

  it("the apron keeps a +X face for a camera STRICTLY within APRON of the min plane, drops it AT the edge (strict)", () => {
    // getVisibleFaces uses a STRICT `greaterThan(cam, min - APRON)`: a camera one block INSIDE the
    // apron keeps the face; AT the apron edge (cam === min - APRON) it is dropped — the overhang quad there is
    // edge-on/back-facing, so the GPU cull makes the drop pixel-identical (never a hole).
    expect(has(getVisibleFaces(0 - FACE_CULL_APRON + 1, 8, 8, 0, 0, 0), F.PosX)).toBe(true); // one inside the apron
    expect(has(getVisibleFaces(0 - FACE_CULL_APRON, 8, 8, 0, 0, 0), F.PosX)).toBe(false); // exactly at the edge → culled
    expect(has(getVisibleFaces(0 - FACE_CULL_APRON - 1, 8, 8, 0, 0, 0), F.PosX)).toBe(false); // one past → culled
  });

  it("octant: camera at +++ keeps the three positive faces and drops the three negatives", () => {
    const m = getVisibleFaces(100, 100, 100, 0, 0, 0);
    expect(has(m, F.PosX) && has(m, F.PosY) && has(m, F.PosZ)).toBe(true);
    expect(has(m, F.NegX) || has(m, F.NegY) || has(m, F.NegZ)).toBe(false);
  });
});
