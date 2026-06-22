// Frustum for per-section AABB culling (v1). RENDERER_PLAN.md §14.
//
// TRAP 14.A: v1 is frustum + distance only. Do NOT build the directional visibility graph until
//            profiling proves culling is the bottleneck — a bounded schematic usually doesn't.
//
// AUDIT F4 (flat scalar frustum): the six planes are 24 plain number fields filled IN PLACE from the
// column-major view-projection (Gribb–Hartmann), so a frame's frustum is built with ZERO array
// allocation and the instance is reused across frames (no steady-state GC). The box tests take scalar
// coordinates (no per-section corner array). Planes are NOT normalized — every test uses only the SIGN
// of the dot, so the 6 hypot + 24 divides the old code ran were dead work (dropped). The near-plane row
// differs from the OpenGL formulation because WebGPU clip space has z ∈ [0, 1].

import type { Vec3 } from "../types";
import type { Mat4 } from "./Camera";
import { instrument } from "../core/Instrument";

const SECTION_SIZE = 16;
/**
 * `CHUNK_SECTION_MARGIN` (1.0 + 0.125). A baked block model legally extends ~1 block past its
 * 16³ section box — `element.from/to` are NOT clamped to [0,16] (FaceBakery) and rotation-with-rescale
 * reaches further, so fences/walls/banners/rotated parts spill into the neighbour section. The 0.125 is
 * an FP epsilon. `testSection` pads the section AABB by this BEFORE the plane test so on-screen
 * overhanging geometry isn't false-culled when the bare 16³ box sits just outside a frustum edge
 * (audit Bug 2; `Viewport.java`). Conservative — only ever draws MORE, never culls visible geometry.
 */
export const SECTION_MARGIN = 1.125;

export class Frustum {
  // Plane i = (aᵢ, bᵢ, cᵢ, dᵢ); a point p is inside the plane ⇔ a·pₓ + b·p_y + c·p_z + d ≥ 0.
  private a0 = 0; private b0 = 0; private c0 = 0; private d0 = 0; // left
  private a1 = 0; private b1 = 0; private c1 = 0; private d1 = 0; // right
  private a2 = 0; private b2 = 0; private c2 = 0; private d2 = 0; // bottom
  private a3 = 0; private b3 = 0; private c3 = 0; private d3 = 0; // top
  private a4 = 0; private b4 = 0; private c4 = 0; private d4 = 0; // near
  private a5 = 0; private b5 = 0; private c5 = 0; private d5 = 0; // far

  constructor() {
    // X5 regression gauge: a Frustum is ALLOCATED here. F4 keeps one instance per renderer and rebuilds it
    // in place each frame, so steady-state `allocFrustum` deltas are 0 — a per-frame `new Frustum()` shows up.
    instrument.bumpAlloc("frustum");
  }

  /** Fill the six planes from a column-major view-projection. In place — no allocation. Returns `this`. */
  setFromViewProjection(vp: Mat4): this {
    // Rows of the column-major matrix: row r = (vp[r], vp[4+r], vp[8+r], vp[12+r]).
    const m0 = vp[0], m1 = vp[1], m2 = vp[2], m3 = vp[3];
    const m4 = vp[4], m5 = vp[5], m6 = vp[6], m7 = vp[7];
    const m8 = vp[8], m9 = vp[9], m10 = vp[10], m11 = vp[11];
    const m12 = vp[12], m13 = vp[13], m14 = vp[14], m15 = vp[15];
    // left = r3 + r0
    this.a0 = m3 + m0; this.b0 = m7 + m4; this.c0 = m11 + m8; this.d0 = m15 + m12;
    // right = r3 - r0
    this.a1 = m3 - m0; this.b1 = m7 - m4; this.c1 = m11 - m8; this.d1 = m15 - m12;
    // bottom = r3 + r1
    this.a2 = m3 + m1; this.b2 = m7 + m5; this.c2 = m11 + m9; this.d2 = m15 + m13;
    // top = r3 - r1
    this.a3 = m3 - m1; this.b3 = m7 - m5; this.c3 = m11 - m9; this.d3 = m15 - m13;
    // near = r2 (WebGPU z ∈ [0, 1])
    this.a4 = m2; this.b4 = m6; this.c4 = m10; this.d4 = m14;
    // far = r3 - r2
    this.a5 = m3 - m2; this.b5 = m7 - m6; this.c5 = m11 - m10; this.d5 = m15 - m14;
    return this;
  }

  /**
   * Is the AABB [min,max] at least partially inside the frustum? Scalar coordinates (no corner array).
   * Conservative (no false negatives): for each plane, test the AABB's positive vertex (the corner
   * farthest along the plane normal) — if it's outside, the whole box is outside.
   */
  testAab(minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number): boolean {
    if ((this.a0 >= 0 ? maxX : minX) * this.a0 + (this.b0 >= 0 ? maxY : minY) * this.b0 + (this.c0 >= 0 ? maxZ : minZ) * this.c0 + this.d0 < 0) return false;
    if ((this.a1 >= 0 ? maxX : minX) * this.a1 + (this.b1 >= 0 ? maxY : minY) * this.b1 + (this.c1 >= 0 ? maxZ : minZ) * this.c1 + this.d1 < 0) return false;
    if ((this.a2 >= 0 ? maxX : minX) * this.a2 + (this.b2 >= 0 ? maxY : minY) * this.b2 + (this.c2 >= 0 ? maxZ : minZ) * this.c2 + this.d2 < 0) return false;
    if ((this.a3 >= 0 ? maxX : minX) * this.a3 + (this.b3 >= 0 ? maxY : minY) * this.b3 + (this.c3 >= 0 ? maxZ : minZ) * this.c3 + this.d3 < 0) return false;
    if ((this.a4 >= 0 ? maxX : minX) * this.a4 + (this.b4 >= 0 ? maxY : minY) * this.b4 + (this.c4 >= 0 ? maxZ : minZ) * this.c4 + this.d4 < 0) return false;
    if ((this.a5 >= 0 ? maxX : minX) * this.a5 + (this.b5 >= 0 ? maxY : minY) * this.b5 + (this.c5 >= 0 ? maxZ : minZ) * this.c5 + this.d5 < 0) return false;
    return true;
  }

  /** A 16³ section at world-block MIN-CORNER origin (ox,oy,oz), padded by `SECTION_MARGIN` so models that
   *  overhang the section box aren't false-culled at frustum edges (audit Bug 2). NOTE: this takes the section
   *  MIN-CORNER, NOT a center (same-named `testSection` takes the CENTER + a padded radius) — every
   *  caller passes `sx*16` etc.; a future caller copying center convention would shift the box by 8
   *  blocks (P3). */
  testSection(ox: number, oy: number, oz: number): boolean {
    return this.testAab(
      ox - SECTION_MARGIN, oy - SECTION_MARGIN, oz - SECTION_MARGIN,
      ox + SECTION_SIZE + SECTION_MARGIN, oy + SECTION_SIZE + SECTION_MARGIN, oz + SECTION_SIZE + SECTION_MARGIN,
    );
  }
}
