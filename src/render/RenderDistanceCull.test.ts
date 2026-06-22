// Render-distance cull = cylindrical, nearest-point fog cull (OcclusionCuller.testDistance).
// Regression for the stress-demo bug: the old sphere-to-section-CENTER test culled far-horizon
// sections whose nearest edge was still in view (visible "holes" behind hills). RENDERER_PLAN §14.

import { describe, it, expect } from "vitest";
import { withinRenderDistance } from "./TerrainRenderer";
import type { Vec3 } from "../types";

const R = 180;

describe("withinRenderDistance (cylindrical nearest-point cull)", () => {
  it("Infinity render distance keeps everything", () => {
    expect(withinRenderDistance([9999, 0, 9999], [0, 0, 0], Infinity)).toBe(true);
  });

  it("keeps a section whose NEAREST edge is in range even when its CENTER is beyond R", () => {
    // Section [16,16,16] from the stress camera: center is ~183 blocks away (sphere-to-center would
    // cull it), but its nearest corner is ~169 horizontally — in range, so it must be KEPT.
    const eye: Vec3 = [148.7, 46.5, 156.4];
    const o: Vec3 = [16, 16, 16];
    // sphere-to-center (the OLD buggy test) would have culled it:
    const cx = o[0] + 8 - eye[0], cy = o[1] + 8 - eye[1], cz = o[2] + 8 - eye[2];
    expect(cx * cx + cy * cy + cz * cz).toBeGreaterThan(R * R); // old test: culled
    expect(withinRenderDistance(o, eye, R)).toBe(true); // new test: kept
  });

  it("culls a section that is genuinely beyond R horizontally (the true far corner)", () => {
    // Section [0,0,0] is ~191 blocks horizontally from the same eye → correctly culled.
    expect(withinRenderDistance([0, 0, 0], [148.7, 46.5, 156.4], R)).toBe(false);
  });

  it("is CYLINDRICAL: horizontal and vertical distances are tested independently (not a sphere)", () => {
    const o: Vec3 = [0, 0, 0]; // with the 1-block apron, the nearest box corner is at (17,17,17)
    // Out of range vertically only (nearest Y edge > R away), XZ in range → culled.
    expect(withinRenderDistance(o, [8, 17 + R + 5, 8], R)).toBe(false);
    // Out of range horizontally only → culled.
    expect(withinRenderDistance(o, [17 + R + 5, 8, 8], R)).toBe(false);
    // Within R on EACH axis but with a 3D distance > R: a sphere would cull, the cylinder KEEPS it.
    const c = 17 + 0.6 * R;
    expect(Math.hypot(0.6 * R, 0.6 * R, 0.6 * R)).toBeGreaterThan(R); // 3D distance exceeds R
    expect(withinRenderDistance(o, [c, c, c], R)).toBe(true);
  });

  it("measures to the nearest point: a camera INSIDE the section is always within range", () => {
    expect(withinRenderDistance([0, 0, 0], [8, 8, 8], R)).toBe(true);
    expect(withinRenderDistance([0, 0, 0], [8, 8, 8], 1)).toBe(true); // even with tiny R
  });
});
