// GUI block-item lights apply the FULL vanilla ITEMS_3D pose, not just its Y-flip. AUDIT M7.

import { describe, it, expect } from "vitest";
import { GUI_ITEM_LIGHT0, GUI_ITEM_LIGHT1, WORLD_ITEM_LIGHT0, WORLD_ITEM_LIGHT1 } from "./itemLighting";

const len3 = (v: readonly number[]) => Math.hypot(v[0], v[1], v[2]);

describe("GUI item lighting (vanilla Lighting.ITEMS_3D)", () => {
  it("keeps the lights unit-length (the pose is orthonormal)", () => {
    expect(len3(GUI_ITEM_LIGHT0)).toBeCloseTo(1, 6);
    expect(len3(GUI_ITEM_LIGHT1)).toBeCloseTo(1, 6);
  });

  it("applies the ITEMS_3D rotations, not just a Y-flip of the world lights (the M7 fix)", () => {
    // The old approximation was GUI = world lights with y negated. The full ITEMS_3D pose also rotates,
    // so at least one of x/z must differ from the world light — a plain Y-flip would leave them identical.
    const yFlipOnly0 = [WORLD_ITEM_LIGHT0[0], -WORLD_ITEM_LIGHT0[1], WORLD_ITEM_LIGHT0[2]];
    const sameAsYFlip = [0, 1, 2].every((i) => Math.abs(GUI_ITEM_LIGHT0[i] - yFlipOnly0[i]) < 1e-6);
    expect(sameAsYFlip).toBe(false);
  });

  it("preserves the diffuse-mode flags (light0.w = 1 enables diffuse; light1.w = 0)", () => {
    expect(GUI_ITEM_LIGHT0[3]).toBe(1);
    expect(GUI_ITEM_LIGHT1[3]).toBe(0);
  });
});
