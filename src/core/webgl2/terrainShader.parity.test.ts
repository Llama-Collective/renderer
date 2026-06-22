// The terrain GLSL twin (core/webgl2/shaders.ts) hardcodes the decode/lighting constants because importing
// them from render/faceShade.ts or mesh/VertexFormat.ts would be a core→render/mesh dependency (the same
// module-boundary rule that forces label-based twin resolution — STYLE_GUIDE §1 / WEBGL_PLAN W2.1). That
// risks DRIFT: if the WGSL source of truth changes (e.g. a SHADE weight, the fixed-point scale), the twin
// would silently diverge and terrain would mis-shade with NO error. A *test* is exempt from the boundary
// (it isn't shipped), so this is where the two are pinned together — fail loudly if either side moves.

import { describe, expect, it } from "vitest";
import { GLSL_TWINS } from "./shaders";
import { DIRECTION_SHADE } from "../../render/faceShade";
import { POS_DECODE_SCALE, POS_ORIGIN_BLOCKS } from "../../mesh/VertexFormat";

/** Render a number the way the twin's GLSL literals are written (integers get a ".0", matching faceShade). */
const glslFloat = (v: number): string => (Number.isInteger(v) ? v.toFixed(1) : v.toString());

describe("terrain GLSL twin ↔ TS source-of-truth parity", () => {
  const vs = GLSL_TWINS.terrain.vertex;

  it("registers a terrain twin", () => {
    expect(GLSL_TWINS.terrain).toBeDefined();
  });

  it("SHADE array matches render/faceShade.ts DIRECTION_SHADE exactly", () => {
    // The twin's `const float SHADE[7] = float[7](...)` must list DIRECTION_SHADE in order, or terrain
    // directional face shading drifts from the entity/WGSL shading (the bug faceShade.ts was created to kill).
    const expected = `float[7](${DIRECTION_SHADE.map(glslFloat).join(", ")})`;
    expect(vs).toContain(expected);
  });

  it("fixed-point position decode matches mesh/VertexFormat.ts", () => {
    // POS_DECODE_SCALE is 1/2048; the twin writes it as `1.0 / 2048.0` (a u16 maps a 32-block range → 2048/block).
    const denom = Math.round(1 / POS_DECODE_SCALE);
    expect(denom).toBe(2048); // guard the source-of-truth value itself
    expect(vs).toContain(`POS_DECODE_SCALE = 1.0 / ${denom}.0`);
    // POS_ORIGIN_BLOCKS is the decode bias (8) — coords in [-8,+24) are representable.
    expect(POS_ORIGIN_BLOCKS).toBe(8);
    expect(vs).toContain(`POS_DECODE_ORIGIN = ${glslFloat(POS_ORIGIN_BLOCKS)}`);
  });

  it("srgbToLinear uses the standard sRGB transfer constants", () => {
    // The exact piecewise-curve constants (must match WGSL_SRGB_TO_LINEAR); a wrong constant tints everything.
    for (const k of ["12.92", "0.055", "1.055", "2.4", "0.04045"]) expect(vs).toContain(k);
  });

  it("declares the origins UBO + u_BaseInstance selector (divergence A)", () => {
    // The storage-array→UBO transposition and its per-draw index uniform are the load-bearing W3 mechanism.
    expect(vs).toContain("uniform Origins");
    expect(vs).toContain("origins[ORIGIN_COUNT]");
    expect(vs).toContain("origins[u_BaseInstance]");
  });
});

describe("entity GLSL twin ↔ TS source-of-truth parity (W5)", () => {
  const evs = GLSL_TWINS.entity.vertex;
  const efs = GLSL_TWINS.entity.fragment;

  it("registers an entity twin with the dynamic-offset Uniforms block + atlas sampler", () => {
    expect(GLSL_TWINS.entity).toBeDefined();
    expect(evs).toContain("uniform Uniforms"); // the per-draw dynamic-offset UBO (model/tint/flash/light…)
    expect(GLSL_TWINS.entity.uniformBlocks).toEqual([{ name: "Uniforms", binding: 0 }]);
    expect(GLSL_TWINS.entity.samplers).toEqual([{ name: "atlasTex", textureBinding: 1, samplerBinding: 2, unit: 0 }]);
  });

  it("shares the same SHADE / srgbToLinear / fixed-point decode as terrain", () => {
    const expectedShade = `float[7](${DIRECTION_SHADE.map(glslFloat).join(", ")})`;
    expect(evs).toContain(expectedShade);
    expect(evs).toContain(`POS_DECODE_SCALE = 1.0 / ${Math.round(1 / POS_DECODE_SCALE)}.0`);
    expect(evs).toContain(`POS_DECODE_ORIGIN = ${glslFloat(POS_ORIGIN_BLOCKS)}`);
    for (const k of ["12.92", "0.055", "1.055", "2.4", "0.04045"]) expect(evs).toContain(k);
  });

  it("selects front/back shade via gl_FrontFacing for PER_FACE box lighting", () => {
    expect(efs).toContain("gl_FrontFacing");
  });
});

describe("portal GLSL twin (W5)", () => {
  const pfs = GLSL_TWINS.portal.fragment;

  it("registers a portal twin with two sampler units sharing one sampler binding", () => {
    expect(GLSL_TWINS.portal).toBeDefined();
    // The single WGSL `samp`@1 feeds both skyTex (unit 0) and portalTex (unit 1).
    expect(GLSL_TWINS.portal.samplers).toEqual([
      { name: "skyTex", textureBinding: 2, samplerBinding: 1, unit: 0 },
      { name: "portalTex", textureBinding: 3, samplerBinding: 1, unit: 1 },
    ]);
  });

  it("OMITS the WGSL Y-flip — gl_FragCoord is already Y-up in GL (TRAP B.2)", () => {
    expect(pfs).toContain("gl_FragCoord");
    // The WGSL did `screen.y = 1.0 - screen.y;`; the GLSL twin must NOT (it would double-flip the effect).
    expect(pfs).not.toMatch(/screen\.y\s*=\s*1\.0\s*-/);
  });
});
