// Per-face directional brightness — the SINGLE source of truth for terrain + entity shading. AUDIT H3.
//
// Indexed by the packed `normalIndex` (= Direction enum 0..5: Down, Up, North, South, West, East), with
// index 6 = NO_SHADE (full-bright sentinel, 1.0). Both terrainShader and entityShader build their WGSL
// `SHADE` array from this so the table can never drift between them (it had — both X faces were 0.65).
//
// Values are vanilla MC's BlockModelLighter.AdjacencyInfo shadeWeights: DOWN 0.5, UP 1.0, NORTH 0.8,
// SOUTH 0.8, WEST 0.6, EAST 0.6 — matches (AoNeighborInfo / DefaultFluidRenderer: X-axis 0.6,
// Z-axis 0.8). West/East are 0.6, NOT 0.65.
export const DIRECTION_SHADE = [0.5, 1.0, 0.8, 0.8, 0.6, 0.6, 1.0] as const;

const wgslFloat = (v: number): string => (Number.isInteger(v) ? v.toFixed(1) : v.toString());

/** `DIRECTION_SHADE` as a WGSL `array<f32, 7>(...)` literal, so the shaders stay in lockstep with it. */
export function wgslShadeArray(): string {
  return `array<f32, 7>(${DIRECTION_SHADE.map(wgslFloat).join(", ")})`;
}

/** The sRGB→linear decode (WGSL), shared by terrain + entity shaders so the two passes can't diverge
 *  (the vertex tint is authored sRGB; the atlas already samples linear — TRAP 7.D). Interpolate into a
 *  shader module body. */
export const WGSL_SRGB_TO_LINEAR = /* wgsl */ `
fn srgbToLinear(c : vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(hi, lo, c <= vec3<f32>(0.04045));
}`;
