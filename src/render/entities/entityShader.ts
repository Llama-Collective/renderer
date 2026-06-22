// Entity / block-entity forward-pass WGSL. RENDERER_PLAN.md §18 (Phase 4.5).
//
// Same packed 20-byte vertex as terrain, but geometry is MODEL-LOCAL: a per-entity `model` matrix
// places it in the world (vs terrain's section-origin add). Entities draw in a pass that LOADS terrain's
// depth (shared depth buffer — resolves TRAP 18.B), after opaque terrain.
//
// Lighting: the packed `normalIndex` is the LOCAL face direction; the shader rotates it by the model
// matrix and shades by the nearest world axis using the SAME per-face table as terrain — so a rotated
// part still lights consistently. Items opt into vanilla's two-directional DIFFUSE lighting instead
// (`light0.w > 0.5`): shade = min(1, (max(0,dot(L0,n)) + max(0,dot(L1,n)))·0.6 + 0.4) over the world
// normal, so a block item relights smoothly as it spins (see itemLighting.ts). A per-entity `tint`
// recolors (sheep wool dye, bake-once + uniform remap), and `flash` mixes toward a colour (TNT white
// blink / hurt flash). Colour stays linear (§17.2).

import { wgslShadeArray, WGSL_SRGB_TO_LINEAR } from "../faceShade";

export const ENTITY_WGSL = /* wgsl */ `
struct Uniforms {
  viewProj : mat4x4<f32>,
  model    : mat4x4<f32>,
  tint     : vec4<f32>,   // linear rgb multiply + alpha
  flash    : vec4<f32>,   // linear flash rgb + mix amount (a)
  light0   : vec4<f32>,   // diffuse light dir 0 (xyz) + mode flag (w > 0.5 ⇒ diffuse item lighting)
  light1   : vec4<f32>,   // diffuse light dir 1 (xyz)
};
@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var atlasTex : texture_2d<f32>;
@group(0) @binding(2) var atlasSamp : sampler;

override ALPHA_TEST : f32 = 1.0;   // entities default to cutout (alpha-tested)
override EMIT_ALPHA : f32 = 0.0;   // 1 → emit texel alpha (translucent entities)

const POS_DECODE_SCALE = 1.0 / 2048.0;
const POS_DECODE_ORIGIN = 8.0;

const NORMALS = array<vec3<f32>, 7>(
  vec3<f32>(0.0, -1.0, 0.0),  // 0 Down
  vec3<f32>(0.0,  1.0, 0.0),  // 1 Up
  vec3<f32>(0.0, 0.0, -1.0),  // 2 North
  vec3<f32>(0.0, 0.0,  1.0),  // 3 South
  vec3<f32>(-1.0, 0.0, 0.0),  // 4 West
  vec3<f32>( 1.0, 0.0, 0.0),  // 5 East
  vec3<f32>(0.0, 1.0, 0.0),   // 6 no-shade sentinel
);
const SHADE = ${wgslShadeArray()};  // shared vanilla AdjacencyInfo table (render/faceShade.ts)

struct VsIn {
  @location(0) packedPos : vec4<u32>,
  @location(1) color : vec4<f32>,   // unorm8 vertex tint (sRGB)
  @location(2) uv    : vec2<f32>,
};
struct VsOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv    : vec2<f32>,
  @location(1) color : vec3<f32>,   // linear vertex tint
  @location(2) shade : f32,         // front-face shade
  @location(3) tintA : f32,
  @location(4) shadeBack : f32,     // back-face shade (PER_FACE_LIGHTING; = shade when not diffuse)
};

${WGSL_SRGB_TO_LINEAR}

// Quantize a world normal to the nearest of the 6 axes → terrain's per-face shade.
fn axisShade(n : vec3<f32>) -> f32 {
  let a = abs(n);
  if (a.y >= a.x && a.y >= a.z) { return select(SHADE[0], SHADE[1], n.y > 0.0); }
  if (a.x >= a.z)               { return select(SHADE[4], SHADE[5], n.x > 0.0); }
  return select(SHADE[2], SHADE[3], n.z > 0.0);
}

@vertex
fn vs(in : VsIn) -> VsOut {
  var out : VsOut;
  // PREC-1 (entity parity — entities never defer, so they move in lockstep with terrain or jitter against it).
  // OFF ⇒ U.viewProj is the WORLD vp and U.model carries the full world translation ⇒ byte-identical to before.
  // ON ⇒ the CPU uploads a translation-free viewProj AND pre-subtracts the camera origin from U.model's
  // translation column, so \`model * local\` lands near the camera and the f32 transform keeps full precision.
  let local = vec3<f32>(in.packedPos.xyz) * POS_DECODE_SCALE - vec3<f32>(POS_DECODE_ORIGIN);
  out.clip = U.viewProj * (U.model * vec4<f32>(local, 1.0));
  out.uv = in.uv;
  out.color = srgbToLinear(in.color.rgb);
  out.tintA = in.color.a;
  let ni = in.packedPos.w & 0xFFu;
  if (U.light0.w > 0.5) {
    // Vanilla diffuse lighting (rotates with the model). Normalize: model carries a uniform scale. Compute
    // both the front (n) and back (−n) shade; the fragment picks per-face when PER_FACE is on (light1.w).
    let n = normalize((U.model * vec4<f32>(NORMALS[ni], 0.0)).xyz);
    let df = max(0.0, dot(U.light0.xyz, n)) + max(0.0, dot(U.light1.xyz, n));
    let db = max(0.0, dot(U.light0.xyz, -n)) + max(0.0, dot(U.light1.xyz, -n));
    out.shade = min(1.0, df * 0.6 + 0.4);
    out.shadeBack = min(1.0, db * 0.6 + 0.4);
  } else if (ni == 6u) {
    out.shade = 1.0;
    out.shadeBack = 1.0;
  } else {
    // Block-display geometry (falling block + the renderSingleBlock-style BE embeds, e.g. the spinning
    // spawner miniature): shade by the LOCAL face direction. Vanilla + BAKE block-model shade at
    // model-bake time from the local face, so it is CONSTANT per face and rotates WITH the block. The old
    // axisShade of the model-rotated normal re-derived it every frame, which snapped between world-axis
    // steps at the 45-degree boundaries as a block spun (the abrupt shading). Static/90-deg-placed blocks
    // are unaffected (their model carries no extra rotation, so local == world here).
    out.shade = axisShade(NORMALS[ni]);
    out.shadeBack = out.shade;
  }
  return out;
}

@fragment
fn fs(in : VsOut, @builtin(front_facing) front : bool) -> @location(0) vec4<f32> {
  let tex = textureSample(atlasTex, atlasSamp, in.uv);
  if (ALPHA_TEST > 0.5 && tex.a < 0.5) {
    discard;
  }
  // PER_FACE lighting (light1.w, box entities): the visible side picks its own shade — so an open box's
  // inner walls are lit by their viewer-side normal, not darkened by the away-facing outward normal.
  let perFace = select(in.shadeBack, in.shade, front);
  let shade = select(in.shade, perFace, U.light1.w > 0.5);
  var rgb = tex.rgb * in.color * U.tint.rgb * shade;
  rgb = mix(rgb, U.flash.rgb, U.flash.a);          // white-flash / hurt overlay
  let alpha = select(1.0, tex.a * in.tintA * U.tint.a, EMIT_ALPHA > 0.5);
  return vec4<f32>(rgb, alpha);
}
`;
