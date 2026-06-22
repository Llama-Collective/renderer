// Terrain WGSL — the one place the packed-vertex decode + lighting math lives (BACKEND 10.C).
// In sync with mesh/VertexFormat.ts (20-byte layout) and camera/math.ts (column-major, z∈[0,1]).
//
// One shader family serves all three terrain passes via override constants:
//   ALPHA_TEST  (cutout)      — discard texels with alpha < 0.5 (cutout threshold).
//   EMIT_ALPHA  (translucent) — output the texel's alpha (× tint alpha) for "over" blending; the
//                               opaque passes force alpha = 1. The pipeline's blend/depth-write
//                               state (set per pass) is what actually makes it translucent (§12).
//
// Color path (TRAP 7.D / WEBGPU_FINDINGS §17.2): the atlas is `-srgb` so `textureSample` returns
// LINEAR rgb (alpha is unaffected). The vertex tint is authored sRGB, converted to linear here. The
// fragment output is linear (straight-alpha, NOT premultiplied — the blend factor is src-alpha); the
// `-srgb` canvas view encodes it on store.

import { wgslShadeArray, WGSL_SRGB_TO_LINEAR } from "./faceShade";

export const TERRAIN_WGSL = /* wgsl */ `
// AUDIT F2: viewProj is identical for every draw in a frame, so it lives in ONE shared (non-dynamic)
// uniform written once per frame — NOT re-copied into every per-draw slot.
//
// F2 COMPLETION (P4): the per-section origin lives in a read-only STORAGE array indexed by the draw's
// instance_index (each batched draw is one instance, drawn with firstInstance = its slot). This replaces
// the old per-draw dynamic-offset uniform: NO per-draw bind-group set (one bind for the whole pass) and a
// 16B/draw packed upload instead of a 256B-aligned slot per draw (16x less). The w lane is padding (a
// storage array element of vec3 still strides 16B in std430; vec4 makes the layout explicit).
// PREC-1: camera-relative origin for float32 vertex precision at large world coordinates. \`camOrigin\` is a
// SIBLING field (std140: a vec4 after a mat4 is 16-aligned → offset 64, no new bind group). DEFAULT-OFF the
// CPU uploads camOrigin=(0,0,0,0) + the WORLD-space viewProj, so the expression below reduces to today's
// exact f32 op sequence (origin + local) → byte-identical clip coords. When ON the CPU uploads a TRANSLATION-
// FREE viewProj (viewProjRel = proj·viewRel) AND a non-zero camOrigin; the large \`origin - camOrigin\` term is
// then cancelled in view space rather than reconstructed before the multiply, so the add+multiply run near
// zero where f32 has full precision (TRAP PREC-1.A — the matrix change and the subtraction are a PAIR).
struct Frame {
  viewProj : mat4x4<f32>,
  camOrigin : vec4<f32>,  // PREC-1: camera world origin (rounded to a block); (0,0,0,0) when the path is off
};
@group(0) @binding(0) var<uniform> F : Frame;
@group(0) @binding(1) var atlasTex : texture_2d<f32>;
@group(0) @binding(2) var atlasSamp : sampler;
@group(0) @binding(3) var<storage, read> origins : array<vec4<f32>>;  // per-draw section world origin (xyz)
// P6/LM-1: the light LUT. A 16×16 RGBA8 table indexed by (blockLevel, skyLevel) — the per-vertex lightmap
// carries those levels as texel-center COORDINATES (mesh/Lighting.ts), so global brightness / day-night is a
// free LUT rewrite (no remesh). Default LUT maps (block 0, sky 15) → white ⇒ a full-bright scene is unchanged.
@group(0) @binding(4) var lightLut : texture_2d<f32>;
@group(0) @binding(5) var lightSamp : sampler;

// 0 → no alpha test (solid); 1 → discard alpha < 0.5 (cutout). Set per-pipeline.
override ALPHA_TEST : f32 = 0.0;
// 0 → opaque output (alpha = 1, solid/cutout); 1 → emit texel alpha for blending (translucent).
override EMIT_ALPHA : f32 = 0.0;

// Fixed-point position decode — must match mesh/VertexFormat.ts (origin 8, range 32 → scale 2048).
const POS_DECODE_SCALE = 1.0 / 2048.0;
const POS_DECODE_ORIGIN = 8.0;

struct VsIn {
  // uint16x4: xyz = section-local fixed-point block coords; w = normalIndex | (material << 8).
  @location(0) packedPos : vec4<u32>,
  @location(1) color : vec4<f32>,  // unorm8 tint (sRGB)
  @location(2) uv    : vec2<f32>,  // unorm16 atlas uv
  @location(3) light : vec4<u32>,  // uint8x4 lightmap: x=blockCoord, y=skyCoord, z=ao, w=flags (P6/LM-4)
};

struct VsOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv    : vec2<f32>,
  @location(1) color : vec3<f32>,  // linear tint
  @location(2) shade : f32,
  @location(3) tintA : f32,        // tint alpha (1 for opaque tints; used by the translucent pass)
  @location(4) lightUv : vec2<f32>, // (blockCoord, skyCoord)/256 → light-LUT texel center (P6/LM-1)
  @location(5) ao : f32,            // ambient-occlusion multiplier (byte/255; 1.0 = unoccluded)
};

// Per-face brightness [Down, Up, North, South, West, East]; index 6 = no directional shade.
// Vanilla AdjacencyInfo shadeWeights via the shared DIRECTION_SHADE table (render/faceShade.ts).
const SHADE = ${wgslShadeArray()};

${WGSL_SRGB_TO_LINEAR}

@vertex
fn vs(in : VsIn, @builtin(instance_index) draw : u32) -> VsOut {
  var out : VsOut;
  // PREC-1: subtract the per-frame camera origin BEFORE the add. OFF ⇒ camOrigin=(0,0,0,0) + world viewProj ⇒
  // (origin - 0) + local with a world matrix is the same f32 op sequence as the pre-change \`origin + local\`
  // (byte-identical). ON ⇒ camOrigin≈camera block + a translation-free viewProj ⇒ the magnitude is small near
  // the camera, so the f32 add+multiply keep sub-block precision (no jitter at large world coordinates).
  let origin = origins[draw].xyz - F.camOrigin.xyz;
  let local = vec3<f32>(in.packedPos.xyz) * POS_DECODE_SCALE - vec3<f32>(POS_DECODE_ORIGIN);
  out.clip = F.viewProj * vec4<f32>(origin + local, 1.0);
  out.uv = in.uv;
  out.color = srgbToLinear(in.color.rgb);
  out.tintA = in.color.a;
  let normalIndex = in.packedPos.w & 0xFFu;  // material is the high byte (unused in shader)
  out.shade = SHADE[normalIndex];
  // P6/LM-1: the stored coords are texel centers in a 256-wide space (level*16+8) → /256 hits the exact
  // center of the matching 16×16 LUT texel. ao byte 255 ⇒ multiplier 1.0 (a no-op until smooth AO bakes it).
  out.lightUv = vec2<f32>(f32(in.light.x), f32(in.light.y)) / 256.0;
  out.ao = f32(in.light.z) / 255.0;
  return out;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
  let tex = textureSample(atlasTex, atlasSamp, in.uv);
  if (ALPHA_TEST > 0.5 && tex.a < 0.5) {
    discard;
  }
  let alpha = select(1.0, tex.a * in.tintA, EMIT_ALPHA > 0.5);
  // P6/LM-1: modulate by the LUT light level (linear-stored multiplier) and the baked AO. With the default
  // full-bright lightmap (sky 15 → white LUT, ao 255 → 1.0) this multiplies by one: output unchanged vs pre-lighting.
  let lm = textureSample(lightLut, lightSamp, in.lightUv).rgb;
  return vec4<f32>(tex.rgb * in.color * in.shade * lm * in.ao, alpha);
}
`;
