// GLSL ES 3.00 shader twins for the WebGL2 backend. WEBGL_PLAN.md (W2); WEBGL_FINDINGS.md §4.
//
// There is no WGSL→GLSL transpiler: the WebGL2 device ships a hand-written GLSL twin of each WGSL
// module, resolved by the LABEL the renderer passes to `createShaderModule(code, label)` (the WGSL
// `code` string itself can't be imported here — that would be a core→render dependency, against the
// module boundary). Each twin mirrors its WGSL byte-for-byte in behavior; the registry maps every
// label a renderer uses to its twin (one WGSL can carry several labels — e.g. overlay/explosion).
//
// Two WebGL2 conventions every twin follows (WEBGL_FINDINGS §3-B / §4):
//  - `#version 300 es` is line 1 (override-constant `#define`s are injected right after it by the
//    device's createPipeline); `precision highp float;` is mandatory in the fragment stage.
//  - the WebGPU clip space is z∈[0,1] but WebGL2's is z∈[-1,1], so every vertex twin remaps depth
//    with `gl_Position.z = gl_Position.z*2.0 - gl_Position.w;`, guarded by `#ifndef CLIP_ZERO_TO_ONE`
//    so the W6 `EXT_clip_control` fast-path can compile it out.
//
// W2 shipped the EASY shaders (color-vertex, grid, box, beam — single UBO, no textures). W3 adds the
// TERRAIN twin: integer vertex attrs (uint16x4 pos / uint8x4 light), the atlas + light-LUT sampler2D
// pairs, and — the load-bearing divergence (A) — the per-section origin. WGSL reads it from a
// `var<storage, read> origins : array<vec4<f32>>` indexed by `@builtin(instance_index)`; WebGL2 has
// neither SSBOs nor base-instance, so the twin reads a `std140 uniform Origins { vec4 origins[N]; }`
// indexed by a per-draw `u_BaseInstance` int the device sets from `firstInstance` (FINDINGS §2 / TRAP
// W3.A). Entity/portal are W5.

/** A WGSL module's GLSL ES 3.00 twin + the binding metadata the device needs to wire it up. */
export interface GlslTwin {
  vertex: string;
  fragment: string;
  /** Each uniform block's GLSL name + the neutral @binding it sits at (used as the GL UBO binding point). */
  uniformBlocks: readonly { name: string; binding: number }[];
  /** Combined `sampler2D` uniforms — GLSL ES folds a WGSL (texture, sampler) pair into one sampler2D
   *  bound to a texture unit. Empty for the W2 shaders; populated for terrain/entity (W3/W5). */
  samplers?: readonly { name: string; textureBinding: number; samplerBinding: number; unit: number }[];
}

// The depth remap, shared by every vertex twin. Kept as a string fragment so the twins stay in lockstep.
const DEPTH_REMAP = `
#ifndef CLIP_ZERO_TO_ONE
  gl_Position.z = gl_Position.z * 2.0 - gl_Position.w; // WebGPU z∈[0,1] → WebGL2 z∈[-1,1]
#endif`;

// --- color-vertex (labels "overlay", "explosion") — render/colorVertex.ts COLOR_VERTEX_WGSL ---------
const COLOR_VERTEX_TWIN: GlslTwin = {
  vertex: `#version 300 es
precision highp float;
layout(std140) uniform U { mat4 viewProj; };
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec4 a_color;
out vec4 v_color;
void main() {
  gl_Position = viewProj * vec4(a_pos, 1.0);${DEPTH_REMAP}
  v_color = a_color;
}`,
  fragment: `#version 300 es
precision highp float;
in vec4 v_color;
out vec4 fragColor;
void main() { fragColor = v_color; }`,
  uniformBlocks: [{ name: "U", binding: 0 }],
};

// --- grid (label "overlay-grid") — render/overlay/OverlayRenderer.ts GRID_WGSL ---------------------
const GRID_TWIN: GlslTwin = {
  vertex: `#version 300 es
precision highp float;
layout(std140) uniform GU { mat4 viewProj; vec4 params; };
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec4 a_color;
out vec4 v_color;
out vec3 v_world;
void main() {
  gl_Position = viewProj * vec4(a_pos, 1.0);${DEPTH_REMAP}
  v_color = a_color;
  v_world = a_pos;
}`,
  fragment: `#version 300 es
precision highp float;
layout(std140) uniform GU { mat4 viewProj; vec4 params; }; // params = (centreX, centreZ, fadeStart, fadeEnd)
in vec4 v_color;
in vec3 v_world;
out vec4 fragColor;
void main() {
  float d = distance(v_world.xz, params.xy);
  float fade = 1.0 - smoothstep(params.z, params.w, d);
  fragColor = vec4(v_color.rgb, v_color.a * fade);
}`,
  uniformBlocks: [{ name: "GU", binding: 0 }],
};

// --- box / wireframe (labels "linebox", "linebox-xray") — render/entities/special/BoxEffect.ts -----
const BOX_TWIN: GlslTwin = {
  vertex: `#version 300 es
precision highp float;
layout(std140) uniform U { mat4 viewProj; float time; float resX; float resY; };
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec4 a_color;
out vec4 v_color;
void main() {
  gl_Position = viewProj * vec4(a_pos, 1.0);${DEPTH_REMAP}
  v_color = a_color;
}`,
  fragment: `#version 300 es
precision highp float;
in vec4 v_color;
out vec4 fragColor;
void main() { fragColor = v_color; }`,
  uniformBlocks: [{ name: "U", binding: 0 }],
};

// --- beacon beam (label "beam") — render/entities/special/BeamEffect.ts ---------------------------
const BEAM_TWIN: GlslTwin = {
  vertex: `#version 300 es
precision highp float;
layout(std140) uniform U { mat4 viewProj; float time; float resX; float resY; };
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec4 a_color;
layout(location = 2) in float a_v;
out vec4 v_color;
out float v_v;
void main() {
  gl_Position = viewProj * vec4(a_pos, 1.0);${DEPTH_REMAP}
  v_color = a_color;
  v_v = a_v;
}`,
  fragment: `#version 300 es
precision highp float;
layout(std140) uniform U { mat4 viewProj; float time; float resX; float resY; };
in vec4 v_color;
in float v_v;
out vec4 fragColor;
void main() {
  // Energy bands flowing up the beam (vanilla's scrolling beacon_beam texture-V).
  float band = 0.5 + 0.5 * sin((v_v - time * 0.75) * 3.1415927);
  fragColor = vec4(v_color.rgb * (0.55 + 0.6 * band), v_color.a * (0.45 + 0.55 * band));
}`,
  uniformBlocks: [{ name: "U", binding: 0 }],
};

// --- terrain (label "terrain") — render/terrainShader.ts TERRAIN_WGSL --------------------------------
// The vertical-slice shader (W3). Mirrors TERRAIN_WGSL byte-for-byte in behavior; the decode/lighting
// constants are hardcoded here (importing render/faceShade.ts or mesh/VertexFormat.ts would be a
// core→render/mesh dependency — the same boundary that forces label resolution) and pinned against
// their TS source of truth by terrainShader.parity.test.ts so they cannot drift (FINDINGS §4.2).
//
// Override constants ALPHA_TEST/EMIT_ALPHA arrive as injected `#define`s (cutout passes `{ALPHA_TEST:1}`,
// translucent `{EMIT_ALPHA:1}`); the `#ifndef` fallbacks below mirror the WGSL `override … = 0.0` so a
// pass that injects neither still compiles. They gate via the PREPROCESSOR (`#if`), not a runtime float
// compare — GLSL ES 3.00 has no implicit int→float, so `ALPHA_TEST > 0.5` on the int `#define` 0/1 would
// not even compile. ORIGIN_COUNT is injected by the device (= MAX_UNIFORM_BLOCK_SIZE/16); the 1024
// fallback is the guaranteed-minimum floor (TRAP W3.B).
const TERRAIN_TWIN: GlslTwin = {
  vertex: `#version 300 es
precision highp float;
precision highp int;
#ifndef ALPHA_TEST
#define ALPHA_TEST 0
#endif
#ifndef EMIT_ALPHA
#define EMIT_ALPHA 0
#endif
#ifndef ORIGIN_COUNT
#define ORIGIN_COUNT 1024
#endif
// std140 mirror of the WGSL Frame struct: mat4 viewProj @0, vec4 camOrigin @64 (TRAP W2.C / PREC-1).
layout(std140) uniform Frame {
  mat4 viewProj;
  vec4 camOrigin;
};
// (A) The per-draw section origin — the WGSL storage array, transposed to a std140 UBO of vec4[N]
// indexed by the per-draw u_BaseInstance int (= firstInstance; FINDINGS §2 / TRAP W3.A).
layout(std140) uniform Origins {
  vec4 origins[ORIGIN_COUNT];
};
uniform int u_BaseInstance;
layout(location = 0) in uvec4 a_packedPos; // uint16x4: xyz fixed-point block coords, w = normalIndex | material<<8
layout(location = 1) in vec4 a_color;      // unorm8 tint (sRGB)
layout(location = 2) in vec2 a_uv;         // unorm16 atlas uv
layout(location = 3) in uvec4 a_light;     // uint8x4 lightmap: x=blockCoord, y=skyCoord, z=ao, w=flags
out vec2 v_uv;
out vec3 v_color;
out float v_shade;
out float v_tintA;
out vec2 v_lightUv;
out float v_ao;
// Fixed-point decode — mirrors mesh/VertexFormat.ts (POS_DECODE_SCALE 1/2048, POS_ORIGIN_BLOCKS 8).
const float POS_DECODE_SCALE = 1.0 / 2048.0;
const float POS_DECODE_ORIGIN = 8.0;
// Per-face brightness [Down,Up,North,South,West,East]; idx 6 = no shade — render/faceShade.ts DIRECTION_SHADE.
const float SHADE[7] = float[7](0.5, 1.0, 0.8, 0.8, 0.6, 0.6, 1.0);
vec3 srgbToLinear(vec3 c) {
  vec3 lo = c / 12.92;
  vec3 hi = pow((c + vec3(0.055)) / 1.055, vec3(2.4));
  return mix(hi, lo, vec3(lessThanEqual(c, vec3(0.04045)))); // WGSL select(hi, lo, c <= 0.04045)
}
void main() {
  // PREC-1: subtract camOrigin BEFORE the add, exactly as WGSL — the f32 op order is load-bearing (TRAP W3.C).
  vec3 origin = origins[u_BaseInstance].xyz - camOrigin.xyz;
  vec3 local = vec3(a_packedPos.xyz) * POS_DECODE_SCALE - vec3(POS_DECODE_ORIGIN);
  gl_Position = viewProj * vec4(origin + local, 1.0);${DEPTH_REMAP}
  v_uv = a_uv;
  v_color = srgbToLinear(a_color.rgb);
  v_tintA = a_color.a;
  uint normalIndex = a_packedPos.w & 0xFFu; // material is the high byte (unused)
  v_shade = SHADE[int(normalIndex)];
  v_lightUv = vec2(float(a_light.x), float(a_light.y)) / 256.0; // texel-center coords → /256 (P6/LM-1)
  v_ao = float(a_light.z) / 255.0;
}`,
  fragment: `#version 300 es
precision highp float;
#ifndef ALPHA_TEST
#define ALPHA_TEST 0
#endif
#ifndef EMIT_ALPHA
#define EMIT_ALPHA 0
#endif
uniform sampler2D atlasTex; // unit 0  (WGSL atlasTex@1 + atlasSamp@2)
uniform sampler2D lightLut; // unit 1  (WGSL lightLut@4 + lightSamp@5)
in vec2 v_uv;
in vec3 v_color;
in float v_shade;
in float v_tintA;
in vec2 v_lightUv;
in float v_ao;
out vec4 fragColor;
void main() {
  vec4 tex = texture(atlasTex, v_uv); // SRGB8_ALPHA8 ⇒ linear rgb (alpha untouched)
#if ALPHA_TEST
  if (tex.a < 0.5) { discard; } // cutout threshold
#endif
#if EMIT_ALPHA
  float alpha = tex.a * v_tintA; // translucent: emit texel alpha for the "over" blend
#else
  float alpha = 1.0;             // opaque/cutout force alpha = 1
#endif
  vec3 lm = texture(lightLut, v_lightUv).rgb; // linear multiplier (full-bright LUT ⇒ white)
  fragColor = vec4(tex.rgb * v_color * v_shade * lm * v_ao, alpha);
}`,
  uniformBlocks: [
    { name: "Frame", binding: 0 },
    { name: "Origins", binding: 3 },
  ],
  samplers: [
    { name: "atlasTex", textureBinding: 1, samplerBinding: 2, unit: 0 },
    { name: "lightLut", textureBinding: 4, samplerBinding: 5, unit: 1 },
  ],
};

// --- entity / block-entity (label "entity") — render/entities/entityShader.ts ENTITY_WGSL ----------
// W5. Reuses terrainVertexLayout but declares only locs 0..2 (no light, loc 3 — the program just won't
// list it in ACTIVE_ATTRIBUTES, so the device's activeLocations skip drops it; TRAP 5.1). The per-draw
// state is a DYNAMIC-OFFSET UBO (256-byte slots; the device's existing dynamicOffsetBindings + bindBufferRange
// path selects the slot). gl_FrontFacing picks front vs back shade for PER_FACE box lighting — the device
// sets frontFace(CCW) unconditionally so it's deterministic even when culling is off. Shade math: diffuse
// (item lighting, light0.w>0.5), axisShade (block-display geometry), or full-bright (sentinel ni==6).
const ENTITY_TWIN: GlslTwin = {
  vertex: `#version 300 es
precision highp float;
precision highp int;
#ifndef ALPHA_TEST
#define ALPHA_TEST 1
#endif
#ifndef EMIT_ALPHA
#define EMIT_ALPHA 0
#endif
layout(std140) uniform Uniforms {
  mat4 viewProj;
  mat4 model;
  vec4 tint;   // linear rgb multiply + alpha
  vec4 flash;  // linear flash rgb + mix amount (a)
  vec4 light0; // diffuse dir 0 (xyz) + mode flag (w>0.5 ⇒ diffuse item lighting)
  vec4 light1; // diffuse dir 1 (xyz) + PER_FACE flag (w>0.5 ⇒ box per-face lighting)
};
layout(location = 0) in uvec4 a_packedPos;
layout(location = 1) in vec4 a_color;
layout(location = 2) in vec2 a_uv;
out vec2 v_uv;
out vec3 v_color;
out float v_shade;
out float v_tintA;
out float v_shadeBack;
const float POS_DECODE_SCALE = 1.0 / 2048.0;
const float POS_DECODE_ORIGIN = 8.0;
// Local face directions (indexed by ni); [6] = no-shade sentinel. Mirrors entityShader.ts NORMALS.
const vec3 NORMALS[7] = vec3[7](
  vec3(0.0, -1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, -1.0),
  vec3(0.0, 0.0, 1.0), vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
const float SHADE[7] = float[7](0.5, 1.0, 0.8, 0.8, 0.6, 0.6, 1.0); // render/faceShade.ts DIRECTION_SHADE
vec3 srgbToLinear(vec3 c) {
  vec3 lo = c / 12.92;
  vec3 hi = pow((c + vec3(0.055)) / 1.055, vec3(2.4));
  return mix(hi, lo, vec3(lessThanEqual(c, vec3(0.04045))));
}
// Quantize a normal to the nearest axis → terrain's per-face shade.
float axisShade(vec3 n) {
  vec3 a = abs(n);
  if (a.y >= a.x && a.y >= a.z) return n.y > 0.0 ? SHADE[1] : SHADE[0];
  if (a.x >= a.z) return n.x > 0.0 ? SHADE[5] : SHADE[4];
  return n.z > 0.0 ? SHADE[3] : SHADE[2];
}
void main() {
  vec3 local = vec3(a_packedPos.xyz) * POS_DECODE_SCALE - vec3(POS_DECODE_ORIGIN);
  gl_Position = viewProj * (model * vec4(local, 1.0));${DEPTH_REMAP}
  v_uv = a_uv;
  v_color = srgbToLinear(a_color.rgb);
  v_tintA = a_color.a;
  uint ni = a_packedPos.w & 0xFFu;
  if (light0.w > 0.5) {
    // Vanilla diffuse item lighting (rotates with the model; uniform scale → normalize). front + back shade.
    vec3 n = normalize((model * vec4(NORMALS[int(ni)], 0.0)).xyz);
    float df = max(0.0, dot(light0.xyz, n)) + max(0.0, dot(light1.xyz, n));
    float db = max(0.0, dot(light0.xyz, -n)) + max(0.0, dot(light1.xyz, -n));
    v_shade = min(1.0, df * 0.6 + 0.4);
    v_shadeBack = min(1.0, db * 0.6 + 0.4);
  } else if (ni == 6u) {
    v_shade = 1.0;
    v_shadeBack = 1.0;
  } else {
    v_shade = axisShade(NORMALS[int(ni)]); // block-display geometry: local face direction
    v_shadeBack = v_shade;
  }
}`,
  fragment: `#version 300 es
precision highp float;
#ifndef ALPHA_TEST
#define ALPHA_TEST 1
#endif
#ifndef EMIT_ALPHA
#define EMIT_ALPHA 0
#endif
layout(std140) uniform Uniforms {
  mat4 viewProj;
  mat4 model;
  vec4 tint;
  vec4 flash;
  vec4 light0;
  vec4 light1;
};
uniform sampler2D atlasTex; // unit 0 (WGSL atlasTex@1 + atlasSamp@2)
in vec2 v_uv;
in vec3 v_color;
in float v_shade;
in float v_tintA;
in float v_shadeBack;
out vec4 fragColor;
void main() {
  vec4 tex = texture(atlasTex, v_uv);
#if ALPHA_TEST
  if (tex.a < 0.5) { discard; } // entities default to cutout
#endif
  float perFace = gl_FrontFacing ? v_shade : v_shadeBack;       // @builtin(front_facing)
  float shade = (light1.w > 0.5) ? perFace : v_shade;           // PER_FACE box lighting
  vec3 rgb = tex.rgb * v_color * tint.rgb * shade;
  rgb = mix(rgb, flash.rgb, flash.a);                            // white-flash / hurt overlay
#if EMIT_ALPHA
  float alpha = tex.a * v_tintA * tint.a;
#else
  float alpha = 1.0;
#endif
  fragColor = vec4(rgb, alpha);
}`,
  uniformBlocks: [{ name: "Uniforms", binding: 0 }],
  samplers: [{ name: "atlasTex", textureBinding: 1, samplerBinding: 2, unit: 0 }],
};

// --- end portal (label "portal") — render/entities/special/PortalEffect.ts -------------------------
// W5. A faithful transcription of vanilla's rendertype_end_portal: a base sky sample + 15/16 receding,
// scrolling, rotating layers, weighted by the COLORS palette. The ONE divergence from the WGSL (TRAP B.2):
// the WGSL flips Y (`screen.y = 1.0 - screen.y`) because WebGPU's @builtin(position) is top-down; GL's
// gl_FragCoord is already bottom-up (matching vanilla's screen-projective coord), so the twin OMITS the flip.
// The `transpose(portalLayerMat(...)) * q` mirrors the WGSL exactly (vanilla's GLSL uses a row-vector mul).
const PORTAL_TWIN: GlslTwin = {
  vertex: `#version 300 es
precision highp float;
layout(std140) uniform U {
  mat4 viewProj;
  float time;
  float resX;
  float resY;
};
layout(location = 0) in vec3 a_pos;
layout(location = 1) in float a_layers; // per-block layer count: 15 (end portal) / 16 (gateway)
flat out float v_layers;
void main() {
  gl_Position = viewProj * vec4(a_pos, 1.0);${DEPTH_REMAP}
  v_layers = a_layers;
}`,
  fragment: `#version 300 es
precision highp float;
layout(std140) uniform U {
  mat4 viewProj;
  float time;
  float resX;
  float resY;
};
uniform sampler2D skyTex;    // unit 0 (environment/end_sky base)
uniform sampler2D portalTex; // unit 1 (entity/end_portal layers)
flat in float v_layers;
out vec4 fragColor;
const float GAME_TIME_PER_SEC = 1.0 / 1200.0; // vanilla (gameTime % 24000)/24000
const vec3 COLORS[16] = vec3[16](
  vec3(0.022087, 0.098399, 0.110818), vec3(0.011892, 0.095924, 0.089485),
  vec3(0.027636, 0.101689, 0.100326), vec3(0.046564, 0.109883, 0.114838),
  vec3(0.064901, 0.117696, 0.097189), vec3(0.063761, 0.086895, 0.123646),
  vec3(0.084817, 0.111994, 0.166380), vec3(0.097489, 0.154120, 0.091064),
  vec3(0.106152, 0.131144, 0.195191), vec3(0.097721, 0.110188, 0.187229),
  vec3(0.133516, 0.138278, 0.148582), vec3(0.070006, 0.243332, 0.235792),
  vec3(0.196766, 0.142899, 0.214696), vec3(0.047281, 0.315338, 0.321970),
  vec3(0.204675, 0.390010, 0.302066), vec3(0.080955, 0.314821, 0.661491));
// Vanilla end_portal_layer(layer) — built column-major, exactly like the WGSL portalLayerMat.
mat4 portalLayerMat(float layer, float t) {
  mat4 scaleTranslate = mat4(
    vec4(0.5, 0.0, 0.0, 0.25),
    vec4(0.0, 0.5, 0.0, 0.25),
    vec4(0.0, 0.0, 1.0, 0.0),
    vec4(0.0, 0.0, 0.0, 1.0));
  mat4 translate = mat4(
    vec4(1.0, 0.0, 0.0, 17.0 / layer),
    vec4(0.0, 1.0, 0.0, (2.0 + layer / 1.5) * (t * 1.5)),
    vec4(0.0, 0.0, 1.0, 0.0),
    vec4(0.0, 0.0, 0.0, 1.0));
  float a = radians((layer * layer * 4321.0 + layer * 9.0) * 2.0);
  float cs = cos(a);
  float sn = sin(a);
  float sc = (4.5 - layer / 4.0) * 2.0;
  mat4 scaleRot = mat4(
    vec4(sc * cs, -sc * sn, 0.0, 0.0),
    vec4(sc * sn, sc * cs, 0.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0),
    vec4(0.0, 0.0, 0.0, 1.0));
  return scaleRot * translate * scaleTranslate;
}
void main() {
  float t = time * GAME_TIME_PER_SEC;
  // TRAP B.2: NO Y flip — gl_FragCoord.y is already bottom-up in GL, matching vanilla's screen coord.
  vec2 screen = gl_FragCoord.xy / vec2(max(resX, 1.0), max(resY, 1.0));
  vec4 q = vec4(screen, 0.0, 1.0);
  vec3 col = textureLod(skyTex, screen, 0.0).rgb * COLORS[0];
  int layers = int(v_layers);
  for (int i = 0; i < layers; i++) {
    vec4 p = transpose(portalLayerMat(float(i + 1), t)) * q; // GLSL row-vector q * end_portal_layer(i+1)
    vec2 uvL = p.xy / p.w;
    col += textureLod(portalTex, uvL, 0.0).rgb * COLORS[i];
  }
  fragColor = vec4(col, 1.0);
}`,
  uniformBlocks: [{ name: "U", binding: 0 }],
  samplers: [
    { name: "skyTex", textureBinding: 2, samplerBinding: 1, unit: 0 },
    { name: "portalTex", textureBinding: 3, samplerBinding: 1, unit: 1 }, // shares sampler@1 across two units
  ],
};

// --- device-internal PRESENT blit (W4) — NOT a renderer twin, not in GLSL_TWINS ---------------------
// W4 renders every pass into an offscreen SRGB8_ALPHA8 FBO so blending is linear (the attachment decodes
// on sample, blends in linear, re-encodes on store). To show it on the default canvas we DON'T blit:
// blitFramebuffer from an SRGB8_ALPHA8 read buffer to the LINEAR default framebuffer sRGB-DECODES on read
// but does NOT re-encode (the default buffer's COLOR_ENCODING is LINEAR), so a raw blit comes out too dark
// (the classic gamma double-darken — verified against the ES3 blit/sRGB rules, webgl-spec deqp
// es3fFramebufferBlitTests). Instead a full-screen triangle SAMPLES the offscreen sRGB texture (the sampler
// auto-decodes to linear) and the fragment shader RE-ENCODES linear→sRGB, writing those sRGB bytes to the
// linear default RGBA8 buffer — exactly the bytes WebGPU's `rgba8unorm-srgb` view would store (±1 LSB from
// the one extra 8-bit round trip; drawingBufferStorage(SRGB8_ALPHA8) removes even that — the W6 fast-path).
// The vertex stage needs NO attributes: it derives a screen-covering triangle from gl_VertexID (draw 3 verts).
export const PRESENT_VERTEX = `#version 300 es
precision highp float;
out vec2 v_uv;
void main() {
  // gl_VertexID 0,1,2 → uv (0,0),(2,0),(0,2) → clip (-1,-1),(3,-1),(-1,3): one triangle covering the viewport.
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  v_uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export const PRESENT_FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D uSrc; // the offscreen SRGB8_ALPHA8 color texture (sample auto-decodes sRGB→linear)
in vec2 v_uv;
out vec4 fragColor;
vec3 linearToSrgb(vec3 c) {
  vec3 lo = c * 12.92;
  vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
  return mix(hi, lo, vec3(lessThanEqual(c, vec3(0.0031308)))); // inverse of the twins' srgbToLinear
}
void main() {
  vec3 lin = texture(uSrc, v_uv).rgb;        // SRGB8 sample → linear
  fragColor = vec4(linearToSrgb(lin), 1.0);  // re-encode → sRGB bytes; alpha 1 (WebGPU canvas alphaMode 'opaque')
}`;

/**
 * Registry: the `label` passed to `createShaderModule(code, label)` → its GLSL twin. Multiple labels
 * can share one twin (the same WGSL is created under different debug names). This is now the COMPLETE
 * renderer shader roster (W5): terrain, entity, portal + the four easy W2 twins.
 */
export const GLSL_TWINS: Record<string, GlslTwin> = {
  overlay: COLOR_VERTEX_TWIN,
  explosion: COLOR_VERTEX_TWIN,
  "overlay-grid": GRID_TWIN,
  linebox: BOX_TWIN,
  "linebox-xray": BOX_TWIN,
  beam: BEAM_TWIN,
  terrain: TERRAIN_TWIN,
  entity: ENTITY_TWIN,
  portal: PORTAL_TWIN,
};
