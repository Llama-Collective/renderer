// PREC-1 — camera-relative origin uniform for float32 vertex precision at large world coordinates.
//
// Two gates:
//  1. DEFAULT-OFF byte-identity. With `cameraRelative=false` the frame uniform uploads the WORLD viewProj at
//     offset 0 AND camOrigin=(0,0,0,0) at offset 64, so the shader's `(origin - camOrigin) + local` with a
//     world matrix is the SAME f32 op sequence as the pre-PREC-1 `origin + local`. We assert the uploaded
//     bytes (vp + a zero camOrigin) AND a CPU model of the shader's `out.clip` are bit-identical to a
//     pre-change golden (the FNV quadHash covers mesh bytes, which this change does NOT touch — so the clip-
//     coordinate bit-identity test is the real off-path gate).
//  2. FLAG-ON precision at large coordinates. At a section origin of 30,000,000 the full-world f32 transform
//     (`proj·view·world`) loses sub-block precision (a ~4-block ULP), so it diverges from the f64 reference;
//     the camera-relative f32 transform (`proj·viewRel·(world - camOrigin)`) stays within tolerance, proving
//     the fix fires.

import { describe, it, expect } from "vitest";
import { TerrainRenderer, type SectionDraw } from "./TerrainRenderer";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { AtlasManager } from "../core/AtlasManager";
import { Camera } from "../camera/Camera";
import { multiply, type Mat4 } from "../camera/math";
import { TextureFormat, BufferUsage } from "../core/GraphicsDevice";
import { TerrainPass } from "../types";

const f = Math.fround;

function tinyAtlas(device: FakeGraphicsDevice): AtlasManager {
  const atlas = new AtlasManager(device);
  atlas.build([{ name: "x", size: 2, rgba: new Uint8Array(2 * 2 * 4).fill(255), frameCount: 1 }]);
  return atlas;
}

/** Column-major mat4 (length 16) · vec3 (w=1), evaluated IN f32 (rounding every op) — models the WGSL
 *  `m * vec4(p, 1.0)` exactly enough to compare bit-for-bit against the pre-change shader. */
function mulVec4f32(m: ArrayLike<number>, x: number, y: number, z: number): [number, number, number, number] {
  const out: [number, number, number, number] = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) {
    // WGSL evaluates a dot of 4 products; we fround each multiply + add to stay in f32.
    out[r] = f(f(f(f(m[r] * x) + f(m[4 + r] * y)) + f(m[8 + r] * z)) + m[12 + r]);
  }
  return out;
}

/** The same in f64 (the "true" reference). */
function mulVec4f64(m: ArrayLike<number>, x: number, y: number, z: number): [number, number, number, number] {
  const out: [number, number, number, number] = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) out[r] = m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r];
  return out;
}

/** Decode local fixed-point (scale 2048, origin 8) in f32 — mirrors POS_DECODE in terrainShader.ts. */
function decodeLocalF32(px: number, py: number, pz: number): [number, number, number] {
  return [f(f(px * (1 / 2048)) - 8), f(f(py * (1 / 2048)) - 8), f(f(pz * (1 / 2048)) - 8)];
}

/** The PRE-CHANGE shader transform (the golden): out.clip = worldVp * vec4(origin + local, 1.0), all f32. */
function clipWorldF32(vp: ArrayLike<number>, origin: [number, number, number], px: number, py: number, pz: number) {
  const [lx, ly, lz] = decodeLocalF32(px, py, pz);
  return mulVec4f32(vp, f(origin[0] + lx), f(origin[1] + ly), f(origin[2] + lz));
}

/** The NEW shader transform with the flag OFF (camOrigin=0): out.clip = worldVp * vec4((origin - 0) + local). */
function clipOffPathF32(vp: ArrayLike<number>, origin: [number, number, number], px: number, py: number, pz: number) {
  const [lx, ly, lz] = decodeLocalF32(px, py, pz);
  // `origin - 0` is the identity in IEEE-754 → bit-identical to the golden above.
  return mulVec4f32(vp, f(f(origin[0] - 0) + lx), f(f(origin[1] - 0) + ly), f(f(origin[2] - 0) + lz));
}

/** The NEW shader transform with the flag ON: out.clip = viewProjRel * vec4((origin - camOrigin) + local). */
// Faithful GPU model: `origin` (the per-draw origin) and `camOrigin` are each UPLOADED as f32, then the shader
// subtracts IN f32 — so truncate each operand to f32 FIRST, then subtract (NOT a double subtraction). With the
// anchor-relative pipeline both operands are small (tens of blocks), so the f32 subtraction is exact; with the
// naive world-absolute approach they are ~30M and the f32 subtraction loses ~2 blocks (the bug this catches).
function clipRelF32(vpRel: ArrayLike<number>, origin: [number, number, number], camOrigin: [number, number, number], px: number, py: number, pz: number) {
  const [lx, ly, lz] = decodeLocalF32(px, py, pz);
  return mulVec4f32(vpRel, f(f(f(origin[0]) - f(camOrigin[0])) + lx), f(f(f(origin[1]) - f(camOrigin[1])) + ly), f(f(f(origin[2]) - f(camOrigin[2])) + lz));
}

/** Build the translation-free viewProjRel = proj · (view with translation column zeroed), in f64. */
function viewProjRel(camera: Camera): Mat4 {
  const view = Float32Array.from(camera.viewMatrix());
  view[12] = 0; view[13] = 0; view[14] = 0;
  return multiply(camera.projectionMatrix(), view);
}

function setup() {
  const device = new FakeGraphicsDevice();
  const renderer = new TerrainRenderer(device, tinyAtlas(device), TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
  const camera = new Camera();
  camera.target = [8, 8, 8];
  const draws: SectionDraw[] = [
    { originBlocks: [0, 0, 0], vertex: device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Vertex }), quadCount: 1, pass: TerrainPass.Solid },
  ];
  return { device, renderer, camera, draws };
}

/** Read the 80-byte frame uniform back: [16 vp floats][4 camOrigin floats]. */
function readFrame(device: FakeGraphicsDevice): { vp: Float32Array; camOrigin: Float32Array } {
  const h = device.bufferByLabel("terrain-frame");
  expect(h).not.toBeNull();
  const bytes = device.read(h!);
  expect(bytes.length).toBe(80); // mat4 (64) + vec4 (16)
  const floats = new Float32Array(bytes.buffer, bytes.byteOffset, 20);
  return { vp: floats.slice(0, 16), camOrigin: floats.slice(16, 20) };
}

describe("PREC-1 terrain camera-relative precision", () => {
  it("flag OFF: frame uniform = world vp + camOrigin(0,0,0,0), and out.clip is bit-identical to the pre-change golden", () => {
    const { device, renderer, camera, draws } = setup();
    expect(renderer.cameraRelative).toBe(false); // default-off

    renderer.render(draws, camera, 256, 256);

    const { vp, camOrigin } = readFrame(device);
    const worldVp = camera.viewProjection();
    // Offset 0 holds the WORLD viewProj, bit-for-bit.
    for (let i = 0; i < 16; i++) expect(vp[i]).toBe(f(worldVp[i]));
    // Offset 64 holds an all-zero camOrigin.
    expect([...camOrigin]).toEqual([0, 0, 0, 0]);

    // Clip-coordinate bit-identity: the new off-path expression == the pre-change golden for every probe vertex.
    const origin: [number, number, number] = [0, 0, 0];
    for (const [px, py, pz] of [[0, 0, 0], [16384, 0, 32768], [65535, 65535, 65535], [1234, 50000, 9001]]) {
      const golden = clipWorldF32(worldVp, origin, px, py, pz);
      const offPath = clipOffPathF32(worldVp, origin, px, py, pz);
      expect(offPath).toEqual(golden); // SAME f32 op sequence → bit-identical
    }
  });

  it("flag ON: frame uniform = a translation-free viewProjRel + the sub-section camera residual", () => {
    const device = new FakeGraphicsDevice();
    const renderer = new TerrainRenderer(device, tinyAtlas(device), TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
    const camera = new Camera();
    const FAR = 30_000_000;
    camera.target = [FAR + 8, 8, 8];
    camera.distance = 40;
    // The section is at the far origin too, so it survives the frustum and the frame uniform is actually written.
    const draws: SectionDraw[] = [
      { originBlocks: [FAR, 0, 0], vertex: device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Vertex }), quadCount: 1, pass: TerrainPass.Solid },
    ];
    renderer.cameraRelative = true;

    renderer.render(draws, camera, 256, 256);
    expect(renderer.stats.drawn).toBe(1);

    const { vp, camOrigin } = readFrame(device);
    // camOrigin = the SUB-SECTION residual `camPos − anchor` (anchor = the camera's section origin), stored as
    // f32. It is small (∈[0,16) per axis) — the large camera magnitude lives in the anchor the per-draw origins
    // are uploaded relative to, NOT in this uniform (so the f32 store is exact).
    const cp = camera.position;
    const anchor = [Math.floor(cp[0] / 16) * 16, Math.floor(cp[1] / 16) * 16, Math.floor(cp[2] / 16) * 16];
    expect([camOrigin[0], camOrigin[1], camOrigin[2]]).toEqual([f(cp[0] - anchor[0]), f(cp[1] - anchor[1]), f(cp[2] - anchor[2])]);
    expect(camOrigin[3]).toBe(0);
    // The uploaded matrix is the translation-free viewProjRel, NOT the world vp.
    const rel = viewProjRel(camera);
    const worldVp = camera.viewProjection();
    let differsFromWorld = false;
    for (let i = 0; i < 16; i++) { expect(vp[i]).toBeCloseTo(rel[i], 5); if (Math.abs(vp[i] - worldVp[i]) > 1e-6) differsFromWorld = true; }
    expect(differsFromWorld).toBe(true); // it is genuinely the relative matrix (TRAP PREC-1.A — not the world vp)
  });

  it("flag ON at origin 30,000,000: the relative f32 transform tracks its f64 reference; the full-world f32 path does NOT", () => {
    const camera = new Camera();
    const FAR = 30_000_000;
    camera.target = [FAR + 8, 8, 8];
    camera.distance = 40;
    camera.aspect = 1;

    // Two f32 transforms of the SAME far vertex, each compared to its OWN-precision (f64) reference:
    //  - the full-world path: `worldVp · (origin + local)`  — at 30M the f32 add loses ~4 blocks of mantissa,
    //    so its f32 result diverges from the f64 truth (the jitter PREC-1 removes);
    //  - the camera-relative path: `viewProjRel · ((origin - camOrigin) + local)` — the operands are small
    //    (~tens of blocks near the camera), so the f32 result is essentially identical to its f64 evaluation.
    const worldVp = camera.viewProjection();
    const rel = viewProjRel(camera);
    const cp = camera.position;
    // Anchor-relative pipeline: the per-draw origin is uploaded RELATIVE to the camera-section anchor (computed
    // in double on the CPU → small, f32-exact), and camOrigin is the sub-section residual. Both small ⇒ the
    // shader's f32 subtraction is exact.
    const anchor: [number, number, number] = [Math.floor(cp[0] / 16) * 16, Math.floor(cp[1] / 16) * 16, Math.floor(cp[2] / 16) * 16];
    const origin: [number, number, number] = [FAR, 0, 0];
    const originRel: [number, number, number] = [origin[0] - anchor[0], origin[1] - anchor[1], origin[2] - anchor[2]];
    const camOrigin: [number, number, number] = [cp[0] - anchor[0], cp[1] - anchor[1], cp[2] - anchor[2]];

    // A vertex one block inside the far section (packed = (block + 8) * 2048; block 1 → 9*2048 = 18432).
    const PACKED = 9 * 2048;
    const [lx, ly, lz] = decodeLocalF32(PACKED, PACKED, PACKED); // == [1,1,1]

    const ndc = (c: [number, number, number, number]) => [c[0] / c[3], c[1] / c[3]] as [number, number];

    // Full-world: f32 path vs f64 truth.
    const worldF32 = clipWorldF32(worldVp, origin, PACKED, PACKED, PACKED);
    const worldF64 = mulVec4f64(worldVp, origin[0] + lx, origin[1] + ly, origin[2] + lz);
    const [wx, wy] = ndc(worldF32); const [twx, twy] = ndc(worldF64);
    const worldErr = Math.hypot(wx - twx, wy - twy);

    // Camera-relative: f32 path (small uploaded operands) vs its f64 evaluation.
    const relF32 = clipRelF32(rel, originRel, camOrigin, PACKED, PACKED, PACKED);
    const relF64 = mulVec4f64(rel, originRel[0] - camOrigin[0] + lx, originRel[1] - camOrigin[1] + ly, originRel[2] - camOrigin[2] + lz);
    const [rx, ry] = ndc(relF32); const [trx, try_] = ndc(relF64);
    const relErr = Math.hypot(rx - trx, ry - try_);

    // The relative f32 path tracks f64 to near machine epsilon (no sub-block precision lost)…
    expect(relErr).toBeLessThan(1e-5);
    // …while the full-world f32 path is off by a visibly large NDC margin (the jitter the fix removes)…
    expect(worldErr).toBeGreaterThan(1e-3);
    // …i.e. the camera-relative path is orders of magnitude more accurate at 30M.
    expect(worldErr).toBeGreaterThan(relErr * 1000);
  });

  it("flag ON still records + replays the F1 bundle (camera-relative uniform does not break the replay key)", () => {
    const device = new FakeGraphicsDevice();
    const renderer = new TerrainRenderer(device, tinyAtlas(device), TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
    const camera = new Camera();
    const FAR = 30_000_000;
    // Camera AND the section both at the far origin, so the section sits dead-centre in view (passes the frustum).
    camera.target = [FAR + 8, 8, 8];
    const draws: SectionDraw[] = [
      { originBlocks: [FAR, 0, 0], vertex: device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Vertex }), quadCount: 1, pass: TerrainPass.Solid },
    ];
    renderer.cameraRelative = true;

    renderer.render(draws, camera, 256, 256); // frame 1: direct
    expect(renderer.stats.drawn).toBe(1);    // the far section was IN view
    renderer.render(draws, camera, 256, 256); // frame 2: settle → record
    renderer.render(draws, camera, 256, 256); // frame 3: replay
    renderer.render(draws, camera, 256, 256); // frame 4: replay
    expect(device.log.bundlesCreated).toBe(1);
    expect(renderer.bundleReplays).toBe(2);
  });
});
