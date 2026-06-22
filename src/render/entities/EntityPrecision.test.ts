// PREC-1 (entity parity) — the camera-relative origin path for the entity forward pass. Entities NEVER defer
// (TRAP PREC-1.D), so their precision must move in lockstep with terrain or they jitter against stable terrain.
//
//  - OFF (default): the per-entity uniform slot holds the WORLD viewProj + the model's UNMODIFIED world
//    transform → byte-identical to the pre-PREC-1 path.
//  - ON: the slot holds a translation-free viewProjRel + the model with its translation column pre-subtracted
//    by the rounded camera origin, so `model · local` lands near the camera and the f32 clip coord is stable.

import { describe, it, expect } from "vitest";
import { EntityRenderer, type EntityDraw, type Mat4 } from "./EntityRenderer";
import { FakeGraphicsDevice } from "../../core/testing/FakeGraphicsDevice";
import { TextureFormat, BufferUsage } from "../../core/GraphicsDevice";
import { TerrainPass } from "../../types";
import { Camera } from "../../camera/Camera";
import { translation, multiply, Mat4Frame } from "../../mesh/entity/mat4";

const f = Math.fround;

/** Build the translation-free viewProjRel = proj · (view with translation column zeroed). */
function viewProjRel(camera: Camera): Mat4 {
  const view = Float32Array.from(camera.viewMatrix());
  view[12] = 0; view[13] = 0; view[14] = 0;
  return multiply(camera.projectionMatrix(), view);
}

function setup() {
  const device = new FakeGraphicsDevice();
  const renderer = new EntityRenderer(device, TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
  const texture = device.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8, label: "atlas" });
  const mkDraw = (model: Mat4): EntityDraw => ({
    vertex: device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Vertex }),
    quadCount: 1,
    pass: TerrainPass.Solid,
    texture,
    model,
  });
  return { device, renderer, mkDraw };
}

/** Read the per-entity uniform slot 0: [16 vp][16 model] (the first 32 floats of the 256B slot). */
function readSlot0(device: FakeGraphicsDevice): { vp: Float32Array; model: Float32Array } {
  const h = device.bufferByLabel("entity-uniform");
  expect(h).not.toBeNull();
  const bytes = device.read(h!);
  const floats = new Float32Array(bytes.buffer, bytes.byteOffset, 32);
  return { vp: floats.slice(0, 16), model: floats.slice(16, 32) };
}

describe("PREC-1 entity camera-relative precision", () => {
  it("flag OFF: uploads the world viewProj + the UNMODIFIED model (byte-identical)", () => {
    const { device, renderer, mkDraw } = setup();
    const camera = new Camera();
    camera.target = [8, 8, 8];
    const model = translation(1000, 50, -2000);
    const vp = camera.viewProjection();
    expect(renderer.cameraRelative).toBe(false); // default-off

    renderer.render([mkDraw(model)], vp, camera.position, 256, 256, {}, camera.viewMatrix(), camera.projectionMatrix());

    const slot = readSlot0(device);
    for (let i = 0; i < 16; i++) expect(slot.vp[i]).toBe(f(vp[i]));          // world vp, bit-for-bit
    for (let i = 0; i < 16; i++) expect(slot.model[i]).toBe(f(model[i]));    // model UNCHANGED
  });

  it("flag ON: uploads viewProjRel + the model translation pre-subtracted by the rounded camera origin", () => {
    const { device, renderer, mkDraw } = setup();
    const camera = new Camera();
    const FAR = 30_000_000;
    camera.target = [FAR + 8, 8, 8];
    camera.distance = 40;
    renderer.cameraRelative = true;

    const model = translation(FAR, 0, 0); // a far-placed entity at the section origin
    const camPos = camera.position;
    const vp = camera.viewProjection();
    renderer.render([mkDraw(model)], vp, camPos, 256, 256, {}, camera.viewMatrix(), camera.projectionMatrix());

    const slot = readSlot0(device);
    const rel = viewProjRel(camera);
    // The uploaded vp is the translation-free relative matrix (NOT the world vp — TRAP PREC-1.A pairing).
    let differs = false;
    for (let i = 0; i < 16; i++) { expect(slot.vp[i]).toBeCloseTo(rel[i], 4); if (Math.abs(slot.vp[i] - vp[i]) > 1e-6) differs = true; }
    expect(differs).toBe(true);
    // The model's translation column (12..14) is shifted by −round(camPos); rotation/scale (0..11) unchanged.
    const ox = Math.round(camPos[0]), oy = Math.round(camPos[1]), oz = Math.round(camPos[2]);
    expect(slot.model[12]).toBe(f(FAR - ox));
    expect(slot.model[13]).toBe(f(0 - oy));
    expect(slot.model[14]).toBe(f(0 - oz));
    for (let i = 0; i < 12; i++) expect(slot.model[i]).toBe(f(model[i])); // linear block untouched
  });

  it("flag ON at 30,000,000: the relative f32 clip coord of a far entity vertex tracks its f64 reference", () => {
    const camera = new Camera();
    const FAR = 30_000_000;
    camera.target = [FAR + 8, 8, 8];
    camera.distance = 40;
    camera.aspect = 1;
    const camPos = camera.position;
    const ox = Math.round(camPos[0]), oy = Math.round(camPos[1]), oz = Math.round(camPos[2]);

    const rel = viewProjRel(camera);
    const worldVp = camera.viewProjection();

    // A model-local vertex (1,1,1) on an entity placed at the far section origin.
    const localX = 1, localY = 1, localZ = 1;

    // OFF path (today): world vp · (world model · local), all f32 — swamped at 30M.
    const worldModel = translation(FAR, 0, 0);
    const wWorld = transformPoint32(worldModel, localX, localY, localZ); // world position (f32)
    const worldClipF32 = mulVec4f32(worldVp, wWorld[0], wWorld[1], wWorld[2]);
    const worldClipF64 = mulVec4f64(worldVp, FAR + localX, localY, localZ);

    // ON path: viewProjRel · (relModel · local), small magnitude near the camera.
    const relModel = translation(FAR - ox, 0 - oy, 0 - oz);
    const wRel = transformPoint32(relModel, localX, localY, localZ);
    const relClipF32 = mulVec4f32(rel, wRel[0], wRel[1], wRel[2]);
    const relClipF64 = mulVec4f64(rel, (FAR - ox) + localX, (0 - oy) + localY, (0 - oz) + localZ);

    const ndc = (c: number[]) => [c[0] / c[3], c[1] / c[3]];
    const relErr = Math.hypot(ndc(relClipF32)[0] - ndc(relClipF64)[0], ndc(relClipF32)[1] - ndc(relClipF64)[1]);
    const worldErr = Math.hypot(ndc(worldClipF32)[0] - ndc(worldClipF64)[0], ndc(worldClipF32)[1] - ndc(worldClipF64)[1]);

    expect(relErr).toBeLessThan(1e-5);             // relative path keeps full precision
    expect(worldErr).toBeGreaterThan(1e-3);        // world path jitters at 30M
    expect(worldErr).toBeGreaterThan(relErr * 1000);
  });

  it("M6 FIX: Mat4Frame.Tw builds the world placement relative in DOUBLE — a NON-f32-exact far entity keeps full precision", () => {
    // The primary entity paths (factory sim entities + box-model mobs/carts) now place the model via
    // `Mat4Frame.Tw`, which subtracts the camera anchor in DOUBLE before the f32 store. So a fractional entity
    // at extreme coords lands EXACTLY, and the renderer uploads the model as-is (no second f32 subtraction).
    const FAR = 30_000_000;
    const anchorX = FAR + 4.3; // camera world position (the unrounded anchor EntityWorld passes)
    const m = new Mat4Frame();
    m.setAnchor(anchorX, 0, 0);

    const worldX = FAR + 0.5; // a moving / fractional entity at extreme coords
    const rel = m.Tw(worldX, 0, 0);
    expect(rel[12]).toBe(f(worldX - anchorX));           // the small relative X is f32-EXACT (subtracted in double)
    expect(rel[12]).toBeCloseTo(worldX - anchorX, 6);    // ≈ −3.8 — the .5 sub-grid offset survives

    // Contrast: the LEGACY path (absolute f32 model, THEN subtract — still used for BE-static / sortOf display
    // entities) loses the .5, because `fround(worldX)` quantizes it away at 30M before the subtraction.
    const legacyRel = f(translation(worldX, 0, 0)[12] - anchorX);
    expect(Math.abs(legacyRel - (worldX - anchorX))).toBeCloseTo(0.5, 6); // the residual the Tw path eliminates
  });
});

// ── f32 / f64 transform helpers (mirror entityShader.ts: out.clip = vp · (model · local)) ───────────────
function transformPoint32(m: Mat4, x: number, y: number, z: number): [number, number, number] {
  return [
    f(f(f(f(m[0] * x) + f(m[4] * y)) + f(m[8] * z)) + m[12]),
    f(f(f(f(m[1] * x) + f(m[5] * y)) + f(m[9] * z)) + m[13]),
    f(f(f(f(m[2] * x) + f(m[6] * y)) + f(m[10] * z)) + m[14]),
  ];
}
function mulVec4f32(m: ArrayLike<number>, x: number, y: number, z: number): number[] {
  const o = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) o[r] = f(f(f(f(m[r] * x) + f(m[4 + r] * y)) + f(m[8 + r] * z)) + m[12 + r]);
  return o;
}
function mulVec4f64(m: ArrayLike<number>, x: number, y: number, z: number): number[] {
  const o = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) o[r] = m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r];
  return o;
}
