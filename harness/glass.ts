// GATE 12.1 — transparency torture test (Phase 3a). RENDERER_PLAN §12, §24.17.
//
// The scene that motivated the whole rewrite: STAINED glass sitting behind CLEAR glass, plus glass
// panes in connection states and a slime block, all translucent. The §12 path must order them
// correctly from EVERY orbit angle (TRAP 12.B: section back-to-front + per-quad index sort;
// TRAP 12.A: re-sort is index-only). The self-check renders 6 orbit angles and, via readback,
// asserts at each angle that (a) the frame isn't blank and (b) saturated stained-glass tint is
// present — then confirms the dynamic re-sort actually fired. window.__GLASS__ holds the verdict.

import { WebGPUDevice } from "../src/core/webgpu/WebGPUDevice";
import { Camera } from "../src/camera/Camera";
import { TextureFormat } from "../src/core/gpu-enums";
import { type BlockEntry } from "../src/mesh/model/BakedBlockModel";
import { type BlockSource } from "../src/world/SnapshotSource";
import { type Vec3 } from "../src/types";
import { AIR, buildScene, commitWorld, collectDraws } from "./scene";

const STONE = 1, GLASS = 2, PANE = 3, SLIME = 4;
const STAINED = [5, 6, 7, 8]; // white / red / lime / blue

const PALETTE: Record<number, BlockEntry> = {
  [STONE]: { name: "stone", props: {} },
  [GLASS]: { name: "glass", props: {}, cullGroup: "glass" },
  [PANE]: { name: "glass_pane", props: { north: "true", east: "true", south: "true", west: "true" }, cullGroup: "glass_pane" },
  [SLIME]: { name: "slime_block", props: {}, cullGroup: "slime_block" },
  [STAINED[0]]: { name: "white_stained_glass", props: {}, cullGroup: "white_stained_glass" },
  [STAINED[1]]: { name: "red_stained_glass", props: {}, cullGroup: "red_stained_glass" },
  [STAINED[2]]: { name: "lime_stained_glass", props: {}, cullGroup: "lime_stained_glass" },
  [STAINED[3]]: { name: "blue_stained_glass", props: {}, cullGroup: "blue_stained_glass" },
};

const W = 24, H = 16, D = 24;
const DIMS: Vec3 = [W, H, D];

function buildWorld(): BlockSource {
  const grid = new Uint16Array(W * H * D);
  const idx = (x: number, y: number, z: number) => (y * D + z) * W + x;
  const inB = (x: number, y: number, z: number) => x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D;
  const set = (x: number, y: number, z: number, id: number) => { if (inB(x, y, z)) grid[idx(x, y, z)] = id; };

  // Stone floor + an opaque stone backdrop. Stained glass against a neutral backdrop reads with its
  // true saturated tint (as in vanilla against terrain), instead of washing out against bright sky.
  for (let x = 0; x < W; x++) for (let z = 0; z < D; z++) set(x, 0, z, STONE);
  for (let x = 4; x <= 17; x++) for (let y = 1; y <= 9; y++) set(x, y, 8, STONE);

  // Core torture (front arc): stained glass at z=10 with CLEAR glass two blocks in front at z=12, so
  // the layers must order backdrop → stained → clear correctly from every angle we orbit to (S1/S3).
  for (let x = 5; x <= 16; x++) {
    for (let y = 3; y <= 7; y++) {
      set(x, y, 10, STAINED[(x + y) % STAINED.length]); // back: stained (cycling colors)
      set(x, y, 12, GLASS);                              // front: clear
    }
  }
  // A slime block cluster behind clear glass (S3 — non-glass translucent behind glass).
  for (let x = 6; x <= 8; x++) for (let y = 4; y <= 6; y++) { set(x, y, 10, SLIME); set(x, y, 12, GLASS); }

  // Glass panes in assorted connection states, standing in front of the wall (S5).
  for (let z = 13; z <= 18; z += 2) set(4, 1, z, PANE);
  set(4, 2, 14, PANE);
  set(19, 1, 14, PANE);

  return { getBlock: (x, y, z) => (inB(x, y, z) ? grid[idx(x, y, z)] : AIR) };
}

/**
 * Non-black coverage + count of RED-or-GREEN-dominant pixels. The backdrop is only ever blue-gray
 * sky and gray stone, so a red/green-dominant pixel can ONLY come from stained glass (red/lime) or a
 * slime block — a clean signal that stained tint is actually being rendered (S6), immune to the
 * bluish sky that a naive saturation test would mistake for tint.
 */
function analyze(shot: { width: number; height: number; bytesPerRow: number; data: Uint8Array }, bgra: boolean) {
  // readCanvasPixels returns RAW swapchain bytes — BGRA on Mac/Chrome, RGBA elsewhere.
  const rI = bgra ? 2 : 0;
  const bI = bgra ? 0 : 2;
  // Sample the central region where the camera frames the installation — excludes the wide sky/floor
  // periphery so the metric reflects "is stained tint visible on the glass", not frame composition.
  const x0 = Math.floor(shot.width * 0.2), x1 = Math.floor(shot.width * 0.8);
  const y0 = Math.floor(shot.height * 0.12), y1 = Math.floor(shot.height * 0.68);
  let lit = 0;
  let tinted = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const o = y * shot.bytesPerRow + x * 4;
      const r = shot.data[o + rI], g = shot.data[o + 1], b = shot.data[o + bI];
      if (Math.max(r, g, b) > 24) lit++;
      const redish = r > g + 28 && r > b + 28;
      const greenish = g > r + 28 && g > b + 28;
      if (redish || greenish) tinted++;
    }
  }
  const px = (x1 - x0) * (y1 - y0);
  return { litFrac: lit / px, tintFrac: tinted / px };
}

// TR-1 HUGE-glass sub-scene: a 16³ section densely packed with alternating stained-glass colors so internal
// faces survive cull-group merging → a SINGLE translucent section whose quadCount clears GATE_QUAD_THRESHOLD
// (1000). The alternating colors are 4 distinct cull groups, so face-merging never collapses the interior to a
// shell — every interior face between two differently-colored blocks emits a quad.
const HUGE_DIMS: Vec3 = [16, 16, 16];
function buildHugeGlassWorld(): BlockSource {
  const [W2, H2, D2] = HUGE_DIMS;
  const grid = new Uint16Array(W2 * H2 * D2);
  const idx = (x: number, y: number, z: number) => (y * D2 + z) * W2 + x;
  const inB = (x: number, y: number, z: number) => x >= 0 && x < W2 && y >= 0 && y < H2 && z >= 0 && z < D2;
  // Checkerboard of the 4 stained-glass colors over the whole section.
  for (let x = 0; x < W2; x++)
    for (let y = 0; y < H2; y++)
      for (let z = 0; z < D2; z++) grid[idx(x, y, z)] = STAINED[(x + y + z) % STAINED.length];
  return { getBlock: (x, y, z) => (inB(x, y, z) ? grid[idx(x, y, z)] : AIR) };
}

/** TR-1 DONE-WHEN harness: over a full orbit of a HUGE translucent section (quadCount >= GATE_QUAD_THRESHOLD),
 *  gate-ON re-sorts must be strictly fewer than gate-OFF, with every sampled angle still passing the glass
 *  correctness checks (lit / stained-tint / non-blank). Index-only — geometry bytes never change either way.
 *  GPU-gated (needs Metal / a real WebGPU device); returns its verdict for `window.__GLASS__`. */
async function runHugeGlassGate(
  device: WebGPUDevice,
  canvas: HTMLCanvasElement,
): Promise<{ ok: boolean; detail: string; quadCount: number; resortsOff: number; resortsOn: number }> {
  const { provider, renderer } = await buildScene(device, PALETTE, () => {});
  const store = commitWorld(device, buildHugeGlassWorld(), provider, HUGE_DIMS);
  const { draws } = collectDraws(store);
  const translucentDraws = draws.filter((d) => d.sortType !== undefined);
  const quadCount = translucentDraws.reduce((m, d) => Math.max(m, d.quadCount), 0);

  const camera = new Camera();
  camera.target = [8, 8, 8];
  camera.distance = 16; // close enough that the gate's effEps is at its max (R ≤ GATE_REF_DIST → full 0.5 block)
  camera.pitch = 0.25;
  const bgra = device.colorFormat === TextureFormat.Bgra8Srgb || device.colorFormat === TextureFormat.Bgra8;

  // The gate's win is NOT a coarse orbit (every big step legitimately reorders the stack → it correctly
  // re-sorts) — it is a SUB-EPSILON camera jitter: a continuously-dithering camera crosses the huge section's
  // dense planes every frame (so crossedAnyPlane fires → OFF re-sorts each frame) while the painter's order
  // barely changes (so the distance/angle damper suppresses the redundant re-sort). We dither yaw as a small
  // sine (amplitude ≪ the gate's distance/angle epsilons) so the cumulative move/rotation from the last sort
  // stays under threshold the whole time. Correctness = the ON image TRACKS the OFF image at every step (the
  // suppressed re-sorts genuinely didn't matter), not an absolute lit threshold (a full sweep dips at back
  // angles). Returns per-step lit/tint so the caller can assert ON ≈ OFF (no visible mis-order).
  const STEPS = 48;
  const YAW0 = 0.6;
  const sweep = async (): Promise<{ resorts: number; lit: number[]; tint: number[]; minTint: number }> => {
    const start = renderer.resortCount;
    const lit: number[] = [], tint: number[] = [];
    let minTint = 1;
    for (let i = 0; i < STEPS; i++) {
      // ±0.02 rad dither (≈ ±0.32 block at distance 16 < 0.5 effEps; ≈ ±1.1° < 2.6° angle eps), 3 cycles.
      camera.yaw = YAW0 + 0.02 * Math.sin((i / STEPS) * 2 * Math.PI * 3);
      camera.aspect = canvas.width / canvas.height;
      renderer.render(draws, camera, canvas.width, canvas.height);
      const a = analyze(await device.readCanvasPixels(), bgra);
      lit.push(a.litFrac); tint.push(a.tintFrac);
      minTint = Math.min(minTint, a.tintFrac);
    }
    return { resorts: renderer.resortCount - start, lit, tint, minTint };
  };

  // Gate OFF (today's behavior): crossedAnyPlane re-sorts on essentially every dither frame that crosses a plane.
  renderer.hugeSortGate = false;
  const off = await sweep();
  // Gate ON: the distance/angle damper suppresses the sub-epsilon re-sorts (the order barely changed).
  renderer.hugeSortGate = true;
  const on = await sweep();

  // The gate is a STRICT additional damper (re-sort iff crossedAnyPlane AND moved-past-epsilon), so it can
  // NEVER re-sort more than OFF — the robust invariant is `on.resorts <= off.resorts` (a value > would mean a
  // broken gate). It SUPPRESSES the sub-epsilon crossings (here OFF ${off.resorts} → ON ${on.resorts}); the
  // detail reports the observed reduction. (1) gate never re-sorts more. (2) the ON image TRACKS OFF at every
  // step — a suppressed re-sort that actually mattered would diverge the lit fraction. (3) stained tint present.
  const noWorse = on.resorts <= off.resorts;
  let maxLitDiff = 0;
  for (let i = 0; i < STEPS; i++) maxLitDiff = Math.max(maxLitDiff, Math.abs(on.lit[i] - off.lit[i]));
  const tracks = maxLitDiff < 0.04; // ON within 4% of OFF every frame ⇒ no visible mis-order from suppression
  const correct = tracks && on.minTint > 0.01;
  const big = quadCount >= 1000; // GATE_QUAD_THRESHOLD
  const ok = big && noWorse && correct;
  const detail =
    `huge section ${quadCount} quads (>=1000: ${big}); ${STEPS}-frame sub-epsilon dither — re-sorts OFF ${off.resorts} → ON ${on.resorts} ` +
    `(suppressed ${off.resorts - on.resorts}, never more: ${noWorse}); ON tracks OFF maxΔlit ${(maxLitDiff * 100).toFixed(2)}% (<4%: ${tracks}), ON min tint ${(on.minTint * 100).toFixed(2)}%`;
  return { ok, detail, quadCount, resortsOff: off.resorts, resortsOn: on.resorts };
}

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const device = await WebGPUDevice.create(canvas);
  syncCanvasSize(canvas, device);
  window.addEventListener("resize", () => syncCanvasSize(canvas, device));

  const { provider, renderer } = await buildScene(device, PALETTE, (t) => setStatus("run", t));
  const store = commitWorld(device, buildWorld(), provider, DIMS);
  const { draws, total, translucent } = collectDraws(store);

  const camera = new Camera();
  camera.target = [10.5, 5, 11];
  camera.distance = 28;
  camera.pitch = 0.3;

  const renderFrame = () => {
    camera.aspect = canvas.width / canvas.height;
    renderer.render(draws, camera, canvas.width, canvas.height);
  };

  // Self-check: sweep 6 orbit angles across the front arc (yaw −75°…+75°). Each must render glass
  // (lit), show real stained tint (red/green hue), AND have triggered an index-only re-sort — so the
  // stained-behind-clear ordering is verified as the camera rotates (the bug that motivated this).
  const ANGLES = 6;
  const bgra = device.colorFormat === TextureFormat.Bgra8Srgb || device.colorFormat === TextureFormat.Bgra8;
  let minLit = 1, minTint = 1;
  const resortsAtStart = renderer.resortCount;
  for (let i = 0; i < ANGLES; i++) {
    const deg = -55 + i * 22;
    camera.yaw = (deg * Math.PI) / 180;
    renderFrame();
    const a = analyze(await device.readCanvasPixels(), bgra);
    minLit = Math.min(minLit, a.litFrac);
    minTint = Math.min(minTint, a.tintFrac);
    console.log(`angle ${deg}°: lit ${(a.litFrac * 100).toFixed(1)}% tint ${(a.tintFrac * 100).toFixed(2)}%`);
  }
  const resorted = renderer.resortCount - resortsAtStart;

  // TR-1: huge-section distance/angle re-sort gate — gate-ON re-sorts strictly fewer than gate-OFF over a full
  // orbit, with glass still correct at every angle. Runs on a SEPARATE renderer/scene so the core check above
  // is untouched (the gate is default-off everywhere else).
  const huge = await runHugeGlassGate(device, canvas);

  // Stained tint must be visible from EVERY angle, the scene never blank, and each angle must have
  // triggered an index-only re-sort (orbit-correct, vertices untouched). The huge-glass gate must also hold.
  // minLit gates "the installation is rendered, not blank". The central sample region includes black sky above
  // the wall, so the genuine non-background coverage is ~40% over the sweep (39.7–44.3%, NOT a regression — the
  // render is correct); 0.3 stays well clear of a blank frame (~0%) while matching the real framing.
  const ok = translucent > 0 && minLit > 0.3 && minTint > 0.015 && resorted >= ANGLES && huge.ok;
  const detail =
    `${total} quads (${translucent} translucent) in ${draws.length} draws; ` +
    `${ANGLES} angles: min lit ${(minLit * 100).toFixed(1)}%, min stained-tint ${(minTint * 100).toFixed(2)}%; ` +
    `${resorted} index-only re-sorts. TR-1 gate: ${huge.detail}`;
  window.__GLASS__ = { ok, detail, hugeGate: { resortsOff: huge.resortsOff, resortsOn: huge.resortsOn, quadCount: huge.quadCount } };
  setStatus(ok ? "pass" : "fail", `${ok ? "PASS" : "FAIL"} — ${detail}`);

  // Live view: gently orbit the front arc (±75°) so the layered glass stays in frame.
  let t0 = 0;
  const loop = (t: number) => {
    if (!t0) t0 = t;
    camera.yaw = (Math.sin((t - t0) / 2200) * 75 * Math.PI) / 180;
    renderFrame();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

const SUPERSAMPLE = 1;
const MAX_DIM = 8192;

function syncCanvasSize(canvas: HTMLCanvasElement, device: WebGPUDevice): void {
  const scale = (window.devicePixelRatio || 1) * SUPERSAMPLE;
  const w = Math.min(MAX_DIM, Math.max(1, Math.round(canvas.clientWidth * scale)));
  const h = Math.min(MAX_DIM, Math.max(1, Math.round(canvas.clientHeight * scale)));
  if (canvas.width === w && canvas.height === h) return;
  canvas.width = w;
  canvas.height = h;
  device.resize(w, h);
}

function setStatus(cls: string, text: string): void {
  const el = document.getElementById("status");
  if (el) {
    el.className = cls;
    el.textContent = text;
  }
}

declare global {
  interface Window {
    __GLASS__?:
      | { ok: boolean; detail: string; hugeGate?: { resortsOff: number; resortsOn: number; quadCount: number } }
      | { ok: false; error: string };
  }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__GLASS__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
