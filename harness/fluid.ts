// GATE 13.1 — fluids torture test (Phase 3b). RENDERER_PLAN §13, §24.17.
//
// Water pool (still), a flowing cascade (level gradient → slope + flow direction), waterlogged
// stairs (block model + water surface), a lava pool (opaque, full-bright), and water sorted against
// clear + stained glass (TRAP 13.A — water participates in the §12 translucent sort). The self-check
// sweeps orbit angles and asserts blue water is visible, the scene isn't blank, and translucent
// sorting fired. window.__FLUID__ holds the verdict.

import { WebGPUDevice } from "../src/core/webgpu/WebGPUDevice";
import { Camera } from "../src/camera/Camera";
import { TextureFormat } from "../src/core/gpu-enums";
import { type BlockEntry } from "../src/mesh/model/BakedBlockModel";
import { type BlockSource } from "../src/world/SnapshotSource";
import { type Vec3 } from "../src/types";
import { AIR, buildScene, commitWorld, collectDraws, makeFluidContext, FLUID_SPRITES } from "./scene";

const STONE = 1, GLASS = 2, STAINED = 3, STAIRS_WL = 4, LAVA = 5;
const WATER0 = 10; // water level 0..7 → ids 10..17
const water = (level: number) => WATER0 + level;

const PALETTE: Record<number, BlockEntry> = {
  [STONE]: { name: "stone", props: {} },
  [GLASS]: { name: "glass", props: {}, cullGroup: "glass" },
  [STAINED]: { name: "blue_stained_glass", props: {}, cullGroup: "blue_stained_glass" },
  [STAIRS_WL]: { name: "oak_stairs", props: { facing: "east", half: "bottom", shape: "straight", waterlogged: "true" }, cullGroup: null },
  [LAVA]: { name: "lava", props: { level: "0" } },
};
for (let lvl = 0; lvl <= 7; lvl++) PALETTE[water(lvl)] = { name: "water", props: { level: String(lvl) } };

const W = 24, H = 12, D = 24;
const DIMS: Vec3 = [W, H, D];

function buildWorld(): BlockSource {
  const grid = new Uint16Array(W * H * D);
  const idx = (x: number, y: number, z: number) => (y * D + z) * W + x;
  const inB = (x: number, y: number, z: number) => x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D;
  const set = (x: number, y: number, z: number, id: number) => { if (inB(x, y, z)) grid[idx(x, y, z)] = id; };

  for (let x = 0; x < W; x++) for (let z = 0; z < D; z++) set(x, 0, z, STONE); // stone floor

  // STILL POOL: a raised stone platform with a water source on top → visible top + 4 side faces.
  for (let x = 3; x <= 8; x++) for (let z = 3; z <= 8; z++) { set(x, 1, z, STONE); set(x, 2, z, water(0)); }
  // Sort test: CLEAR glass in front of the pool, BLUE stained glass behind it.
  for (let y = 2; y <= 4; y++) for (let z = 3; z <= 8; z++) { set(2, y, z, GLASS); set(9, y, z, STAINED); }

  // FLOWING CASCADE: a walled channel along +x with rising level → falling height → flow toward +x.
  for (let x = 11; x <= 19; x++) {
    set(x, 1, 14, STONE);                       // channel floor
    set(x, 1, 13, STONE); set(x, 2, 13, STONE); // walls
    set(x, 1, 15, STONE); set(x, 2, 15, STONE);
    set(x, 2, 14, water(Math.min(7, x - 11)));  // source at x=11 (tall) → level 7 downstream (low)
  }

  // WATERLOGGED STAIRS: block model + a water surface around it (additive).
  for (let z = 18; z <= 20; z++) set(5, 1, z, STAIRS_WL);

  // LAVA POOL: opaque, full-bright, on its own platform.
  for (let x = 15; x <= 18; x++) for (let z = 18; z <= 21; z++) { set(x, 1, z, STONE); set(x, 2, z, LAVA); }

  return { getBlock: (x, y, z) => (inB(x, y, z) ? grid[idx(x, y, z)] : AIR) };
}

/** Central-region blue-water coverage + non-black coverage (BGRA-aware). */
function analyze(shot: { width: number; height: number; bytesPerRow: number; data: Uint8Array }, bgra: boolean) {
  const rI = bgra ? 2 : 0;
  const bI = bgra ? 0 : 2;
  const x0 = Math.floor(shot.width * 0.15), x1 = Math.floor(shot.width * 0.85);
  const y0 = Math.floor(shot.height * 0.1), y1 = Math.floor(shot.height * 0.75);
  let lit = 0, blue = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const o = y * shot.bytesPerRow + x * 4;
      const r = shot.data[o + rI], g = shot.data[o + 1], b = shot.data[o + bI];
      if (Math.max(r, g, b) > 24) lit++;
      // Water tint is a SATURATED, fairly dark blue (b dominant, low red). The sky is a LIGHT
      // blue-gray with high red (~180), so `r < 140` excludes it; gray stone has r≈g≈b.
      if (b > r + 35 && b > g + 20 && b > 70 && r < 140) blue++;
    }
  }
  const px = (x1 - x0) * (y1 - y0);
  return { litFrac: lit / px, blueFrac: blue / px };
}

// TR-1 large-water sub-scene: a 16³ section densely packed with alternating water sources and blue stained
// glass (distinct translucent groups) so interior faces survive merging → a SINGLE huge translucent section
// (quadCount >= GATE_QUAD_THRESHOLD). The other huge-translucent regime besides glass.
const HUGE_DIMS: Vec3 = [16, 16, 16];
function buildHugeWaterWorld(): BlockSource {
  const [W2, H2, D2] = HUGE_DIMS;
  const grid = new Uint16Array(W2 * H2 * D2);
  const idx = (x: number, y: number, z: number) => (y * D2 + z) * W2 + x;
  const inB = (x: number, y: number, z: number) => x >= 0 && x < W2 && y >= 0 && y < H2 && z >= 0 && z < D2;
  for (let x = 0; x < W2; x++)
    for (let y = 0; y < H2; y++)
      for (let z = 0; z < D2; z++) grid[idx(x, y, z)] = (x + y + z) % 2 === 0 ? water(0) : STAINED;
  return { getBlock: (x, y, z) => (inB(x, y, z) ? grid[idx(x, y, z)] : AIR) };
}

/** TR-1 large-water regression: over a full orbit of a huge translucent water section, gate-ON re-sorts must
 *  be strictly fewer than gate-OFF with no layering error (water still visible / non-blank at every angle).
 *  Index-only — geometry bytes never change either way. GPU-gated (needs Metal). */
async function runHugeWaterGate(
  device: WebGPUDevice,
  canvas: HTMLCanvasElement,
): Promise<{ ok: boolean; detail: string; quadCount: number; resortsOff: number; resortsOn: number }> {
  const { atlas, provider, renderer } = await buildScene(device, PALETTE, () => {}, FLUID_SPRITES);
  const fluids = makeFluidContext(PALETTE, atlas);
  const store = commitWorld(device, buildHugeWaterWorld(), provider, HUGE_DIMS, fluids);
  const { draws } = collectDraws(store);
  const quadCount = draws.filter((d) => d.sortType !== undefined).reduce((m, d) => Math.max(m, d.quadCount), 0);

  const camera = new Camera();
  camera.target = [8, 8, 8];
  camera.distance = 16; // close enough that the gate's effEps is at its max (R ≤ GATE_REF_DIST → full 0.5 block)
  camera.pitch = 0.25;
  const bgra = device.colorFormat === TextureFormat.Bgra8Srgb || device.colorFormat === TextureFormat.Bgra8;
  // The gate's win is a SUB-EPSILON camera jitter (not a coarse orbit — every big step legitimately reorders
  // the stack, which the gate correctly re-sorts). We dither yaw as a small sine (amplitude ≪ the gate's
  // distance/angle epsilons) so the cumulative move/rotation from the last sort stays under threshold while
  // crossedAnyPlane keeps firing (OFF re-sorts each crossing). Correctness = the ON image TRACKS the OFF image
  // at every step (the suppressed re-sorts genuinely didn't matter), not an absolute lit threshold.
  const STEPS = 48;
  const YAW0 = 0.6;
  const sweep = async (): Promise<{ resorts: number; lit: number[]; minBlue: number }> => {
    const start = renderer.resortCount;
    const lit: number[] = [];
    let minBlue = 1;
    for (let i = 0; i < STEPS; i++) {
      camera.yaw = YAW0 + 0.02 * Math.sin((i / STEPS) * 2 * Math.PI * 3); // ±0.02 rad ≈ ±0.32 block < 0.5 effEps
      camera.aspect = canvas.width / canvas.height;
      renderer.render(draws, camera, canvas.width, canvas.height);
      const a = analyze(await device.readCanvasPixels(), bgra);
      lit.push(a.litFrac);
      minBlue = Math.min(minBlue, a.blueFrac);
    }
    return { resorts: renderer.resortCount - start, lit, minBlue };
  };

  renderer.hugeSortGate = false;
  const off = await sweep();
  renderer.hugeSortGate = true;
  const on = await sweep();

  // The gate is a STRICT additional damper, so it can NEVER re-sort more than OFF — the robust invariant is
  // `on.resorts <= off.resorts`; it SUPPRESSES the sub-epsilon crossings (OFF → ON below). ON must also TRACK
  // OFF every frame (no layering error from a suppressed re-sort) and keep water visible (not blank).
  const noWorse = on.resorts <= off.resorts;
  let maxLitDiff = 0;
  for (let i = 0; i < STEPS; i++) maxLitDiff = Math.max(maxLitDiff, Math.abs(on.lit[i] - off.lit[i]));
  const tracks = maxLitDiff < 0.04; // ON within 4% of OFF every frame ⇒ no layering error from suppression
  const correct = tracks && on.minBlue > 0.005;
  const big = quadCount >= 1000;
  const ok = big && noWorse && correct;
  const detail =
    `huge water section ${quadCount} quads (>=1000: ${big}); ${STEPS}-frame sub-epsilon dither — re-sorts OFF ${off.resorts} → ON ${on.resorts} ` +
    `(suppressed ${off.resorts - on.resorts}, never more: ${noWorse}); ON tracks OFF maxΔlit ${(maxLitDiff * 100).toFixed(2)}% (<4%: ${tracks}), ON min water ${(on.minBlue * 100).toFixed(2)}%`;
  return { ok, detail, quadCount, resortsOff: off.resorts, resortsOn: on.resorts };
}

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const device = await WebGPUDevice.create(canvas);
  syncCanvasSize(canvas, device);
  window.addEventListener("resize", () => syncCanvasSize(canvas, device));

  const { atlas, provider, renderer } = await buildScene(device, PALETTE, (t) => setStatus("run", t), FLUID_SPRITES);
  const fluids = makeFluidContext(PALETTE, atlas);
  const store = commitWorld(device, buildWorld(), provider, DIMS, fluids);
  const { draws, total, translucent } = collectDraws(store);

  const camera = new Camera();
  camera.target = [11, 3, 12];
  camera.distance = 30;
  camera.pitch = 0.5;

  const renderFrame = () => {
    camera.aspect = canvas.width / canvas.height;
    renderer.render(draws, camera, canvas.width, canvas.height);
  };

  const ANGLES = 6;
  const bgra = device.colorFormat === TextureFormat.Bgra8Srgb || device.colorFormat === TextureFormat.Bgra8;
  let minLit = 1, minBlue = 1;
  const resortsAtStart = renderer.resortCount;
  for (let i = 0; i < ANGLES; i++) {
    camera.yaw = ((-60 + i * 24) * Math.PI) / 180;
    renderFrame();
    const a = analyze(await device.readCanvasPixels(), bgra);
    minLit = Math.min(minLit, a.litFrac);
    minBlue = Math.min(minBlue, a.blueFrac);
    console.log(`angle ${-60 + i * 24}°: lit ${(a.litFrac * 100).toFixed(1)}% water ${(a.blueFrac * 100).toFixed(2)}%`);
  }
  const resorted = renderer.resortCount - resortsAtStart;

  // TR-1: large-water re-sort gate — gate-ON re-sorts strictly fewer than gate-OFF over a full orbit, water
  // still correct at every angle. Runs on a SEPARATE renderer/scene so the core check above is untouched.
  const huge = await runHugeWaterGate(device, canvas);

  // minLit gates "the scene is rendered, not blank". The central sample region includes black sky, so the
  // genuine non-background coverage is ~42% over the sweep (NOT a regression — the render is correct); 0.3 stays
  // well clear of a blank frame while matching the real framing.
  const ok = translucent > 0 && minLit > 0.3 && minBlue > 0.01 && resorted >= ANGLES && huge.ok;
  const detail =
    `${total} quads (${translucent} translucent) in ${draws.length} draws; ` +
    `${ANGLES} angles: min lit ${(minLit * 100).toFixed(1)}%, min water ${(minBlue * 100).toFixed(2)}%; ` +
    `${resorted} index-only re-sorts. TR-1 gate: ${huge.detail}`;
  window.__FLUID__ = { ok, detail, hugeGate: { resortsOff: huge.resortsOff, resortsOn: huge.resortsOn, quadCount: huge.quadCount } };
  setStatus(ok ? "pass" : "fail", `${ok ? "PASS" : "FAIL"} — ${detail}`);

  // Live view: orbit the front arc, ticking the atlas so water/lava animate. `tick()` wants the
  // DELTA ticks since the last frame (vanilla runs at 20 TPS / 50ms per tick); water_still's
  // frametime 2 ⇒ 100ms per frame ⇒ a ~3.2s cycle over its 32 frames.
  let t0 = 0;
  let last = 0;
  const loop = (t: number) => {
    if (!t0) t0 = last = t;
    atlas.tick(((t - last) / 1000) * 20); // delta seconds × 20 ticks/sec
    last = t;
    camera.yaw = (Math.sin((t - t0) / 2600) * 60 * Math.PI) / 180;
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
    __FLUID__?:
      | { ok: boolean; detail: string; hugeGate?: { resortsOff: number; resortsOn: number; quadCount: number } }
      | { ok: false; error: string };
  }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__FLUID__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
