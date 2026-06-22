// Phase-2/3a end-to-end demo: real Minecraft models from pack.zip rendered through the baker.
// pack.zip → ResourcePack → ModelBaker → meshSection → SectionStore.commit → TerrainRenderer
// (solid + cutout + TRANSLUCENT glass). Proves GATE 2 and shows the §12 translucent path in context.
// Self-verifies via readback and exposes window.__DEMO__.

import { WebGPUDevice } from "../src/core/webgpu/WebGPUDevice";
import { Camera } from "../src/camera/Camera";
import { type BlockEntry } from "../src/mesh/model/BakedBlockModel";
import { type BlockSource } from "../src/world/SnapshotSource";
import { type Vec3 } from "../src/types";
import { AIR, buildScene, commitWorld, collectDraws } from "./scene";

const STONE = 1, DIRT = 2, GRASS = 3, PLANKS = 4, LOG = 5, COBBLE = 6, STAIRS = 7, LEAVES = 8, FENCE = 9;
const GLASS = 10, GLASS_PANE = 11;
const STAINED = [12, 13, 14, 15]; // white / red / lime / blue stained glass

const PALETTE: Record<number, BlockEntry> = {
  [STONE]: { name: "stone", props: {} },
  [DIRT]: { name: "dirt", props: {} },
  [GRASS]: { name: "grass_block", props: { snowy: "false" } },
  [PLANKS]: { name: "oak_planks", props: {} },
  [LOG]: { name: "oak_log", props: { axis: "y" } },
  [COBBLE]: { name: "cobblestone", props: {} },
  [STAIRS]: { name: "oak_stairs", props: { facing: "east", half: "bottom", shape: "straight" } },
  [LEAVES]: { name: "oak_leaves", props: { persistent: "false" } },
  [FENCE]: { name: "oak_fence", props: { north: "false", east: "false", south: "false", west: "false" } },
  [GLASS]: { name: "glass", props: {}, cullGroup: "glass" },
  [GLASS_PANE]: { name: "glass_pane", props: { north: "true", east: "false", south: "true", west: "false" }, cullGroup: "glass_pane" },
  [STAINED[0]]: { name: "white_stained_glass", props: {}, cullGroup: "white_stained_glass" },
  [STAINED[1]]: { name: "red_stained_glass", props: {}, cullGroup: "red_stained_glass" },
  [STAINED[2]]: { name: "lime_stained_glass", props: {}, cullGroup: "lime_stained_glass" },
  [STAINED[3]]: { name: "blue_stained_glass", props: {}, cullGroup: "blue_stained_glass" },
};

const W = 32, H = 24, D = 32;
const DIMS: Vec3 = [W, H, D];

function buildWorld(): BlockSource {
  const grid = new Uint16Array(W * H * D);
  const idx = (x: number, y: number, z: number) => (y * D + z) * W + x;
  const inB = (x: number, y: number, z: number) => x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D;
  const set = (x: number, y: number, z: number, id: number) => { if (inB(x, y, z)) grid[idx(x, y, z)] = id; };
  const top = (x: number, z: number) => { for (let y = H - 1; y >= 0; y--) if (grid[idx(x, y, z)] !== AIR) return y + 1; return 0; };

  // Heightmapped ground: stone → dirt → grass.
  for (let x = 0; x < W; x++) {
    for (let z = 0; z < D; z++) {
      const h = Math.max(3, Math.min(10, Math.round(5 + 2 * Math.sin(x * 0.35) + 2 * Math.cos(z * 0.3))));
      for (let y = 0; y < h; y++) set(x, y, z, y === h - 1 ? GRASS : y >= h - 3 ? DIRT : STONE);
    }
  }
  // A staircase of oak_stairs climbing east.
  for (let s = 0; s < 6; s++) set(6 + s, top(6 + s, 8) + s, 8, STAIRS);
  // Trees: oak_log trunk + oak_leaves canopy (cutout).
  for (const [tx, tz] of [[20, 22], [12, 24], [24, 10]]) {
    const g = top(tx, tz);
    for (let y = g; y < g + 4; y++) set(tx, y, tz, LOG);
    for (let dx = -2; dx <= 2; dx++) for (let dz = -2; dz <= 2; dz++) for (let dy = 0; dy <= 2; dy++) {
      if (Math.abs(dx) + Math.abs(dz) + dy <= 3) set(tx + dx, g + 3 + dy, tz + dz, LEAVES);
    }
  }
  // A plank platform with a cobblestone rim + fence posts.
  for (let x = 16; x < 20; x++) for (let z = 4; z < 8; z++) set(x, top(x, z), z, x === 16 || x === 19 || z === 4 || z === 7 ? COBBLE : PLANKS);
  set(16, top(16, 4) + 1, 4, FENCE);
  set(19, top(19, 7) + 1, 7, FENCE);

  // Translucent display (§12): a wall of stained glass with CLEAR glass two blocks in front, so the
  // stained tint must blend correctly THROUGH the clear glass from every orbit angle (GATE 12.1).
  for (let x = 12; x <= 19; x++) {
    for (let y = 11; y <= 14; y++) {
      set(x, y, 18, STAINED[(x + y) % STAINED.length]); // back: stained
      set(x, y, 20, GLASS);                              // front: clear
    }
  }
  // Glass panes in assorted connection states alongside the wall.
  for (let z = 16; z <= 22; z += 2) set(11, 11, z, GLASS_PANE);

  return { getBlock: (x, y, z) => (inB(x, y, z) ? grid[idx(x, y, z)] : AIR) };
}

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const device = await WebGPUDevice.create(canvas);
  syncCanvasSize(canvas, device);
  window.addEventListener("resize", () => syncCanvasSize(canvas, device));

  const { provider, renderer } = await buildScene(device, PALETTE, (t) => setStatus("run", t));
  const store = commitWorld(device, buildWorld(), provider, DIMS);
  const { draws, total, cutout, translucent } = collectDraws(store);

  const camera = new Camera();
  camera.target = [W / 2, 9, D / 2];
  camera.distance = 56;
  camera.pitch = 0.45;
  const renderFrame = () => {
    camera.aspect = canvas.width / canvas.height;
    renderer.render(draws, camera, canvas.width, canvas.height);
  };

  renderFrame();
  const shot = await device.readCanvasPixels();
  let terrain = 0;
  for (let p = 0; p < shot.width * shot.height; p++) {
    const o = p * 4;
    // Terrain = NON-background pixels. The clear color is near-black, so a pixel is terrain when it is brighter
    // than the dark background (max channel > 28, matching harness/lighting.ts). The old `min < 150` test
    // counted the black background ITSELF as terrain (min=0 < 150) → a bogus ~99% coverage.
    if (Math.max(shot.data[o], shot.data[o + 1], shot.data[o + 2]) > 28) terrain++;
  }
  const frac = terrain / (shot.width * shot.height);
  const ok = frac > 0.1 && frac < 0.98 && total > 0 && translucent > 0;
  const detail = `${total} quads (${cutout} cutout, ${translucent} translucent) in ${draws.length} draws; ${renderer.resortCount} re-sorts; terrain ${(frac * 100).toFixed(1)}% of ${shot.width}×${shot.height}`;
  window.__DEMO__ = { ok, detail };
  setStatus(ok ? "pass" : "fail", `${ok ? "PASS" : "FAIL"} — ${detail}`);

  let last = 0;
  const loop = (t: number) => {
    const dt = last ? (t - last) / 1000 : 0;
    last = t;
    camera.yaw += dt * 0.35;
    renderFrame();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

// Backing-store resolution = CSS size × devicePixelRatio × SUPERSAMPLE. DPR alone gives native
// retina sharpness; bump SUPERSAMPLE > 1 for extra-crisp supersampled AA (costs fill rate). Both
// axes are clamped to MAX_DIM so a large display × high DPR can't exceed the depth/canvas texture
// limit (the guaranteed WebGPU maxTextureDimension2D is 8192).
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
    __DEMO__?: { ok: boolean; detail: string } | { ok: false; error: string };
  }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__DEMO__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
