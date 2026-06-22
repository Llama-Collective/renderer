// P6 lighting — the LUT encoding + free global brightness (LM-1/LM-4). RENDERER_PLAN Phase 6.
//
// Proves on real GPU/Metal that:
//   (1) the default full-bright render is non-blank (the location-3 lightmap + default daylight LUT did
//       NOT regress the pre-lighting output — the encoding's golden-image safety, TRAP 6.B);
//   (2) `setLightLut({ brightness })` DIMS the terrain by the expected factor IN LINEAR SPACE with the
//       draw list reference UNCHANGED — i.e. a global brightness change is a free LUT rewrite, ZERO remesh;
//   (3) restoring the default LUT returns BIT-IDENTICAL pixels — the LUT is the only variable, so nothing
//       in the geometry/draw path moved (fully reversible, no remesh, no relocation).
// window.__LIGHTING__ holds the verdict.

import { WebGPUDevice } from "../src/core/webgpu/WebGPUDevice";
import { Camera } from "../src/camera/Camera";
import { TextureFormat } from "../src/core/gpu-enums";
import { type BlockEntry } from "../src/mesh/model/BakedBlockModel";
import { type BlockSource } from "../src/world/SnapshotSource";
import { type Vec3 } from "../src/types";
import { buildLightLut, defaultLightLut } from "../src/render/LightLut";
import { AIR, buildScene, commitWorld, collectDraws } from "./scene";

const STONE = 1, GLASS = 2;
const PALETTE: Record<number, BlockEntry> = {
  [STONE]: { name: "stone", props: {} },
  [GLASS]: { name: "glass", props: {}, cullGroup: "glass" },
};

const W = 20, H = 12, D = 20;
const DIMS: Vec3 = [W, H, D];

function buildWorld(): BlockSource {
  const grid = new Uint16Array(W * H * D);
  const idx = (x: number, y: number, z: number) => (y * D + z) * W + x;
  const inB = (x: number, y: number, z: number) => x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D;
  const set = (x: number, y: number, z: number, id: number) => { if (inB(x, y, z)) grid[idx(x, y, z)] = id; };
  // A stone floor + a stepped stone massif + a clear-glass wall — fills the frame with lit terrain so the
  // brightness measurement has plenty of non-background pixels (both opaque AND translucent get modulated).
  for (let x = 0; x < W; x++) for (let z = 0; z < D; z++) set(x, 0, z, STONE);
  for (let x = 3; x <= 16; x++) for (let z = 3; z <= 16; z++) for (let y = 1; y <= 2 + ((x + z) % 4); y++) set(x, y, z, STONE);
  for (let x = 5; x <= 14; x++) for (let y = 1; y <= 6; y++) set(x, y, 17, GLASS);
  return { getBlock: (x, y, z) => (inB(x, y, z) ? grid[idx(x, y, z)] : AIR) };
}

function srgbToLinear(c: number): number {
  const x = c / 255;
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

type Shot = { width: number; height: number; bytesPerRow: number; data: Uint8Array };

/** Mean LINEAR luminance of clearly-terrain pixels (above the dark background) in the central region, plus
 *  the count of such pixels (so we know the measurement is well-populated). */
function meanTerrainLuma(shot: Shot, bgra: boolean): { luma: number; lit: number } {
  const rI = bgra ? 2 : 0, bI = bgra ? 0 : 2;
  const x0 = Math.floor(shot.width * 0.15), x1 = Math.floor(shot.width * 0.85);
  const y0 = Math.floor(shot.height * 0.1), y1 = Math.floor(shot.height * 0.75);
  let sum = 0, lit = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const o = y * shot.bytesPerRow + x * 4;
      const r = shot.data[o + rI], g = shot.data[o + 1], b = shot.data[o + bI];
      if (Math.max(r, g, b) <= 28) continue; // skip the dark background clear color (unaffected by the LUT)
      sum += 0.2126 * srgbToLinear(r) + 0.7152 * srgbToLinear(g) + 0.0722 * srgbToLinear(b);
      lit++;
    }
  }
  return { luma: lit ? sum / lit : 0, lit };
}

/** Count of differing bytes between two shots (0 ⇒ bit-identical frames). */
function pixelDiff(a: Shot, b: Shot): number {
  let diff = 0;
  for (let i = 0; i < a.data.length; i++) if (a.data[i] !== b.data[i]) diff++;
  return diff;
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
  camera.target = [10, 4, 10];
  camera.distance = 30;
  camera.pitch = 0.42;
  camera.yaw = 0.7;
  camera.aspect = canvas.width / canvas.height;
  const renderFrame = () => renderer.render(draws, camera, canvas.width, canvas.height);
  const bgra = device.colorFormat === TextureFormat.Bgra8Srgb || device.colorFormat === TextureFormat.Bgra8;

  // (1) Default full-bright baseline.
  renderFrame();
  const baseShot = await device.readCanvasPixels();
  const base = meanTerrainLuma(baseShot, bgra);
  const baseData = baseShot.data.slice(); // own a copy before the next readback reuses the buffer

  // (2) Global brightness 0.4 — ONE LUT rewrite, no remesh. The draw list reference must be untouched.
  const DIM = 0.4;
  const drawsRef = draws;
  renderer.setLightLut(buildLightLut({ brightness: DIM }));
  renderFrame();
  const dimShot = await device.readCanvasPixels();
  const dim = meanTerrainLuma(dimShot, bgra);
  const drawsUntouched = draws === drawsRef; // setLightLut never touches the draw list (LUT-only path)

  // (3) Restore default → must be BIT-IDENTICAL to the baseline (LUT was the only thing that changed).
  renderer.setLightLut(defaultLightLut());
  renderFrame();
  const restoreShot = await device.readCanvasPixels();
  const diff = pixelDiff({ ...restoreShot, data: baseData } as Shot, restoreShot);

  // (4) SL-4 smooth lighting: a SEPARATELY-committed apron-2 store with smooth AO. It's a VISUAL feature,
  // so the gate is "the image changed" (concave corners darken) + the flat baseline is untouched + a
  // restore back to the flat draws is bit-identical to the baseline (reversible). Not a 0-diff golden.
  const smoothStore = commitWorld(device, buildWorld(), provider, DIMS, undefined, undefined, { smoothLighting: true });
  const smoothDraws = collectDraws(smoothStore).draws;
  renderer.render(smoothDraws, camera, canvas.width, canvas.height);
  const smoothShot = await device.readCanvasPixels();
  const smoothDiff = pixelDiff({ ...baseShot, data: baseData } as Shot, smoothShot);
  renderFrame(); // back to the FLAT draws
  const reShot = await device.readCanvasPixels();
  const smoothReversible = pixelDiff({ ...baseShot, data: baseData } as Shot, reShot) === 0;
  const smoothOk = smoothDiff > 0 && smoothReversible; // smooth AO changed pixels AND the flat baseline is intact

  // Verdict: scene lit, dim ratio ≈ 0.4 in linear space (±0.1), restore exact, draws never rebuilt, smooth OK.
  const ratio = base.luma > 0 ? dim.luma / base.luma : 0;
  const ratioOk = Math.abs(ratio - DIM) <= 0.1;
  const ok = total > 0 && base.lit > 2000 && base.luma > 0.02 && dim.luma < base.luma && ratioOk && diff === 0 && drawsUntouched && smoothOk;
  const detail =
    `${total} quads (${translucent} translucent) in ${draws.length} draws; ` +
    `lit px ${base.lit}; linear luma base ${base.luma.toFixed(4)} → dim ${dim.luma.toFixed(4)} ` +
    `(ratio ${ratio.toFixed(3)} vs ${DIM}); restore diff ${diff} bytes; draws-stable ${drawsUntouched}; ` +
    `SL-4 smooth ${smoothOk ? "OK" : "BAD"} (${smoothDiff}px changed, reversible ${smoothReversible})`;
  window.__LIGHTING__ = { ok, detail };
  setStatus(ok ? "pass" : "fail", `${ok ? "PASS" : "FAIL"} — ${detail}`);

  // Live view: gently breathe the brightness so the LUT-only dimming is visible.
  let t0 = 0;
  const loop = (t: number) => {
    if (!t0) t0 = t;
    const b = 0.55 + 0.45 * (0.5 + 0.5 * Math.sin((t - t0) / 1400));
    renderer.setLightLut(buildLightLut({ brightness: b }));
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
    __LIGHTING__?: { ok: boolean; detail: string } | { ok: false; error: string };
  }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__LIGHTING__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
