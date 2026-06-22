// PREC-1 far-offset harness — proof (on real Metal) that the camera-relative origin path removes float32
// vertex jitter at large world coordinates while staying byte-identical near the origin.
//
// GPU-gated: vitest (node) has no GPU, so this lives in the harness like stress/glass/fluid. It clones the
// stress world-gen but applies a configurable WORLD_OFFSET to BOTH the section origins and the camera, so the
// geometry sits ~30,000,000 blocks from origin (where the f32 ULP is ~4 blocks). Two proofs:
//
//   1. STABILITY (the fix): at the offset, translate the camera by a sub-block delta and re-render. With the
//      flag OFF the rendered vertex JITTERS (the f32 add snaps to the ~4-block grid → a measurable pixel
//      delta breathing with the camera); with the flag ON the image is stable (tiny pixel delta). The gate is
//      jitterOff > jitterOn by a wide margin — the documented 30M baseline vs the stable fix.
//   2. NEAR-ORIGIN IDENTITY: render the SAME world at offset 0 with the flag OFF vs ON. The camera-relative
//      transform must be ~0-pixel-diff at small coordinates (the off-default render is untouched there).

import { WebGPUDevice } from "../src/core/webgpu/WebGPUDevice";
import { Camera } from "../src/camera/Camera";
import { type BlockEntry } from "../src/mesh/model/BakedBlockModel";
import { type BlockSource } from "../src/world/SnapshotSource";
import { sectionKey, type SectionKey } from "../src/world/SectionKey";
import { type Vec3 } from "../src/types";
import { type SectionDraw } from "../src/render/TerrainRenderer";
import { AIR, buildScene, commitWorld, collectDraws, makeFluidContext, FLUID_SPRITES } from "./scene";

const STONE = 1, DIRT = 2, GRASS = 3;
const PALETTE: Record<number, BlockEntry> = {
  [STONE]: { name: "stone", props: {} },
  [DIRT]: { name: "dirt", props: {} },
  [GRASS]: { name: "grass_block", props: { snowy: "false" } },
};

const W = 64, H = 32, D = 64; // a compact world (the precision regime, not the cull regime — that's stress.ts)
const DIMS: Vec3 = [W, H, D];

// The far offset applied to section origins + camera. 30,000,000 is Minecraft's world border scale, where the
// f32 ULP (~4 blocks) utterly swamps the 1/2048 fixed-point local decode (the SEVERE jitter regime).
const WORLD_OFFSET = 30_000_000;

/** A small heightmap world rooted at world coordinate `(offX, 0, offZ)`: `getBlock` reads the grid at the
 *  LOCAL index `(wx - offX, wy, wz - offZ)`, so the same geometry can be placed near origin or 30M out. */
function buildWorld(offX: number, offZ: number): BlockSource {
  const grid = new Uint16Array(W * H * D);
  const idx = (x: number, y: number, z: number) => (y * D + z) * W + x;
  for (let x = 0; x < W; x++) {
    for (let z = 0; z < D; z++) {
      const h = Math.round(12 + 5 * Math.sin(x * 0.18) * Math.cos(z * 0.15));
      for (let y = 0; y < h; y++) grid[idx(x, y, z)] = y === h - 1 ? GRASS : y >= h - 3 ? DIRT : STONE;
    }
  }
  return {
    getBlock: (wx, wy, wz) => {
      const x = wx - offX, y = wy, z = wz - offZ;
      return x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D ? grid[idx(x, y, z)] : AIR;
    },
  };
}

/** The far-world section keys (block coords `[offX, 0, offZ]` … +dims, in section units). `commitWorld` scans
 *  `[0,dims)` from section 0, so far sections are committed via its `extraSections` hook with `dims=[0,0,0]`. */
function sectionsAt(offX: number, offZ: number): SectionKey[] {
  const keys: SectionKey[] = [];
  const sx0 = offX >> 4, sz0 = offZ >> 4;
  for (let X = 0; X < Math.ceil(W / 16); X++)
    for (let Y = 0; Y < Math.ceil(H / 16); Y++)
      for (let Z = 0; Z < Math.ceil(D / 16); Z++) keys.push(sectionKey(sx0 + X, Y, sz0 + Z));
  return keys;
}

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const device = await WebGPUDevice.create(canvas);
  syncCanvasSize(canvas, device);
  window.addEventListener("resize", () => syncCanvasSize(canvas, device));

  const { atlas, provider, renderer } = await buildScene(device, PALETTE, (t) => setStatus("run", t), FLUID_SPRITES);
  const fluids = makeFluidContext(PALETTE, atlas);

  // Two stores: one rooted at WORLD_OFFSET (far regime), one at logical origin (near regime, the identity gate).
  // Far sections are committed via `extraSections` (commitWorld's [0,dims) scan only covers near-origin sections).
  const NO_SCAN: Vec3 = [0, 0, 0];
  const farStore = commitWorld(device, buildWorld(WORLD_OFFSET, WORLD_OFFSET), provider, NO_SCAN, fluids, sectionsAt(WORLD_OFFSET, WORLD_OFFSET));
  const nearStore = commitWorld(device, buildWorld(0, 0), provider, [W, H, D], fluids);
  const farDraws: SectionDraw[] = collectDraws(farStore).draws;
  const nearDraws: SectionDraw[] = collectDraws(nearStore).draws;

  const camera = new Camera();
  camera.far = 4000;
  camera.distance = 60;
  camera.pitch = 0.4;
  camera.yaw = 0.6;
  renderer.bundleReplay = false; // direct-encode every frame so the A/B deltas are clean

  const renderFrame = (d: SectionDraw[]): void => {
    camera.aspect = canvas.width / canvas.height;
    renderer.render(d, camera, canvas.width, canvas.height);
  };

  type Pixels = { width: number; height: number; data: Uint8Array };
  const N = canvas.width * canvas.height;
  const pixelDiff = (a: Pixels, b: Pixels): number => {
    let n = 0;
    for (let p = 0; p < N; p++) {
      const o = p * 4;
      if (a.data[o] !== b.data[o] || a.data[o + 1] !== b.data[o + 1] || a.data[o + 2] !== b.data[o + 2]) n++;
    }
    return n;
  };

  // The proof is a GROUND-TRUTH comparison, not a frame-to-frame nudge (a sub-block nudge at 30M is BELOW the
  // ~2-block f32 ULP, so it can't surface the jitter). The SAME scene rendered near the origin (where f32 is
  // exact) is the reference. The camera-relative path must render the far-offset copy IDENTICALLY (the fix);
  // the off path at 30M visibly diverges (the f32 transform snaps geometry to the coarse ULP grid). WORLD_OFFSET
  // (30M) is a multiple of 16, so the camera-section anchors align exactly between the near and far worlds.
  const shoot = async (d: SectionDraw[], target: [number, number, number], cameraRelative: boolean): Promise<Pixels> => {
    renderer.cameraRelative = cameraRelative;
    camera.target = target;
    renderFrame(d);
    return device.readCanvasPixels();
  };
  const NEAR: [number, number, number] = [W / 2, 14, D / 2];
  const FAR: [number, number, number] = [WORLD_OFFSET + W / 2, 14, WORLD_OFFSET + D / 2];
  const groundTruth = await shoot(nearDraws, NEAR, false); // the scene near the origin (f32-precise reference)
  const nearOn = await shoot(nearDraws, NEAR, true); //   near-origin identity: the flag must be a ~no-op here
  const farOn = await shoot(farDraws, FAR, true); //       the FIX: far render must match the near ground truth
  const farOff = await shoot(farDraws, FAR, false); //     the BUG: off at 30M visibly diverges (jitter)

  const nearOnPct = (pixelDiff(groundTruth, nearOn) / N) * 100;
  const farOnPct = (pixelDiff(groundTruth, farOn) / N) * 100;
  const farOffPct = (pixelDiff(groundTruth, farOff) / N) * 100;

  // PASS: (1) near-origin flag on ≈ off (the default-off render is untouched); (2) the FIX — far-ON tracks the
  // near ground truth within fp tolerance; (3) the problem is real — far-OFF diverges by a wide margin. Gate is
  // RELATIVE (far-OFF ≫ far-ON) plus small absolute bounds on the two fixed paths.
  const nearOk = nearOnPct < 0.5;
  const fixOk = farOnPct < 1.0;
  const buggyOff = farOffPct > farOnPct * 4 && farOffPct > 1.0;
  const ok = nearOk && fixOk && buggyOff;

  const detail =
    `offset ${(WORLD_OFFSET / 1e6).toFixed(0)}M vs near ground-truth · near on-vs-off ${nearOnPct.toFixed(3)}% (${nearOk ? "identical" : "BAD"}) · ` +
    `far-ON ${farOnPct.toFixed(3)}% (${fixOk ? "tracks" : "BAD"}) · far-OFF ${farOffPct.toFixed(2)}% (${buggyOff ? "jitters" : "BAD"})`;
  window.__FARWORLD__ = { ok, detail };
  setStatus(ok ? "pass" : "fail", `${ok ? "PASS" : "FAIL"} — ${detail}`);

  // Live loop: orbit the FAR world with the flag ON so the fix is visually obvious (no breathing geometry).
  renderer.cameraRelative = true;
  let last = 0;
  const loop = (t: number): void => {
    const dt = last ? (t - last) / 1000 : 0;
    last = t;
    atlas.tick(dt * 20);
    camera.yaw += dt * 0.1;
    camera.target = [WORLD_OFFSET + W / 2, 14, WORLD_OFFSET + D / 2];
    renderFrame(farDraws);
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
  if (el) { el.className = cls; el.textContent = text; }
}

declare global {
  interface Window {
    __FARWORLD__?: { ok: boolean; detail: string } | { ok: false; error: string };
  }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__FARWORLD__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
