// GATE 4 / P5 — moving pistons driven by the REAL PistonTransients driver (PISTON_PLAN). Proves BOTH
// routes on real GPU:
//   • OPAQUE route (PISTON_PLAN §5 D1): a piston pushing an EMERALD block renders through the transient
//     TERRAIN pipeline (TransientOpaque → terrain shader → exact per-face shading), visible as green.
//   • TRANSLUCENT route (BUG_REPORT.md / vanilla `translucentMovingBlock`): a piston pushing RED stained
//     glass BEHIND a static clear-glass wall is its OWN self-sorted, depth-WRITING transient draw — drawn
//     BEFORE terrain translucent. Depth (not a geometry merge) orders it against the wall, so the red shows
//     THROUGH the clear wall while the wall section is drawn normally — NEVER suppressed.
//
// The driver itself derives the geometry from MovingPistonInfo + the tick partial (set()/frame()), so this
// exercises the interpolation, the opaque/translucent layer split, and both transient passes end-to-end.
//
// Self-check (window.__PISTON__): scene stays lit; GREEN appears (opaque block rasterized via terrain);
// RED shows through the wall (depth-ordered transient); the moving glass is a depth-writing draw each frame;
// no terrain section is ever dropped from the frame; opaque draws are produced each frame; and terrain is
// never re-sorted (motion is transient-only — TRAP 12.A).

import { WebGPUDevice } from "../src/core/webgpu/WebGPUDevice";
import { Camera } from "../src/camera/Camera";
import { TextureFormat } from "../src/core/gpu-enums";
import { type BlockEntry } from "../src/mesh/model/BakedBlockModel";
import { type BlockSource } from "../src/world/SnapshotSource";
import { type Vec3 } from "../src/types";
import { PistonTransients, type MovingPistonInput } from "../src/render/transient/PistonTransients";
import type { SectionDraw } from "../src/render/TerrainRenderer";
import { AIR, buildScene, commitWorld, collectDraws } from "./scene";

const STONE = 1, GLASS = 2, REDGLASS = 3, GREEN = 4;
const PALETTE: Record<number, BlockEntry> = {
  [STONE]: { name: "stone", props: {} },
  [GLASS]: { name: "glass", props: {}, cullGroup: "glass" },
  [REDGLASS]: { name: "red_stained_glass", props: {}, cullGroup: "red_stained_glass" },
  // Not placed in the world — the driver bakes it on demand — but must be atlased so its sprite resolves.
  [GREEN]: { name: "emerald_block", props: {} },
};

const W = 24, H = 16, D = 20;
const DIMS: Vec3 = [W, H, D];
const BY = 6; // both moving blocks ride this row

function buildStaticWorld(): BlockSource {
  const grid = new Uint16Array(W * H * D);
  const idx = (x: number, y: number, z: number) => (y * D + z) * W + x;
  const inB = (x: number, y: number, z: number) => x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D;
  const set = (x: number, y: number, z: number, id: number) => { if (inB(x, y, z)) grid[idx(x, y, z)] = id; };

  for (let x = 0; x < W; x++) for (let z = 0; z < D; z++) set(x, 0, z, STONE); // floor
  for (let x = 0; x < W; x++) for (let y = 1; y <= 11; y++) set(x, y, 3, STONE); // gray backdrop (z=3)
  for (let x = 6; x <= 17; x++) for (let y = 3; y <= 9; y++) set(x, y, 12, GLASS); // static clear wall (z=12)

  return { getBlock: (x, y, z) => (inB(x, y, z) ? grid[idx(x, y, z)] : AIR) };
}

/** One MovingPistonInfo (case C — a pushed block), east-extending into its BE cell. */
function push(x: number, z: number, name: string): MovingPistonInput {
  return { x, y: BY, z, movedState: { name, properties: {} }, direction: "east", extending: true, isSourcePiston: false, progress: 1, progressO: 0 };
}

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const device = await WebGPUDevice.create(canvas);
  syncCanvasSize(canvas, device);
  window.addEventListener("resize", () => syncCanvasSize(canvas, device));

  const { provider, renderer, baker } = await buildScene(device, PALETTE, (t) => setStatus("run", t));

  const store = commitWorld(device, buildStaticWorld(), provider, DIMS);
  const baseDraws = collectDraws(store).draws;

  // The real driver — geometry comes from MovingPistonInfo + partial, baked against the live atlas.
  const pistons = new PistonTransients(device, (name, props) => baker.bake({ name, props }));
  // GREEN (opaque) slides in open space at x≈19→20 (outside the wall's x-range → unoccluded green);
  // RED (translucent) slides at x≈10→11 BEHIND the clear wall (x 6..17, z=12) → must show THROUGH it.
  pistons.set([push(20, 8, "minecraft:emerald_block"), push(11, 10, "minecraft:red_stained_glass")]);

  const camera = new Camera();
  camera.target = [14, BY, 10];
  camera.distance = 14;
  camera.pitch = 0.18;

  const renderAt = (partial: number): { opaque: number; frameLen: number; movingQuads: number; depthWrite: boolean } => {
    const { opaqueDraws, translucent } = pistons.frame(camera.position, partial);
    // The moving glass is APPENDED — every terrain section stays in the frame (no suppression).
    const frame: SectionDraw[] = baseDraws.slice();
    for (const d of opaqueDraws) frame.push(d);
    if (translucent) frame.push(translucent);
    camera.aspect = canvas.width / canvas.height;
    renderer.render(frame, camera, canvas.width, canvas.height);
    return {
      opaque: opaqueDraws.length,
      frameLen: frame.length,
      movingQuads: translucent ? translucent.quadCount : 0,
      depthWrite: translucent ? translucent.depthWrite === true : false,
    };
  };

  const bgra = device.colorFormat === TextureFormat.Bgra8Srgb || device.colorFormat === TextureFormat.Bgra8;
  const FRAMES = 7;
  // The frame must NEVER be shorter than the static terrain (no terrain section is ever suppressed). The
  // moving glass is one extra depth-writing translucent draw appended on top.
  const minFrameLen = baseDraws.length;
  let minLit = 1, maxRed = 0, maxGreen = 0, movingHits = 0, opaqueHits = 0, noSuppress = 0;
  const resortsBefore = renderer.resortCount;
  for (let i = 0; i < FRAMES; i++) {
    const info = renderAt(i / (FRAMES - 1)); // partial 0→1 = the slide
    if (info.opaque > 0) opaqueHits++;
    if (info.movingQuads > 0 && info.depthWrite) movingHits++; // the red block as a depth-writing transient draw
    if (info.frameLen >= minFrameLen + info.opaque + 1) noSuppress++; // all terrain + opaque + the moving glass
    const a = analyze(await device.readCanvasPixels(), bgra);
    minLit = Math.min(minLit, a.litFrac);
    maxRed = Math.max(maxRed, a.redFrac);
    maxGreen = Math.max(maxGreen, a.greenFrac);
    console.log(`frame ${i}: lit ${(a.litFrac * 100).toFixed(1)}% green ${(a.greenFrac * 100).toFixed(2)}% red ${(a.redFrac * 100).toFixed(2)}% opaqueDraws ${info.opaque} movingGlass ${info.movingQuads} (depthWrite ${info.depthWrite}) frameLen ${info.frameLen}`);
  }
  const resorted = renderer.resortCount - resortsBefore;

  // First-fire fix regression: a piston_head must bake to REAL geometry even though NO piston/piston_head
  // block is in this scene's palette (buildScene now always preloads + atlases piston_head). A probe driver
  // bakes a case-A head and asserts non-empty — WITHOUT the fix piston_head's blockstate isn't preloaded so
  // ModelBaker returns EMPTY (0 quads) and the head stays invisible for the whole first activation.
  const headProbe = new PistonTransients(device, (name, props) => baker.bake({ name, props }));
  headProbe.set([{ x: 0, y: 0, z: 0, movedState: { name: "minecraft:piston_head", properties: { type: "normal" } }, direction: "east", extending: true, isSourcePiston: false, progress: 1, progressO: 0 }]);
  const headQuads = headProbe.frame([0, 0, 0], 0.5).opaqueDraws.reduce((s, d) => s + d.quadCount, 0);
  headProbe.dispose();
  const headOk = headQuads > 0;

  // red-through is naturally faint (a 1-block translucent body behind a clear wall, dimmed by the glass),
  // so a 0.15% floor — well above the 0% the scene shows WITHOUT correct ordering — gates it; the structural
  // proof is now that the red block is a depth-writing transient draw EVERY frame while NO terrain section is
  // ever suppressed (the frame only grows: terrain + opaque + the one moving-glass draw).
  const ok = minLit > 0.5 && maxGreen > 0.004 && maxRed > 0.0015 && opaqueHits === FRAMES && movingHits === FRAMES && noSuppress === FRAMES && resorted === 0 && headOk;
  const detail =
    `${FRAMES} frames: min lit ${(minLit * 100).toFixed(1)}%, max green ${(maxGreen * 100).toFixed(2)}% (opaque route), ` +
    `max red-through ${(maxRed * 100).toFixed(2)}% (translucent route); opaque draws ${opaqueHits}/${FRAMES}; ` +
    `depth-writing moving glass ${movingHits}/${FRAMES}; no-suppress ${noSuppress}/${FRAMES}; terrain re-sorts ${resorted}; piston_head bakes ${headQuads} quads (first-fire fix)`;
  window.__PISTON__ = { ok, detail };
  setStatus(ok ? "pass" : "fail", `${ok ? "PASS" : "FAIL"} — ${detail}`);

  // Live view: both blocks ride the piston path back and forth (ping-pong the partial).
  let t0 = 0;
  const loop = (t: number) => {
    if (!t0) t0 = t;
    const p = ((t - t0) / 2500) % 1;
    renderAt(0.5 - 0.5 * Math.cos(p * Math.PI * 2));
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

/**
 * Lit coverage + GREEN-dominant (the opaque emerald block) and RED-dominant (the translucent block through
 * the wall) pixel fractions over a wide central region. Green/red are the only such sources in the scene
 * (floor/backdrop are gray, sky is blue), so each fraction isolates one route.
 */
function analyze(shot: { width: number; height: number; bytesPerRow: number; data: Uint8Array }, bgra: boolean) {
  const rI = bgra ? 2 : 0;
  const bI = bgra ? 0 : 2;
  const x0 = Math.floor(shot.width * 0.12), x1 = Math.floor(shot.width * 0.88);
  const y0 = Math.floor(shot.height * 0.15), y1 = Math.floor(shot.height * 0.8);
  let lit = 0, red = 0, green = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const o = y * shot.bytesPerRow + x * 4;
      const r = shot.data[o + rI], g = shot.data[o + 1], b = shot.data[o + bI];
      if (Math.max(r, g, b) > 24) lit++;
      if (r > g + 30 && r > b + 30) red++;
      if (g > r + 25 && g > b + 25) green++;
    }
  }
  const px = (x1 - x0) * (y1 - y0);
  return { litFrac: lit / px, redFrac: red / px, greenFrac: green / px };
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
    __PISTON__?: { ok: boolean; detail: string } | { ok: false; error: string };
  }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__PISTON__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
