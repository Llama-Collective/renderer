// GATE W7.1 — cross-backend pixel-diff parity (the WebGL2-fallback correctness milestone).
//
// The W3–W6 smoke checks prove each MECHANISM on real WebGL2 (clear, terrain draw, origin-select,
// linear blend, bundle replay, …). This page proves the WHOLE thing: it builds glass.ts's §12.1
// transparency torture world (STAINED glass behind CLEAR glass — the scene that motivated the
// rewrite) ONCE per backend from a SHARED world definition, renders the SAME camera on a real
// WebGPU device AND a forced WebGL2 device, and diffs the two presented frames across a front-arc
// orbit. Parity here means: identical geometry (same CPU mesh ⇒ equal quad counts), the stained
// tint present on BOTH backends at every angle (the translucency ordering reproduced), and the two
// renders agree within a rasterizer-difference tolerance (Metal vs ANGLE/SwiftShader can't be
// bit-identical, so the gate measures "visually equivalent", not "bit-equal"). window.__CROSSBACKEND__
// holds the verdict; the WebGPU canvas is #c so the Playwright runner's #c screenshot still works.

import { Camera } from "../src/camera/Camera";
import { TextureFormat, BackendKind } from "../src/core/gpu-enums";
import { createDevice, type CanvasDevice } from "../src/core/createDevice";
import { type BlockEntry } from "../src/mesh/model/BakedBlockModel";
import { type BlockSource } from "../src/world/SnapshotSource";
import { type Vec3 } from "../src/types";
import { AIR, buildScene, commitWorld, collectDraws } from "./scene";

// ── §12.1 torture content — lifted verbatim from harness/glass.ts (every page defines its own world;
//    keeping it identical here means both backends render the exact stained-behind-clear layers). ──
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

  for (let x = 0; x < W; x++) for (let z = 0; z < D; z++) set(x, 0, z, STONE);
  for (let x = 4; x <= 17; x++) for (let y = 1; y <= 9; y++) set(x, y, 8, STONE);
  // Core torture: stained glass at z=10, CLEAR glass two blocks in front at z=12 (order must be
  // backdrop → stained → clear from every angle — TRAP 12.B per-quad index sort).
  for (let x = 5; x <= 16; x++) {
    for (let y = 3; y <= 7; y++) {
      set(x, y, 10, STAINED[(x + y) % STAINED.length]);
      set(x, y, 12, GLASS);
    }
  }
  for (let x = 6; x <= 8; x++) for (let y = 4; y <= 6; y++) { set(x, y, 10, SLIME); set(x, y, 12, GLASS); }
  for (let z = 13; z <= 18; z += 2) set(4, 1, z, PANE);
  set(4, 2, 14, PANE);
  set(19, 1, 14, PANE);

  return { getBlock: (x, y, z) => (inB(x, y, z) ? grid[idx(x, y, z)] : AIR) };
}

// ── Fixed, DPR-independent backing size so both canvases share an identical pixel grid (1:1 diff). ──
const SIZE = 512;
const ANGLES_DEG = [-55, -20, 15, 50]; // front-arc orbit (matches glass.ts's correctness-critical arc)
const TOL = 16; // per-channel LSB tolerance absorbing rasterizer/sampler/sRGB-rounding differences

interface Shot { width: number; height: number; bytesPerRow: number; data: Uint8Array }

const isBgra = (d: CanvasDevice): boolean =>
  d.colorFormat === TextureFormat.Bgra8Srgb || d.colorFormat === TextureFormat.Bgra8;

/** Normalize a backend's RAW swapchain readback into a canonical tight top-first RGBA8 buffer:
 *  swap R/B when the format is BGRA (WebGPU on Mac), and index the source with the backend's OWN
 *  bytesPerRow (WebGPU pads rows to 256B; WebGL2 is tight). Both backends already return top-first
 *  rows (WebGPU copyTextureToBuffer is top-first; WebGL2Device Y-flips gl.readPixels) — no flip. */
function toRGBA(shot: Shot, bgra: boolean): Uint8Array {
  const { width, height, bytesPerRow, data } = shot;
  const rI = bgra ? 2 : 0, bI = bgra ? 0 : 2;
  const out = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = y * bytesPerRow + x * 4;
      const p = (y * width + x) * 4;
      out[p] = data[o + rI];
      out[p + 1] = data[o + 1];
      out[p + 2] = data[o + bI];
      out[p + 3] = data[o + 3];
    }
  }
  return out;
}

interface DiffMetrics {
  maeR: number; maeG: number; maeB: number; // mean abs error over BOTH-lit (interior) pixels
  interMae: number;                          // max of the three (the headline color-drift number)
  maxDelta: number;                          // single worst channel delta, whole frame
  pctOver: number;                           // fraction of both-lit pixels with maxΔ > TOL
  coverGpu: number; coverGl: number;         // non-background coverage fraction per backend
  coverXor: number;                          // |lit-on-exactly-one| / |lit-on-either| (silhouette disagreement)
  tintGpu: number; tintGl: number;           // stained-tint fraction per backend (central region)
}

const LIT = 24; // max(r,g,b) > LIT ⇒ non-background (same threshold as glass/demo)

/** Compare two canonical RGBA buffers (same w×h). Color metrics are computed over the INTERSECTION
 *  (pixels lit on BOTH) so genuine edge antialiasing differences between two rasterizers don't drown
 *  the interior-color signal; the silhouette disagreement is reported separately as coverXor. */
function diff(a: Uint8Array, b: Uint8Array, w: number, h: number): DiffMetrics {
  let sumR = 0, sumG = 0, sumB = 0, inter = 0, over = 0, maxDelta = 0;
  let litA = 0, litB = 0, litEither = 0, litXor = 0;
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    const ar = a[o], ag = a[o + 1], ab = a[o + 2];
    const br = b[o], bg = b[o + 1], bb = b[o + 2];
    const la = Math.max(ar, ag, ab) > LIT;
    const lb = Math.max(br, bg, bb) > LIT;
    if (la) litA++;
    if (lb) litB++;
    if (la || lb) litEither++;
    if (la !== lb) litXor++;
    const dR = Math.abs(ar - br), dG = Math.abs(ag - bg), dB = Math.abs(ab - bb);
    const dMax = Math.max(dR, dG, dB);
    if (dMax > maxDelta) maxDelta = dMax;
    if (la && lb) {
      sumR += dR; sumG += dG; sumB += dB; inter++;
      if (dMax > TOL) over++;
    }
  }
  const n = Math.max(1, inter);
  const maeR = sumR / n, maeG = sumG / n, maeB = sumB / n;
  return {
    maeR, maeG, maeB,
    interMae: Math.max(maeR, maeG, maeB),
    maxDelta,
    pctOver: over / n,
    coverGpu: litA / (w * h), coverGl: litB / (w * h),
    coverXor: litXor / Math.max(1, litEither),
    tintGpu: tintFrac(a, w, h), tintGl: tintFrac(b, w, h),
  };
}

/** Stained-tint fraction over the central region (canonical RGBA, so no bgra swap) — a red/green-
 *  dominant pixel can only come from stained glass / slime, proving translucent tint actually rendered
 *  (glass.ts's analyze metric, the §12.1 signal). */
function tintFrac(rgba: Uint8Array, w: number, h: number): number {
  const x0 = Math.floor(w * 0.2), x1 = Math.floor(w * 0.8);
  const y0 = Math.floor(h * 0.12), y1 = Math.floor(h * 0.68);
  let tinted = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const o = (y * w + x) * 4;
      const r = rgba[o], g = rgba[o + 1], b = rgba[o + 2];
      if ((r > g + 28 && r > b + 28) || (g > r + 28 && g > b + 28)) tinted++;
    }
  }
  return tinted / Math.max(1, (x1 - x0) * (y1 - y0));
}

interface Built { renderer: import("../src/render/TerrainRenderer").TerrainRenderer; draws: ReturnType<typeof collectDraws>["draws"]; total: number; translucent: number }

/** Build the shared torture world's scene on one device. Meshing (commitWorld → meshSection) is pure
 *  CPU and device-independent, so both backends carry byte-identical geometry; only upload+draw differ. */
async function buildOn(device: CanvasDevice): Promise<Built> {
  const { provider, renderer } = await buildScene(device, PALETTE, () => {});
  const store = commitWorld(device, buildWorld(), provider, DIMS);
  const { draws, total, translucent } = collectDraws(store);
  return { renderer, draws, total, translucent };
}

function sizeCanvas(canvas: HTMLCanvasElement, device: CanvasDevice): void {
  canvas.width = SIZE;
  canvas.height = SIZE;
  device.resize(SIZE, SIZE);
}

// Per-backend worst-case aggregation across the orbit. `skip` distinguishes a legitimate environmental
// SKIP (no WebGPU adapter) from a real FAIL — a forced-WebGL2 init failure must surface as FAIL, not be
// swallowed as SKIP by the runner's free-text "not available" match.
interface Verdict { ok: boolean; detail: string; skip?: boolean; metrics?: Record<string, number> }

async function run(): Promise<Verdict> {
  const canvasGpu = document.getElementById("c") as HTMLCanvasElement;
  const canvasGl = document.getElementById("cgl") as HTMLCanvasElement;

  // The WebGPU side: createDevice picks WebGPU when navigator.gpu is present. The GL side: force the
  // fallback so we deterministically get WebGL2 even on a WebGPU-capable Chrome.
  const gpu = await createDevice(canvasGpu);
  if (gpu.backend !== BackendKind.WebGPU) {
    // No WebGPU adapter ⇒ both sides would be WebGL2; this isn't a cross-backend test. Signal SKIP (code 2).
    return { ok: false, skip: true, detail: "WebGPU adapter not available — cross-backend diff needs both backends" };
  }
  // A forced-WebGL2 creation FAILURE is a real FAIL (the fallback backend is broken), NOT a SKIP.
  let gl: CanvasDevice;
  try {
    gl = await createDevice(canvasGl, { preferWebGL2: true });
  } catch (e) {
    return { ok: false, detail: "forced WebGL2 device failed to create: " + (e instanceof Error ? e.message : String(e)) };
  }

  sizeCanvas(canvasGpu, gpu);
  sizeCanvas(canvasGl, gl);

  const bGpu = await buildOn(gpu);
  const bGl = await buildOn(gl);

  const camera = new Camera();
  camera.target = [10.5, 5, 11];
  camera.distance = 28;
  camera.pitch = 0.3;
  camera.aspect = 1; // square canvas

  const bgraGpu = isBgra(gpu), bgraGl = isBgra(gl);

  // Worst-case metrics across the front-arc orbit (the §12.1 ordering is angle-dependent).
  let worstInterMae = 0, worstPctOver = 0, worstCoverXor = 0, worstMaxDelta = 0;
  let minTintGpu = 1, minTintGl = 1, maxTintDelta = 0;
  let minCover = 1, worstCoverDelta = 0; // both backends must render substantial, AGREEING coverage
  const perAngle: string[] = [];

  for (const deg of ANGLES_DEG) {
    camera.yaw = (deg * Math.PI) / 180;

    bGpu.renderer.render(bGpu.draws, camera, SIZE, SIZE);
    const a = toRGBA(await gpu.readCanvasPixels(), bgraGpu);
    bGl.renderer.render(bGl.draws, camera, SIZE, SIZE);
    const b = toRGBA(await gl.readCanvasPixels(), bgraGl);

    const m = diff(a, b, SIZE, SIZE);
    worstInterMae = Math.max(worstInterMae, m.interMae);
    worstPctOver = Math.max(worstPctOver, m.pctOver);
    worstCoverXor = Math.max(worstCoverXor, m.coverXor);
    worstMaxDelta = Math.max(worstMaxDelta, m.maxDelta);
    minTintGpu = Math.min(minTintGpu, m.tintGpu);
    minTintGl = Math.min(minTintGl, m.tintGl);
    maxTintDelta = Math.max(maxTintDelta, Math.abs(m.tintGpu - m.tintGl));
    minCover = Math.min(minCover, m.coverGpu, m.coverGl);
    worstCoverDelta = Math.max(worstCoverDelta, Math.abs(m.coverGpu - m.coverGl));
    perAngle.push(
      `${deg}°: MAE ${m.interMae.toFixed(1)} over ${(m.pctOver * 100).toFixed(1)}% xor ${(m.coverXor * 100).toFixed(1)}% ` +
      `tint ${(m.tintGpu * 100).toFixed(1)}/${(m.tintGl * 100).toFixed(1)}%`,
    );
    console.log(`[crossbackend] ${perAngle[perAngle.length - 1]}`);
  }

  // GATE W7.1 thresholds. Structural equality (same CPU mesh) is exact; pixel agreement is bounded by the
  // rasterizer reality. NOTE ON WHAT THIS PROVES: in a headless Chrome on Mac, WebGPU and the forced WebGL2
  // (Chrome's ANGLE) both target the SAME Metal driver, so the observed deltas are ~0 (bit-exact). That makes
  // this a strong CODE-LEVEL parity gate — a wrong GLSL twin, a dropped translucent pass, an sRGB/gamma error,
  // or a winding flip in the WebGL2 path produces a delta INDEPENDENT of the shared driver and trips
  // MAE/over-tol/xor/coverage/tint here. The tolerances below are deliberately sized for an INDEPENDENT
  // rasterizer (Metal vs SwiftShader) so the same gate stays valid if run with the GL side forced onto a
  // software rasterizer; they are not exercised against a non-zero delta in the shared-driver environment.
  const sameGeom = bGpu.total === bGl.total && bGpu.translucent === bGl.translucent;
  const G_MAE = 14;      // interior color drift (LSB)
  const G_PCT = 0.06;    // fraction of interior pixels past TOL
  const G_XOR = 0.05;    // silhouette disagreement
  const G_COVER = 0.05;  // each backend must render a substantial frame (not near-blank) at every angle
  const G_COVERD = 0.03; // the two backends' coverage fractions must agree (no geometry dropped on one)
  const G_TINT = 0.015;  // stained tint must render on BOTH (the §12.1 signal)
  const G_TINTD = 0.05;  // tint fraction must agree between backends

  const ok =
    sameGeom &&
    bGpu.translucent > 0 &&
    worstInterMae < G_MAE &&
    worstPctOver < G_PCT &&
    worstCoverXor < G_XOR &&
    minCover > G_COVER && worstCoverDelta < G_COVERD &&
    minTintGpu > G_TINT && minTintGl > G_TINT &&
    maxTintDelta < G_TINTD;

  const detail =
    `WebGPU(${gpu.colorFormat}) vs WebGL2(${gl.colorFormat}) @ ${SIZE}² · ` +
    `geom ${bGpu.total}q/${bGpu.translucent}t == ${bGl.total}q/${bGl.translucent}t: ${sameGeom}; ` +
    `worst interior-MAE ${worstInterMae.toFixed(2)} (<${G_MAE}), over-tol ${(worstPctOver * 100).toFixed(2)}% (<${G_PCT * 100}%), ` +
    `silhouette-xor ${(worstCoverXor * 100).toFixed(2)}% (<${G_XOR * 100}%), maxΔ ${worstMaxDelta}; ` +
    `coverage min ${(minCover * 100).toFixed(1)}% (>${G_COVER * 100}%) Δ ${(worstCoverDelta * 100).toFixed(2)}% (<${G_COVERD * 100}%); ` +
    `min tint ${(minTintGpu * 100).toFixed(2)}/${(minTintGl * 100).toFixed(2)}% (>${G_TINT * 100}%), tintΔ ${(maxTintDelta * 100).toFixed(2)}%. ` +
    `[${perAngle.join(" | ")}]`;

  return {
    ok,
    detail,
    metrics: {
      interMae: worstInterMae, pctOver: worstPctOver, coverXor: worstCoverXor, maxDelta: worstMaxDelta,
      minCover, coverDelta: worstCoverDelta, minTintGpu, minTintGl, tintDelta: maxTintDelta,
      totalGpu: bGpu.total, totalGl: bGl.total,
    },
  };
}

function setStatus(cls: string, text: string): void {
  const el = document.getElementById("status");
  if (el) { el.className = cls; el.textContent = text; }
}

declare global {
  interface Window {
    __CROSSBACKEND__?: { ok: boolean; detail: string; metrics?: Record<string, number> } | { ok: false; error: string };
  }
}

run().then(
  (v) => {
    // Explicit SKIP (no WebGPU adapter) → write it as `error` so the runner returns code 2. A FAIL (e.g. a
    // broken forced-WebGL2 device) goes through the normal path → code 1, NOT swallowed as SKIP.
    if (v.skip) {
      window.__CROSSBACKEND__ = { ok: false, error: v.detail };
      setStatus("fail", `SKIP — ${v.detail}`);
      return;
    }
    window.__CROSSBACKEND__ = { ok: v.ok, detail: v.detail, metrics: v.metrics };
    setStatus(v.ok ? "pass" : "fail", `${v.ok ? "PASS" : "FAIL"} — ${v.detail}`);
  },
  (e) => {
    const msg = e instanceof Error ? e.message : String(e);
    window.__CROSSBACKEND__ = { ok: false, error: msg };
    setStatus("fail", `FAIL — ${msg}`);
  },
);
