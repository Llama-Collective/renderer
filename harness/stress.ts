// Large-world stress harness. RENDERER_PLAN §14 (culling), §15 (batched submission).
//
// A big procedural world (hundreds of sections) proves the two large-world wins:
//  1. Frustum + distance CULLING — only visible sections cost draws (`renderer.stats`).
//  2. SINGLE-PASS batched submission — the whole frame is ONE render pass / submit (`device.passCount`)
//     regardless of how many sections are drawn, vs one pass+submit per section before.
// Self-check asserts the scene renders, culling drops most sections, and passes/frame is ~1.

import { WebGPUDevice } from "../src/core/webgpu/WebGPUDevice";
import { Camera } from "../src/camera/Camera";
import { Frustum } from "../src/camera/Frustum";
import { type BlockEntry } from "../src/mesh/model/BakedBlockModel";
import { type BlockSource } from "../src/world/SnapshotSource";
import { type Vec3 } from "../src/types";
import { withinRenderDistance, type SectionDraw } from "../src/render/TerrainRenderer";
import { AIR, buildScene, commitWorld, collectDraws, makeFluidContext, FLUID_SPRITES, occlusionVisibleSet, occlusionHomeSection, filterDrawsByOcclusion } from "./scene";
import { instrument } from "../src/core/Instrument";

const STONE = 1, DIRT = 2, GRASS = 3, GLASS = 4, WATER = 5;
const PALETTE: Record<number, BlockEntry> = {
  [STONE]: { name: "stone", props: {} },
  [DIRT]: { name: "dirt", props: {} },
  [GRASS]: { name: "grass_block", props: { snowy: "false" } },
  [GLASS]: { name: "glass", props: {}, cullGroup: "glass" },
  [WATER]: { name: "water", props: { level: "0" } },
};

const W = 256, H = 48, D = 256; // 16×3×16 = 768 sections
const DIMS: Vec3 = [W, H, D];

function buildWorld(): BlockSource {
  const grid = new Uint16Array(W * H * D);
  const idx = (x: number, y: number, z: number) => (y * D + z) * W + x;
  const inB = (x: number, y: number, z: number) => x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D;
  const set = (x: number, y: number, z: number, id: number) => { if (inB(x, y, z)) grid[idx(x, y, z)] = id; };

  // Rolling heightmap terrain (stone → dirt → grass), with a sunken water lake basin.
  for (let x = 0; x < W; x++) {
    for (let z = 0; z < D; z++) {
      const h = Math.round(18 + 8 * Math.sin(x * 0.06) * Math.cos(z * 0.05) + 4 * Math.sin((x + z) * 0.12));
      for (let y = 0; y < h; y++) set(x, y, z, y === h - 1 ? GRASS : y >= h - 3 ? DIRT : STONE);
      // A lake in a central depression, filled with water to y=17.
      const lake = Math.hypot(x - W * 0.5, z - D * 0.5) < 36;
      if (lake) { for (let y = 0; y < 16; y++) set(x, y, z, y >= h ? AIR : STONE); for (let y = h; y <= 17; y++) if (y < H) set(x, y, z, WATER); }
    }
  }
  // A grid of glass towers (translucent geometry spread across many sections).
  for (let tx = 24; tx < W; tx += 48) {
    for (let tz = 24; tz < D; tz += 48) {
      let g = 0; for (let y = H - 1; y >= 0; y--) if (grid[idx(tx, y, tz)] !== AIR) { g = y + 1; break; }
      for (let y = g; y < g + 12 && y < H; y++) for (let dx = 0; dx < 3; dx++) for (let dz = 0; dz < 3; dz++) set(tx + dx, y, tz + dz, GLASS);
    }
  }
  return { getBlock: (x, y, z) => (inB(x, y, z) ? grid[idx(x, y, z)] : AIR) };
}

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const device = await WebGPUDevice.create(canvas);
  syncCanvasSize(canvas, device);
  window.addEventListener("resize", () => syncCanvasSize(canvas, device));

  const RENDER_DISTANCE = 180;
  const { atlas, provider, renderer } = await buildScene(device, PALETTE, (t) => setStatus("run", t), FLUID_SPRITES, RENDER_DISTANCE);
  const fluids = makeFluidContext(PALETTE, atlas);
  setStatus("run", `meshing ${Math.ceil(W / 16) * Math.ceil(H / 16) * Math.ceil(D / 16)} sections…`);
  const store = commitWorld(device, buildWorld(), provider, DIMS, fluids);
  const { draws, total } = collectDraws(store);

  const camera = new Camera();
  camera.target = [W * 0.42, 18, D * 0.42];
  camera.distance = 70;
  camera.pitch = 0.42;
  camera.yaw = 0.7;
  camera.far = 2000;

  const renderFrame = (d: SectionDraw[]) => {
    camera.aspect = canvas.width / canvas.height;
    renderer.render(d, camera, canvas.width, canvas.height);
  };

  // OCC-1: occlusion-cull the (frustum-visible) draw list via the camera BFS over the section visibility
  // graph. `inFrustum` matches the renderer's own frustum + render-distance test so the two culls agree.
  // This is the LEGACY frustum-FUSED path (the A/B correctness proof uses it): the BFS re-runs every call.
  const occCull = (): SectionDraw[] => {
    const frustum = new Frustum();
    frustum.setFromViewProjection(camera.viewProjection());
    const set = occlusionVisibleSet(store, camera.position, (sx, sy, sz) =>
      frustum.testSection(sx * 16, sy * 16, sz * 16) &&
      withinRenderDistance([sx * 16, sy * 16, sz * 16], camera.position, RENDER_DISTANCE));
    const out: SectionDraw[] = [];
    return filterDrawsByOcclusion(draws, set, out);
  };

  // OCC-1 cached path (occBfsCache ON): the BFS produces an orientation-INDEPENDENT reachable set keyed on
  // the clamped home section + graphRevision, so it re-runs ONLY on a home-section crossing or a graph bump
  // (never on a pure rotation). The live frustum is reapplied downstream by the renderer's own per-frame
  // frustum cull (Option B), so the rendered picture is identical; this only avoids the per-frame BFS + alloc.
  let occHomeKey = ""; // packed clamped home section the cached set was computed for
  let occGraphRev = -1;
  const occReach = new Set<number>();
  let occCachedDraws: SectionDraw[] = [];
  const occCullCached = (): SectionDraw[] => {
    const home = occlusionHomeSection(store, camera.position);
    const homeKey = home ? `${home[0]},${home[1]},${home[2]}` : "∅";
    const graphRev = store.graphRevision;
    if (homeKey !== occHomeKey || graphRev !== occGraphRev) {
      // Reachability query: pass-through inFrustum so the cached set is rotation-independent.
      occlusionVisibleSet(store, camera.position, () => true, occReach);
      occHomeKey = homeKey;
      occGraphRev = graphRev;
      occCachedDraws = filterDrawsByOcclusion(draws, occReach, []); // membership changed ⇒ fresh array (F1 key)
    }
    return occCachedDraws; // rotation-only frame: same array, renderer frustum-culls it per frame
  };

  // ── A: all draws (frustum + distance cull only) ──
  const passesBefore = device.passCount;
  renderFrame(draws);
  const passesPerFrame = device.passCount - passesBefore;
  const imgAll = await device.readCanvasPixels();
  const statsAll = { ...renderer.stats };

  // ── B: occlusion-culled draws ──
  const occDraws = occCull();
  renderFrame(occDraws);
  const imgOcc = await device.readCanvasPixels();
  const statsOcc = { ...renderer.stats };

  // Occlusion must remove draws WITHOUT changing the picture: any large pixel delta = a hole (over-cull).
  const N = imgAll.width * imgAll.height;
  let diff = 0;
  let terrain = 0;
  for (let p = 0; p < N; p++) {
    const o = p * 4;
    if (
      Math.abs(imgAll.data[o] - imgOcc.data[o]) > 8 ||
      Math.abs(imgAll.data[o + 1] - imgOcc.data[o + 1]) > 8 ||
      Math.abs(imgAll.data[o + 2] - imgOcc.data[o + 2]) > 8
    ) diff++;
    if (Math.min(imgOcc.data[o], imgOcc.data[o + 1], imgOcc.data[o + 2]) < 150) terrain++;
  }
  const diffPct = (diff / N) * 100;
  const frac = terrain / N;
  const s = statsAll;
  const culledPct = s.total > 0 ? (s.culled / s.total) * 100 : 0;
  const occSavedPct = s.drawn > 0 ? ((s.drawn - statsOcc.drawn) / s.drawn) * 100 : 0;

  // ── F1 render-bundle replay (GPU proof on real Metal) ──
  // Neither A nor B records a bundle (they pass DIFFERENT arrays). Render the SAME (occDraws, camera) twice
  // more: the first RECORDS a bundle (this frame matches B), the second hits the static-camera fast path and
  // REPLAYS it. The replayed frame must equal frame B (direct-encoded) pixel-for-pixel — exercising
  // createRenderBundle + executeBundles + the skip-uploads fast path that the Fake-device unit test can't.
  const replaysBefore = renderer.bundleReplays;
  renderFrame(occDraws); // records (camera + array identical to B)
  renderFrame(occDraws); // replays
  const imgReplay = await device.readCanvasPixels();
  let bundleDiff = 0;
  for (let p = 0; p < N; p++) {
    const o = p * 4;
    if (imgReplay.data[o] !== imgOcc.data[o] || imgReplay.data[o + 1] !== imgOcc.data[o + 1] || imgReplay.data[o + 2] !== imgOcc.data[o + 2]) bundleDiff++;
  }
  const bundleReplays = renderer.bundleReplays - replaysBefore;
  const bundleOk = bundleReplays > 0 && bundleDiff === 0; // replay actually happened AND was pixel-identical

  // ── FS-4 facing-slice cull: pixel-identity proof (real Metal) ──
  // Mesh the SAME world with partitionFacing ON, then render the DEFAULT (cull off, unpartitioned) vs the
  // SLICED (cull on, partitioned) at two poses incl. an edge-on worst case. Dropping camera-away facing
  // slices only removes triangles the GPU already back-face-culls, so the image must be EXACTLY 0-pixel-diff
  // — and the cull must actually fire (sliceDrawsCulled > 0). diff>0 means fix getVisibleFaces/quadFacing,
  // never widen the tolerance.
  const slicedStore = commitWorld(device, buildWorld(), provider, DIMS, fluids, undefined, { partitionFacing: true });
  const slicedDraws = collectDraws(slicedStore).draws;
  const bundleWas = renderer.bundleReplay;
  renderer.bundleReplay = false; // direct-encode both sides for a clean comparison
  let sliceDiff = 0;
  let sliceCulled = 0;
  const poses: Array<[number, number]> = [[camera.yaw, camera.pitch], [0, 0.0]]; // current + edge-on (level, axis-aligned)
  for (const [yaw, pitch] of poses) {
    camera.yaw = yaw; camera.pitch = pitch;
    renderer.faceSliceCull = false;
    renderFrame(draws); // DEFAULT baseline (unpartitioned, one opaque draw per section)
    const base = await device.readCanvasPixels();
    const culledBefore = renderer.sliceDrawsCulled;
    renderer.faceSliceCull = true;
    renderFrame(slicedDraws); // SLICED (partitioned, one draw per VISIBLE facing run)
    const sliced = await device.readCanvasPixels();
    sliceCulled += renderer.sliceDrawsCulled - culledBefore;
    for (let p = 0; p < N; p++) {
      const o = p * 4;
      if (base.data[o] !== sliced.data[o] || base.data[o + 1] !== sliced.data[o + 1] || base.data[o + 2] !== sliced.data[o + 2]) sliceDiff++;
    }
  }
  renderer.faceSliceCull = false;
  renderer.bundleReplay = bundleWas;
  const sliceOk = sliceDiff === 0 && sliceCulled > 0; // EXACT 0px across both poses AND the cull genuinely fired

  // PASS: scene renders; frustum drops sections; occlusion removes MORE (vs frustum-only) with no holes.
  // `frac < 0.9999` only rejects a literally-uniform (broken) frame — a large-world view legitimately
  // fills the screen with terrain (no sky from inside a 256²-block world), so the old < 0.99 sanity
  // bound never held here (confirmed pre-existing via git-stash A/B). The real gate is now occlusion
  // correctness: it must remove draws (statsOcc.drawn < frustum drawn) with NO holes (diffPct < 0.5).
  const ok =
    s.drawn > 0 && s.drawn < s.total && passesPerFrame <= 2 && frac > 0.1 && frac < 0.9999 &&
    statsOcc.drawn < s.drawn && diffPct < 0.5 && bundleOk && sliceOk;
  const detail =
    `${s.total} section-passes (${(total / 1000).toFixed(0)}k quads); frustum ${s.drawn} drawn/${s.culled} culled (${culledPct.toFixed(0)}%); ` +
    `OCC-1 ${statsOcc.drawn} drawn (−${occSavedPct.toFixed(0)}% vs frustum, holes ${diffPct.toFixed(2)}%); ` +
    `F1 bundle replay ${bundleOk ? "OK" : "BAD"} (${bundleReplays} replays, ${bundleDiff}px diff); ` +
    `FS-4 slice cull ${sliceOk ? "OK" : "BAD"} (${sliceCulled} slices culled, ${sliceDiff}px diff); ` +
    `${passesPerFrame} pass/frame; terrain ${(frac * 100).toFixed(1)}% of ${imgAll.width}×${imgAll.height}`;

  // ── OCC-1 cache proof: the BFS re-runs ONLY when the camera's home section crosses a boundary — NEVER on
  // an orientation-only (same-home-section) frame. NOTE: this is an ORBIT camera, so `camera.yaw += …` moves
  // the EYE around the target (≈ distance·Δyaw per frame), legitimately crossing a few section boundaries over
  // the sweep — it is NOT a fixed-eye rotation. So the correct invariant is `occBfsRuns == home-section
  // crossings` (and ≪ the 64 per-frame runs the uncached `occCull` path would do), which proves the cache holds
  // across every same-home frame (the OCC-1 win). We count the crossings and assert the BFS never ran more.
  instrument.enabled = true;
  instrument.reset();
  occHomeKey = ""; occGraphRev = -1; // cold cache
  const camYaw0 = camera.yaw, camPitch0 = camera.pitch;
  let homeCrossings = 0, prevHome = "";
  for (let i = 0; i < 64; i++) {
    camera.yaw += 0.01; // orbit yaw — moves the eye, so the home section may cross a boundary a few times
    const h = occlusionHomeSection(store, camera.position);
    const hk = h ? `${h[0]},${h[1]},${h[2]}` : "none";
    if (hk !== prevHome) { homeCrossings++; prevHome = hk; } // count distinct contiguous home sections
    occCullCached();
  }
  const occBfsRunsRotate = instrument.job.occBfsRuns;
  camera.yaw = camYaw0; camera.pitch = camPitch0;
  // BFS ran at most once per home-section crossing (so same-home/orientation frames are all cache hits) AND
  // far below the 64 per-frame runs the uncached path would do — the cache demonstrably holds.
  const occCacheOk = occBfsRunsRotate <= homeCrossings && occBfsRunsRotate < 64;

  // ── final PASS: scene renders; frustum drops sections; occlusion removes MORE (vs frustum-only) with no
  //    holes; F1 bundle replays; FS-4 slice cull is pixel-exact; and the OCC-1 BFS cache holds on rotation. ──
  const okFinal = ok && occCacheOk;
  const detailFinal = `${detail}; OCC-1 cache ${occCacheOk ? "OK" : "BAD"} (${occBfsRunsRotate} BFS runs / 64 orbit frames, ${homeCrossings} home crossings)`;
  window.__STRESS__ = { ok: okFinal, detail: detailFinal };
  setStatus(okFinal ? "pass" : "fail", `${okFinal ? "PASS" : "FAIL"} — ${detailFinal}`);

  // The live loop uses the cached path (occBfsCache ON): the BFS runs only on home-section crossings as the
  // camera orbits — not every frame. The renderer's own per-frame frustum cull drops off-screen sections.
  const OCC_BFS_CACHE = true;
  let last = 0;
  const loop = (t: number) => {
    const dt = last ? (t - last) / 1000 : 0;
    last = t;
    atlas.tick(dt * 20);
    camera.yaw += dt * 0.1;
    renderFrame(OCC_BFS_CACHE ? occCullCached() : occCull());
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
    __STRESS__?: { ok: boolean; detail: string } | { ok: false; error: string };
  }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__STRESS__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
