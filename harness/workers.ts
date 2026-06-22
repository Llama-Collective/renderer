// Off-thread meshing smoke (Phase 5). Meshes the SAME world two ways — synchronous inline, and via a
// real Web-Worker pool (mesher.worker, fed the transferred baked-model + fluid init) — and asserts the
// rendered pixels are IDENTICAL. End-to-end proof that off-thread meshing diverges from inline by nothing
// (MeshInit unit tests prove byte-identical output; this proves the real Worker round-trip — postMessage,
// structured clone, transferables — preserves it).

import { WebGPUDevice } from "../src/core/webgpu/WebGPUDevice";
import { Camera } from "../src/camera/Camera";
import { type BlockEntry } from "../src/mesh/model/BakedBlockModel";
import { type BlockSource, SnapshotSource, isAllAirCore } from "../src/world/SnapshotSource";
import { SectionStore } from "../src/world/SectionStore";
import { GpuSectionUploader } from "../src/render/GpuSectionUploader";
import { sectionKey, type SectionKey } from "../src/world/SectionKey";
import { makeBuildOutput } from "../src/workers/BuildOutput";
import { WorkerPool } from "../src/workers/WorkerPool";
import { extractMeshInit, extractPaletteModels } from "../src/workers/MeshInit";
import { SharedMeshQueue, sharedMemoryAvailable } from "../src/workers/SharedMeshQueue";
import { meshSection, type BakedModelProvider } from "../src/mesh/SectionMesher";
import type { FluidContext } from "../src/mesh/FluidMesher";
import { APRON } from "../src/world/SnapshotSource";
import { DirtyReason, type Vec3 } from "../src/types";
import { AIR, buildScene, commitWorld, collectDraws, makeFluidContext, FLUID_SPRITES } from "./scene";

const STONE = 1, DIRT = 2, GRASS = 3, GLASS = 4, WATER = 5;
const PALETTE: Record<number, BlockEntry> = {
  [STONE]: { name: "stone", props: {} },
  [DIRT]: { name: "dirt", props: {} },
  [GRASS]: { name: "grass_block", props: { snowy: "false" } },
  [GLASS]: { name: "glass", props: {}, cullGroup: "glass" },
  [WATER]: { name: "water", props: { level: "0" } },
};
const PALETTE_IDS = [AIR, STONE, DIRT, GRASS, GLASS, WATER];

const W = 48, H = 32, D = 48;
const DIMS: Vec3 = [W, H, D];

function buildWorld(): BlockSource {
  const grid = new Uint16Array(W * H * D);
  const idx = (x: number, y: number, z: number) => (y * D + z) * W + x;
  const inB = (x: number, y: number, z: number) => x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D;
  for (let x = 0; x < W; x++) {
    for (let z = 0; z < D; z++) {
      const h = Math.round(10 + 4 * Math.sin(x * 0.2) * Math.cos(z * 0.18));
      for (let y = 0; y < h; y++) grid[idx(x, y, z)] = y === h - 1 ? GRASS : y >= h - 3 ? DIRT : STONE;
      if (Math.hypot(x - W / 2, z - D / 2) < 8) for (let y = h; y <= 9; y++) if (y < H) grid[idx(x, y, z)] = WATER; // translucent pond
    }
  }
  // A hollow glass box (translucent + cull groups) so the smoke exercises all three passes.
  for (let x = 6; x < 12; x++) for (let y = 12; y < 18; y++) for (let z = 6; z < 12; z++)
    if (x === 6 || x === 11 || y === 12 || y === 17 || z === 6 || z === 11) grid[idx(x, y, z)] = GLASS;
  return { getBlock: (x, y, z) => (inB(x, y, z) ? grid[idx(x, y, z)] : AIR) };
}

/** Mesh the world OFF-THREAD via a real Worker pool, committing each result through the invariant store.
 *  `shared` selects Model B (SAB ring, zero-copy snapshots) vs Model A (postMessage + clone). */
async function meshViaWorkers(
  device: WebGPUDevice,
  world: BlockSource,
  provider: BakedModelProvider,
  fluids: FluidContext,
  shared: boolean,
): Promise<{ store: SectionStore; workers: number }> {
  const store = new SectionStore(new GpuSectionUploader(device));
  const snapshots = new SnapshotSource(world);
  let workers = 0;
  const createWorker = (index: number): Worker => {
    workers++;
    return new Worker(new URL("./dist/mesher.worker.entry.js", location.href), { type: "module", name: `Mesher-B-${index}` });
  };
  // Init the workers WITHOUT glass to exercise incremental model push: glass is sent via `addModels`
  // below (as if a glass block were first placed after worker startup). The render must still match — in
  // Model B that exercises the epoch ordering gate (the glass builds wait for the addModels postMessage).
  const init = extractMeshInit(provider, PALETTE_IDS.filter((id) => id !== GLASS), fluids);
  const queue = shared ? SharedMeshQueue.create({ apron: APRON }) : null;
  const pool = queue
    ? new WorkerPool({ mode: "shared", init, size: 3, createWorker, queue, fallbackMesh: (s) => meshSection(s, provider, fluids) })
    : new WorkerPool({ mode: "worker", init, size: 3, createWorker });
  pool.post({ type: "addModels", models: extractPaletteModels(provider, [GLASS]) }); // bumps the epoch BEFORE any build

  const jobToKey = new Map<number, SectionKey>();
  let jobId = 1;
  let pending = 0;
  return await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`worker mesh timed out — ${pending} pending`)), 20_000);
    pool.onMessage((res) => {
      if (res.type !== "buildDone") return;
      const key = jobToKey.get(res.jobId);
      if (key === undefined) return;
      store.acceptBuild(makeBuildOutput(res.payload));
      const s = store.get(key);
      if (s) store.commit(s);
      if (--pending === 0) {
        clearTimeout(timer);
        pool.dispose();
        resolve({ store, workers });
      }
    });
    for (let X = 0; X < Math.ceil(W / 16); X++)
      for (let Y = 0; Y < Math.ceil(H / 16); Y++)
        for (let Z = 0; Z < Math.ceil(D / 16); Z++) {
          const key = sectionKey(X, Y, Z);
          const gen = store.markDirty(key, DirtyReason.InitialLoad);
          const snap = snapshots.buildSnapshot(key, gen, DirtyReason.InitialLoad);
          if (isAllAirCore(snap)) continue;
          const id = jobId++;
          jobToKey.set(id, key);
          pending++;
          pool.post({ type: "build", jobId: id, snapshot: snap });
        }
    if (pending === 0) {
      clearTimeout(timer);
      resolve({ store, workers });
    }
  });
}

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const device = await WebGPUDevice.create(canvas);
  syncCanvasSize(canvas, device);

  const { atlas, provider, renderer } = await buildScene(device, PALETTE, undefined, FLUID_SPRITES);
  const fluids = makeFluidContext(PALETTE, atlas);
  const world = buildWorld();

  const camera = new Camera();
  camera.target = [W / 2, 10, D / 2];
  camera.distance = 60;
  camera.pitch = 0.5;
  camera.yaw = 0.8;
  camera.far = 2000;
  camera.aspect = canvas.width / canvas.height;
  const renderFrame = (store: SectionStore): void => {
    renderer.render(collectDraws(store).draws, camera, canvas.width, canvas.height);
  };

  // INLINE reference.
  const inlineStore = commitWorld(device, world, provider, DIMS, fluids);
  renderFrame(inlineStore);
  const imgInline = await device.readCanvasPixels();
  const N = imgInline.width * imgInline.height;

  /** Off-thread (model A or B) pixel diff vs the inline reference + terrain coverage. */
  const offThreadDiff = async (shared: boolean): Promise<{ workers: number; diffPct: number; litPct: number }> => {
    const { store, workers } = await meshViaWorkers(device, world, provider, fluids, shared);
    renderFrame(store);
    const img = await device.readCanvasPixels();
    let diff = 0;
    let lit = 0;
    for (let p = 0; p < N; p++) {
      const o = p * 4;
      if (Math.abs(imgInline.data[o] - img.data[o]) > 4 || Math.abs(imgInline.data[o + 1] - img.data[o + 1]) > 4 || Math.abs(imgInline.data[o + 2] - img.data[o + 2]) > 4) diff++;
      if (Math.min(img.data[o], img.data[o + 1], img.data[o + 2]) < 150) lit++;
    }
    return { workers, diffPct: (diff / N) * 100, litPct: (lit / N) * 100 };
  };

  // Model A (postMessage + clone) — always available.
  const a = await offThreadDiff(false);
  // Model B (SAB + Atomics, zero-copy snapshots) — only when the page is cross-origin isolated. The verify
  // runner injects COOP/COEP so crossOriginIsolated becomes true here; without it Model B is correctly skipped.
  const isolated = sharedMemoryAvailable();
  const b = isolated ? await offThreadDiff(true) : null;

  const aOk = a.workers > 0 && a.litPct > 5 && a.diffPct < 0.1;
  const bOk = !b || (b.workers > 0 && b.diffPct < 0.1);
  const ok = aOk && bOk;
  const detail =
    `crossOriginIsolated=${isolated}; Model A diff ${a.diffPct.toFixed(3)}% (terrain ${a.litPct.toFixed(1)}%, ${a.workers}w)` +
    (b ? `; Model B (SAB zero-copy) diff ${b.diffPct.toFixed(3)}% (${b.workers}w)` : `; Model B SKIPPED (not isolated)`) +
    ` — ${ok ? "IDENTICAL" : "MISMATCH"}`;
  setStatus(ok ? "pass" : "fail", `${ok ? "PASS" : "FAIL"} — ${detail}`);
  window.__WORKERS__ = { ok, detail };
}

function syncCanvasSize(canvas: HTMLCanvasElement, device: WebGPUDevice): void {
  const scale = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(canvas.clientWidth * scale));
  const h = Math.max(1, Math.round(canvas.clientHeight * scale));
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
    __WORKERS__?: { ok: boolean; detail: string } | { ok: false; error: string };
  }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__WORKERS__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
