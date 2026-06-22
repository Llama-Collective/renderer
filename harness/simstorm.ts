// Phase P perf harness (RENDERER_OPTIMIZATION_PLAN.md §Phase P): a larger world under a scripted
// SIMULATION STORM — a redstone-clock toggle band + periodic explosions + moving entities + firing
// pistons — driving the app façade exactly as the live sim will. Enables `instrument`, samples the full
// `RenderStats` surface every frame, renders a LIVE HUD, and dumps an aggregated baseline JSON.
//
// Fixes TRAP P.A: the old `stress.ts` runs collectDraws ONCE at setup; here the world is mutated every
// tick so the per-tick remesh / upload / cull / alloc cost the real sim triggers is actually measured.
// Self-check: window.__SIMSTORM__ = { ok, detail, baseline }.

import { SchematicViewer, type RendererSimulationDiff, type RenderStats } from "../src/app/SchematicViewer";
import { instrument } from "../src/core/Instrument";
import type { EditableSchematic } from "../../schematic-io/types";

const W = 64;
const H = 24;
const D = 64;
const TICKS = 120;

// Deterministic hash → a heightmap, so the world is identical run-to-run (no Math.random at world-gen).
function hash2(x: number, z: number): number {
  let h = (x * 374761393 + z * 668265263) | 0;
  h = (h ^ (h >>> 13)) * 1274126177;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}

function bigWorld(): EditableSchematic {
  const blocks: EditableSchematic["blocks"] = [];
  for (let x = 0; x < W; x++) {
    for (let z = 0; z < D; z++) {
      const h = 1 + Math.floor(hash2(x, z) * 4);
      for (let y = 0; y < h; y++) blocks.push({ x, y, z, name: "minecraft:stone", properties: {} });
      // scattered glass towers to exercise the translucent path + culling
      if (x % 11 === 5 && z % 11 === 5) for (let y = h; y < h + 6; y++) blocks.push({ x, y, z, name: "minecraft:glass", properties: {} });
    }
  }
  return { name: "simstorm", size: [W, H, D], blocks };
}

// The redstone-clock cell set: a band of cells re-dirtied every tick (toggled stone<->air), spread
// across many sections so the remesh storm + cull touch lots of section-passes.
const clockCells: { x: number; y: number; z: number }[] = [];
for (let x = 4; x < W - 4; x += 6) for (let z = 4; z < D - 4; z += 6) clockCells.push({ x, y: 6, z });

// A handful of pistons firing every frame (exercises the transient bake + opaque/translucent transient
// draws). Each pushes a stone block (case C) mid-slide so transient geometry is live every frame.
const pistonCells: { x: number; y: number; z: number }[] = [];
for (let x = 8; x < W - 8; x += 16) for (let z = 8; z < D - 8; z += 16) pistonCells.push({ x, y: 8, z });

function scenario(tick: number): RendererSimulationDiff {
  const on = tick % 2 === 0;
  const setBlocks: RendererSimulationDiff["setBlocks"] = [];
  const removedBlocks: RendererSimulationDiff["removedBlocks"] = [];
  for (const c of clockCells) {
    if (on) setBlocks!.push({ x: c.x, y: c.y, z: c.z, name: "minecraft:stone", properties: {} });
    else removedBlocks!.push(c);
  }
  // Moving entities (items drifting in a circle) — exercises the per-frame entity path.
  const entities: RendererSimulationDiff["entities"] = [];
  for (let i = 0; i < 8; i++) {
    const a = (tick * 0.1 + (i * Math.PI) / 4) % (Math.PI * 2);
    entities!.push({ id: `item-${i}`, type: "minecraft:item", position: [W / 2 + Math.cos(a) * 10, 8, D / 2 + Math.sin(a) * 10], properties: { item: "minecraft:diamond" } });
  }
  // Pistons mid-slide (progress 0.5) so the transient path bakes + draws every frame.
  const movingPistons: RendererSimulationDiff["movingPistons"] = pistonCells.map((c) => ({
    x: c.x, y: c.y, z: c.z,
    movedState: { name: "minecraft:stone", properties: {} },
    direction: "east", extending: true, isSourcePiston: false, progress: 0.5, progressO: 0,
  }));
  return { setBlocks, removedBlocks, entities, movingPistons };
}

function frames(n: number): Promise<void> {
  return new Promise((res) => {
    let i = 0;
    const tick = (): void => {
      if (++i >= n) res();
      else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

function pct(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}
function avg(arr: number[]): number {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}
const r2 = (n: number): number => +n.toFixed(2);

/** avg / p95 / max for one RenderStats field across the run. */
function agg(samples: RenderStats[], key: keyof RenderStats): { avg: number; p95: number; max: number } {
  const arr = samples.map((s) => s[key]);
  return { avg: r2(avg(arr)), p95: r2(pct(arr, 0.95)), max: r2(Math.max(0, ...arr)) };
}

/** Render the live RenderStats HUD (called every frame). */
function renderHud(s: RenderStats): void {
  const el = document.getElementById("stats");
  if (!el) return;
  const f = (n: number, w = 7) => String(r2(n)).padStart(w);
  el.textContent = [
    `frame ms   ${f(s.frameMs)}   (cull ${f(s.cullMs, 5)}  encode ${f(s.encodeMs, 5)}  resort ${f(s.resortMs, 5)}  entity ${f(s.entityMs, 5)})`,
    `gpu        draws ${f(s.drawCalls, 5)}  passes ${f(s.passes, 4)}  liveBuffers ${f(s.liveBuffers, 5)}`,
    `bufpool    createCalls/fr ${f(s.createBufferCalls, 4)}  poolHits/fr ${f(s.poolHits, 4)}`,
    `sections   drawn ${f(s.sectionsDrawn, 5)} / ${f(s.sectionsTotal, 5)}   culled ${f(s.sectionsCulled, 5)}   resorts ${f(s.resorts, 5)}`,
    `entities   ${f(s.entities, 4)}  draws ${f(s.entityDraws, 5)}  beStatic ${f(s.beStatic, 4)}  beAnim ${f(s.beAnimating, 4)}`,
    `alloc/fr   rawVtx ${f(s.allocRawVertex, 6)}  drawList ${f(s.allocDrawList, 5)}  frustum ${f(s.allocFrustum, 3)}  entEmit ${f(s.allocEntityEmit, 4)}`,
    `jobs/tick  dirtied ${f(s.jobsDirtied, 4)}  meshed ${f(s.jobsMeshed, 4)}  discarded ${f(s.jobsDiscarded, 3)}`,
    `occ/fr     bfsRuns ${f(s.occBfsRuns, 4)}  visited ${f(s.occVisited, 6)}`,
  ].join("\n");
}

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const viewer = new SchematicViewer(canvas);
  instrument.enabled = true; // Phase P: turn on the alloc/job counters for this run
  // Preload the item geometry for every dropped item this scenario spawns (mirrors app/page.tsx, which sets
  // itemIds from the inventory). Without it `buildItemGeometry` returns undefined → dropped items have no
  // model. The real app already does this; only this harness omitted it.
  viewer.itemIds = ["minecraft:diamond"];

  const packRes = await fetch("pack.zip");
  if (!packRes.ok) throw new Error(`fetch pack.zip failed (${packRes.status})`);
  await viewer.loadResourcePack(await packRes.arrayBuffer());
  viewer.loadSchematic(bigWorld());
  await viewer.whenReady();
  await frames(12); // warm up (pipeline + atlas + first cull + piston bakes)

  const samples: RenderStats[] = [];

  for (let t = 0; t < TICKS; t++) {
    await viewer.applySimulationDiff(scenario(t), { budgetMs: 6, timing: { animate: true, interval: 50, startedAtMs: performance.now() } });
    if (t % 40 === 39) {
      // explosion burst: blow a sphere out of the terrain + spawn the effect
      const cx = 16 + ((t * 7) % (W - 32));
      const cz = 16 + ((t * 5) % (D - 32));
      const removedBlocks: RendererSimulationDiff["removedBlocks"] = [];
      for (let dx = -3; dx <= 3; dx++) for (let dy = -3; dy <= 3; dy++) for (let dz = -3; dz <= 3; dz++) if (dx * dx + dy * dy + dz * dz <= 9) removedBlocks!.push({ x: cx + dx, y: 4 + dy, z: cz + dz });
      await viewer.applySimulationDiff({ removedBlocks }, { budgetMs: 6 });
      viewer.spawnExplosion(cx, 4, cz, 4);
    }
    await frames(1);
    const s = viewer.getStats().renderer;
    samples.push(s);
    renderHud(s);
  }

  const sectionsTotal = Math.max(0, ...samples.map((s) => s.sectionsTotal));
  const baseline = {
    scenario: `redstone-clock + explosion + piston · ${W}x${H}x${D} world · ${pistonCells.length} pistons`,
    ticks: TICKS,
    samples: samples.length,
    sectionsTotal,
    frameMs: agg(samples, "frameMs"),
    cullMs: agg(samples, "cullMs"),
    encodeMs: agg(samples, "encodeMs"),
    resortMs: agg(samples, "resortMs"),
    entityMs: agg(samples, "entityMs"),
    drawCalls: agg(samples, "drawCalls"),
    passes: agg(samples, "passes"),
    liveBuffers: agg(samples, "liveBuffers"),
    // MEM-2: with the buffer pool ON, createBufferCalls/tick drops under sustained churn and poolHitRate > 0.
    // OFF (the committed default), createBufferCalls == the logical create count and poolHits stays 0.
    createBufferCalls: agg(samples, "createBufferCalls"),
    poolHits: agg(samples, "poolHits"),
    poolHitRate: (() => {
      let hits = 0;
      let creates = 0;
      for (const s of samples) { hits += s.poolHits; creates += s.createBufferCalls; }
      return hits + creates > 0 ? r2(hits / (hits + creates)) : 0;
    })(),
    sectionsDrawn: agg(samples, "sectionsDrawn"),
    sectionsCulled: agg(samples, "sectionsCulled"),
    resorts: agg(samples, "resorts"),
    allocRawVertex: agg(samples, "allocRawVertex"),
    allocDrawList: agg(samples, "allocDrawList"),
    allocFrustum: agg(samples, "allocFrustum"),
    allocEntityEmit: agg(samples, "allocEntityEmit"),
    jobsDirtied: agg(samples, "jobsDirtied"),
    jobsMeshed: agg(samples, "jobsMeshed"),
    jobsDiscarded: agg(samples, "jobsDiscarded"),
    // OCC-1: BFS runs scale with home-section crossings + graphRevision bumps (sparse under a 20 TPS storm),
    // NOT frame count — so this stays bounded even though the camera moves every frame.
    occBfsRuns: agg(samples, "occBfsRuns"),
    occVisited: agg(samples, "occVisited"),
  };

  // PASS: the storm completed with geometry on screen, the Phase-P metrics were produced, and there was
  // no catastrophic frame stall. (This is a baseline-RECORDING harness, not a regression gate yet.)
  // NOTE: `allocRawVertex` is intentionally ~0 after Phase-0 #1 (single-pass inline vertex encode), so
  // it is NO LONGER a "metric is wired" sanity signal — `jobsMeshed > 0` proves meshing ran instead.
  const lastDrawn = baseline.sectionsDrawn.avg;
  const ok = lastDrawn > 0 && baseline.frameMs.max < 250 && baseline.frameMs.avg < 60 && baseline.drawCalls.max > 0 && baseline.jobsMeshed.max > 0;
  const detail =
    `ticks ${TICKS} · sections ${baseline.sectionsDrawn.avg}/${sectionsTotal} · frameMs avg ${baseline.frameMs.avg} p95 ${baseline.frameMs.p95} max ${baseline.frameMs.max} · ` +
    `draws ${baseline.drawCalls.avg} · liveBuf ${baseline.liveBuffers.max} · rawVtx/fr ${baseline.allocRawVertex.avg} · jobs/tick ${baseline.jobsMeshed.avg}`;
  window.__SIMSTORM__ = { ok, detail, baseline };
  setStatus(ok ? "pass" : "fail", `${ok ? "PASS" : "FAIL"} — ${detail}`);
  // The committed baseline (`harness/baselines/simstorm.json`) is copied from this dump.
  // eslint-disable-next-line no-console
  console.log("[simstorm] BASELINE_JSON " + JSON.stringify(baseline));
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
    __SIMSTORM__?: { ok: boolean; detail: string; baseline: unknown } | { ok: false; error: string };
  }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__SIMSTORM__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
