// Functional integration smoke test: drives the app FAÇADE (SchematicViewer) exactly as app/page.tsx
// does — load resource pack → load a schematic → render → apply a simulation diff → confirm the affected
// section remeshed (more geometry → more lit pixels). RENDERER_INTEGRATION_PLAN Step 11.
// Self-check: window.__INTEGRATION__ = { ok, detail }.

import { SchematicViewer } from "../src/app/SchematicViewer";
import type { EditableSchematic } from "../../schematic-io/types";

// A small synthetic world (no WASM/.litematic parse needed): an 8×8 stone floor + a glass block, a
// plank cube, and a water cell — exercising solid + cutout-free + translucent + fluid paths — plus a sheep.
function syntheticSchematic(): EditableSchematic {
  const blocks: EditableSchematic["blocks"] = [];
  for (let x = 0; x < 8; x++) for (let z = 0; z < 8; z++) blocks.push({ x, y: 0, z, name: "minecraft:stone", properties: {} });
  blocks.push({ x: 2, y: 1, z: 2, name: "minecraft:glass", properties: {} });
  blocks.push({ x: 5, y: 1, z: 5, name: "minecraft:oak_planks", properties: {} });
  blocks.push({ x: 5, y: 1, z: 2, name: "minecraft:water", properties: { level: "0" } });
  return {
    name: "integration-smoke",
    size: [8, 8, 8],
    blocks,
    entities: [{ id: "sheep-1", type: "minecraft:sheep", position: [4, 1, 4], properties: { color: "white" } }],
  };
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

// Non-sky pixel fraction. Background is the renderer's linear sky [0.45,0.62,0.85] → ~(181,206,237) sRGB.
function litFraction(shot: { width: number; height: number; bytesPerRow: number; data: Uint8Array }): number {
  let lit = 0;
  let n = 0;
  for (let y = 0; y < shot.height; y += 3) {
    for (let x = 0; x < shot.width; x += 3) {
      const o = y * shot.bytesPerRow + x * 4;
      const a = shot.data[o];
      const b = shot.data[o + 1];
      const c = shot.data[o + 2];
      // Channel-order-agnostic: sky is bluish ~(181,206,237)/BGRA-swapped; "lit" = far from that set.
      const d = Math.abs(a - 181) + Math.abs(b - 206) + Math.abs(c - 237);
      const dSwap = Math.abs(a - 237) + Math.abs(b - 206) + Math.abs(c - 181);
      if (Math.min(d, dSwap) > 50) lit++;
      n++;
    }
  }
  return n ? lit / n : 0;
}

/** Fraction of pixels that changed between two shots — a background-independent signal that a remesh actually
 *  committed and re-rendered (adding a stone tower MUST change the image). Robust where litFraction is not: the
 *  SchematicViewer renders a dark GRID, not the light sky litFraction assumes, so its "lit" saturates near 100%
 *  and a lit-increase can't detect the remesh. */
function pixelChange(a: { width: number; height: number; bytesPerRow: number; data: Uint8Array }, b: typeof a): number {
  let changed = 0, n = 0;
  for (let y = 0; y < a.height; y += 3) {
    for (let x = 0; x < a.width; x += 3) {
      const o = y * a.bytesPerRow + x * 4;
      if (Math.abs(a.data[o] - b.data[o]) + Math.abs(a.data[o + 1] - b.data[o + 1]) + Math.abs(a.data[o + 2] - b.data[o + 2]) > 30) changed++;
      n++;
    }
  }
  return n ? changed / n : 0;
}

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const viewer = new SchematicViewer(canvas);
  // The esbuild harness serve can't bundle the viewer's `new URL(...mesher.worker.ts)` worker (production's
  // turbopack does), so inject the prebuilt worker bundle — the SAME one workers.html uses — so this exercises
  // REAL off-thread meshing (diff → worker mesh → commit → render), matching production.
  viewer.workerFactory = (index) => new Worker(new URL("./dist/mesher.worker.entry.js", location.href), { type: "module", name: `renderer-mesher-${index}` });
  const detail: string[] = [];

  const packRes = await fetch("pack.zip");
  if (!packRes.ok) throw new Error(`fetch pack.zip failed (${packRes.status})`);
  await viewer.loadResourcePack(await packRes.arrayBuffer());
  viewer.loadSchematic(syntheticSchematic());
  await viewer.whenReady();
  await frames(4);

  const shot0 = await viewer.readback();
  const lit0 = litFraction(shot0);
  const stats0 = viewer.getStats().renderer ?? {};
  detail.push(`initial lit ${(lit0 * 100).toFixed(1)}% drawn ${stats0.sectionsDrawn ?? 0}/${stats0.sectionsTotal ?? 0}`);

  // Sim diff: drop a 3×6×3 stone tower in the centre of the framed view → the section remeshes with
  // more geometry, so the lit fraction must rise. Proves the diff→dirty→remesh→commit path runs.
  const setBlocks: { x: number; y: number; z: number; name: string; properties: Record<string, string> }[] = [];
  for (let x = 3; x <= 5; x++) for (let y = 1; y <= 6; y++) for (let z = 3; z <= 5; z++) setBlocks.push({ x, y, z, name: "minecraft:stone", properties: {} });
  await viewer.applySimulationDiff({ setBlocks }, { budgetMs: 8 });
  await frames(30); // off-thread worker mesh → drain → commit → re-render (worker round-trip needs a few frames)

  const shot1 = await viewer.readback();
  // The diff dropped a 3×6×3 stone tower in the framed centre: the section must remesh + commit + re-render, so
  // the image must visibly change. Background-independent (works against the dark grid the viewer renders).
  const changed = pixelChange(shot0, shot1);
  detail.push(`diff changed ${(changed * 100).toFixed(2)}% of pixels`);

  // Spawn an explosion (smoke-check the effect path doesn't throw).
  viewer.spawnExplosion(4, 2, 4, 3);
  await frames(3);

  const rendered = (stats0.sectionsDrawn ?? 0) > 0 && lit0 > 0.02; // initial scene drew sections + is non-blank
  const remeshed = changed > 0.002; // the diff→dirty→remesh→commit→render path moved real pixels
  const ok = rendered && remeshed;
  window.__INTEGRATION__ = { ok, detail: detail.join(" · ") + (ok ? " · PASS" : ` · FAIL(rendered=${rendered} remeshed=${remeshed})`) };
  setStatus(ok ? "pass" : "fail", `${ok ? "PASS" : "FAIL"} — ${window.__INTEGRATION__.detail}`);
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
    __INTEGRATION__?: { ok: boolean; detail: string } | { ok: false; error: string };
  }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__INTEGRATION__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
