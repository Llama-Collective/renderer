// Phase 4.5 — interactive showcase of EVERY entity the renderer implements, over a terrain floor.
// RENDERER_PLAN.md §18. Controls: WASD move, Space/Shift up/down, click-drag look (like the BE demo).
//
// Roster (all current box-model + block-display entities): falling_block, primed_tnt (flashing), a
// dropped block-item (bob+spin), every minecart variant (body + contained block), and sheep in all 16
// dye colours (velocity-driven leg walk). Entities draw in a pass that LOADS terrain's depth (TRAP 18.B).
// `?focus=<type>` frames one cell up close. Self-check (window.__ENTITIES__): lit + entity draws > 0.

import { WebGPUDevice } from "../src/core/webgpu/WebGPUDevice";
import { FreeCamera } from "../src/camera/FreeCamera";
import { type BlockEntry } from "../src/mesh/model/BakedBlockModel";
import { type BlockSource } from "../src/world/SnapshotSource";
import type { Vec3 } from "../src/types";
import type { EntitySnapshot } from "../src/render/entities/EntityScene";
import { AIR, buildScene, buildEntityWorld, buildItemGeometry, commitWorld, collectDraws } from "./scene";
import { ItemSlotRenderer, type ItemSlot } from "../src/render/items/ItemSlotRenderer";

const DYES = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black"];

interface EntityDemo {
  type: string;
  props?: Record<string, string>;
  velocity?: Vec3;
  /** Per-frame animation hook applied in `snapshot()`. */
  anim?: "fall" | "tnt" | "spin";
  label?: string;
}

const DEMOS: EntityDemo[] = [
  { type: "falling_block", props: { block: "minecraft:sand" }, anim: "fall" },
  { type: "primed_tnt", anim: "tnt" },
  { type: "item", props: { item: "minecraft:stick" }, label: "item:stick" },
  { type: "item", props: { item: "minecraft:apple" }, label: "item:apple" },
  { type: "item", props: { item: "minecraft:diamond_sword" }, label: "item:sword" },
  { type: "item", props: { item: "minecraft:diamond" }, label: "item:diamond" },
  { type: "item", props: { item: "minecraft:diamond_block" }, label: "item:diamond_block" },
  { type: "minecart", anim: "spin" },
  { type: "chest_minecart", anim: "spin" },
  { type: "furnace_minecart", anim: "spin" },
  { type: "hopper_minecart", anim: "spin" },
  { type: "tnt_minecart", anim: "spin" },
  { type: "command_block_minecart", anim: "spin" },
  { type: "oak_boat", label: "oak_boat" },
  { type: "spruce_boat", label: "spruce_boat" },
  { type: "spruce_chest_boat", label: "spruce_chest_boat" },
  { type: "bamboo_raft", label: "bamboo_raft" },
  { type: "pig", props: { yBodyRot: "180" }, velocity: [0.3, 0, 0] },
  { type: "cow", props: { yBodyRot: "180" }, velocity: [0.3, 0, 0] },
  // Item frames (frame block model + facing + contained item). The diamond is already in `itemIds` (above).
  { type: "item_frame", props: { facing: "south", item: "minecraft:diamond", item_rotation: "1" }, label: "item_frame" },
  { type: "glow_item_frame", props: { facing: "south" }, label: "glow_item_frame" },
  ...DYES.map((c): EntityDemo => ({ type: "sheep", props: { color: c, yBodyRot: "180" }, velocity: [0.3, 0, 0], label: `sheep:${c}` })),
];

const COLS = 6;
const SP = 3;
const BASE_Y = 1.0;

// Blocks referenced by block-DISPLAY entities (falling_block, dropped block-item) — must be in the palette
// so their sprites get atlased + models bake. The floor uses stone. The minecart CONTENTS blocks (chest,
// hopper, furnace, command_block, tnt) are intentionally NOT listed: buildScene auto-preloads + atlases the
// ENTITY_CONTAINED_BLOCKS set, so this demo exercises that path (the "hopper cart has no block" fix).
const BLOCKS = ["stone", "sand", "diamond_block"];

function gridSize(): { w: number; d: number } {
  return { w: COLS * SP + 4, d: (Math.ceil(DEMOS.length / COLS) + 1) * SP + 4 };
}
function cell(i: number): { cx: number; cz: number } {
  return { cx: (i % COLS) * SP + 2, cz: Math.floor(i / COLS) * SP + 2 };
}

/** This frame's entity snapshots, with per-entity animation folded into position/properties. */
function snapshot(clock: number): EntitySnapshot[] {
  return DEMOS.map((d, i): EntitySnapshot => {
    const { cx, cz } = cell(i);
    let y = BASE_Y;
    const props: Record<string, string> = { ...(d.props ?? {}) };
    if (d.anim === "fall") y = BASE_Y + 0.2 + 1.4 * Math.abs(Math.sin(clock * 1.5 + i)); // bobbing fall
    if (d.anim === "tnt") props.fuse = String(80 - Math.floor((clock * 20) % 80)); // count down → flash + pulse
    if (d.anim === "spin") props.yRot = String((clock * 35) % 360);
    return { id: d.label ?? d.type, type: `minecraft:${d.type}`, position: [cx + 0.5, y, cz + 0.5], properties: props, velocity: d.velocity };
  });
}

function floorSource(w: number, h: number, d: number, stone: number): BlockSource {
  const idx = (x: number, y: number, z: number) => (y * d + z) * w + x;
  const inB = (x: number, y: number, z: number) => x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d;
  const grid = new Uint16Array(w * h * d);
  for (let x = 0; x < w; x++) for (let z = 0; z < d; z++) grid[idx(x, 0, z)] = stone;
  return { getBlock: (x, y, z) => (inB(x, y, z) ? grid[idx(x, y, z)] : AIR) };
}

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const device = await WebGPUDevice.create(canvas);
  syncCanvasSize(canvas, device);
  window.addEventListener("resize", () => syncCanvasSize(canvas, device));

  const palette: Record<number, BlockEntry> = {};
  BLOCKS.forEach((name, i) => { palette[i + 1] = { name, props: {} }; });
  const STONE = 1; // BLOCKS[0]

  const scene = await buildScene(device, palette, (t) => setStatus("run", t));
  const { w, d } = gridSize();
  const H = 6;
  const store = commitWorld(device, floorSource(w, H, d, STONE), scene.provider, [w, H, d]);
  const terrainDraws = collectDraws(store).draws;
  // Preload the dropped items' geometry (generated sprites → extruded; the item system).
  const itemIds = [...new Set(DEMOS.filter((d) => d.type === "item").map((d) => d.props!.item))];
  const world = await buildEntityWorld(device, scene, undefined, itemIds);

  // Inventory-slot (GUI) item rendering — generated items flat, block items isometric (the exposed form).
  const GUI_ITEMS = ["minecraft:stick", "minecraft:apple", "minecraft:diamond_sword", "minecraft:diamond", "minecraft:stone", "minecraft:diamond_block"];
  const slotGeom = await buildItemGeometry(device, scene, GUI_ITEMS);
  const slotRenderer = slotGeom ? new ItemSlotRenderer(device, device.colorFormat, device.depthFormat, slotGeom) : undefined;
  const guiSlots = (): ItemSlot[] => {
    const SZ = 110, PAD = 14, X0 = 30, Y0 = 90, COLS_ = 3;
    return GUI_ITEMS.map((item, i) => ({ item, x: X0 + (i % COLS_) * (SZ + PAD), y: Y0 + Math.floor(i / COLS_) * (SZ + PAD), size: SZ }));
  };

  const camera = new FreeCamera();
  camera.pos = [w / 2, 6, d + 6];
  camera.lookAt([w / 2, 1.5, d / 2]);
  installControls(canvas, camera);

  // Debug: `?focus=<entityType>` frames one cell up close.
  const focus = new URLSearchParams(location.search).get("focus");
  if (focus) {
    const i = /^\d+$/.test(focus) ? Number(focus) : DEMOS.findIndex((e) => e.type === focus || e.label === focus);
    if (i >= 0) {
      const { cx, cz } = cell(i);
      if (/minecart/.test(DEMOS[i].type)) {
        camera.pos = [cx + 2.4, BASE_Y + 2.2, cz + 2.4]; // 3/4 side-and-above so the whole cart + contents frame
        camera.lookAt([cx + 0.5, BASE_Y + 0.7, cz + 0.5]);
      } else {
        camera.pos = [cx + 0.5, BASE_Y + 1.1, cz + 2.2];
        camera.lookAt([cx + 0.5, BASE_Y + 0.6, cz + 0.5]);
      }
    }
  }

  const renderFrame = (clock: number): void => {
    world.ingestEntities(snapshot(clock));
    camera.aspect = canvas.width / canvas.height;
    scene.renderer.render(terrainDraws, camera, canvas.width, canvas.height);
    world.render(camera, 0, clock, canvas.width, canvas.height);
    // GUI inventory overlay (the exposed slot form) — drawn over the world in an ortho pass.
    slotRenderer?.render(guiSlots(), canvas.width, canvas.height);
  };

  renderFrame(0.5);
  const lit = litFraction(await device.readCanvasPixels());
  const ok = lit > 0.2 && world.stats.drawn > 0;
  const detail = `${DEMOS.length} entities · draws ${world.stats.drawn} (entities ${world.stats.entities}) · lit ${(lit * 100).toFixed(1)}%`;
  window.__ENTITIES__ = { ok, detail };
  setStatus(ok ? "pass" : "fail", `${ok ? "PASS" : "FAIL"} — ${detail}  ·  WASD move · drag look · Space/Shift up/down`);

  let t0 = 0;
  const loop = (ts: number): void => {
    if (!t0) t0 = ts;
    const dt = Math.min(0.05, (ts - (lastTs || ts)) / 1000);
    lastTs = ts;
    step(camera, dt);
    renderFrame((ts - t0) / 1000);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

// ── Input (mirrors the BE demo) ────────────────────────────────────────────────
const keys = new Set<string>();
let dragging = false, lastX = 0, lastY = 0, lastTs = 0;

function installControls(canvas: HTMLCanvasElement, camera: FreeCamera): void {
  window.addEventListener("keydown", (e) => keys.add(e.key.toLowerCase()));
  window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
  canvas.addEventListener("mousedown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener("mouseup", () => { dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    camera.rotate((e.clientX - lastX) * 0.005, -(e.clientY - lastY) * 0.005);
    lastX = e.clientX;
    lastY = e.clientY;
  });
}

function step(camera: FreeCamera, dt: number): void {
  const speed = (keys.has("shift") ? 14 : 7) * dt;
  let f = 0, r = 0, u = 0;
  if (keys.has("w")) f += speed;
  if (keys.has("s")) f -= speed;
  if (keys.has("d")) r += speed;
  if (keys.has("a")) r -= speed;
  if (keys.has(" ")) u += speed;
  if (keys.has("control") || keys.has("c")) u -= speed;
  if (f || r || u) camera.moveLocal(f, r, u);
}

function litFraction(shot: { width: number; height: number; bytesPerRow: number; data: Uint8Array }): number {
  let lit = 0, n = 0;
  for (let y = 0; y < shot.height; y += 3) for (let x = 0; x < shot.width; x += 3) {
    const o = y * shot.bytesPerRow + x * 4;
    if (Math.max(shot.data[o], shot.data[o + 1], shot.data[o + 2]) > 20) lit++;
    n++;
  }
  return lit / n;
}

const MAX_DIM = 8192;
function syncCanvasSize(canvas: HTMLCanvasElement, device: WebGPUDevice): void {
  const scale = window.devicePixelRatio || 1;
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
    __ENTITIES__?: { ok: boolean; detail: string } | { ok: false; error: string };
  }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__ENTITIES__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
