// Single-subject viewer. Renders ONE block / entity / block-entity / item, centred on a small platform,
// with an ORBIT camera (drag to rotate, wheel to zoom) — for clean vanilla-comparison verification without
// the grid demos' framing fights. RENDERER_PLAN.md §18.
//
//   viewer.html?block=minecraft:hopper[facing=down]
//   viewer.html?entity=hopper_minecart&spin=1
//   viewer.html?be=conduit&props=active=true,hunting=true
//   viewer.html?be=vault&props=item=minecraft:diamond
//   viewer.html?item=minecraft:diamond_sword
// Common opts: &props=k=v,k=v  &spin=0|1  &blocks=a,b  &items=a,b  &az=<deg>&el=<deg>&dist=<n>
// Self-check (window.__VIEWER__): lit fraction + draw count > 0.

import { WebGPUDevice } from "../src/core/webgpu/WebGPUDevice";
import { FreeCamera } from "../src/camera/FreeCamera";
import { type BlockEntry } from "../src/mesh/model/BakedBlockModel";
import { type BlockSource } from "../src/world/SnapshotSource";
import type { EntitySnapshot } from "../src/render/entities/EntityScene";
import { AIR, buildScene, buildEntityWorld, commitWorld, collectDraws } from "./scene";

const Q = new URLSearchParams(location.search);
const qProps = (): Record<string, string> => {
  const out: Record<string, string> = {};
  for (const pair of (Q.get("props") ?? "").split(",")) {
    const eq = pair.indexOf("=");
    if (eq > 0) out[pair.slice(0, eq).trim()] = pair.slice(eq + 1).trim();
  }
  return out;
};
const list = (s: string | null): string[] => (s ?? "").split(",").map((t) => t.trim()).filter(Boolean);
const bare = (s: string): string => s.replace(/^minecraft:/, "").replace(/\[.*$/, "");

// Cart contents (mirrors EntityModelFactory.CART_CONTENTS) — atlased so a minecart's block renders.
const CART_BLOCKS = ["chest", "hopper", "furnace", "tnt", "command_block"];

// The subject is placed at this cell on a 3×3 stone platform (floor at y=0, subject sits at y=1).
const C = 1;
const CENTER: [number, number, number] = [C + 0.5, 1.5, C + 0.5];

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const device = await WebGPUDevice.create(canvas);
  syncCanvasSize(canvas, device);
  window.addEventListener("resize", () => syncCanvasSize(canvas, device));

  const props = qProps();
  const mode = Q.has("be") ? "be" : Q.has("entity") ? "entity" : Q.has("item") ? "item" : "block";
  const subject = Q.get(mode === "block" ? "block" : mode) ?? "minecraft:stone";

  // ── Decide which blocks to atlas + which items to preload ──────────────────────
  const blocks = new Set<string>(["stone"]);
  const items = new Set<string>();
  for (const b of list(Q.get("blocks"))) blocks.add(bare(b));
  for (const it of list(Q.get("items"))) items.add(it);
  if (mode === "block") blocks.add(bare(subject));
  if (mode === "entity") {
    if (/minecart/.test(subject)) CART_BLOCKS.forEach((b) => blocks.add(b));
    if (props.block) blocks.add(bare(props.block));
  }
  if (mode === "item") items.add(subject);
  if (mode === "be") {
    if (props.item) { items.add(props.item); blocks.add(bare(props.item)); }
    for (const it of list(props.items)) { items.add(it); blocks.add(bare(it)); }
  }

  const palette: Record<number, BlockEntry> = {};
  const ids = new Map<string, number>();
  let next = 1;
  const idOf = (name: string): number => {
    let id = ids.get(name);
    if (id === undefined) { id = next++; ids.set(name, id); palette[id] = { name, props: {} }; }
    return id;
  };
  for (const b of blocks) idOf(b);
  const STONE = ids.get("stone")!;
  // A terrain-block subject needs its own palette entry WITH props (the provider is built at buildScene
  // time), so register it now; a BE sits on the floor and draws its own geometry (no block under it).
  let subjectBlockId = 0;
  if (mode === "block") {
    subjectBlockId = next++;
    palette[subjectBlockId] = { name: bare(subject), props: subject.includes("[") ? parseProps(subject) : {} };
  }

  const scene = await buildScene(device, palette, (t) => setStatus("run", t));

  // ── World: a single stone block under the subject (no wide platform — its near edge would occlude the
  // subject at a low orbit angle). The subject block (block mode) sits on top at (C,1,C). ──
  const W = 3, H = 3, D = 3;
  const grid = new Uint16Array(W * H * D);
  const gi = (x: number, y: number, z: number) => (y * D + z) * W + x;
  grid[gi(C, 0, C)] = STONE; // single grounding block
  if (mode === "block") grid[gi(C, 1, C)] = subjectBlockId;
  const world: BlockSource = { getBlock: (x, y, z) => (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D ? grid[gi(x, y, z)] : AIR) };
  const store = commitWorld(device, world, scene.provider, [W, H, D]);
  const terrainDraws = collectDraws(store).draws;

  const entityWorld = await buildEntityWorld(device, scene, undefined, [...items]);
  if (mode === "be") {
    entityWorld.setBlockEntity({ x: C, y: 1, z: C, type: subject, props, animating: true });
    // `&double=1` on a chest: place the partner half so the seam join can be inspected. For a south-facing
    // double chest the RIGHT half is the WEST block and LEFT the EAST (so their omitted seam faces meet),
    // so a `left` subject's partner goes WEST and a `right` subject's partner goes EAST.
    if (Q.get("double") === "1") {
      const partner = props.type === "right" ? "left" : "right";
      const dx = props.type === "right" ? 1 : -1;
      entityWorld.setBlockEntity({ x: C + dx, y: 1, z: C, type: subject, props: { ...props, type: partner }, animating: true });
      grid[gi(C + dx, 0, C)] = STONE;
      CENTER[0] += dx * 0.5;
    }
  }

  // ── Orbit camera ───────────────────────────────────────────────────────────────
  const cam = new FreeCamera();
  let az = (Number(Q.get("az")) || 35) * (Math.PI / 180);
  let el = (Number(Q.get("el")) || 22) * (Math.PI / 180);
  let dist = Number(Q.get("dist")) || 3.4;
  const place = (): void => {
    cam.pos = [CENTER[0] + dist * Math.cos(el) * Math.sin(az), CENTER[1] + dist * Math.sin(el), CENTER[2] + dist * Math.cos(el) * Math.cos(az)];
    cam.lookAt(CENTER);
    cam.aspect = canvas.width / canvas.height;
  };
  installOrbit(canvas, (dAz, dEl) => { az += dAz; el = Math.max(-1.4, Math.min(1.4, el + dEl)); }, (dz) => { dist = Math.max(1.5, Math.min(12, dist * dz)); });

  const spin = Q.get("spin") !== "0"; // entities/items spin by default to expose all sides
  const snapshot = (clock: number): EntitySnapshot[] => {
    if (mode === "entity") return [{ id: subject, type: ns(subject), position: CENTER, properties: { ...props, ...(spin ? { yRot: String((clock * 45) % 360) } : {}) } }];
    if (mode === "item") return [{ id: "item", type: "minecraft:item", position: CENTER, properties: { item: subject } }];
    return [];
  };

  const renderFrame = (clock: number): void => {
    place();
    if (mode === "entity" || mode === "item") entityWorld.ingestEntities(snapshot(clock));
    scene.renderer.render(terrainDraws, cam, canvas.width, canvas.height);
    entityWorld.render(cam, 0, clock, canvas.width, canvas.height);
  };

  renderFrame(0.4);
  const lit = litFraction(await device.readCanvasPixels());
  const drawn = entityWorld.stats.drawn + (entityWorld.stats.special ?? 0) + terrainDraws.length;
  const ok = lit > 0.05 && drawn > 0;
  window.__VIEWER__ = { ok, detail: `${mode}:${subject} · draws ${drawn} (ent ${entityWorld.stats.drawn}, special ${entityWorld.stats.special ?? 0}) · lit ${(lit * 100).toFixed(1)}%` };
  setStatus(ok ? "pass" : "fail", `${ok ? "PASS" : "FAIL"} — ${window.__VIEWER__.detail}`);

  let t0 = 0;
  const loop = (ts: number): void => {
    if (!t0) t0 = ts;
    const dt = Math.min(0.05, (ts - (lastTs || ts)) / 1000);
    lastTs = ts;
    scene.atlas.tick(dt * 20);
    renderFrame((ts - t0) / 1000);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function ns(t: string): string { return t.startsWith("minecraft:") ? t : `minecraft:${t}`; }
function parseProps(state: string): Record<string, string> {
  const out: Record<string, string> = {};
  const inner = state.slice(state.indexOf("[") + 1, state.lastIndexOf("]"));
  for (const pair of inner.split(",")) { const eq = pair.indexOf("="); if (eq > 0) out[pair.slice(0, eq).trim()] = pair.slice(eq + 1).trim(); }
  return out;
}

// ── Orbit input ────────────────────────────────────────────────────────────────
let dragging = false, lastX = 0, lastY = 0, lastTs = 0;
function installOrbit(canvas: HTMLCanvasElement, orbit: (dAz: number, dEl: number) => void, zoom: (dz: number) => void): void {
  canvas.addEventListener("mousedown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener("mouseup", () => { dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    orbit(-(e.clientX - lastX) * 0.01, (e.clientY - lastY) * 0.01);
    lastX = e.clientX; lastY = e.clientY;
  });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoom(e.deltaY > 0 ? 1.1 : 0.9); }, { passive: false });
}

function litFraction(shot: { width: number; height: number; bytesPerRow: number; data: Uint8Array }): number {
  let lit = 0, n = 0;
  for (let y = 0; y < shot.height; y += 3) for (let x = 0; x < shot.width; x += 3) {
    const o = y * shot.bytesPerRow + x * 4;
    // count non-background pixels (the sky is #aac8e6 ≈ 170,200,230) — anything notably off that is "lit"
    const r = shot.data[o], g = shot.data[o + 1], b = shot.data[o + 2];
    if (Math.abs(r - 170) + Math.abs(g - 200) + Math.abs(b - 230) > 40) lit++;
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
  canvas.width = w; canvas.height = h;
  device.resize(w, h);
}

function setStatus(cls: string, text: string): void {
  const el = document.getElementById("status");
  if (el) { el.className = cls; el.textContent = text; }
}

declare global {
  interface Window { __VIEWER__?: { ok: boolean; detail: string } | { ok: false; error: string }; }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__VIEWER__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
