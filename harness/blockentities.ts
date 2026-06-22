// Phase 4.5 — every block-entity renderer + its underlying BLOCK MODEL, in a grid, free-fly camera.
// RENDERER_PLAN.md §18. Controls: WASD move, Space/Shift up/down, click-drag look.
//
// Many BEs render only an OVERLAY (bell = body, support is the block model; shelf = items, board is the
// block model; campfire/beacon/lectern/enchant/vault/brushable likewise). So each cell places the real
// BLOCK (terrain model) AND the BE — giving the full vanilla shape. Chest/sign/bed/skull/shulker/banner/
// pot have an empty block model (the BE is the whole geometry); placing the block is then a no-op.

import { WebGPUDevice } from "../src/core/webgpu/WebGPUDevice";
import { FreeCamera } from "../src/camera/FreeCamera";
import { type BlockEntry } from "../src/mesh/model/BakedBlockModel";
import { type BlockSource } from "../src/world/SnapshotSource";
import type { BlockProps } from "../src/mesh/model/BlockStateResolver";
import type { Vec3 } from "../src/types";
import { AIR, buildScene, buildEntityWorld, commitWorld, collectDraws } from "./scene";

interface Entry {
  /** BE type id == the block id (bare, no "minecraft:"). */
  type: string;
  props?: Record<string, string>;
  animating?: boolean;
  /** Don't place a terrain block (BE-only). */
  noBlock?: boolean;
  /** Extra block+BE at an offset (e.g. a bed's foot). */
  extra?: { type: string; props: Record<string, string>; dx: number; dz: number }[];
}

// Every BE loads IDLE (animating:false) — exactly the live app's default after loadSchematic. This proves
// the BBE §5 "inverted animation-trigger": idle-loop BEs (banners, conduit) wave/spin BY DEFAULT (no editor
// toggle → counted in beAnimating), while event-driven BEs (chest/shulker/bell/pot) bake into the static
// section mesh (counted in beStatic) until a setBlockEntityAnimating preview toggles them. (For an animating
// chest/bell preview, see harness/viewer.ts `?subject=…`, which sets animating:true.)
const DEMOS: Entry[] = [
  { type: "chest", props: { facing: "south", type: "single" } },
  { type: "chest", props: { facing: "south", type: "right" }, extra: [{ type: "chest", props: { facing: "south", type: "left" }, dx: 1, dz: 0 }] },
  { type: "trapped_chest", props: { facing: "south" } },
  { type: "ender_chest", props: { facing: "south" } },
  { type: "copper_chest", props: { facing: "south" } },
  { type: "red_shulker_box", props: { facing: "up" } },
  { type: "blue_shulker_box", props: { facing: "east" } },
  { type: "red_bed", props: { part: "head", facing: "north" }, extra: [{ type: "red_bed", props: { part: "foot", facing: "north" }, dx: 0, dz: 1 }] },
  { type: "oak_sign", props: { rotation: "0", text: "Hello|World|line 3|line 4" } },
  { type: "oak_wall_sign", props: { facing: "south", text: "Wall Sign|abcXYZ 123" } },
  { type: "oak_hanging_sign", props: { rotation: "0", text: "Hanging|Sign" } },
  { type: "white_banner", props: { rotation: "0", patterns: "stripe_bottom:red,cross:blue,border:black" } },
  { type: "red_banner", props: { rotation: "0", patterns: "creeper:lime,skull:white,border:yellow" } },
  { type: "skeleton_skull", props: { rotation: "0" } },
  { type: "creeper_head", props: { rotation: "0" } },
  { type: "zombie_head", props: { rotation: "0" } },
  { type: "player_head", props: { rotation: "0" } },
  { type: "wither_skeleton_wall_skull", props: { facing: "south" } },
  { type: "bell", props: { facing: "south", attachment: "floor" } },
  { type: "conduit", props: { active: "true", hunting: "true" } },
  { type: "copper_golem_statue", props: { facing: "south" } },
  { type: "enchanting_table" },
  { type: "lectern", props: { facing: "south", has_book: "true" } },
  { type: "decorated_pot", props: { facing: "south" } },
  { type: "beacon", props: { color: "white", height: "16" } },
  { type: "spawner" },
  { type: "vault", props: { facing: "south", item: "minecraft:diamond" } },
  { type: "suspicious_sand", props: { dusted: "2", item: "minecraft:emerald" } },
  { type: "campfire", props: { facing: "south", lit: "true", items: "minecraft:porkchop,minecraft:potato" } },
  { type: "oak_shelf", props: { facing: "south", items: "minecraft:diamond_sword,minecraft:gold_block,minecraft:apple" } },
  { type: "structure_block", props: { mode: "save", sizeX: "2", sizeY: "1", sizeZ: "2" }, noBlock: true },
  { type: "end_portal", noBlock: true },
];

const COLS = 6;
const SP = 3;
const Y = 1;

// ── Palette / world assembly ────────────────────────────────────────────────
function propsKey(p: BlockProps): string {
  return Object.keys(p).sort().map((k) => `${k}=${p[k]}`).join(",");
}

interface Placed { x: number; y: number; z: number; id: number; }
interface BE { x: number; y: number; z: number; type: string; props: Record<string, string>; animating: boolean; }

// Blocks the embedded BE specials (spawner/vault/brushable/shelf/campfire) render — must be atlased +
// preloaded even though they aren't placed in the world. Bare names (ResourcePack.preloadBlocks keys by
// the un-namespaced file path).
const EXTRA_BLOCKS = ["slime_block", "gold_block", "diamond_block", "emerald_block", "oak_planks", "hay_block"];

function assemble(): { palette: Record<number, BlockEntry>; placed: Placed[]; bes: BE[]; dims: Vec3 } {
  const palette: Record<number, BlockEntry> = {};
  const ids = new Map<string, number>();
  let next = 1;
  const IRONBLOCK = idOf("iron_block", {});
  function idOf(name: string, props: BlockProps): number {
    const key = `${name}|${propsKey(props)}`;
    let id = ids.get(key);
    if (id === undefined) {
      id = next++;
      ids.set(key, id);
      palette[id] = { name, props };
    }
    return id;
  }

  const placed: Placed[] = [];
  const bes: BE[] = [];
  const place = (x: number, y: number, z: number, type: string, props: Record<string, string>, animating: boolean, noBlock: boolean): void => {
    if (!noBlock) {
      // Bare block name (the resolver ignores unknown BE-only variant props).
      placed.push({ x, y, z, id: idOf(type, props) });
    }
    bes.push({ x, y, z, type, props, animating });
  };
  // Atlas + preload the embedded-item blocks (not placed in the world).
  for (const name of EXTRA_BLOCKS) idOf(name, {});

  DEMOS.forEach((d, i) => {
    const cx = (i % COLS) * SP + 1;
    const cz = Math.floor(i / COLS) * SP + 1;
    place(cx, Y, cz, d.type, d.props ?? {}, !!d.animating, !!d.noBlock);
    for (const e of d.extra ?? []) place(cx + e.dx, Y, cz + e.dz, e.type, e.props, false, false);
  });

  const w = COLS * SP + 2;
  const depth = (Math.ceil(DEMOS.length / COLS) + 1) * SP + 2;
  // Floor of stone under the whole grid.
  for (let x = 0; x < w; x++) for (let z = 0; z < depth; z++) placed.push({ x, y: 0, z, id: IRONBLOCK });
  return { palette, placed, bes, dims: [w, 4, depth] };
}

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const device = await WebGPUDevice.create(canvas);
  syncCanvasSize(canvas, device);
  window.addEventListener("resize", () => syncCanvasSize(canvas, device));

  const { palette, placed, bes, dims } = assemble();
  const scene = await buildScene(device, palette, (t) => setStatus("run", t));

  // World grid from the placed blocks (last write wins).
  const [W, H, D] = dims;
  const grid = new Uint16Array(W * H * D);
  const gi = (x: number, y: number, z: number) => (y * D + z) * W + x;
  const inB = (x: number, y: number, z: number) => x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D;
  for (const p of placed) if (inB(p.x, p.y, p.z)) grid[gi(p.x, p.y, p.z)] = p.id;
  const world: BlockSource = { getBlock: (x, y, z) => (inB(x, y, z) ? grid[gi(x, y, z)] : AIR) };

  const store = commitWorld(device, world, scene.provider, dims);
  const terrainDraws = collectDraws(store).draws;
  // Embedded BE items (vault/brushable `item`, shelf/campfire `items`) — preload their geometry so they
  // render as real items (generated sprites + block models) rather than block placeholders.
  const embeddedItems = [...new Set(DEMOS.flatMap((d) => [d.props?.item, ...(d.props?.items?.split(",") ?? [])].filter((s): s is string => !!s)))];
  const entityWorld = await buildEntityWorld(device, scene, undefined, embeddedItems);
  for (const b of bes) entityWorld.setBlockEntity({ x: b.x, y: b.y, z: b.z, type: b.type, props: b.props, animating: b.animating });

  const camera = new FreeCamera();
  camera.pos = [dims[0] / 2, 6, dims[2] + 6];
  camera.lookAt([dims[0] / 2, 1.5, dims[2] / 2]);
  installControls(canvas, camera);

  // Debug: `?focus=<beType>` (or a numeric grid index) frames one cell up close — for visual verify.
  const focus = new URLSearchParams(location.search).get("focus");
  if (focus) {
    const i = /^\d+$/.test(focus) ? Number(focus) : DEMOS.findIndex((d) => d.type === focus);
    if (i >= 0) {
      const cx = (i % COLS) * SP + 1, cz = Math.floor(i / COLS) * SP + 1;
      camera.pos = [cx + 0.5, Y + 2.2, cz + 4];
      camera.lookAt([cx + 0.5, Y + 1.2, cz + 0.5]);
    }
  }
  window.__focusBE = (type: string): void => {
    const i = DEMOS.findIndex((d) => d.type === type);
    if (i < 0) return;
    const cx = (i % COLS) * SP + 1, cz = Math.floor(i / COLS) * SP + 1;
    camera.pos = [cx + 0.5, Y + 2.2, cz + 4];
    camera.lookAt([cx + 0.5, Y + 1.2, cz + 0.5]);
  };

  renderFrame(0);
  const lit = litFraction(await device.readCanvasPixels());
  // The HYBRID split (BBE §1/§5): with EVERY BE idle (none toggled, mirroring the app), the static bake must
  // hold the event-driven BEs (beStatic > 0) while the idle-loop BEs (banners + conduit) still animate per
  // frame BY DEFAULT (beAnimating > 0) — a per-frame box draw can only exist here via the idleLoop route.
  const s = entityWorld.stats;
  const idleLoopBEs = bes.filter((b) => b.type.includes("banner") || b.type === "conduit").length;
  const hybridOk = s.beStatic > 0 && s.beAnimating > 0 && idleLoopBEs > 0;
  const ok = lit > 0.2 && s.drawn + s.special > 0 && hybridOk;
  window.__BLOCKENTITIES__ = { ok, detail: `${bes.length} BEs over ${placed.length} blocks · BE draws ${s.drawn} (static ${s.beStatic}, idle-loop anim ${s.beAnimating}/${idleLoopBEs}, special ${s.special}) · lit ${(lit * 100).toFixed(1)}%` };
  setStatus(ok ? "pass" : "fail", `${ok ? "PASS" : "FAIL"} — ${window.__BLOCKENTITIES__.detail}  ·  WASD move · drag look · Space/Shift up/down`);

  let t0 = 0;
  const loop = (ts: number): void => {
    if (!t0) t0 = ts;
    const dt = Math.min(0.05, (ts - (lastTs || ts)) / 1000);
    lastTs = ts;
    step(camera, dt);
    scene.atlas.tick(dt * 20); // advance animated block textures (campfire fire, water, lava) at 20 TPS
    renderFrame((ts - t0) / 1000);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  function renderFrame(clock: number): void {
    camera.aspect = canvas.width / canvas.height;
    scene.renderer.render(terrainDraws, camera, canvas.width, canvas.height);
    entityWorld.render(camera, 0, clock, canvas.width, canvas.height);
  }
}

// ── Input ─────────────────────────────────────────────────────────────────────
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
    __BLOCKENTITIES__?: { ok: boolean; detail: string } | { ok: false; error: string };
    __focusBE?: (type: string) => void;
  }
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  window.__BLOCKENTITIES__ = { ok: false, error: msg };
  setStatus("fail", `FAIL — ${msg}`);
});
