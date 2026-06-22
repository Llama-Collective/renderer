// Turns a RenderEntity into EntityDraws. RENDERER_PLAN.md §18, Phase 4.5a/4.5b.
//
// Geometry is baked ONCE per (type, variant) and cached as a GPU mesh, then instanced with a per-entity
// model matrix + tint/flash (the §18 "cache baked geometry, instance with a matrix" optimization). Two
// geometry sources (vanilla splits cleanly):
//   - block-display entities (falling_block / primed_tnt / dropped block-item / minecart contents) reuse
//     the EXISTING block baker → block atlas (4.5a);
//   - box-model entities (sheep / minecart body / boat / arrow) use the ModelPart engine → entity atlas
//     (4.5b), injected via `boxModels`.
// Unhandled types log once and (for now) render nothing; a labelled fallback box lands with the entity
// atlas. Block entities are handled separately (the hybrid static-bake path, §18 / Phase 4.5c).

import type { GpuBufferHandle, GpuTextureHandle, GraphicsDevice } from "../../core/GraphicsDevice";
import { BufferUsage } from "../../core/GraphicsDevice";
import type { BakedBlockModel } from "../../mesh/model/BakedBlockModel";
import type { BlockProps } from "../../mesh/model/BlockStateResolver";
import { packVertices, PackedVertexSink, type RawVertex } from "../../mesh/VertexFormat";
import { instrument } from "../../core/Instrument";
import { Mat4Frame, type Mat4 } from "../../mesh/entity/mat4";
import { TerrainPass, TERRAIN_PASSES, type Vec3 } from "../../types";
import { bakedModelToEntityVerts, hasGeometry, parseBlockState, type EntityVertsByPass } from "./blockDisplay";
import { ItemGeometry } from "../items/ItemGeometry";
import { itemDisplayMatrix } from "../items/itemDisplay";
import { WORLD_ITEM_LIGHT0, WORLD_ITEM_LIGHT1, WORLD_ENTITY_LIGHT0, WORLD_ENTITY_LIGHT1 } from "../items/itemLighting";
import type { RenderEntity } from "./EntityScene";
import type { EntityDraw } from "./EntityRenderer";

/** A box-model entity geometry source (the 4.5b ModelPart path), injected so 4.5a stays standalone. */
export interface BoxModelSource {
  /** Atlas the box models sample from (entity atlas). */
  texture: GpuTextureHandle;
  /** Build the draw layers for `entity` (e.g. sheep body + dyed wool overlay), or null if unsupported.
   *  `mat` is the factory's per-frame matrix arena — the layer `model` matrices build into it (A3). */
  build(entity: RenderEntity, clock: number, mat: Mat4Frame): BoxModelLayer[] | null;
}

/** One drawable layer of a box-model entity: its own geometry + world matrix + colour state. */
export interface BoxModelLayer {
  /** Geometry-cache key. Static layers share by key; DYNAMIC (re-posed) layers must key per entity id. */
  geomKey: string;
  /** Lazy geometry bake (AUDIT ENT-1): a THUNK, so a static-layer cache hit skips the bake entirely
   *  (dozens-to-hundreds of resting mobs no longer re-bake + alloc every frame just to be discarded). */
  verts: () => EntityVertsByPass;
  model: Mat4;
  tint?: readonly [number, number, number, number];
  flash?: readonly [number, number, number, number];
  /** Re-posed per frame (walk swing etc.) → re-encode into a reused buffer instead of static caching. */
  dynamic?: boolean;
  /** Dynamic layers may bake STRAIGHT into a packed sink (single Cutout pass) — no `RawVertex[]`, no
   *  per-frame packing buffer (A4). When present on a `dynamic` layer, the factory uses it over `verts`. */
  bakeInto?: (sink: PackedVertexSink) => void;
  sortPos?: Vec3;
}

export interface EntityFactoryDeps {
  device: GraphicsDevice;
  /** Bake a blockstate into a model, or null if unknown (reuses the terrain baker). */
  bakeBlock: (name: string, props: BlockProps) => BakedBlockModel | null;
  blockAtlas: GpuTextureHandle;
  /** Optional box-model engine (4.5b); when absent, only block-display entities render. */
  boxModels?: BoxModelSource;
  /** Optional item geometry (generated sprite extrusion / block model); enables real dropped-item sprites. */
  itemGeometry?: ItemGeometry;
}

interface PassMesh {
  pass: TerrainPass;
  vertex: GpuBufferHandle;
  quadCount: number;
  texture: GpuTextureHandle;
}

export class EntityModelFactory {
  private readonly geomCache = new Map<string, PassMesh[]>();
  private readonly warned = new Set<string>();
  /** Per-frame mat4 arena (A3): every per-entity model matrix is built into a reused buffer, consumed by
   *  EntityRenderer this same frame, then rewound at the next `beginFrame` — so a static/animated entity
   *  scene allocates no matrices in steady state. */
  private readonly mat = new Mat4Frame(() => instrument.bumpAlloc("entityEmit")); // X5: arena growth gauge
  /** Per-frame EntityDraw pool (A4): draws emit into reused objects pushed onto the caller's drawScratch,
   *  consumed by EntityRenderer this same frame, then rewound at `beginFrame` — no per-entity draw object,
   *  array, `.map`, or spread. Reused objects carry stale fields, so `emit` sets EVERY field each time. */
  private readonly drawPool: EntityDraw[] = [];
  private drawCursor = 0;
  /** Per-frame Vec3 pool for sort positions DERIVED from a model matrix (A4). Entity-owned positions
   *  (`e.position`) are passed through directly and consume no slot. */
  private readonly sortPool: [number, number, number][] = [];
  private sortCursor = 0;
  /** Reused packed-vertex sink for the dynamic-mob bake (A4) — one per factory, refilled per layer (the
   *  bytes are uploaded via `writeBuffer`, which copies synchronously, before the next layer reuses it). */
  private readonly dynSink = new PackedVertexSink();

  constructor(private readonly deps: EntityFactoryDeps) {}

  /** Build this frame's draws for one interpolated entity into `out` (the caller's pooled drawScratch).
   *  Returns the number of draws appended (the caller counts an entity as drawn iff > 0). `clock` (seconds)
   *  drives idle anim. */
  buildDraws(e: RenderEntity, clock: number, out: EntityDraw[]): number {
    const before = out.length;
    const type = strip(e.type);
    switch (type) {
      case "falling_block": this.fallingBlock(e, out); break;
      case "tnt":
      case "primed_tnt": this.primedTnt(e, out); break;
      case "item": this.droppedItem(e, clock, out); break;
      case "item_frame":
      case "glow_item_frame": this.itemFrame(e, type, out); break;
      case "minecart":
      case "chest_minecart":
      case "hopper_minecart":
      case "tnt_minecart":
      case "furnace_minecart":
      case "command_block_minecart": this.minecart(e, type, clock, out); break;
      default: this.boxModelOrFallback(e, clock, out);
    }
    return out.length - before;
  }

  /**
   * Emit draws for a blockstate at a world `model` matrix (geometry in [0,1] block space) into `out`. Used
   * for block-entity special draws that embed a block (mob-spawner placeholder, brushable item, shelf /
   * campfire items, vault display). Cached by blockstate.
   */
  blockModelDraws(block: string, model: Mat4, out: EntityDraw[], tint?: readonly [number, number, number, number]): void {
    const { name, props } = parseBlockState(block);
    const mesh = this.blockMesh(`be|${block}`, name, props);
    if (!mesh) return;
    this.emit(out, mesh, model, this.sortOf(model), tint);
  }

  /**
   * Emit draws for an item embedded in a block entity (vault / shelf / brushable / campfire) at a world
   * `model` matrix into `out`. Routes through the real item system — a generated item renders as its
   * extruded sprite, a block item as the baked block model (diffuse-lit + back-face culled like a dropped
   * block-item). Falls back to a block placeholder if no item geometry is wired. Reuses the `item|…` cache.
   */
  itemDraws(item: string, model: Mat4, out: EntityDraw[]): void {
    const geom = this.deps.itemGeometry?.get(item);
    if (!geom) { this.blockModelDraws(item, model, out); return; }
    const mesh = this.getMesh(`item|${item}`, () => ({ [TerrainPass.Cutout]: geom.verts }), geom.texture);
    if (!mesh.length) return;
    // block + entity (builtin/entity BE box) items: diffuse-lit + back-face culled; generated sprites stay flat.
    if (geom.kind !== "generated") this.emit(out, mesh, model, this.sortOf(model), undefined, undefined, WORLD_ITEM_LIGHT0, WORLD_ITEM_LIGHT1, true);
    else this.emit(out, mesh, model, this.sortOf(model));
  }

  // ── Pooled draw emit (A4) ─────────────────────────────────────────────────────

  /** Next pooled draw slot (grows on demand; rewound each `beginFrame`). `emit` fills every field. */
  private nextDraw(): EntityDraw {
    let d = this.drawPool[this.drawCursor];
    if (d === undefined) { d = {} as EntityDraw; this.drawPool[this.drawCursor] = d; instrument.bumpAlloc("entityEmit"); } // X5: pool growth
    this.drawCursor++;
    return d;
  }

  /** A model matrix's translation column as the sort position, into a pooled Vec3 (A4). */
  private sortOf(m: Mat4): Vec3 {
    let v = this.sortPool[this.sortCursor];
    if (v === undefined) { v = [0, 0, 0]; this.sortPool[this.sortCursor] = v; instrument.bumpAlloc("entityEmit"); } // X5: pool growth
    this.sortCursor++;
    v[0] = m[12]; v[1] = m[13]; v[2] = m[14];
    return v;
  }

  /** Emit one pooled draw per pass-mesh into `out`, setting EVERY field (reused objects carry stale state). */
  private emit(
    out: EntityDraw[], mesh: PassMesh[], model: Mat4, sortPos: Vec3,
    tint?: readonly [number, number, number, number], flash?: readonly [number, number, number, number],
    light0?: readonly [number, number, number, number], light1?: readonly [number, number, number, number],
    cull?: boolean, relative?: boolean,
  ): void {
    for (const pm of mesh) {
      const d = this.nextDraw();
      d.vertex = pm.vertex;
      d.quadCount = pm.quadCount;
      d.pass = pm.pass;
      d.texture = pm.texture;
      d.model = model;
      d.tint = tint;
      d.flash = flash;
      d.light0 = light0;
      d.light1 = light1;
      d.sortPos = sortPos;
      d.cull = cull;
      d.modelRelative = relative; // M6: model built camera-relative via `Tw` ⇒ renderer skips the f32 subtraction
      out.push(d);
    }
  }

  // ── Block-display entities (4.5a) ────────────────────────────────────────────

  private fallingBlock(e: RenderEntity, out: EntityDraw[]): void {
    const state = e.properties.block ?? "minecraft:stone";
    const { name, props } = parseBlockState(state);
    const mesh = this.blockMesh(`fb|${state}`, name, props);
    if (!mesh) return;
    // [0,1] block model centered on x,z, base at the entity y. M6: `Tw` anchors the world placement in double.
    const p = e.position;
    this.emit(out, mesh, this.mat.Tw(p[0] - 0.5, p[1], p[2] - 0.5), e.position, undefined, undefined, undefined, undefined, undefined, true);
  }

  private primedTnt(e: RenderEntity, out: EntityDraw[]): void {
    const mesh = this.blockMesh("tnt", "minecraft:tnt", {});
    if (!mesh) return;
    const fuse = num(e.properties.fuse, 80);
    // Pulse scale in the final ~10 ticks: 1 + clamp(1 - fuse/10,0,1)^4 * 0.3 (§18).
    const f = Math.max(0, Math.min(1, 1 - fuse / 10));
    const s = 1 + f * f * f * f * 0.3;
    const blink = Math.floor(fuse / 5) % 2 === 0; // white blink every 5 ticks
    const p = e.position;
    const model = this.mat.mul(this.mat.Tw(p[0] - 0.5, p[1], p[2] - 0.5), scaleAboutCenter(this.mat, s));
    this.emit(out, mesh, model, e.position, undefined, blink ? TNT_BLINK : TNT_NO_FLASH, undefined, undefined, undefined, true);
  }

  private droppedItem(e: RenderEntity, clock: number, out: EntityDraw[]): void {
    const item = e.properties.item ?? "minecraft:stone";
    // Real item rendering (generated → extruded sprite, block → block model), with the GROUND display.
    const geom = this.deps.itemGeometry?.get(item);
    if (geom) {
      const mesh = this.getMesh(`item|${item}`, () => ({ [TerrainPass.Cutout]: geom.verts }), geom.texture);
      if (mesh.length) {
        const p = e.position, off = bobOffset(e.id);
        // Vanilla ItemEntityRenderer: bob = sin(age/10 + off)·0.1 + 0.1; spin = age/20 + off (age = clock·20).
        const bob = Math.sin(clock * 2 + off) * 0.1 + 0.1;
        const model = this.mat.mul(this.mat.Tw(p[0], p[1] + bob + 0.0625, p[2]), this.mat.Ry(clock + off), itemDisplayMatrix(geom.kind, "ground"));
        // Block / entity (builtin/entity BE box) items: vanilla diffuse lighting (relights as it spins) +
        // back-face cull (a solid cube's back faces otherwise z-fight a bright sliver through the silhouette).
        // Generated 1px slabs keep their flat shade and stay double-sided (no cull). M6: `Tw` ⇒ relative model.
        if (geom.kind !== "generated") this.emit(out, mesh, model, e.position, undefined, undefined, WORLD_ITEM_LIGHT0, WORLD_ITEM_LIGHT1, true, true);
        else this.emit(out, mesh, model, e.position, undefined, undefined, undefined, undefined, undefined, true);
        return;
      }
    }
    // Fallback when no item system is wired: a small spinning block (block items only).
    const { name, props } = parseBlockState(item);
    const mesh = this.blockMesh(`item|${item}`, name, props);
    if (!mesh) {
      this.warnOnce(`item:${item}`, `dropped item ${item} has no model (no item geometry + not a block)`);
      return;
    }
    const p = e.position;
    const bob = Math.sin(clock * 2) * 0.05 + 0.08;
    const model = this.mat.mul(this.mat.Tw(p[0], p[1] + bob + 0.0625, p[2]), this.mat.Ry(clock), this.mat.S(0.35, 0.35, 0.35), this.mat.T(-0.5, -0.5, -0.5));
    this.emit(out, mesh, model, e.position, undefined, undefined, undefined, undefined, undefined, true);
  }

  /** Item frame (4.5a hybrid): the vanilla frame BLOCK model (border + back panel) rotated so its front
   *  points along `facing`, plus the contained item laid flat against the back panel and spun by
   *  item_rotation·45°. An invisible frame draws only the item; a map frame uses the `_map` model. */
  private itemFrame(e: RenderEntity, type: string, out: EntityDraw[]): void {
    const props = e.properties;
    const facing = props?.facing ?? "south";
    const isMap = props?.map === "true";
    const glow = type === "glow_item_frame";
    const p = e.position;
    const F = this.mat;
    // Place the [0,1] frame model at the frame's block, rotating its authored front onto `facing` about centre.
    const frameModel = F.mul(F.T(Math.floor(p[0]), Math.floor(p[1]), Math.floor(p[2])), frameFacing(F, facing));
    if (props?.invisible !== "true") {
      this.blockModelDraws(`minecraft:${glow ? "glow_item_frame" : "item_frame"}[map=${isMap ? "true" : "false"}]`, frameModel, out);
    }
    const item = props?.item;
    if (item) {
      // Flat just in front of the back panel (z≈15/16), centred, rotated by item_rotation·45°, half scale.
      const rotN = num(props?.item_rotation, 0) & 7;
      const itemModel = F.mul(frameModel, F.T(0.5, 0.5, 15 / 16), F.Rz((rotN * 45 * Math.PI) / 180), F.S(0.5, 0.5, 0.5), F.T(-0.5, -0.5, -0.5));
      this.itemDraws(item, itemModel, out);
    }
  }

  private minecart(e: RenderEntity, type: string, clock: number, out: EntityDraw[]): void {
    const before = out.length;
    // Body (box model — 4.5b); contents (block-display — 4.5a). Either may be absent.
    const body = this.deps.boxModels?.build(e, clock, this.mat);
    if (body) this.boxDraws(out, body, e.position);

    const contained = CART_CONTENTS[type];
    if (contained) {
      const { name, props } = parseBlockState(contained);
      const mesh = this.blockMesh(`cart|${contained}`, name, props);
      if (mesh) {
        const off = CART_DISPLAY_OFFSET[type] ?? 6;
        const p = e.position;
        // Share the body's EXACT frame (cartModelMatrix): T(pos)·T(0,0.375,0)·Ry(180−yaw)·Zp(−pitch), with
        // yaw = yRotRail ?? yRot and pitch = xRot — so a tilted cart on a slope carries its block in sync.
        // BEFORE the body's scale(-1,-1,1) flip: scale 0.75, translate (-0.5, (off-8)/16, 0.5) [centres X/Z,
        // offsets Y by the per-cart display offset], rotate Y 90°. Else the block floats out of the cart.
        const yaw = num(e.properties.yRotRail ?? e.properties.yRot, 0);
        const pitch = num(e.properties.xRot, 0);
        const model = this.mat.mul(
          this.mat.Tw(p[0], p[1], p[2]), // M6: outermost world placement anchored in double
          this.mat.T(0, 0.375, 0),
          this.mat.Ry(((180 - yaw) * Math.PI) / 180),
          this.mat.Rz((-pitch * Math.PI) / 180),
          this.mat.S(0.75, 0.75, 0.75),
          this.mat.T(-0.5, (off - 8) / 16, 0.5),
          this.mat.Ry(Math.PI / 2),
        );
        // The contained block is lit by the SAME vanilla LEVEL diffuse lights as the cart body (the cart's
        // lighting "applies to the block inside too") — so it relights smoothly and IN SYNC with the body as
        // the cart turns, instead of snapping between terrain-axis steps at 45° boundaries (NO_LIGHT path) or
        // staying frozen (local-only shade, which looks detached from the diffusely-lit body).
        this.emit(out, mesh, model, e.position, undefined, undefined, WORLD_ENTITY_LIGHT0, WORLD_ENTITY_LIGHT1, undefined, true);
      }
    }
    if (out.length === before) this.warnOnce(type, `${type} body model not available yet (needs box-model engine)`);
  }

  // ── Box-model entities (4.5b) ────────────────────────────────────────────────

  private boxModelOrFallback(e: RenderEntity, clock: number, out: EntityDraw[]): void {
    const layers = this.deps.boxModels?.build(e, clock, this.mat);
    if (layers) { this.boxDraws(out, layers, e.position); return; }
    this.warnOnce(strip(e.type), `entity ${e.type} has no model yet (fallback box pending entity atlas)`);
  }

  /** Emit draws for box-model layers into `out`: static layers share cached GPU meshes; dynamic ones re-encode.
   *  M6: `absSortPos` is the entity's ABSOLUTE world position — passed so `sortPos` stays absolute even though
   *  the layer `model` is now built camera-relative (`Mat4Frame.Tw`), where `sortOf(model)` would be relative. */
  private boxDraws(out: EntityDraw[], layers: BoxModelLayer[], absSortPos?: Vec3): void {
    const tex = this.deps.boxModels!.texture;
    for (const layer of layers) {
      // Static: getMesh runs the bake thunk ONLY on a cache miss (ENT-1). Dynamic: re-posed every frame,
      // re-encoded into a reused buffer — STRAIGHT from the sink when `bakeInto` is present (no RawVertex[]).
      const mesh = layer.dynamic
        ? (layer.bakeInto ? this.dynamicMeshSink(layer.geomKey, layer.bakeInto, tex) : this.dynamicMesh(layer.geomKey, layer.verts(), tex))
        : this.getMesh(layer.geomKey, layer.verts, tex);
      // Box-model entities (minecart body, mobs) use vanilla LEVEL diffuse lighting — a spinning cart's
      // sides shade smoothly, not in the quantized axis steps the terrain shade table would produce. All
      // pass-meshes of a layer share its frame ⇒ one sort position (same value the per-pm copy produced).
      const sortPos = absSortPos ?? layer.sortPos ?? this.sortOf(layer.model);
      this.emit(out, mesh, layer.model, sortPos, layer.tint, layer.flash, WORLD_ENTITY_LIGHT0, WORLD_ENTITY_LIGHT1, undefined, absSortPos !== undefined);
    }
  }

  // ── GPU mesh cache ───────────────────────────────────────────────────────────

  private blockMesh(key: string, name: string, props: BlockProps): PassMesh[] | null {
    const cached = this.geomCache.get(key);
    if (cached) return cached.length ? cached : null;
    const model = this.deps.bakeBlock(name, props);
    if (!model) {
      this.geomCache.set(key, []);
      return null;
    }
    const verts = bakedModelToEntityVerts(model);
    if (!hasGeometry(verts)) {
      this.geomCache.set(key, []);
      return null;
    }
    return this.getMesh(key, () => verts, this.deps.blockAtlas);
  }

  private getMesh(key: string, build: () => EntityVertsByPass, texture: GpuTextureHandle): PassMesh[] {
    let m = this.geomCache.get(key);
    if (m && m.length) return m;
    const verts = build();
    m = [];
    for (const pass of TERRAIN_PASSES) {
      const list = verts[pass];
      if (!list || list.length === 0) continue;
      instrument.bumpAlloc("rawVertex", list.length); // X5: RawVertex objects this bake emitted (cache miss only)
      const data = packVertices(list);
      const vertex = this.deps.device.createBuffer({ sizeBytes: data.byteLength, usage: BufferUsage.Vertex, label: `entity-${key}-${pass}` });
      this.deps.device.writeBuffer(vertex, 0, new Uint8Array(data));
      m.push({ pass, vertex, quadCount: list.length / 4, texture });
    }
    this.geomCache.set(key, m);
    return m;
  }

  // ── Dynamic (re-posed-per-frame) meshes ──────────────────────────────────────
  // Animated box models (walk swing) re-encode their geometry every frame into a REUSED buffer per
  // (entity, layer) — never a fresh GPU buffer per frame (TRAP 18.C / no churn). `beginFrame`/`endFrame`
  // bracket the frame; buffers whose key isn't re-encoded this frame (despawned/idle) are reclaimed.

  private readonly dynamicCache = new Map<string, DynMesh[]>();
  private seenDynamic = new Set<string>();

  /** Call once per frame before building entity draws. `anchor` (M6): the camera world position the OUTERMOST
   *  entity placement (`Mat4Frame.Tw`) is built relative to, so far-from-origin entities have no f32 precision
   *  loss. [0,0,0] (default, camera-relative off) ⇒ `Tw` === `T`, byte-identical. Set by EntityWorld in
   *  lockstep with the renderer's camera-relative gate. */
  beginFrame(anchor?: Vec3): void {
    this.seenDynamic = new Set<string>();
    this.mat.setAnchor(anchor ? anchor[0] : 0, anchor ? anchor[1] : 0, anchor ? anchor[2] : 0);
    this.mat.reset(); // A3: rewind the model-matrix arena (everything from last frame was already drawn)
    this.drawCursor = 0; // A4: rewind the draw + sort-position pools (same lifetime as the matrices)
    this.sortCursor = 0;
  }

  /** Call once per frame after building entity draws — frees dynamic buffers not touched this frame. */
  endFrame(): void {
    for (const [key, meshes] of this.dynamicCache) {
      if (this.seenDynamic.has(key)) continue;
      for (const m of meshes) this.deps.device.destroyBuffer(m.vertex);
      this.dynamicCache.delete(key);
    }
  }

  /** A4: bake a dynamic (single Cutout pass) layer STRAIGHT into the reused sink, then upload to the reused
   *  per-key buffer — no `RawVertex[]` and no fresh packing buffer per frame. Shares the dynamic cache +
   *  per-frame reclamation (seenDynamic) with `dynamicMesh`. */
  private dynamicMeshSink(key: string, bakeInto: (sink: PackedVertexSink) => void, texture: GpuTextureHandle): PassMesh[] {
    this.seenDynamic.add(key);
    let meshes = this.dynamicCache.get(key);
    if (!meshes) this.dynamicCache.set(key, (meshes = []));
    this.dynSink.reset();
    bakeInto(this.dynSink);
    // Free any slots beyond the single one this path uses (a prior multi-pass `dynamicMesh` life of this key).
    for (let i = 1; i < meshes.length; i++) this.deps.device.destroyBuffer(meshes[i].vertex);
    if (meshes.length > 1) meshes.length = 1;
    if (this.dynSink.isEmpty()) {
      if (meshes.length) { this.deps.device.destroyBuffer(meshes[0].vertex); meshes.length = 0; }
      return meshes;
    }
    const data = this.dynSink.bytes();
    let m = meshes[0];
    if (!m || m.capacityBytes < data.byteLength) {
      if (m) this.deps.device.destroyBuffer(m.vertex);
      const cap = Math.max(data.byteLength, 1024);
      const vertex = this.deps.device.createBuffer({ sizeBytes: cap, usage: BufferUsage.Vertex, label: `entity-dyn-${key}-cutout` });
      m = { pass: TerrainPass.Cutout, vertex, quadCount: 0, texture, capacityBytes: cap };
      meshes[0] = m;
    }
    this.deps.device.writeBuffer(m.vertex, 0, data); // copies synchronously ⇒ the sink can be reused next layer
    m.pass = TerrainPass.Cutout;
    m.quadCount = this.dynSink.quadCount;
    m.texture = texture;
    return meshes; // single Cutout slot
  }

  private dynamicMesh(key: string, verts: EntityVertsByPass, texture: GpuTextureHandle): PassMesh[] {
    this.seenDynamic.add(key);
    let meshes = this.dynamicCache.get(key);
    if (!meshes) this.dynamicCache.set(key, (meshes = []));
    const out: PassMesh[] = [];
    let slot = 0;
    for (const pass of TERRAIN_PASSES) {
      const list = verts[pass];
      if (!list || list.length === 0) continue;
      instrument.bumpAlloc("rawVertex", list.length); // X5: RawVertex objects (the verts path; sink path stays 0)
      const data = new Uint8Array(packVertices(list));
      let m = meshes[slot];
      if (!m || m.capacityBytes < data.byteLength) {
        if (m) this.deps.device.destroyBuffer(m.vertex);
        const vertex = this.deps.device.createBuffer({ sizeBytes: Math.max(data.byteLength, 1024), usage: BufferUsage.Vertex, label: `entity-dyn-${key}-${pass}` });
        m = { pass, vertex, quadCount: 0, texture, capacityBytes: Math.max(data.byteLength, 1024) };
        meshes[slot] = m;
      }
      this.deps.device.writeBuffer(m.vertex, 0, data);
      m.pass = pass;
      m.quadCount = list.length / 4;
      m.texture = texture;
      out.push(m);
      slot++;
    }
    // Free any trailing buffers from a previous frame that had MORE passes than this one.
    for (let i = slot; i < meshes.length; i++) this.deps.device.destroyBuffer(meshes[i].vertex);
    meshes.length = slot;
    return out;
  }

  private warnOnce(key: string, msg: string): void {
    if (this.warned.has(key)) return;
    this.warned.add(key);
    console.warn(`[EntityModelFactory] ${msg}`);
  }

  dispose(): void {
    for (const meshes of this.geomCache.values()) for (const m of meshes) this.deps.device.destroyBuffer(m.vertex);
    for (const meshes of this.dynamicCache.values()) for (const m of meshes) this.deps.device.destroyBuffer(m.vertex);
    this.geomCache.clear();
    this.dynamicCache.clear();
  }
}

interface DynMesh extends PassMesh {
  capacityBytes: number;
}

/** AbstractMinecart.getDisplayBlockState per cart type (ported from old EntityMeshFactory). Exported so the
 *  scene builder can pre-atlas + pre-bake these contained blocks even when none are placed in the world —
 *  otherwise a spawned hopper/furnace/… minecart bakes a block whose sprites the atlas never stitched. */
export const CART_CONTENTS: Record<string, string | undefined> = {
  // chest_minecart is intentionally absent: a chest has no terrain block model (it's a block entity), so it
  // is rendered as the BE chest box model by the box-model source (boxModels.ts), not the terrain baker.
  hopper_minecart: "minecraft:hopper[facing=down,enabled=true]",
  furnace_minecart: "minecraft:furnace[facing=north,lit=false]",
  tnt_minecart: "minecraft:tnt",
  command_block_minecart: "minecraft:command_block",
};

/** AbstractMinecart.getDefaultDisplayOffset per cart type (1/16 units). */
const CART_DISPLAY_OFFSET: Record<string, number> = {
  chest_minecart: 8,
  hopper_minecart: 1,
};

/** Stable per-item-entity bob/spin phase (vanilla `bobOffs` is a fixed random per entity) from its id. */
function bobOffset(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
  return ((h >>> 0) % 1000) / 1000 * Math.PI * 2;
}

/** TNT white-blink flash tuples (constant — shared read-only by the entity uniform, never mutated). */
const TNT_BLINK: readonly [number, number, number, number] = [1, 1, 1, 0.55];
const TNT_NO_FLASH: readonly [number, number, number, number] = [1, 1, 1, 0];

/** M = T(c)·S(s)·T(-c) — scale about the [0,1] block model's center. Built into the per-frame arena. */
function scaleAboutCenter(F: Mat4Frame, s: number): Mat4 {
  return F.mul(F.T(0.5, 0.5, 0.5), F.S(s, s, s), F.T(-0.5, -0.5, -0.5));
}

/** Rotation that turns the item-frame block model so its front (opening + item) points along `facing`, about
 *  the block centre. The vanilla model is authored on the +Z slab with the opening toward -Z (NORTH-facing);
 *  Ry(θ) maps -Z → (-sinθ,0,-cosθ), so Ry(180°)→south, Ry(-90°)→east, etc. */
function frameFacing(F: Mat4Frame, facing: string): Mat4 {
  const rot =
    facing === "south" ? F.Ry(Math.PI) :
    facing === "east" ? F.Ry(-Math.PI / 2) :
    facing === "west" ? F.Ry(Math.PI / 2) :
    facing === "up" ? F.Rx(Math.PI / 2) :
    facing === "down" ? F.Rx(-Math.PI / 2) :
    F.T(0, 0, 0); // north = identity (authored orientation: opening faces -Z)
  return F.mul(F.T(0.5, 0.5, 0.5), rot, F.T(-0.5, -0.5, -0.5));
}

function strip(type: string): string {
  return type.replace(/^minecraft:/, "");
}

function num(s: string | undefined, dflt: number): number {
  const n = Number.parseFloat(s ?? "");
  return Number.isFinite(n) ? n : dflt;
}
