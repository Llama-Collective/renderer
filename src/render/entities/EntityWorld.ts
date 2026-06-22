// Per-frame entity + block-entity orchestrator. RENDERER_PLAN.md §18, Phase 4.5.
//
// Ties the pieces into one usable API: EntityScene (ingest + interpolation), EntityModelFactory (entity
// geometry), BlockEntitySceneBaker (hybrid BE bake), and EntityRenderer (forward pass that LOADS
// terrain's depth). Call AFTER TerrainRenderer.render each frame so entities + BEs share the depth
// buffer (TRAP 18.B). Per-entity frustum cull + per-section BE cull reuse camera/Frustum + the terrain
// render-distance test. Box models + BEs are optional (degrades to block-display entities only).

import type { GpuTextureHandle, GraphicsDevice, TextureFormat } from "../../core/GraphicsDevice";
import type { BakedBlockModel } from "../../mesh/model/BakedBlockModel";
import type { BlockProps } from "../../mesh/model/BlockStateResolver";
import { BlockEntityIndex, type BlockEntityRecord } from "../../world/BlockEntityIndex";
import { Frustum } from "../../camera/Frustum";
import type { CameraView } from "../../camera/Camera";
import type { Mat4 } from "../../mesh/entity/mat4";
import type { SpriteUv, Vec3 } from "../../types";
import { EntityScene, type EntitySnapshot } from "./EntityScene";
import { EntityModelFactory } from "./EntityModelFactory";
import { EntityRenderer, type EntityDraw } from "./EntityRenderer";
import { BlockEntitySceneBaker } from "./BlockEntitySceneBaker";
import { SpecialRenderer, type PortalTextures } from "./SpecialRenderer";
import { blockEntityAnimatesPerFrame, type BESpecialDraw } from "./blockentities";
import { entityAabb, ENTITY_BOUNDS_COLOR } from "./entityBounds";
import { createBoxModelSource } from "./boxModels";
import type { ItemGeometry } from "../items/ItemGeometry";
import { withinRenderDistance } from "../TerrainRenderer";

const SECTION_SIZE = 16;
const ENTITY_HALF = 1.5; // per-entity cull AABB half-extent (blocks) — generous, no false negatives

export interface EntityWorldDeps {
  device: GraphicsDevice;
  colorFormat: TextureFormat;
  depthFormat: TextureFormat;
  /** Bake a blockstate (reuses the terrain baker) — block-display entities + BE contents. */
  bakeBlock: (name: string, props: BlockProps) => BakedBlockModel | null;
  blockAtlas: GpuTextureHandle;
  /** Entity atlas + rect lookup (box-model entities + BE skins). Omit → block-display entities only. */
  entityAtlas?: GpuTextureHandle;
  entityUvFor?: (texture: string) => SpriteUv | undefined;
  /** Distance-cull radius in blocks (matches the terrain renderer). */
  renderDistanceBlocks?: number;
  /** End-portal shader textures (end_sky + end_portal). Omit → portal draws a dim fallback. */
  portalTextures?: PortalTextures;
  /** Item geometry (generated sprite extrusion / block model) → real dropped-item sprites. */
  itemGeometry?: ItemGeometry;
  /** Atlas managers (entity + item) this world's textures came from — released on dispose (O1, §8). */
  ownedAtlases?: readonly { dispose(): void }[];
}

export class EntityWorld {
  readonly scene = new EntityScene();
  readonly blockEntities = new BlockEntityIndex();
  private readonly factory: EntityModelFactory;
  private readonly renderer: EntityRenderer;
  private readonly beBaker: BlockEntitySceneBaker | null;
  private readonly special: SpecialRenderer;
  private readonly renderDistance: number;
  /** Reused zero-alloc frustum (AUDIT F4) — rebuilt in place each frame, no per-frame plane arrays. */
  private readonly frustum = new Frustum();
  // AUDIT F8/ENT-4: per-frame draw/special scratch, reused (cleared, not re-allocated) so a static OR
  // animated entity scene allocates no draw lists per frame — pairs with EntityScene's pooled output.
  private readonly drawScratch: EntityDraw[] = [];
  private readonly specialScratch: BESpecialDraw[] = [];
  // Stashed by renderEntities for a deferred renderSpecials (split frame loop): the specials draw AFTER
  // translucent terrain, so the viewer calls renderEntities between the terrain passes and renderSpecials last.
  private stashedVp: Mat4 | null = null;
  private stashedClock = 0;
  private stashedW = 0;
  private stashedH = 0;
  // Cached "any BE animates per frame?" for the render gate (recomputed only when the BE set changes —
  // O(1) per gate call, not a per-frame scan). Invalidated on every BE mutation below.
  private animActiveDirty = true;
  private animActive = false;
  stats = { entities: 0, beStatic: 0, beAnimating: 0, drawn: 0, special: 0 };

  constructor(private readonly deps: EntityWorldDeps) {
    this.renderDistance = deps.renderDistanceBlocks ?? Infinity;
    const boxModels =
      deps.entityAtlas && deps.entityUvFor
        ? createBoxModelSource({ uvFor: (n) => req(deps.entityUvFor!(n), n), has: (n) => deps.entityUvFor!(n) !== undefined }, deps.entityAtlas)
        : undefined;
    this.factory = new EntityModelFactory({ device: deps.device, bakeBlock: deps.bakeBlock, blockAtlas: deps.blockAtlas, boxModels, itemGeometry: deps.itemGeometry });
    this.renderer = new EntityRenderer(deps.device, deps.colorFormat, deps.depthFormat);
    this.special = new SpecialRenderer(deps.device, deps.colorFormat, deps.depthFormat);
    if (deps.portalTextures) this.special.setPortalTextures(deps.portalTextures);
    this.beBaker =
      deps.entityAtlas && deps.entityUvFor ? new BlockEntitySceneBaker(deps.device, this.blockEntities, deps.entityAtlas, deps.entityUvFor) : null;
  }

  // ── Data ingestion ──────────────────────────────────────────────────────────

  /** PREC-1: forward the DEFAULT-OFF camera-relative origin flag to the entity renderer so entities move in
   *  lockstep with terrain (TRAP PREC-1.D — both go camera-relative or neither). Off ⇒ byte-identical. */
  set cameraRelative(on: boolean) {
    this.renderer.cameraRelative = on;
  }
  get cameraRelative(): boolean {
    return this.renderer.cameraRelative;
  }

  ingestEntities(changed: readonly EntitySnapshot[], removedIds: readonly string[] = []): void {
    this.scene.ingest(changed, removedIds);
  }

  setBlockEntity(rec: BlockEntityRecord): void {
    this.beBaker?.invalidate(this.blockEntities.set(rec));
    this.animActiveDirty = true;
  }
  removeBlockEntity(x: number, y: number, z: number): void {
    this.beBaker?.invalidate(this.blockEntities.remove(x, y, z));
    this.animActiveDirty = true;
  }
  /** Toggle a BE's deliberate-preview animation (chest lid / bell swing / shulker open / pot wobble). Idle-loop
   *  BEs (banners, conduits) animate regardless of this flag. Re-meshes exactly that BE's section. */
  setBlockEntityAnimating(x: number, y: number, z: number, animating: boolean): void {
    this.beBaker?.invalidate(this.blockEntities.setAnimating(x, y, z, animating));
    this.animActiveDirty = true;
  }

  /** Set a container's lid OPEN target (chest / shulker), e.g. from the sim's open-viewer count. The baker
   *  ramps the lid open/closed at the vanilla rate; an opening/open/closing container draws per-frame, a
   *  closed one bakes static. Invalidating the section kicks the transition off promptly (render-on-demand). */
  setBlockEntityOpen(x: number, y: number, z: number, open: boolean): void {
    this.beBaker?.invalidate(this.blockEntities.setOpen(x, y, z, open));
  }

  /** True if ANY block entity will draw on a per-frame (animating) path — an idle-loop motion (banner wave,
   *  conduit spin), a toggled preview, or an always-animated special (spinning spawner/vault, end portal,
   *  beam). The render gate ORs this into its continuous-render decision so those motions don't freeze under
   *  render-on-demand (dynamicFps). Cached; recomputed only after a BE mutation. */
  hasActiveAnimations(): boolean {
    if (this.animActiveDirty) {
      this.animActive = false;
      for (const rec of this.blockEntities.values()) {
        if (blockEntityAnimatesPerFrame(rec)) { this.animActive = true; break; }
      }
      this.animActiveDirty = false;
    }
    return this.animActive;
  }

  /**
   * A container lid (chest / shulker) is mid open/close this frame — the openness is changing, so the loop
   * must keep drawing until it settles. This is a STATE CHANGE (a container opening/closing), NOT decoration,
   * so the viewer treats it like a piston slide: it plays regardless of the animations master switch (whereas
   * `hasActiveAnimations` decorative motion is gated by it). Self-terminating (each lid settles in ~0.5 s), so
   * it can't pin render-on-demand. Kept separate from `hasActiveAnimations` precisely so the gate can exempt it.
   */
  hasContainerLidTransitions(): boolean {
    return !!this.beBaker?.hasOpenTransitions();
  }

  // ── Frame ───────────────────────────────────────────────────────────────────

  /**
   * Build + submit this frame's entity/BE draws AND the specials (one combined pass after terrain). Kept for
   * callers/tests that don't interleave; the editor uses `renderEntities` + `renderSpecials` so entities draw
   * between opaque and translucent terrain (depth-correct vs glass).
   */
  render(camera: CameraView, partialTick: number, clock: number, width: number, height: number): void {
    this.renderEntities(camera, partialTick, clock, width, height);
    this.renderSpecials();
  }

  /**
   * Render the deferred specials (beams / portals / wireframes + entity hitboxes) gathered by the most recent
   * `renderEntities`. Call AFTER translucent terrain: these LOAD depth and the xray hitboxes/beams are meant to
   * draw last (over glass), as before. No-op if `renderEntities` hasn't run this frame.
   */
  renderSpecials(): void {
    if (!this.stashedVp) return;
    this.special.render(this.specialScratch, this.stashedVp, this.stashedClock, this.stashedW, this.stashedH);
    this.stats.special = this.special.stats.quads;
  }

  /**
   * Build + submit this frame's entity/BE GEOMETRY (items/mobs/BEs) only — the specials are DEFERRED to
   * `renderSpecials`. Call BETWEEN opaque and translucent terrain so entities depth-order against glass (vanilla
   * order). `partialTick` interpolates entity motion; `clock` (seconds) drives idle/looping animation.
   */
  renderEntities(camera: CameraView, partialTick: number, clock: number, width: number, height: number): void {
    const vp = camera.viewProjection();
    const frustum = this.frustum.setFromViewProjection(vp);
    const camPos = camera.position;

    // Entities: interpolate → build → per-entity frustum + distance cull. Each visible entity also gets a
    // simulation-accurate AXIS-ALIGNED bounding-box outline (never tilted with the model — see entityAabb).
    // NOTE: these are ENTITIES only (mobs/carts/boats/items/tnt); block entities are NOT in this set, so
    // they get no entity outline (their own wireframes, e.g. conduit/structure block, are BE specials).
    const entities = this.scene.frame(partialTick);
    // M6: when the camera-relative path is ACTIVE (same gate the renderer uses — `cameraRelative` + the camera
    // exposes view/proj), build entity model placements relative to the camera (in double, via `Mat4Frame.Tw`)
    // so far-from-origin entities have no f32 precision loss. Off ⇒ no anchor ⇒ `Tw` === `T` (byte-identical).
    const relActive = this.renderer.cameraRelative && !!camera.viewMatrix && !!camera.projectionMatrix;
    this.factory.beginFrame(relActive ? camPos : undefined);
    const draws = this.drawScratch;
    const specials = this.specialScratch;
    draws.length = 0; // F8: clear + reuse — no fresh per-frame array; loops below avoid spread arg-arrays too
    specials.length = 0;
    let entityCount = 0;
    for (const e of entities) {
      if (!this.entityVisible(e.position, frustum, camPos)) continue;
      // A4: buildDraws emits straight into the pooled drawScratch (no per-entity array); returns the count.
      if (this.factory.buildDraws(e, clock, draws) > 0) entityCount++;
      const bb = entityAabb(e.type, e.position, e.properties);
      // xray: entity hitboxes render THROUGH blocks/entities (always visible), like the old renderer.
      specials.push({ kind: "lineBox", min: bb.min, max: bb.max, color: ENTITY_BOUNDS_COLOR, xray: true });
    }
    this.factory.endFrame();

    // Block entities: hybrid bake (idle static + culled / animating per-frame) + specials.
    if (this.beBaker) {
      const beDraws = this.beBaker.frame(clock, (origin) => this.sectionVisible(origin, frustum, camPos), camPos);
      for (const x of beDraws) draws.push(x);
      // Embedded block/item specials (mob spawner, vault, brushable, shelf, campfire) reuse the baker.
      for (const s of this.beBaker.specials) {
        if (s.kind === "blockModel") this.factory.blockModelDraws(s.block, s.model, draws);
        else if (s.kind === "item") this.factory.itemDraws(s.item, s.model, draws);
      }
      for (const s of this.beBaker.specials) specials.push(s); // beam / portal / BE wireframes
      this.stats.beStatic = this.beBaker.stats.staticDrawn;
      this.stats.beAnimating = this.beBaker.stats.animating;
    }

    this.stats.entities = entityCount;
    this.stats.drawn = draws.length;
    // PREC-1: hand the entity renderer the SEPARATE view/proj when the camera exposes them, so its camera-
    // relative path (default-off; gated by `cameraRelative`) can build a translation-free viewProjRel WITHOUT
    // rebuilding the shared world `vp` above (the frustum/specials still read the world matrix). Off ⇒ no-op.
    this.renderer.render(draws, vp, camPos, width, height, {}, camera.viewMatrix?.(), camera.projectionMatrix?.());

    // Beam / portal / wireframe specials (incl. entity bounding boxes) are DEFERRED to renderSpecials so they
    // draw AFTER translucent terrain (they load depth; xray hitboxes/beams draw last over glass). Stash the
    // frame's view-projection + clock + size for that call (specials live in `specialScratch`).
    this.stashedVp = vp;
    this.stashedClock = clock;
    this.stashedW = width;
    this.stashedH = height;
  }

  private entityVisible(p: Vec3, frustum: Frustum, camPos: Vec3): boolean {
    // Scalar testAab (AUDIT F4) — no per-entity min/max arrays.
    if (!frustum.testAab(p[0] - ENTITY_HALF, p[1] - ENTITY_HALF, p[2] - ENTITY_HALF, p[0] + ENTITY_HALF, p[1] + ENTITY_HALF, p[2] + ENTITY_HALF)) return false;
    if (this.renderDistance === Infinity) return true; // common case (bounded editor world) — skip the array
    return withinRenderDistance([p[0] - SECTION_SIZE / 2, p[1] - SECTION_SIZE / 2, p[2] - SECTION_SIZE / 2], camPos, this.renderDistance);
  }

  private sectionVisible(origin: Vec3, frustum: Frustum, camPos: Vec3): boolean {
    // Padded section test (audit Bug 2): block-entity models (banners, signs, chests, bells) overhang the
    // 16³ box, so use `testSection`'s margin rather than the bare box — same fix as terrain culling.
    if (!frustum.testSection(origin[0], origin[1], origin[2])) return false;
    return withinRenderDistance(origin, camPos, this.renderDistance);
  }

  dispose(): void {
    this.factory.dispose();
    this.renderer.dispose();
    this.special.dispose();
    this.beBaker?.dispose();
    for (const a of this.deps.ownedAtlases ?? []) a.dispose(); // O1: release the entity/item atlas textures
  }
}

function req<T>(v: T | undefined, name: string): T {
  if (v === undefined) throw new Error(`EntityWorld: missing entity atlas rect for "${name}"`);
  return v;
}
