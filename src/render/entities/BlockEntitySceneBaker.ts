// Hybrid static/dynamic block-entity baking for a scene. RENDERER_PLAN.md §18, Phase 4.5c (BBE-adopted).
//
// Reads the per-section `BlockEntityIndex` and, per section, bakes all IDLE BEs into ONE cached static
// GPU mesh in section-local coords (drawn with model = T(sectionOrigin) so it culls + behaves like
// terrain). ANIMATING BEs are excluded from that bake and re-encoded PER FRAME into reused buffers
// (TRAP 18.C). Editing a section / toggling a BE's `animating` flag invalidates exactly that section's
// static mesh (the existing dirty-section path), so flipping a BE re-meshes only its section and never
// double-draws (it's in the static bake XOR the per-frame set — see `splitSectionBEs`).
//
// v1: the static BE mesh uses the ENTITY atlas and is culled per-section by a visibility predicate —
// behaviourally the hybrid path (baked once, culls with the section). Merging it into the SAME buffer
// as terrain (one unified atlas) is the documented refinement (the plan's per-section target).

import type { GpuBufferHandle, GpuTextureHandle, GraphicsDevice } from "../../core/GraphicsDevice";
import { BufferUsage } from "../../core/GraphicsDevice";
import { packVertices, type RawVertex } from "../../mesh/VertexFormat";
import { translation } from "../../mesh/entity/mat4";
import { parseSectionKey, type SectionKey } from "../../world/SectionKey";
import type { BlockEntityIndex, BlockEntityRecord } from "../../world/BlockEntityIndex";
import { SECTION_SIZE, TerrainPass, type SpriteUv, type Vec3 } from "../../types";
import { bakeBlockEntity, blockEntitySpecials, getBEDef } from "./blockentities";
import type { BESpecialDraw } from "./blockentities";
import type { EntityDraw } from "./EntityRenderer";

/** Section visible this frame? (origin = section world origin in blocks.) Default: always visible. */
export type SectionVisible = (origin: Vec3) => boolean;

// Container lid openness integration (chest/shulker). Vanilla advances the lid by 0.1 per tick at 20 TPS, so
// a full open/close is 0.5 s; OPEN_RATE matches that on the render clock. Below OPEN_EPS the lid is treated
// as shut → the container drops back into the BBB-static section bake.
const OPEN_RATE = 2.0; // openness units per second (0.1/tick × 20 tps) → ~0.5 s for a full open/close
const OPEN_EPS = 1e-3;
// Clamp the per-frame openness step (tab refocus / a first frame after a render-on-demand idle would
// otherwise carry a huge `clock - lastClock`). Small enough that a wake-from-idle advances only a sliver
// (no visible jump to half-open) yet ≥ one frame at 20 fps, so the duration stays accurate at normal rates.
const MAX_OPEN_DT = 0.05;

interface StaticMesh {
  vertex: GpuBufferHandle;
  quadCount: number;
}
interface AnimBuf {
  vertex: GpuBufferHandle;
  capacityBytes: number;
}
/** Per-position lid-animation state for an `openable` container (chest/shulker). */
interface OpenState {
  openness: number; // current eased lid openness ∈ [0,1]
  lastClock: number; // render clock at the last advance (for dt)
  perFrame: boolean; // was it drawn per-frame last frame? (membership-change detection for the static cache)
}

/** Per-position key matching `animDraw`'s buffer key. */
const recKey = (rec: BlockEntityRecord): string => `${rec.x},${rec.y},${rec.z}`;

export class BlockEntitySceneBaker {
  /** Per-section idle bake: null = baked-but-empty (cached negative). Absent = needs (re)bake. */
  private readonly staticCache = new Map<SectionKey, StaticMesh | null>();
  /** AUDIT ENT-5: numeric section origin per key, computed once (keys→origins are constant — BEs don't
   *  move), so `frame()` stops re-parsing `sectionOrigin` (string split + Number + Vec3) every frame. */
  private readonly originCache = new Map<SectionKey, Vec3>();
  private readonly animBufs = new Map<string, AnimBuf>();
  private readonly seenAnim = new Set<string>();
  /** Per-position lid openness state for openable containers; pruned when a container disappears. */
  private readonly openState = new Map<string, OpenState>();
  private readonly seenOpen = new Set<string>();
  /** F8: reused per-frame draw output (cleared each `frame()`; consumed synchronously by EntityWorld). */
  private readonly outScratch: EntityDraw[] = [];
  /** This frame's camera world position (for camera-facing BE billboards); set in `frame()`. */
  private cameraPos: Vec3 | undefined;
  /** Special (beam/portal/wireframe/embedded) draws gathered this frame — read by EntityWorld. */
  specials: BESpecialDraw[] = [];
  /** Instrumentation: idle BEs baked into terrain vs animating BEs drawn per-frame this frame. */
  stats = { staticSections: 0, staticDrawn: 0, animating: 0 };

  constructor(
    private readonly device: GraphicsDevice,
    private readonly index: BlockEntityIndex,
    private readonly atlasTexture: GpuTextureHandle,
    private readonly uvFor: (texture: string) => SpriteUv | undefined,
  ) {}

  /** Drop a section's cached static mesh so it re-bakes (edit / animating toggle). */
  invalidate(section: SectionKey | null): void {
    if (!section) return;
    const m = this.staticCache.get(section);
    if (m) this.device.destroyBuffer(m.vertex);
    this.staticCache.delete(section);
  }

  /**
   * Assemble this frame's BE draws. `clock` (seconds) drives animating poses. `visible` culls static
   * section meshes (proving idle BEs cull with their section). Returns static + animating draws merged.
   */
  frame(clock: number, visible: SectionVisible = () => true, cameraPos?: Vec3): EntityDraw[] {
    this.cameraPos = cameraPos;
    // F8: reuse the per-frame output + bookkeeping (cleared, not re-allocated). `out` is consumed
    // synchronously by EntityWorld.render before the next frame, so it's safe to reuse.
    const out = this.outScratch;
    out.length = 0;
    this.seenAnim.clear();
    this.seenOpen.clear();
    this.specials.length = 0;
    let staticSections = 0;
    let staticDrawn = 0;
    let animating = 0;

    for (const sk of this.index.occupiedSections()) {
      const origin = this.originOf(sk);
      const records = this.index.inSection(sk);
      const { idle, perFrame, membershipChanged } = this.splitSection(records, clock);
      const vis = visible(origin);

      // An openable container crossed the open/closed boundary this frame → its static-vs-per-frame membership
      // changed, so the cached static mesh is stale (it still has, or still lacks, that container). Drop it so
      // getStaticMesh re-bakes `idle` — now excluding a just-opened container, or re-including a just-closed
      // one with its lid shut. This is the synchronous analog of BBE's terrain fence (BBB_FINDINGS §5/§7).
      if (membershipChanged) this.invalidate(sk);

      // ── Static (idle/closed) bake — once per section, cached, culled like terrain ──
      const mesh = this.getStaticMesh(sk, origin, idle);
      if (idle.length > 0) staticSections++;
      if (mesh && vis) {
        out.push({ vertex: mesh.vertex, quadCount: mesh.quadCount, pass: TerrainPass.Cutout, texture: this.atlasTexture, model: translation(origin[0], origin[1], origin[2]), sortPos: origin });
        staticDrawn++;
      }

      // ── Per-frame BEs (idle-loop / toggled / opening containers) — re-encoded, excluded from the bake ──
      for (const pf of perFrame) {
        if (!vis) continue;
        const draw = this.animDraw(pf.rec, origin, clock, pf.openness);
        if (draw) {
          out.push(draw);
          animating++;
        }
      }

      // ── Special (beam / portal / wireframe / embedded) draws — every frame ──
      if (vis) {
        for (const rec of records) {
          for (const s of blockEntitySpecials(rec, clock)) this.specials.push(s);
        }
      }
    }

    this.sweepAnim();
    this.sweepOpen();
    this.stats = { staticSections, staticDrawn, animating };
    return out;
  }

  /**
   * Split a section's BEs into the static (idle/closed) set and the per-frame set. Openable containers
   * (chest/shulker) are routed by their integrated lid openness: drawn per-frame while opening/open/closing
   * (openness > 0), baked static (lid shut) when fully closed. Non-openable BEs use the idle-loop / animating
   * classifier (BBE §5). `membershipChanged` is true when an openable container crossed the open/closed
   * boundary this frame, so the caller invalidates the section's static cache.
   */
  private splitSection(records: readonly BlockEntityRecord[], clock: number): { idle: BlockEntityRecord[]; perFrame: { rec: BlockEntityRecord; openness?: number }[]; membershipChanged: boolean } {
    const idle: BlockEntityRecord[] = [];
    const perFrame: { rec: BlockEntityRecord; openness?: number }[] = [];
    let membershipChanged = false;
    for (const rec of records) {
      const def = getBEDef(rec.type);
      if (!def?.bake) continue;
      if (def.openable) {
        const openness = this.advanceOpenness(rec, clock);
        const isPerFrame = openness > OPEN_EPS;
        const st = this.openState.get(recKey(rec))!; // advanceOpenness just created/updated it
        if (st.perFrame !== isPerFrame) {
          st.perFrame = isPerFrame;
          membershipChanged = true;
        }
        if (isPerFrame) perFrame.push({ rec, openness });
        else idle.push(rec);
      } else if (def.animated && (rec.animating || def.idleLoop)) {
        perFrame.push({ rec });
      } else {
        idle.push(rec);
      }
    }
    return { idle, perFrame, membershipChanged };
  }

  /** Advance an openable container's lid openness toward its `open` target at the vanilla rate; returns the
   *  new openness. A container seen for the first time SNAPS to its target (a loaded-open chest is already
   *  open — no startup animation). Openness advances regardless of visibility so an off-screen close settles. */
  private advanceOpenness(rec: BlockEntityRecord, clock: number): number {
    const key = recKey(rec);
    this.seenOpen.add(key);
    const target = rec.open ? 1 : 0;
    let st = this.openState.get(key);
    if (!st) {
      st = { openness: target, lastClock: clock, perFrame: target > OPEN_EPS };
      this.openState.set(key, st);
      return st.openness;
    }
    const dt = Math.min(MAX_OPEN_DT, Math.max(0, clock - st.lastClock));
    st.lastClock = clock;
    if (st.openness < target) st.openness = Math.min(target, st.openness + OPEN_RATE * dt);
    else if (st.openness > target) st.openness = Math.max(target, st.openness - OPEN_RATE * dt);
    return st.openness;
  }

  /** Drop lid state for containers no longer present (removed / re-meshed away). */
  private sweepOpen(): void {
    for (const key of this.openState.keys()) if (!this.seenOpen.has(key)) this.openState.delete(key);
  }

  /** True while ANY openable container's lid is mid-open/close (openness strictly between shut and open) —
   *  the render gate keeps drawing until the lid settles so the animation doesn't freeze under render-on-
   *  demand. A fully open/closed (settled) container needs no redraw, so it doesn't count. */
  hasOpenTransitions(): boolean {
    for (const st of this.openState.values()) if (st.openness > OPEN_EPS && st.openness < 1 - OPEN_EPS) return true;
    return false;
  }

  private getStaticMesh(sk: SectionKey, origin: Vec3, idle: readonly BlockEntityRecord[]): StaticMesh | null {
    if (this.staticCache.has(sk)) return this.staticCache.get(sk) ?? null;
    const verts: RawVertex[] = [];
    const ctx = { uvFor: this.uvFor, sectionOrigin: origin };
    for (const rec of idle) {
      const v = bakeBlockEntity(rec, ctx, 0, false);
      if (v) verts.push(...v);
    }
    if (verts.length === 0) {
      this.staticCache.set(sk, null);
      return null;
    }
    const data = new Uint8Array(packVertices(verts));
    const vertex = this.device.createBuffer({ sizeBytes: data.byteLength, usage: BufferUsage.Vertex, label: `be-static-${sk}` });
    this.device.writeBuffer(vertex, 0, data);
    const mesh: StaticMesh = { vertex, quadCount: verts.length / 4 };
    this.staticCache.set(sk, mesh);
    return mesh;
  }

  private animDraw(rec: BlockEntityRecord, origin: Vec3, clock: number, openness?: number): EntityDraw | null {
    const key = recKey(rec);
    const verts = bakeBlockEntity(rec, { uvFor: this.uvFor, sectionOrigin: origin, cameraPos: this.cameraPos, openness }, clock, true);
    if (!verts || verts.length === 0) return null;
    const data = new Uint8Array(packVertices(verts));
    this.seenAnim.add(key);
    let buf = this.animBufs.get(key);
    if (!buf || buf.capacityBytes < data.byteLength) {
      if (buf) this.device.destroyBuffer(buf.vertex);
      const vertex = this.device.createBuffer({ sizeBytes: Math.max(data.byteLength, 1024), usage: BufferUsage.Vertex, label: `be-anim-${key}` });
      buf = { vertex, capacityBytes: Math.max(data.byteLength, 1024) };
      this.animBufs.set(key, buf);
    }
    this.device.writeBuffer(buf.vertex, 0, data);
    return { vertex: buf.vertex, quadCount: verts.length / 4, pass: TerrainPass.Cutout, texture: this.atlasTexture, model: translation(origin[0], origin[1], origin[2]), sortPos: origin };
  }

  /** Numeric section origin, cached per key (ENT-5). */
  private originOf(sk: SectionKey): Vec3 {
    let o = this.originCache.get(sk);
    if (!o) {
      o = sectionOrigin(sk);
      this.originCache.set(sk, o);
    }
    return o;
  }

  private sweepAnim(): void {
    for (const [key, buf] of this.animBufs) {
      if (this.seenAnim.has(key)) continue;
      this.device.destroyBuffer(buf.vertex);
      this.animBufs.delete(key);
    }
  }

  dispose(): void {
    for (const m of this.staticCache.values()) if (m) this.device.destroyBuffer(m.vertex);
    for (const b of this.animBufs.values()) this.device.destroyBuffer(b.vertex);
    this.staticCache.clear();
    this.animBufs.clear();
    this.openState.clear();
  }
}

function sectionOrigin(sk: SectionKey): Vec3 {
  const [sx, sy, sz] = parseSectionKey(sk);
  return [sx * SECTION_SIZE, sy * SECTION_SIZE, sz * SECTION_SIZE];
}
