// Moving-piston animation driver. PISTON_PLAN.md (full plan), RENDERER_PLAN.md §18 (Pistons).
//
// Ports the proven interpolation + blockstate-derivation from the old three.js renderer
// (schematic-renderer/src/editor/MovingPistonAnimator) WITHOUT any three.js. Per active moving_piston
// block entity it derives which blockstate(s) to draw — the moved block (case C), or a sliding piston
// HEAD plus a static EXTENDED base (case B), or the head itself (case A) — slides the moving piece by
// direction·extendedProgress (interpolated by the tick partial), bakes each chosen blockstate to world
// quads, and routes them BY RENDER LAYER:
//   • OPAQUE/CUTOUT → a transient TERRAIN draw (TransientOpaque): terrain shader, exact shading, no pop
//     at the static↔moving handoff (PISTON_PLAN §5 D1).
//   • TRANSLUCENT (glass/slime/honey/ice) → its own self-sorted, depth-WRITING translucent draw
//     (TransientTranslucent) — vanilla's `translucentMovingBlock`. Never merged into terrain, never a
//     separate always-on-top pass.
//
// Driven by the sim's per-tick `MovingPistonInfo[]` via set() (once per applied tick) and the
// render-frame `partial` (0..1 into the current tick) via frame() (every frame). The completion handoff
// is implicit: when a piston finishes the sim drops it from movingPistons AND writes the resting block
// in the SAME diff, so set([]) clears the transient the same tick the resting block re-meshes.

import type { GraphicsDevice } from "../../core/GraphicsDevice";
import type { BakedBlockModel } from "../../mesh/model/BakedBlockModel";
import type { BlockProps } from "../../mesh/model/BlockStateResolver";
import type { RawVertex } from "../../mesh/VertexFormat";
import { TerrainPass, type Vec3 } from "../../types";
import type { SectionDraw } from "../TerrainRenderer";
import { bakedModelToEntityVerts, parseBlockState } from "../entities/blockDisplay";
import { TransientOpaque } from "./TransientOpaque";
import { TransientTranslucent, type TransientQuad } from "../TransientTranslucent";

/** One active moving_piston block entity (mirrors the sim's MovingPistonInfo verbatim — zero-copy). */
export interface MovingPistonInput {
  x: number;
  y: number;
  z: number;
  movedState: { name: string; properties: Record<string, string> };
  /** Piston facing (Direction NAME: "down"|"up"|"north"|"south"|"west"|"east") — the slide axis. */
  direction: string;
  extending: boolean;
  isSourcePiston: boolean;
  progress: number;
  progressO: number;
}

/** This frame's transient piston geometry: opaque terrain draws + the moving translucent draw (or null). */
export interface PistonFrame {
  opaqueDraws: SectionDraw[];
  translucent: SectionDraw | null;
}

type HeadMode = "none" | "shortUntilHalf" | "shortFromHalf";

interface PistonEntry {
  x: number;
  y: number;
  z: number;
  signature: string;
  step: Vec3; // DIRECTION_STEPS[direction]
  extending: boolean;
  headMode: HeadMode;
  progress: number;
  progressO: number;
  /** The sliding piece's blockstate (and, for head cases, its short/long variants). */
  moving: string;
  movingShort: string | null;
  movingLong: string | null;
  /** Case B only: the static EXTENDED piston base, drawn at the cell with NO slide offset. */
  base: string | null;
}

interface StatePlan {
  moving: string;
  base: string | null;
  headMode: HeadMode;
  movingShort?: string;
  movingLong?: string;
}

/** Per-Direction unit step (the slide axis). Verbatim from the old renderer's DIRECTION_STEPS. */
const DIRECTION_STEPS: Record<string, Vec3> = {
  north: [0, 0, -1],
  south: [0, 0, 1],
  west: [-1, 0, 0],
  east: [1, 0, 0],
  up: [0, 1, 0],
  down: [0, -1, 0],
};

export class PistonTransients {
  private readonly entries = new Map<string, PistonEntry>();
  private readonly opaque: TransientOpaque;
  private readonly translucent: TransientTranslucent;
  /** Per-blockstate cache: block-local translucent verts (4/quad). Opaque is cached in TransientOpaque. */
  private readonly translucentLocal = new Map<string, RawVertex[]>();
  private readonly processed = new Set<string>();

  constructor(
    device: GraphicsDevice,
    private readonly bakeModel: (name: string, props: BlockProps) => BakedBlockModel | null,
  ) {
    this.opaque = new TransientOpaque(device);
    this.translucent = new TransientTranslucent(device);
  }

  /** Any pistons animating this frame? */
  active(): boolean {
    return this.entries.size > 0;
  }

  /**
   * Replace the set of animating moving_piston block entities — call ONCE per applied sim tick. A cell
   * whose signature is unchanged just updates progress/progressO in place (no rebuild); a vanished cell
   * is dropped. frame() interpolates progressO→progress between calls.
   */
  set(infos: readonly MovingPistonInput[]): void {
    const seen = new Set<string>();
    for (const info of infos) {
      const key = `${info.x},${info.y},${info.z}`;
      seen.add(key);
      const signature = signatureOf(info);
      const existing = this.entries.get(key);
      if (existing && existing.signature === signature) {
        existing.progress = info.progress;
        existing.progressO = info.progressO;
        continue;
      }
      this.entries.set(key, this.createEntry(info, signature));
    }
    for (const key of [...this.entries.keys()]) {
      if (!seen.has(key)) this.entries.delete(key);
    }
  }

  /**
   * Build this frame's transient piston geometry. `partial` is the 0..1 fraction into the current sim
   * tick. Returns opaque terrain draws to APPEND to the frame draw list, and the moving translucent draw
   * (self-sorted + depth-writing — never merged with terrain glass, so no terrain section is suppressed).
   */
  frame(camPos: Vec3, partial: number): PistonFrame {
    const opaqueDraws: SectionDraw[] = [];
    const transientQuads: TransientQuad[] = [];
    for (const entry of this.entries.values()) {
      const progress = entry.progressO + partial * (entry.progress - entry.progressO);
      const extended = entry.extending ? progress - 1 : 1 - progress; // getExtendedProgress
      const moveOrigin: Vec3 = [
        entry.x + entry.step[0] * extended,
        entry.y + entry.step[1] * extended,
        entry.z + entry.step[2] * extended,
      ];
      this.emit(this.activeMoving(entry, progress), moveOrigin, opaqueDraws, transientQuads);
      // Case B: the piston body stays put while the head pulls in — no slide offset.
      if (entry.base) this.emit(entry.base, [entry.x, entry.y, entry.z], opaqueDraws, transientQuads);
    }
    const translucent = transientQuads.length > 0 ? this.translucent.update(camPos, transientQuads) : null;
    return { opaqueDraws, translucent };
  }

  /** Drop all cached baked geometry (call on atlas/scene rebuild — UVs changed). Keeps active entries. */
  invalidate(): void {
    this.opaque.clear();
    this.translucentLocal.clear();
    this.processed.clear();
  }

  dispose(): void {
    this.entries.clear();
    this.invalidate();
    this.opaque.dispose();
    this.translucent.dispose();
  }

  // ── internals ────────────────────────────────────────────────────────────────

  /** The blockstate to draw for the sliding piece this frame (head short/long swaps at 0.5). */
  private activeMoving(entry: PistonEntry, progress: number): string {
    if (entry.headMode === "none") return entry.moving;
    const showShort = entry.headMode === "shortUntilHalf" ? progress <= 0.5 : progress >= 0.5;
    return showShort ? entry.movingShort ?? entry.moving : entry.movingLong ?? entry.moving;
  }

  /** Bake (once, cached) a blockstate and emit its opaque draws + translucent world quads at `origin`. */
  private emit(state: string, origin: Vec3, opaqueDraws: SectionDraw[], transientQuads: TransientQuad[]): void {
    this.ensureTemplate(state);
    for (const d of this.opaque.draws(state, origin)) opaqueDraws.push(d);
    const local = this.translucentLocal.get(state);
    if (local) {
      for (let i = 0; i + 4 <= local.length; i += 4) {
        transientQuads.push([
          worldVertex(local[i], origin),
          worldVertex(local[i + 1], origin),
          worldVertex(local[i + 2], origin),
          worldVertex(local[i + 3], origin),
        ]);
      }
    }
  }

  /** Bake a blockstate string ONCE: opaque → TransientOpaque buffers, translucent → block-local verts. */
  private ensureTemplate(state: string): void {
    if (this.processed.has(state)) return;
    this.processed.add(state);
    const { name, props } = parseBlockState(state);
    const baked = this.bakeModel(name, props);
    const verts = baked ? bakedModelToEntityVerts(baked) : null;
    this.opaque.ensure(state, verts?.[TerrainPass.Solid], verts?.[TerrainPass.Cutout]);
    this.translucentLocal.set(state, verts?.[TerrainPass.Translucent] ?? []);
  }

  private createEntry(info: MovingPistonInput, signature: string): PistonEntry {
    const plan = statesOf(info);
    return {
      x: info.x,
      y: info.y,
      z: info.z,
      signature,
      step: DIRECTION_STEPS[info.direction] ?? [0, 0, 0],
      extending: info.extending,
      headMode: plan.headMode,
      progress: info.progress,
      progressO: info.progressO,
      moving: plan.moving,
      movingShort: plan.movingShort ?? null,
      movingLong: plan.movingLong ?? null,
      base: plan.base,
    };
  }
}

/** World-space corner = block-local vert + slide origin (everything else carried through). */
function worldVertex(v: RawVertex, origin: Vec3): RawVertex {
  return { ...v, x: v.x + origin[0], y: v.y + origin[1], z: v.z + origin[2] };
}

/** Rebuild-identity string: when unchanged across a tick the entry is reused (progress excluded). */
export function signatureOf(info: MovingPistonInput): string {
  const props = Object.entries(info.movedState.properties)
    .map(([k, v]) => `${k}=${v}`)
    .sort()
    .join(",");
  return `${info.movedState.name}|${props}|${info.direction}|${info.extending}|${info.isSourcePiston}`;
}

/**
 * The blockstate(s) vanilla's PistonHeadRenderer.extractRenderState submits, ported verbatim from the old
 * renderer's statesOf (itself a faithful port). Three cases (PISTON_PLAN §2.3):
 *   A — movedState is a piston_head (head extending): sliding head, SHORT while progress ≤ 0.5.
 *   B — source piston retracting: sliding synthesized head (SHORT while progress ≥ 0.5) + STATIC base.
 *   C — any other block: the moved block's own model slides as one piece.
 */
export function statesOf(info: MovingPistonInput): StatePlan {
  const dir = info.direction;
  const movedName = info.movedState.name;

  if (movedName === "minecraft:piston_head") {
    const type = info.movedState.properties.type ?? "normal";
    return {
      moving: `minecraft:piston_head[facing=${dir},short=true,type=${type}]`,
      movingShort: `minecraft:piston_head[facing=${dir},short=true,type=${type}]`,
      movingLong: `minecraft:piston_head[facing=${dir},short=false,type=${type}]`,
      base: null,
      headMode: "shortUntilHalf",
    };
  }

  if (info.isSourcePiston && !info.extending) {
    const type = movedName === "minecraft:sticky_piston" ? "sticky" : "normal";
    return {
      moving: `minecraft:piston_head[facing=${dir},short=false,type=${type}]`,
      movingShort: `minecraft:piston_head[facing=${dir},short=true,type=${type}]`,
      movingLong: `minecraft:piston_head[facing=${dir},short=false,type=${type}]`,
      base: `${movedName}[facing=${dir},extended=true]`,
      headMode: "shortFromHalf",
    };
  }

  const props = Object.entries(info.movedState.properties)
    .map(([k, v]) => `${k}=${v}`)
    .sort()
    .join(",");
  return { moving: `${movedName}${props ? `[${props}]` : ""}`, base: null, headMode: "none" };
}
