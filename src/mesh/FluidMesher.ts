// Dedicated fluid surface mesher. RENDERER_PLAN.md §13.
//
// Faithful port of the old `FluidMeshBuilder` (vanilla LiquidBlockRenderer logic): per-cell fluid
// height from the `level` property, corner heights weighted-averaged from orthogonal + diagonal
// neighbors, downhill flow direction → top-face UV rotation, top + 4 side + (floating) bottom faces,
// same-fluid face culling, and a small side inset to avoid z-fighting with waterlogged block models.
//
// Pure DATA (GPU-free): emits `FluidQuad`s in SECTION-LOCAL block coords with atlas UVs already
// applied, so the SectionMesher can pack them straight into the right pass.
//
// TRAP 13.A: water is translucent — the SectionMesher routes water quads into the translucent
//            collector (lava is opaque → solid pass). Never an unsorted always-on-top overlay.
// TRAP 13.B: every neighbor this samples is within Chebyshev distance 1 (orthogonals, the diagonal
//            (x±1,z±1), the cell above, and the below-diagonal (x±1,y−1,z±1)). All lie inside the
//            existing 1-block apron, so NO apron widening is needed. (Documented; see SnapshotSource.)
// TRAP 13.C: animation is handled in the atlas (in-place per-frame pixel update with static UVs),
//            not here — so a sloshing fluid never re-meshes. We bake the dedicated `_still` sprite
//            (16px, matching the block tile) and convey flow via UV ROTATION, as the old renderer did;
//            swapping in the 32px `_flow` sprite is a deferred visual refinement (needs atlas support).

import { Direction, opposite, type SpriteUv } from "../types";
import { NO_SHADE } from "./VertexFormat";

export enum FluidType {
  Water = 0,
  Lava = 1,
}

/** A fluid cell's render state: its type and `level` (0 = source, 1..7 flowing; ≥8 clamps to source). */
export interface FluidState {
  type: FluidType;
  level: number;
}

/** Atlas appearance for one fluid type. */
export interface FluidAppearance {
  /** Atlas rect for the `{type}_still` sprite (top + sides). */
  sprite: SpriteUv;
  /** Tint 0xRRGGBBAA — water carries its ~0.8 opacity in the alpha byte; lava is opaque white. */
  colorRGBA: number;
  /** Water → translucent pass; lava → solid pass. */
  translucent: boolean;
  /** Directional face shade (true for water; false = full-bright for glowing lava). */
  shade: boolean;
}

/** Reads the snapshot for fluid meshing (apron-inclusive section-local coords). */
export interface FluidSampler {
  /** Packed block id at a section-local cell (0 = air). */
  idAt(lx: number, ly: number, lz: number): number;
  /** Fluid state for a block id, or null if the block isn't a fluid. */
  fluidOf(id: number): FluidState | null;
  /** Does the block at (lx,ly,lz) FULLY occlude its `dir`-facing side? (a full opaque cube → all 6). */
  occludesFace(lx: number, ly: number, lz: number, dir: Direction): boolean;
}

/** What the SectionMesher needs to mesh fluids: classify ids + look up per-type appearance. */
export interface FluidContext {
  fluidOf(id: number): FluidState | null;
  appearance(type: FluidType): FluidAppearance;
}

type Vec3 = [number, number, number];
type Vec2 = [number, number];

/** One emitted fluid surface quad, ready to pack (section-local positions, atlas UVs). */
export interface FluidQuad {
  positions: [Vec3, Vec3, Vec3, Vec3];
  atlasUV: [Vec2, Vec2, Vec2, Vec2];
  /** Direction index for shade (or NO_SHADE). */
  normal: number;
  colorRGBA: number;
}

const INSET = 0.004; // side faces pulled toward center → no z-fight with waterlogged block models
const SOURCE_AMOUNT = 8; // vanilla "amount" units; height = amount / 9
const FALLING_ADJ = SOURCE_AMOUNT / 9; // ≈0.8889, vanilla's falling-into-air flow term
const FLOW_SCALE = 0.25; // top-face flow UV is a 0.5×0.5 sub-rect rotated about (0.5,0.5)

/** Vanilla fluid render height for a cell's own level: source/≥8 → 8/9, flowing 1..7 → (8-level)/9. */
function ownHeight(state: FluidState): number {
  const level = Math.min(Math.max(0, state.level), 8);
  const amount = level === 0 || level >= 8 ? SOURCE_AMOUNT : SOURCE_AMOUNT - level;
  return amount / 9;
}

/** Height contribution of the fluid at a cell: 1 if same fluid stacked above; own height; -1 solid; 0 air. */
function heightAt(s: FluidSampler, type: FluidType, lx: number, ly: number, lz: number): number {
  const id = s.idAt(lx, ly, lz);
  const f = s.fluidOf(id);
  if (f && f.type === type) {
    const above = s.fluidOf(s.idAt(lx, ly + 1, lz));
    if (above && above.type === type) return 1;
    return ownHeight(f);
  }
  return id !== 0 ? -1 : 0; // any non-fluid block occludes (-1); air is open (0)
}

/** Weighted accumulate: heights ≥0.8 dominate (×10), [0,0.8) contribute ×1, solids (<0) are ignored. */
function addWeighted(acc: [number, number], h: number): void {
  if (h >= 0.8) {
    acc[0] += h * 10;
    acc[1] += 10;
  } else if (h >= 0) {
    acc[0] += h;
    acc[1] += 1;
  }
}

/** Corner height from self + two orthogonal neighbors + the diagonal between them (vanilla average). */
function cornerHeight(
  s: FluidSampler,
  type: FluidType,
  self: number,
  h1: number,
  h2: number,
  cx: number,
  cy: number,
  cz: number,
): number {
  if (h1 >= 1 || h2 >= 1) return 1;
  const acc: [number, number] = [0, 0];
  if (h1 > 0 || h2 > 0) {
    const corner = heightAt(s, type, cx, cy, cz);
    if (corner >= 1) return 1;
    addWeighted(acc, corner);
  }
  addWeighted(acc, self);
  addWeighted(acc, h1);
  addWeighted(acc, h2);
  return acc[1] === 0 ? self : acc[0] / acc[1];
}

/** Normalized downhill flow direction in the XZ plane (0,0 = still). */
/** The 4 orthogonal XZ neighbours probed for fluid flow direction — module const (A11: no per-call array). */
const FLOW_NEIGHBORS = [[0, -1], [0, 1], [1, 0], [-1, 0]] as const;

function flowDir(s: FluidSampler, type: FluidType, lx: number, ly: number, lz: number, selfOwn: number): Vec2 {
  let dx = 0;
  let dz = 0;
  for (const [sx, sz] of FLOW_NEIGHBORS) {
    const nId = s.idAt(lx + sx, ly, lz + sz);
    const nFluid = s.fluidOf(nId);
    const same = nFluid?.type === type;
    if (!same && nId !== 0) continue; // solid neighbor doesn't pull flow
    let diff = 0;
    if (!same) {
      const below = s.fluidOf(s.idAt(lx + sx, ly - 1, lz + sz)); // air neighbor with fluid below → flows that way
      if (below?.type === type) {
        const bh = ownHeight(below);
        if (bh > 0) diff = selfOwn - (bh - FALLING_ADJ);
      }
    } else {
      diff = selfOwn - ownHeight(nFluid!);
    }
    if (diff !== 0) {
      dx += sx * diff;
      dz += sz * diff;
    }
  }
  const len = Math.hypot(dx, dz);
  return len < 1e-4 ? [0, 0] : [dx / len, dz / len];
}

/** Map a sprite-local UV [0,1] into the atlas rect. */
function mapUV(rect: SpriteUv, u: number, v: number): Vec2 {
  return [rect.u + u * rect.width, rect.v + v * rect.height];
}

/**
 * Mesh one fluid cell into surface quads (top + exposed sides + floating bottom), section-local.
 * Faces are wound CCW-from-outside (top from above) so the translucent pass's back-face cull keeps
 * the visible side, matching the model path.
 */
/** Emit one fluid face. A2: meshFluidCell pushes faces through this sink instead of returning a
 *  `FluidQuad[]`, so the mesh hot loop allocates no per-cell array + per-face wrapper object. */
export type FluidSink = (positions: [Vec3, Vec3, Vec3, Vec3], atlasUV: [Vec2, Vec2, Vec2, Vec2], normal: number, colorRGBA: number) => void;

export function meshFluidCell(
  state: FluidState,
  lx: number,
  ly: number,
  lz: number,
  app: FluidAppearance,
  s: FluidSampler,
  sink: FluidSink,
): void {
  const { type } = state;
  const shadeNormal = (d: Direction): number => (app.shade ? d : NO_SHADE);
  const c = app.colorRGBA;

  const selfOwn = ownHeight(state);
  const aboveSame = s.fluidOf(s.idAt(lx, ly + 1, lz))?.type === type;

  // Four top-corner heights. Stacked fluid (same fluid above) is a flat full block.
  let hNW = 1, hNE = 1, hSW = 1, hSE = 1;
  if (!aboveSame) {
    const hN = heightAt(s, type, lx, ly, lz - 1);
    const hS = heightAt(s, type, lx, ly, lz + 1);
    const hE = heightAt(s, type, lx + 1, ly, lz);
    const hW = heightAt(s, type, lx - 1, ly, lz);
    hNW = cornerHeight(s, type, selfOwn, hN, hW, lx - 1, ly, lz - 1);
    hNE = cornerHeight(s, type, selfOwn, hN, hE, lx + 1, ly, lz - 1);
    hSW = cornerHeight(s, type, selfOwn, hS, hW, lx - 1, ly, lz + 1);
    hSE = cornerHeight(s, type, selfOwn, hS, hE, lx + 1, ly, lz + 1);
  }

  // TOP face (skip if same fluid above — interior — or a solid block directly above occludes it).
  // Corners: NW→SW→SE→NE (CCW from above).
  if (!aboveSame && !s.occludesFace(lx, ly + 1, lz, Direction.Down)) {
    const flow = aboveSame ? ([0, 0] as Vec2) : flowDir(s, type, lx, ly, lz, selfOwn);
    let uNW: Vec2 = [0, 0], uSW: Vec2 = [0, 1], uSE: Vec2 = [1, 1], uNE: Vec2 = [1, 0];
    if (flow[0] !== 0 || flow[1] !== 0) {
      const angle = Math.atan2(flow[1], flow[0]) - Math.PI / 2;
      const sn = Math.sin(angle) * FLOW_SCALE;
      const cs = Math.cos(angle) * FLOW_SCALE;
      uNW = [0.5 - cs - sn, 0.5 - cs + sn];
      uSW = [0.5 - cs + sn, 0.5 + cs + sn];
      uSE = [0.5 + cs + sn, 0.5 + cs - sn];
      uNE = [0.5 + cs - sn, 0.5 - cs - sn];
    }
    sink(
      [
        [lx + 0, ly + hNW, lz + 0],
        [lx + 0, ly + hSW, lz + 1],
        [lx + 1, ly + hSE, lz + 1],
        [lx + 1, ly + hNE, lz + 0],
      ],
      [mapUV(app.sprite, ...uNW), mapUV(app.sprite, ...uSW), mapUV(app.sprite, ...uSE), mapUV(app.sprite, ...uNE)],
      shadeNormal(Direction.Up),
      c,
    );
  }

  // SIDE faces — emit unless the same-level neighbor is the same fluid. Top corners follow the fluid
  // height; bottom sits at y=0; V runs 1−height (top) → 1 (bottom). Inset toward center vs z-fighting.
  const sides: { dir: Direction; dx: number; dz: number; pTop: [Vec3, Vec3]; hA: number; hB: number }[] = [
    { dir: Direction.North, dx: 0, dz: -1, pTop: [[1, 0, 0], [0, 0, 0]], hA: hNE, hB: hNW }, // z=0, x:1→0
    { dir: Direction.South, dx: 0, dz: 1, pTop: [[0, 0, 1], [1, 0, 1]], hA: hSW, hB: hSE }, // z=1, x:0→1
    { dir: Direction.West, dx: -1, dz: 0, pTop: [[0, 0, 0], [0, 0, 1]], hA: hNW, hB: hSW }, // x=0, z:0→1
    { dir: Direction.East, dx: 1, dz: 0, pTop: [[1, 0, 1], [1, 0, 0]], hA: hSE, hB: hNE }, // x=1, z:1→0
  ];
  for (const side of sides) {
    const nx = lx + side.dx, nz = lz + side.dz;
    if (s.fluidOf(s.idAt(nx, ly, nz))?.type === type) continue; // same fluid → shared face culled
    if (s.occludesFace(nx, ly, nz, opposite(side.dir))) continue; // opaque neighbor fully occludes it
    const ix = side.dx * INSET;
    const iz = side.dz * INSET;
    const [a, b] = side.pTop;
    const ax = lx + a[0] - ix, az = lz + a[2] - iz;
    const bx = lx + b[0] - ix, bz = lz + b[2] - iz;
    sink(
      [
        [ax, ly + side.hA, az], // top A
        [ax, ly + 0, az], // bottom A
        [bx, ly + 0, bz], // bottom B
        [bx, ly + side.hB, bz], // top B
      ],
      [
        mapUV(app.sprite, 0, 1 - side.hA),
        mapUV(app.sprite, 0, 1),
        mapUV(app.sprite, 1, 1),
        mapUV(app.sprite, 1, 1 - side.hB),
      ],
      shadeNormal(side.dir),
      c,
    );
  }

  // BOTTOM face — render it whenever the block below is NOT the same fluid AND does not fully occlude its
  // up face (vanilla FluidRenderer: renderDown = !sameFluid && !isFaceOccludedByNeighbor(DOWN)). So fluid
  // resting on glass/leaves/fences/slabs shows its underside, not just fluid over pure air (AUDIT M1).
  const belowId = s.idAt(lx, ly - 1, lz);
  if (s.fluidOf(belowId)?.type !== type && !s.occludesFace(lx, ly - 1, lz, Direction.Up)) {
    sink(
      [
        [lx + 0, ly + 0, lz + 1],
        [lx + 0, ly + 0, lz + 0],
        [lx + 1, ly + 0, lz + 0],
        [lx + 1, ly + 0, lz + 1],
      ],
      [mapUV(app.sprite, 0, 1), mapUV(app.sprite, 0, 0), mapUV(app.sprite, 1, 0), mapUV(app.sprite, 1, 1)],
      shadeNormal(Direction.Down),
      c,
    );
  }
}
