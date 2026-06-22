// Vanilla `ModelPart` / `Cube` / `PartPose` box-model bake. RENDERER_PLAN.md §18 (Phase 4.5b), GATE 18.1.
//
// Vanilla entity + block-entity geometry is code-defined box hierarchies: `CubeListBuilder.texOffs(u,v)
// .addBox(ox,oy,oz, dx,dy,dz)` under a `ModelPart` tree with `PartPose` pivots/rotations. This is a
// GPU-free port of that bake (modeled on the proven cubane `SignRenderer` port — same FACE_DEFS,
// box-unwrap UV layout, and per-face winding), emitting the shared 20-byte `RawVertex` so entity
// geometry packs/draws exactly like terrain.
//
// Coordinate conventions:
//  - Cube origin/size + part pivot are in MODEL PIXELS (1/16 block); positions emit in BLOCK units.
//  - Part matrix accumulates parent · T(pivot/16) · Rz · Ry · Rx (vanilla's ZYX rotate order), with an
//    optional per-part `PartPose` override (animation re-pose — TRAP 18.C). A model-wide `base` matrix
//    (e.g. vanilla's scale(-1,-1,1) + lift to feet) converts model space → entity-local (Y-up).
//  - `normal` carries the LOCAL face Direction index (0..5); the entity shader rotates it by the model
//    matrix to shade, so a posed/rotated part still lights correctly.
//  - UVs box-unwrap into [0,1] of the entity texture, then remap into the atlas `uvRect` (bake-once +
//    UV-remap, §18) so one mesh serves every atlas placement.

import { Direction, DIRECTION_OFFSET, TerrainPass, type SpriteUv } from "../../types";
import { NO_SHADE, type RawVertex } from "../VertexFormat";
import { identity, mul, rotationX, rotationY, rotationZ, transformDir, translation, type Mat4 } from "./mat4";

type V3 = readonly [number, number, number];

/**
 * The face's WORLD-facing normal in entity-local space. A part's rotation + the model `base` are applied
 * to POSITIONS by `m`, so the raw cube face Direction is NOT the face's actual orientation — e.g. the
 * minecart's four walls are the same SIDE cube rotated 0/90/180/270°, so storing the raw dir gives all
 * four the SAME normal (→ they'd shade identically). Transform the dir by `m` and quantize to the nearest
 * axis (the vertex format stores one axis index). `m` is translations·rotations·±1-scale, so it equals
 * its own inverse-transpose — `transformDir` is the correct normal transform, exact for 90° rotations.
 */
function bakedNormal(m: Mat4, dir: Direction): number {
  const b = DIRECTION_OFFSET[dir];
  const [x, y, z] = transformDir(m, b[0], b[1], b[2]);
  const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
  if (ay >= ax && ay >= az) return y >= 0 ? Direction.Up : Direction.Down;
  if (ax >= az) return x >= 0 ? Direction.East : Direction.West;
  return z >= 0 ? Direction.South : Direction.North;
}

/** One `addBox`: origin + size in model pixels, `uv` = texOffs, optional mirror + inflate (px). */
export interface EntityCube {
  origin: V3;
  size: V3;
  uv: readonly [number, number];
  mirror?: boolean;
  /** CubeDeformation: grow the box by this many px on every side; UVs still use the un-inflated size. */
  inflate?: number;
  /** Restrict which faces are emitted (vanilla `EnumSet.of(...)` for flat planes). Default: all visible. */
  faces?: Direction[];
}

/** A node in the `ModelPart` tree. Pivot/rotation default to zero; cubes + children optional. */
export interface ModelPartDef {
  name?: string;
  /** Pivot (part origin) in model pixels. */
  pivot?: V3;
  /** Default rotation (radians, applied Z→Y→X). */
  rotation?: V3;
  cubes?: EntityCube[];
  children?: ModelPartDef[];
}

/** A per-frame pose override for a named part (animation). Rotation REPLACES the default; offset ADDS. */
export interface PartPose {
  rotation?: V3;
  /** Extra translation in model pixels, added to the part's pivot (e.g. a bob/bounce). */
  offsetPixels?: V3;
}

export interface BakeOptions {
  /** Entity texture logical dimensions in px (e.g. sheep 64×32) for box-unwrap. */
  texWidth: number;
  texHeight: number;
  /** Atlas rect to remap the [0,1] entity-texture UVs into. */
  uvRect: SpriteUv;
  /** Vertex tint 0xRRGGBBAA (default opaque white). */
  color?: number;
  /** Model→entity-local transform (Y flip, lift, scale). Default identity. */
  base?: Mat4;
  /** Per-part animation overrides keyed by `ModelPartDef.name`. */
  poses?: Record<string, PartPose | undefined>;
  /** Render pass / material (default Cutout → alpha-tested, which suits most entity textures). */
  pass?: TerrainPass;
  /** Apply directional shade from the (model-transformed) face normal; false → full-bright. */
  shade?: boolean;
}

const WHITE = 0xffffffff;

/** Where a bake writes its vertices. `PackedVertexSink.push` satisfies this structurally (its trailing
 *  `light` arg defaults to full-bright, matching a `RawVertex` with no `light`), so a dynamic mob can bake
 *  STRAIGHT into the packed buffer — no intermediate `RawVertex[]`, no second packing pass (A4). */
export interface VertexEmit {
  push(x: number, y: number, z: number, u: number, v: number, normal: number, colorRGBA: number, material: number): void;
}

/** Collects emitted vertices as `RawVertex` objects (the existing `bakeModel`/`bakeModelPart` contract). */
class RawVertexCollector implements VertexEmit {
  readonly out: RawVertex[] = [];
  push(x: number, y: number, z: number, u: number, v: number, normal: number, colorRGBA: number, material: number): void {
    this.out.push({ x, y, z, u, v, normal, colorRGBA, material });
  }
}

// 8 cube corners in (---, +--, ++-, -+-, --+, +-+, +++, -++) order, then 6 faces by corner index +
// direction — identical to cubane SignRenderer (which matches the Rust entity mesher / vanilla).
const FACE_DEFS: { dir: Direction; corners: readonly [number, number, number, number] }[] = [
  { dir: Direction.Down, corners: [4, 5, 1, 0] },
  { dir: Direction.Up, corners: [3, 2, 6, 7] },
  { dir: Direction.North, corners: [1, 0, 3, 2] },
  { dir: Direction.South, corners: [4, 5, 6, 7] },
  { dir: Direction.West, corners: [0, 4, 7, 3] },
  { dir: Direction.East, corners: [5, 1, 2, 6] },
];

/** Bake a part tree into model-local `RawVertex` quads (4 verts/face, shared 0,1,2,0,2,3 winding). */
export function bakeModelPart(root: ModelPartDef, opts: BakeOptions): RawVertex[] {
  const c = new RawVertexCollector();
  walk(root, opts.base ?? identity(), opts, c);
  return c.out;
}

/** Convenience: bake several sibling root parts under one base (a whole entity/BE model). */
export function bakeModel(roots: readonly ModelPartDef[], opts: BakeOptions): RawVertex[] {
  const c = new RawVertexCollector();
  const base = opts.base ?? identity();
  for (const r of roots) walk(r, base, opts, c);
  return c.out;
}

/**
 * Bake several sibling roots STRAIGHT into a vertex sink — no intermediate `RawVertex[]`, no second packing
 * pass (A4 dynamic-mob path). The emitted bytes are identical to `packVertices(bakeModel(roots, opts))`
 * (same vertex order + values; the sink's default light = full-bright = a `RawVertex` with no `light`).
 */
export function bakeModelInto(roots: readonly ModelPartDef[], opts: BakeOptions, sink: VertexEmit): void {
  const base = opts.base ?? identity();
  for (const r of roots) walk(r, base, opts, sink);
}

function walk(part: ModelPartDef, parent: Mat4, opts: BakeOptions, emit: VertexEmit): void {
  const pose = part.name ? opts.poses?.[part.name] : undefined;
  const pv = part.pivot ?? [0, 0, 0];
  const off = pose?.offsetPixels ?? [0, 0, 0];
  const rot = pose?.rotation ?? part.rotation ?? [0, 0, 0];
  // local = T((pivot+offset)/16) · Rz · Ry · Rx  (vanilla ModelPart.rotate order).
  const local = mul(
    translation((pv[0] + off[0]) / 16, (pv[1] + off[1]) / 16, (pv[2] + off[2]) / 16),
    rotationZ(rot[2]),
    rotationY(rot[1]),
    rotationX(rot[0]),
  );
  const m = mul(parent, local);
  if (part.cubes) for (const cube of part.cubes) emitCube(cube, m, opts, emit);
  if (part.children) for (const c of part.children) walk(c, m, opts, emit);
}

// Reused 8-corner world-space scratch (bake is synchronous + single-threaded, and `emitCube` fully consumes
// it before returning) — no per-cube `corners`/`corners.map` temp arrays (A4).
const WORLD: [number, number, number][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
function setWorld(i: number, m: Mat4, x: number, y: number, z: number): void {
  const w = WORLD[i]; // same dot products as transformPoint — byte-identical positions
  w[0] = m[0] * x + m[4] * y + m[8] * z + m[12];
  w[1] = m[1] * x + m[5] * y + m[9] * z + m[13];
  w[2] = m[2] * x + m[6] * y + m[10] * z + m[14];
}

function emitCube(cube: EntityCube, m: Mat4, opts: BakeOptions, emit: VertexEmit): void {
  const inf = cube.inflate ?? 0;
  const [ox, oy, oz] = cube.origin;
  const [w, h, d] = cube.size;
  // Inflated box for POSITIONS; UVs use the un-inflated size (vanilla CubeDeformation).
  const x0 = (ox - inf) / 16, y0 = (oy - inf) / 16, z0 = (oz - inf) / 16;
  const x1 = (ox + w + inf) / 16, y1 = (oy + h + inf) / 16, z1 = (oz + d + inf) / 16;

  // 8 corners (FACE_DEFS index order) transformed into entity-local space, into the reused scratch.
  setWorld(0, m, x0, y0, z0); setWorld(1, m, x1, y0, z0); setWorld(2, m, x1, y1, z0); setWorld(3, m, x0, y1, z0);
  setWorld(4, m, x0, y0, z1); setWorld(5, m, x1, y0, z1); setWorld(6, m, x1, y1, z1); setWorld(7, m, x0, y1, z1);
  const world = WORLD;

  const color = opts.color ?? WHITE;
  const material = (opts.pass ?? TerrainPass.Cutout) === TerrainPass.Cutout ? 1 : 0;

  // Flat (zero-thickness) cubes emit coincident / zero-area faces → z-fighting. Skip them: drop any
  // zero-area face, and on the single degenerate axis keep only the POSITIVE face (vanilla emits both,
  // but our depth-test flickers on coincident quads). An explicit `faces` list overrides this entirely.
  const degX = Math.abs(x1 - x0) < 1e-6, degY = Math.abs(y1 - y0) < 1e-6, degZ = Math.abs(z1 - z0) < 1e-6;
  const skip = (dir: Direction): boolean => {
    if (cube.faces) return !cube.faces.includes(dir);
    if ((dir === Direction.Up || dir === Direction.Down) && (degX || degZ)) return true;
    if ((dir === Direction.North || dir === Direction.South) && (degX || degY)) return true;
    if ((dir === Direction.West || dir === Direction.East) && (degY || degZ)) return true;
    return (degY && dir === Direction.Down) || (degX && dir === Direction.West) || (degZ && dir === Direction.North);
  };

  for (const { dir, corners: ci } of FACE_DEFS) {
    if (skip(dir)) continue;
    const uv = cubeFaceUVs(cube.uv, cube.size, dir, cube.mirror ?? false, opts);
    // Entity-local face normal (part rotation + base folded in via `m`); the entity shader rotates it by
    // the per-frame model matrix to shade. Flat if !shade. See bakedNormal — storing raw `dir` would give
    // rotated parts (e.g. all four minecart walls) the same normal, so they'd shade identically.
    const n = opts.shade === false ? NO_SHADE : bakedNormal(m, dir);
    // Up/Down faces wind opposite the sides; reverse the corner+uv pairs so the shared 0,1,2,0,2,3
    // EBO produces a front-facing quad either way (matches SignRenderer's per-face winding).
    const order = dir === Direction.Up || dir === Direction.Down ? [0, 3, 2, 1] : [0, 1, 2, 3];
    for (const k of order) {
      const p = world[ci[k]];
      emit.push(p[0], p[1], p[2], uv[k][0], uv[k][1], n, color, material);
    }
  }
}

/**
 * Box-unwrap UVs for one cube face (Minecraft layout: a horizontal cross), normalized to the entity
 * texture then remapped into the atlas rect. Same formula + per-face order as cubane `SignRenderer`.
 */
function cubeFaceUVs(
  texOffset: readonly [number, number],
  size: V3,
  face: Direction,
  mirror: boolean,
  opts: BakeOptions,
): [number, number][] {
  const [u0, v0] = texOffset;
  const [w, h, d] = size;
  const tw = opts.texWidth;
  const th = opts.texHeight;

  let left: number, top: number, right: number, bottom: number;
  switch (face) {
    case Direction.Down: [left, top, right, bottom] = [u0 + d, v0, u0 + d + w, v0 + d]; break;
    case Direction.Up: [left, top, right, bottom] = [u0 + d + w, v0, u0 + d + w + w, v0 + d]; break;
    case Direction.North: [left, top, right, bottom] = [u0 + d, v0 + d, u0 + d + w, v0 + d + h]; break;
    case Direction.South: [left, top, right, bottom] = [u0 + d + w + d, v0 + d, u0 + d + w + d + w, v0 + d + h]; break;
    case Direction.West: [left, top, right, bottom] = [u0, v0 + d, u0 + d, v0 + d + h]; break;
    case Direction.East: [left, top, right, bottom] = [u0 + d + w, v0 + d, u0 + d + w + d, v0 + d + h]; break;
    default: [left, top, right, bottom] = [u0, v0, u0 + w, v0 + h];
  }

  const nl = left / tw, nt = top / th, nr = right / tw, nb = bottom / th;
  // Texture-space corner UVs in FACE_DEFS corner order (top-left origin).
  let quad: [number, number][];
  if (face === Direction.Up) {
    quad = mirror
      ? [[nr, nb], [nl, nb], [nl, nt], [nr, nt]]
      : [[nl, nb], [nr, nb], [nr, nt], [nl, nt]];
  } else if (face === Direction.Down) {
    // Down is Up flipped across V (vanilla ModelPart.Cube DOWN polygon). It must NOT use the side ordering
    // below, which flips the face in U — that mirrored every box's bottom face, visible as a REVERSED
    // inner-lid texture on an open chest (the lid's inside is its Down face).
    quad = mirror
      ? [[nr, nt], [nl, nt], [nl, nb], [nr, nb]]
      : [[nl, nt], [nr, nt], [nr, nb], [nl, nb]];
  } else {
    quad = mirror
      ? [[nl, nt], [nr, nt], [nr, nb], [nl, nb]]
      : [[nr, nt], [nl, nt], [nl, nb], [nr, nb]];
  }
  // Remap [0,1] texture UV → atlas rect.
  const r = opts.uvRect;
  return quad.map(([u, v]) => [r.u + u * r.width, r.v + v * r.height]);
}
