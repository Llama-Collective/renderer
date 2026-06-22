// Bake a resolved model + blockstate transform into geometry quads. RENDERER_PLAN §24.5.
//
// Faithful port of vanilla FaceBakery (MC 26.1.2): element from/to → per-face 4 corners (CCW from
// outside), default/explicit UVs with face.rotation, element rotation (+rescale), blockstate x/y/z
// rotation about block center, uvlock UV transform, cullface remap, directional shade, tintindex,
// and zero-thickness degenerate-face dropping.
//
// Output is GPU-free DATA: positions in BLOCK units [0,1] (model/16), UVs SPRITE-LOCAL [0,1] (the
// atlas rect is applied later by the mesher). Quads are CCW viewed from outside.

import { Direction, DIRECTION_OFFSET } from "../../types";
import { FACE_NAMES, FACE_TO_DIRECTION, type RawElement, type RawFace, type ResolvedModel, type VariantPart } from "./ModelTypes";
import { resolveFaceTexture, resolveModel, type RawModelProvider } from "./ModelResolver";
import {
  blockstateMatrix,
  elementMatrix,
  identity3,
  isIdentity3,
  mul3,
  rotX90,
  rotY90,
  transformPoint3,
  transpose3,
  type Mat3,
  type Vec3,
} from "./bakeMath";

export type Vec2 = [number, number];

export interface BakedQuad {
  /** 4 corners in block units [0,1], CCW from outside. */
  positions: [Vec3, Vec3, Vec3, Vec3];
  /** 4 sprite-local UVs [0,1] (atlas rect applied downstream). */
  uvs: [Vec2, Vec2, Vec2, Vec2];
  sprite: string;
  forceTranslucent: boolean;
  /** Face this quad is culled against, or null (never culled). */
  cullface: Direction | null;
  /** Snapped facing — drives directional shade and the per-direction bucket. */
  normal: Direction;
  /** -1 = untinted; ≥0 = tint source index. */
  tintindex: number;
  /** Apply directional shade (false → flat full-bright). */
  shade: boolean;
}

const BLOCK_CENTER: Vec3 = [0.5, 0.5, 0.5];

// Corner selectors per Direction: [xSel,ySel,zSel] where 0=from(min), 1=to(max). Vanilla FaceInfo.
// Exported so smooth lighting (SL-2) maps per-corner AO/light to the SAME vertex order (no 90° rotation).
export const FACE_CORNERS: Record<Direction, ReadonlyArray<readonly [0 | 1, 0 | 1, 0 | 1]>> = {
  [Direction.Down]: [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]],
  [Direction.Up]: [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
  [Direction.North]: [[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]],
  [Direction.South]: [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
  [Direction.West]: [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]],
  [Direction.East]: [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]],
};

const CULLFACE_TO_DIRECTION: Record<string, Direction> = {
  down: Direction.Down,
  bottom: Direction.Down,
  up: Direction.Up,
  north: Direction.North,
  south: Direction.South,
  west: Direction.West,
  east: Direction.East,
};

/** Face axis for the degenerate-face test: east/west=x, up/down=y, north/south=z. */
const FACE_AXIS: Record<Direction, "x" | "y" | "z"> = {
  [Direction.West]: "x",
  [Direction.East]: "x",
  [Direction.Down]: "y",
  [Direction.Up]: "y",
  [Direction.North]: "z",
  [Direction.South]: "z",
};

/** Snap an arbitrary direction to the nearest of the 6 axis Directions. */
function snapDirection(n: Vec3): Direction {
  let best = Direction.Up;
  let bestDot = -Infinity;
  for (const d of [Direction.Down, Direction.Up, Direction.North, Direction.South, Direction.West, Direction.East]) {
    const nd = DIRECTION_OFFSET[d];
    const dot = n[0] * nd[0] + n[1] * nd[1] + n[2] * nd[2];
    if (dot > bestDot) {
      bestDot = dot;
      best = d;
    }
  }
  return best;
}

/** Geometric normal = cross of the quad diagonals (v2−v0)×(v3−v1), normalized. */
function quadNormal(p: BakedQuad["positions"]): Vec3 {
  const ax = p[2][0] - p[0][0], ay = p[2][1] - p[0][1], az = p[2][2] - p[0][2];
  const bx = p[3][0] - p[1][0], by = p[3][1] - p[1][1], bz = p[3][2] - p[1][2];
  const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
  const len = Math.hypot(nx, ny, nz) || 1;
  return [nx / len, ny / len, nz / len];
}

// uvlock: per-original-face local→global rotation (canonical face has +Z normal = SOUTH). §24.5.
function localToGlobal(dir: Direction): Mat3 {
  switch (dir) {
    case Direction.South: return identity3();
    case Direction.East: return rotY90(90);
    case Direction.West: return rotY90(-90);
    case Direction.North: return rotY90(180);
    case Direction.Up: return rotX90(-90);
    case Direction.Down: return rotX90(90);
  }
}

// uvlock UV transform for `origFace` under blockstate rotation `blockRot`. Vanilla `FaceBakery.bakeVertex`
// applies `modelState.inverseFaceTransformation(face)` to the centered UVs — the `invertAffine` of
// `BlockMath.getFaceTransformation`, NOT the forward transform (cubane/earlier ports get this backwards:
// AUDIT H1). `getFaceTransformation = transpose(localToGlobal(newFace))·blockRot·localToGlobal(origFace)`;
// every factor is a pure 90° rotation, so the product is orthogonal and its inverse is its transpose.
function uvLockTransform(blockRot: Mat3, origFace: Direction): Mat3 {
  const ft = mul3(blockRot, localToGlobal(origFace));
  const newFace = snapDirection(transformPoint3(ft, [0, 0, 1]));
  const faceTransform = mul3(transpose3(localToGlobal(newFace)), ft); // vanilla getFaceTransformation (forward)
  return transpose3(faceTransform); // inverseFaceTransformation — the matrix vanilla actually applies
}

/** Default face UV (model 0..16) derived from element bounds when `uv` is omitted. Vanilla. */
function defaultFaceUV(dir: Direction, from: Vec3, to: Vec3): [number, number, number, number] {
  switch (dir) {
    case Direction.Down: return [from[0], 16 - to[2], to[0], 16 - from[2]];
    case Direction.Up: return [from[0], from[2], to[0], to[2]];
    case Direction.North: return [16 - to[0], 16 - to[1], 16 - from[0], 16 - from[1]];
    case Direction.South: return [from[0], 16 - to[1], to[0], 16 - from[1]];
    case Direction.West: return [from[2], 16 - to[1], to[2], 16 - from[1]];
    case Direction.East: return [16 - to[2], 16 - to[1], 16 - from[2], 16 - from[1]];
  }
}

/** Resolve + bake a list of blockstate parts (the mesher's entry point). */
export function bakeParts(parts: VariantPart[], provider: RawModelProvider): BakedQuad[] {
  const out: BakedQuad[] = [];
  for (const part of parts) out.push(...bakeModelPart(resolveModel(part.model, provider), part));
  return out;
}

/** Bake every present face of every element of one model part. */
export function bakeModelPart(model: ResolvedModel, part: VariantPart): BakedQuad[] {
  const blockRot = blockstateMatrix(part.x, part.y, part.z);
  const blockRotId = isIdentity3(blockRot);
  const out: BakedQuad[] = [];
  for (const element of model.elements) bakeElement(element, model, part, blockRot, blockRotId, out);
  // Keep every face the model defines — back-face culling (TerrainRenderer, all passes) hides the away-
  // facing side, so inward faces (spawner/vault cages, redstone-torch glow box) show only their far side
  // over the geometry underneath, exactly as vanilla. No inner-face/coincident-pair deletion needed.
  return out;
}

function bakeElement(
  element: RawElement,
  model: ResolvedModel,
  part: VariantPart,
  blockRot: Mat3,
  blockRotId: boolean,
  out: BakedQuad[],
): void {
  const from = element.from as Vec3;
  const to = element.to as Vec3;
  const flat: Record<"x" | "y" | "z", boolean> = { x: from[0] === to[0], y: from[1] === to[1], z: from[2] === to[2] };

  const elemRot = element.rotation
    ? { m: elementMatrix(element.rotation.axis, element.rotation.angle, element.rotation.rescale === true), origin: scale16(element.rotation.origin as Vec3) }
    : null;
  const shade = element.shade !== false;

  for (const faceName of FACE_NAMES) {
    const face = element.faces[faceName];
    if (!face) continue;
    const dir = FACE_TO_DIRECTION[faceName];
    // Drop faces of a zero-thickness element that aren't along the flat axis (vanilla).
    if ((flat.x || flat.y || flat.z) && !flat[FACE_AXIS[dir]]) continue;
    // A flat element's two opposite faces (e.g. campfire fire / cross plane north+south) are coplanar but
    // OPPOSITELY wound, so back-face culling shows exactly one per viewpoint — no z-fight, visible from
    // both sides (vanilla). Both are kept; culling, not deletion, picks the camera-facing one.
    out.push(bakeFace(faceName, face, dir, from, to, model, part, blockRot, blockRotId, elemRot, shade));
  }
}

function bakeFace(
  faceName: (typeof FACE_NAMES)[number],
  face: RawFace,
  dir: Direction,
  from: Vec3,
  to: Vec3,
  model: ResolvedModel,
  part: VariantPart,
  blockRot: Mat3,
  blockRotId: boolean,
  elemRot: { m: Mat3; origin: Vec3 } | null,
  shade: boolean,
): BakedQuad {
  // 1) Corner positions (model 0..16 → block 0..1), with element + blockstate rotation.
  const positions = FACE_CORNERS[dir].map((sel) => {
    let p: Vec3 = [(sel[0] ? to[0] : from[0]) / 16, (sel[1] ? to[1] : from[1]) / 16, (sel[2] ? to[2] : from[2]) / 16];
    if (elemRot) p = rotateAboutLocal(elemRot.m, elemRot.origin, p);
    if (!blockRotId) p = rotateAboutLocal(blockRot, BLOCK_CENTER, p);
    return p;
  }) as [Vec3, Vec3, Vec3, Vec3];

  // 2) UVs (model 0..16 → sprite 0..1), face.rotation, uvlock.
  const uv16 = face.uv ?? defaultFaceUV(dir, from, to);
  const shift = (((Math.round((face.rotation ?? 0) / 90) % 4) + 4) % 4);
  const uvTransform = part.uvlock && !blockRotId ? uvLockTransform(blockRot, dir) : null;
  const uvs = [0, 1, 2, 3].map((i) => {
    const k = (i + shift) % 4;
    let u = (k === 0 || k === 1 ? uv16[0] : uv16[2]) / 16;
    let v = (k === 0 || k === 3 ? uv16[1] : uv16[3]) / 16;
    if (uvTransform) {
      const t = transformPoint3(uvTransform, [u - 0.5, v - 0.5, 0]);
      u = t[0] + 0.5;
      v = t[1] + 0.5;
    }
    return [u, v] as Vec2;
  }) as [Vec2, Vec2, Vec2, Vec2];

  // 3) cullface (remapped by blockstate rotation), normal, texture.
  let cullface: Direction | null = face.cullface !== undefined ? (CULLFACE_TO_DIRECTION[face.cullface] ?? null) : null;
  if (cullface !== null && !blockRotId) cullface = snapDirection(transformPoint3(blockRot, DIRECTION_OFFSET[cullface]));
  const tex = resolveFaceTexture(face.texture, model.textures);

  return {
    positions,
    uvs,
    sprite: tex.sprite,
    forceTranslucent: tex.forceTranslucent,
    cullface,
    normal: snapDirection(quadNormal(positions)),
    tintindex: face.tintindex ?? -1,
    shade,
  };
}

function rotateAboutLocal(m: Mat3, pivot: Vec3, v: Vec3): Vec3 {
  const p = transformPoint3(m, [v[0] - pivot[0], v[1] - pivot[1], v[2] - pivot[2]]);
  return [p[0] + pivot[0], p[1] + pivot[1], p[2] + pivot[2]];
}

function scale16(v: Vec3): Vec3 {
  return [v[0] / 16, v[1] / 16, v[2] / 16];
}
