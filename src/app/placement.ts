// Placement-hint math: which face region was clicked and the block facing it implies. Ported verbatim
// from the old renderer (rendererGeometry.ts + EditableWorldRenderer.getPlacementHint/faceRegionForPick/
// facingForRegion/regionFacingMap/localHit). Pure functions over a PickResult + camera basis.
// RENDERER_INTEGRATION_PLAN Step 6.

import type { FaceRegion, PickResult, PlacementHint, Vec3Tuple } from "../../../schematic-io/types";

type Facing = PlacementHint["facing"];

const DIRECTION_NAMES: [Vec3Tuple, Facing][] = [
  [[0, -1, 0], "down"],
  [[0, 1, 0], "up"],
  [[0, 0, -1], "north"],
  [[0, 0, 1], "south"],
  [[-1, 0, 0], "west"],
  [[1, 0, 0], "east"],
];

export function directionName(v: Vec3Tuple): Facing {
  let best = DIRECTION_NAMES[0];
  let bestDot = -Infinity;
  for (const entry of DIRECTION_NAMES) {
    const dot = v[0] * entry[0][0] + v[1] * entry[0][1] + v[2] * entry[0][2];
    if (dot > bestDot) {
      bestDot = dot;
      best = entry;
    }
  }
  return best[1];
}

export function negate(v: Vec3Tuple): Vec3Tuple {
  return [-v[0], -v[1], -v[2]];
}

/** In-plane (right, up) world axes for a face, looking at it from outside. */
export function faceBasis(normal: Vec3Tuple): [Vec3Tuple, Vec3Tuple] {
  const [nx, ny, nz] = normal;
  if (ny !== 0) return [[1, 0, 0], [0, 0, ny > 0 ? -1 : 1]];
  if (nx !== 0) return [[0, 0, nx > 0 ? -1 : 1], [0, 1, 0]];
  return [[nz > 0 ? 1 : -1, 0, 0], [0, 1, 0]];
}

export function axisComponent(v: Vec3Tuple, axis: Vec3Tuple): number {
  if (axis[0] !== 0) return v[0];
  if (axis[1] !== 0) return v[1];
  return v[2];
}

export function sign(axis: Vec3Tuple): number {
  return axis[0] + axis[1] + axis[2] > 0 ? 1 : -1;
}

/** Fractional hit location inside the CLICKED block, each component in [0,1). */
export function localHit(pick: PickResult): Vec3Tuple {
  const hp = pick.hitPoint;
  if (!hp) return [0.5, 0.5, 0.5];
  const [bx, by, bz] = pick.position;
  const clampFrac = (v: number, base: number): number => {
    const f = v - base;
    return Math.min(0.9999, Math.max(0.0001, f));
  };
  return [clampFrac(hp[0], bx), clampFrac(hp[1], by), clampFrac(hp[2], bz)];
}

/** Classify the hit into a face region using the two in-plane axes. */
export function faceRegionForPick(pick: PickResult): FaceRegion {
  const hit = localHit(pick);
  const [right, up] = faceBasis(pick.normal);
  const u = axisComponent(hit, right) * sign(right) + (sign(right) < 0 ? 1 : 0);
  const v = axisComponent(hit, up) * sign(up) + (sign(up) < 0 ? 1 : 0);
  const du = u - 0.5;
  const dv = v - 0.5;
  const m = Math.max(Math.abs(du), Math.abs(dv));
  if (m <= 1 / 6) return "centermost";
  if (m <= 1 / 3) return "center";
  if (Math.abs(du) > Math.abs(dv)) return du > 0 ? "right" : "left";
  return dv > 0 ? "up" : "down";
}

export function facingForRegion(normal: Vec3Tuple, region: FaceRegion): Facing {
  if (region === "centermost") return directionName(negate(normal));
  if (region === "center") return directionName(normal);
  const [right, up] = faceBasis(normal);
  switch (region) {
    case "right":
      return directionName(right);
    case "left":
      return directionName(negate(right));
    case "up":
      return directionName(up);
    case "down":
      return directionName(negate(up));
  }
}

export function regionFacingMap(pick: PickResult | null): Partial<Record<FaceRegion, Facing>> {
  if (!pick) return {};
  const normal: Vec3Tuple = pick.type === "block" ? pick.normal : [0, 1, 0];
  const regions: FaceRegion[] = ["centermost", "center", "up", "down", "left", "right"];
  const map: Partial<Record<FaceRegion, Facing>> = {};
  for (const r of regions) map[r] = facingForRegion(normal, r);
  return map;
}

/**
 * Resolve where a new block would be placed from a pick + the orientation hint. `cameraHorizontalFacing`
 * is used for ground/entity picks (camera-relative cardinal).
 */
export function getPlacementHint(pick: PickResult | null, cameraHorizontalFacing: Facing): PlacementHint | null {
  if (!pick) return null;
  if (pick.type === "block") {
    const [nx, ny, nz] = pick.normal;
    const position: Vec3Tuple = [pick.position[0] + nx, pick.position[1] + ny, pick.position[2] + nz];
    const region = faceRegionForPick(pick);
    const facing = facingForRegion(pick.normal, region);
    return { position, clickedFace: pick.normal, hit: localHit(pick), region, facing };
  }
  if (pick.type === "ground") {
    const region = faceRegionForPick(pick);
    const facing = facingForRegion(pick.normal, region);
    return { position: pick.position, clickedFace: [0, 1, 0], hit: localHit(pick), region, facing };
  }
  return { position: pick.position, clickedFace: [0, 1, 0], hit: [0.5, 0, 0.5], region: "center", facing: cameraHorizontalFacing };
}
