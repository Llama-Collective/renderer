// Smooth lighting + per-vertex ambient occlusion (SL-2). `AoFaceData` / vanilla smooth lighting.
//
// For an ALIGNED FULL FACE, each of the 4 corners samples the 2×2 block neighbourhood of the cell the face
// looks INTO (the face neighbour + the two edge neighbours toward the corner + the diagonal). From that:
//   - AO level (vanilla): both edges occlude → 0 (darkest); else 3 − (edge1 + edge2 + corner). 0..3.
//   - Corner light: the average of the 4 cells' block/sky LEVELS, with a min-non-zero replacement so an
//     occluded (0-light) cell doesn't bleed black into the corner.
// The AO level is scaled into byte18 (×255) and multiplied by the SELF block's shade-brightness (SL-1) —
// kept in its OWN byte (never premultiplied into colour), so it stays tunable and the LUT light stays
// remesh-free. An open corner with shade 1.0 → ao 255 (== AO_NONE), so an open uniform face is byte-
// identical to the flat path (smooth lighting only changes pixels at concave corners / light gradients).
//
// The corner→vertex order is FaceBakery.FACE_CORNERS, the SAME winding the geometry uses, so the 4 words
// line up with the 4 emitted vertices (a wrong permutation would rotate AO 90° — the subtle trap).

import { Direction, DIRECTION_OFFSET } from "../types";
import { snapshotIndex } from "../world/SnapshotSource";
import { packLight, AO_NONE, type PackedLight } from "./Lighting";
import { FACE_CORNERS } from "./model/FaceBakery";

const SKY_SHIFT = 4;
const LEVEL_MASK = 0xf;
const SKY_DEFAULT = 15 << SKY_SHIFT; // full sky when no light array (matches the flat path / full-bright)

/** The two TANGENT axes (the axes != the face normal axis) per direction, so the per-corner sampling never
 *  `.find`s over [0,1,2] (M3: no per-face array/closure allocation). u is the first non-normal axis, w the second. */
const UV_AXES: Record<Direction, readonly [number, number]> = {
  [Direction.Down]: [0, 2], [Direction.Up]: [0, 2],   // normal axis 1 (y)
  [Direction.North]: [0, 1], [Direction.South]: [0, 1], // normal axis 2 (z)
  [Direction.West]: [1, 2], [Direction.East]: [1, 2],   // normal axis 0 (x)
};

/** AO brightness FLOOR for a fully-occluded concave corner (H2). Vanilla and never multiply a corner
 *  by 0 — `getShadeBrightness` for an opaque cell is ~0.2, an open corner stays 1.0. The vanilla
 *  0..3 AO level is remapped onto [AO_FLOOR, 1] so the darkest corner floors at ~0.2 instead of pure black.
 *  (A faithful `AoFaceData` port — averaging four per-cell shade-brightnesses — is the eventual refinement;
 *  this floor remap is the minimal correct fix that also keeps open faces byte-identical to the flat path.) */
export const AO_FLOOR = 0.2;

/** Module scratch for the returned tuple — meshing is synchronous (one section at a time, inline or per
 *  worker), and the caller (SectionMesher) consumes all 4 words into the vertex sink BEFORE the next call,
 *  so reusing one tuple is safe and allocates nothing per face (M3). */
const sfOut: [PackedLight, PackedLight, PackedLight, PackedLight] = [0, 0, 0, 0] as unknown as [PackedLight, PackedLight, PackedLight, PackedLight];

/** Clamp a sample coord into the apron range, then index — module-level (no per-face closure capture). */
function cellIndex(x: number, y: number, z: number, lo: number, hi: number, apron: number): number {
  const xr = x < lo ? lo : x > hi ? hi : x;
  const yr = y < lo ? lo : y > hi ? hi : y;
  const zr = z < lo ? lo : z > hi ? hi : z;
  return snapshotIndex(xr, yr, zr, apron);
}

/**
 * The 4 per-corner PackedLight words for an aligned full face of the cell `(lx,ly,lz)` facing `dir`, in
 * FACE_CORNERS[dir] order (== the emitted vertex order). `shade` is the self block's shade-brightness (SL-1).
 * Returns a REUSED tuple (M3) — the caller must consume it before the next call.
 */
export function smoothFaceLight(
  dir: Direction,
  lx: number, ly: number, lz: number,
  light: Uint8Array | undefined,
  aoFlags: Uint8Array,
  apron: number,
  shade: number,
): [PackedLight, PackedLight, PackedLight, PackedLight] {
  const off = DIRECTION_OFFSET[dir];
  const cx = lx + off[0], cy = ly + off[1], cz = lz + off[2]; // face-neighbour cell = centre of the 2×2 plane
  const uv = UV_AXES[dir];
  const u = uv[0], w = uv[1];
  const corners = FACE_CORNERS[dir];
  const lo = -apron, hi = 15 + apron;
  for (let v = 0; v < 4; v++) {
    const sel = corners[v];
    const du = sel[u] === 1 ? 1 : -1;
    const dw = sel[w] === 1 ? 1 : -1;
    // The four sampled cells (centre / edge-u / edge-w / diagonal) as SCALAR coords — `+du` lands on axis u,
    // `+dw` on axis w (no per-cell arrays).
    const e1x = cx + (u === 0 ? du : 0), e1y = cy + (u === 1 ? du : 0), e1z = cz + (u === 2 ? du : 0);
    const e2x = cx + (w === 0 ? dw : 0), e2y = cy + (w === 1 ? dw : 0), e2z = cz + (w === 2 ? dw : 0);
    const dgx = e1x + (w === 0 ? dw : 0), dgy = e1y + (w === 1 ? dw : 0), dgz = e1z + (w === 2 ? dw : 0);

    const iC = cellIndex(cx, cy, cz, lo, hi, apron);
    const i1 = cellIndex(e1x, e1y, e1z, lo, hi, apron);
    const i2 = cellIndex(e2x, e2y, e2z, lo, hi, apron);
    const iD = cellIndex(dgx, dgy, dgz, lo, hi, apron);

    const o1 = aoFlags[i1] === 1 ? 1 : 0;
    const o2 = aoFlags[i2] === 1 ? 1 : 0;
    const oc = aoFlags[iD] === 1 ? 1 : 0;
    const aoLevel = o1 && o2 ? 0 : 3 - (o1 + o2 + oc); // vanilla AO: 0 (both sides) … 3 (open)
    // H2: remap 0..3 onto [AO_FLOOR, 1] — a fully-occluded corner floors at AO_FLOOR, never multiplies by 0.
    // aoLevel 3 → 1.0 (open faces stay byte-identical to flat); aoLevel 0 → AO_FLOOR.
    const ao = Math.round((AO_FLOOR + (aoLevel / 3) * (1 - AO_FLOOR)) * shade * 255);

    // Corner light = average of the 4 cells' levels, min-non-zero replacing 0 (no dark-edge bleed).
    const pC = light ? light[iC] : SKY_DEFAULT;
    const p1 = light ? light[i1] : SKY_DEFAULT;
    const p2 = light ? light[i2] : SKY_DEFAULT;
    const pD = light ? light[iD] : SKY_DEFAULT;
    const bC = pC & LEVEL_MASK, b1 = p1 & LEVEL_MASK, b2 = p2 & LEVEL_MASK, bD = pD & LEVEL_MASK;
    const sC = (pC >> SKY_SHIFT) & LEVEL_MASK, s1 = (p1 >> SKY_SHIFT) & LEVEL_MASK, s2 = (p2 >> SKY_SHIFT) & LEVEL_MASK, sD = (pD >> SKY_SHIFT) & LEVEL_MASK;
    let minB = 16; if (bC > 0 && bC < minB) minB = bC; if (b1 > 0 && b1 < minB) minB = b1; if (b2 > 0 && b2 < minB) minB = b2; if (bD > 0 && bD < minB) minB = bD;
    let minS = 16; if (sC > 0 && sC < minS) minS = sC; if (s1 > 0 && s1 < minS) minS = s1; if (s2 > 0 && s2 < minS) minS = s2; if (sD > 0 && sD < minS) minS = sD;
    if (minB === 16) minB = 0;
    if (minS === 16) minS = 0;
    const block = (bC || minB) + (b1 || minB) + (b2 || minB) + (bD || minB);
    const sky = (sC || minS) + (s1 || minS) + (s2 || minS) + (sD || minS);
    sfOut[v] = packLight(Math.round(block / 4), Math.round(sky / 4), ao, false);
  }
  return sfOut;
}

/** True iff `ao` is the fully-open value — used in tests to assert open faces stay byte-identical to flat. */
export const AO_OPEN = AO_NONE;

/** True iff the quad is a FULL aligned cube face — its 4 positions match FACE_CORNERS[cullface] exactly.
 *  Excludes inset / partial faces (grass paths, slabs' side cuts) where 2×2 corner sampling isn't valid. */
export function isAlignedFullFace(cullface: Direction | null, positions: ReadonlyArray<readonly [number, number, number]>): boolean {
  if (cullface === null) return false;
  const corners = FACE_CORNERS[cullface];
  for (let v = 0; v < 4; v++) {
    const p = positions[v], cc = corners[v];
    if (p[0] !== cc[0] || p[1] !== cc[1] || p[2] !== cc[2]) return false;
  }
  return true;
}
