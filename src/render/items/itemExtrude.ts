// Flat-sprite → 3D item model: a faithful port of vanilla `ItemModelGenerator` (resources/model/cuboid).
// RENDERER_PLAN.md §18. A `builtin/generated` item (most non-block items — tools, food, materials) is its
// 2D sprite extruded into a 1px-thick slab: a front (+Z) + back (−Z) quad spanning z 7.5→8.5 — both showing
// the same texel at the same world (x,y), so the slab needs no back-face culling — plus a "cookie-cutter"
// side quad everywhere an opaque pixel borders a transparent one (the silhouette).
// GPU-free; emits the shared `RawVertex` (UVs remapped into the item-atlas rect) so it draws like any mesh.

import { Direction, type SpriteUv } from "../../types";
import { NO_SHADE, type RawVertex } from "../../mesh/VertexFormat";

const MIN_Z = 7.5, MAX_Z = 8.5; // the 1px slab in model px
const INSET = 0.1; // UV_SHRINK — pull side-face UVs in 0.1px so they don't bleed neighbours

export interface ItemSprite {
  rgba: Uint8Array; // tightly packed, length = width·height·4
  width: number;
  height: number;
}

type V3 = [number, number, number];
type UV = [number, number];

/** Extrude a generated item's sprite into a 3D model. `rect` = the sprite's rect in the item atlas. */
export function extrudeItemSprite(sprite: ItemSprite, rect: SpriteUv, color = 0xffffffff): RawVertex[] {
  const out: RawVertex[] = [];
  const sw = sprite.width, sh = sprite.height;
  const xs = 16 / sw, ys = 16 / sh; // model-px per sprite-px (sprites may be 16/32/…)
  const u = (mu: number): number => rect.u + (mu / 16) * rect.width; // model-px UV [0,16] → atlas
  const v = (mv: number): number => rect.v + (mv / 16) * rect.height;
  const p = (px: number): number => px / 16; // model px → block

  // Front (+Z, MAX_Z, SOUTH) + back (−Z, MIN_Z, NORTH) caps close the 1px slab — exactly as vanilla
  // `ItemModelGenerator.bakeExtrudedSprite`. Crucially the back is NOT world-mirrored: vanilla's
  // NORTH_FACE_UVS U-flip (16→0) just cancels NORTH's reversed FaceInfo traversal, so BOTH caps map the
  // SAME texel to the same world (x,y) — front reads correct from +Z, back reads correct (un-mirrored)
  // from −Z, and a transparent pixel coincides on both. So with NO culling there's no cross-bleed: the
  // earlier X came from a genuinely mirrored back cap, not from drawing two faces. The back is just
  // reverse-wound (normal −Z) with the front's UVs at MIN_Z.
  quad(out, [[0, 0, MAX_Z], [16, 0, MAX_Z], [16, 16, MAX_Z], [0, 16, MAX_Z]], [[u(0), v(16)], [u(16), v(16)], [u(16), v(0)], [u(0), v(0)]], Direction.South, color, p);
  quad(out, [[16, 16, MIN_Z], [16, 0, MIN_Z], [0, 0, MIN_Z], [0, 16, MIN_Z]], [[u(16), v(0)], [u(16), v(16)], [u(0), v(16)], [u(0), v(0)]], Direction.North, color, p);

  // Silhouette side faces (every opaque↔transparent pixel transition).
  for (const f of sideFaces(sprite)) {
    const x = f.x, y = f.y, o = f.dir;
    const u0 = x + INSET, u1 = x + 1 - INSET;
    const horiz = o === SIDE.up || o === SIDE.down;
    const v0 = horiz ? y + INSET : y + 1 - INSET;
    const v1 = horiz ? y + 1 - INSET : y + INSET;
    let sx = x, sy = y, ex = x, ey = y;
    if (o === SIDE.up) { ex += 1; }
    else if (o === SIDE.down) { ex += 1; sy += 1; ey += 1; }
    else if (o === SIDE.left) { ey += 1; }
    else { sx += 1; ex += 1; ey += 1; } // right
    sx *= xs; ex *= xs; sy = 16 - sy * ys; ey = 16 - ey * ys;
    const uvs: UV[] = [[u(u0 * xs), v(v0 * ys)], [u(u1 * xs), v(v0 * ys)], [u(u1 * xs), v(v1 * ys)], [u(u0 * xs), v(v1 * ys)]];
    let corners: V3[];
    let dir: Direction;
    if (o === SIDE.up) { corners = [[sx, sy, MIN_Z], [ex, sy, MIN_Z], [ex, sy, MAX_Z], [sx, sy, MAX_Z]]; dir = Direction.Up; }
    else if (o === SIDE.down) { corners = [[sx, ey, MIN_Z], [ex, ey, MIN_Z], [ex, ey, MAX_Z], [sx, ey, MAX_Z]]; dir = Direction.Down; }
    else if (o === SIDE.left) { corners = [[sx, sy, MIN_Z], [sx, ey, MIN_Z], [sx, ey, MAX_Z], [sx, sy, MAX_Z]]; dir = Direction.West; }
    else { corners = [[ex, sy, MIN_Z], [ex, ey, MIN_Z], [ex, ey, MAX_Z], [ex, sy, MAX_Z]]; dir = Direction.East; }
    quad(out, corners, uvs, dir, color, p);
  }
  return out;
}

// ── silhouette scan ─────────────────────────────────────────────────────────────
const SIDE = { up: 0, down: 1, left: 2, right: 3 } as const;
interface SideFace { dir: number; x: number; y: number }

// Vanilla `SpriteContents.isTransparent` is strict `ARGB.alpha(pixel) == 0` — a pixel is part of the
// silhouette unless it is fully transparent (not a >0 threshold).
const isTransparent = (s: ItemSprite, x: number, y: number): boolean =>
  x < 0 || y < 0 || x >= s.width || y >= s.height || s.rgba[(y * s.width + x) * 4 + 3] === 0;

/** Every opaque pixel emits a side face on each edge whose neighbour is transparent (vanilla getSideFaces). */
function sideFaces(s: ItemSprite): SideFace[] {
  const faces: SideFace[] = [];
  for (let y = 0; y < s.height; y++) {
    for (let x = 0; x < s.width; x++) {
      if (isTransparent(s, x, y)) continue;
      if (isTransparent(s, x, y - 1)) faces.push({ dir: SIDE.up, x, y });
      if (isTransparent(s, x, y + 1)) faces.push({ dir: SIDE.down, x, y });
      if (isTransparent(s, x - 1, y)) faces.push({ dir: SIDE.left, x, y });
      if (isTransparent(s, x + 1, y)) faces.push({ dir: SIDE.right, x, y });
    }
  }
  return faces;
}

/** Push a textured quad (4 verts, shared 0,1,2,0,2,3 EBO). `p` converts model px → block units. */
function quad(out: RawVertex[], corners: V3[], uvs: UV[], dir: Direction, color: number, p: (n: number) => number): void {
  // Side faces shade by their Direction; the flat front/back stay full-bright (NO_SHADE) like a GUI sprite.
  const n = dir === Direction.North || dir === Direction.South ? NO_SHADE : dir;
  for (let i = 0; i < 4; i++) {
    out.push({ x: p(corners[i][0]), y: p(corners[i][1]), z: p(corners[i][2]), u: uvs[i][0], v: uvs[i][1], normal: n, colorRGBA: color, material: 1 });
  }
}
