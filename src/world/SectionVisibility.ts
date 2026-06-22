// Per-section directional visibility (`DirectionalVisGraph` / `VisibilityEncoding`).
// RENDERER_PLAN.md §13; RENDERER_OPTIMIZATION_PLAN.md Phase 3 OCC-1; "Directional
// Visibility Data".
//
// At mesh time we flood-fill a section's 16³ opacity grid to learn which of its 6 faces are mutually
// reachable through open (non-opaque) space. Two faces touched by the same connected air component are
// "visible through" each other. The camera BFS (OcclusionCuller) uses this to answer "entered through
// face F — which faces can light/sight exit?", so a fully enclosed section stops the search and the
// geometry behind it is never drawn.
//
// Encoding: 6 faces ⇒ C(6,2)=15 unordered face PAIRS ⇒ a 15-bit field in a plain JS number (fits in
// 32-bit bitwise ops — no BigInt, no overflow). The 6 self-pairs are omitted: re-exiting the face you
// entered is always removed by the BFS's outward direction-culling, so it carries no information.
//
// Pure, GPU-free, worker-safe. Consumed by OcclusionCuller (BFS) and produced by the mesher.

import { Direction, DIRECTION_COUNT, SECTION_SIZE, type Vec3i } from "../types";

/** A section's face-to-face connectivity, packed as 15 pair-bits. See `pairIndex`. */
export type SectionVisibility = number;

/** Every face pair connected — an empty / very sparse section (and absent/never-meshed sections). */
export const VIS_ALL: SectionVisibility = (1 << 15) - 1; // 0x7FFF
/** No face pair connected — a fully opaque (solid) section blocks all sight lines. */
export const VIS_NONE: SectionVisibility = 0;

// Lower-triangular pair index for the 15 unordered pairs of 6 faces. pair(a,b) with a≠b.
// Layout (a<b): (0,1)=0 (0,2)=1 (0,3)=2 (0,4)=3 (0,5)=4 (1,2)=5 (1,3)=6 (1,4)=7 (1,5)=8
//               (2,3)=9 (2,4)=10 (2,5)=11 (3,4)=12 (3,5)=13 (4,5)=14
const PAIR_INDEX: number[] = (() => {
  const t = new Array(DIRECTION_COUNT * DIRECTION_COUNT).fill(-1);
  let k = 0;
  for (let a = 0; a < DIRECTION_COUNT; a++) {
    for (let b = a + 1; b < DIRECTION_COUNT; b++) {
      t[a * DIRECTION_COUNT + b] = k;
      t[b * DIRECTION_COUNT + a] = k;
      k++;
    }
  }
  return t;
})();

/** Bit index of the connectivity pair (a,b); a and b must differ. */
function pairBit(a: number, b: number): number {
  return 1 << PAIR_INDEX[a * DIRECTION_COUNT + b];
}

// ── OCC-2 per-perspective DIRECTION_SETS (default-off) ───────────────────────────────────────────────
// stores up to 4 per-perspective connectivity sets per section (DirectionalVisGraph's per-quadrant
// graphs) and joins them at BFS time by the camera's side of the section, so a section only forwards sight
// in directions consistent with where the camera is. We collapse that into 4 masks keyed by the HORIZONTAL
// octant (±x, ±z) of each air component — the axes along which oblique sight-lines over-draw most. Element
// [0] is ALWAYS the symmetric union of all 4 (so any `?.[0] ?? VIS_ALL` reader + every off-path reader is
// byte-identical to `computeSectionVisibility`). The other 3 each hold only the components on/straddling
// that horizontal quadrant, so joining the camera's quadrant draws a SUBSET of the symmetric mask — never a
// superset (a component touching the quadrant boundary is recorded in BOTH adjacent quadrants, so a join can
// never DROP a connection a sight-line from that camera position could use → conservative, never a hole).

/** Number of perspective masks per section. `[0]` = symmetric union; `[1..3]` = the 4 horizontal quadrants
 *  (West-North, West-South, East-North, East-South) packed into indices 1..3 + the union at 0 (a quadrant's
 *  own set is at `1 + quadrant`, but quadrant 0 (W-N) reuses slot 0's union role is NOT — see `setIndexOf`). */
export const VIS_SETS_COUNT = 4;

/** Horizontal-quadrant index (0..3) for a point at section-local (x,z), relative to the section centre.
 *  bit0 = east half (x ≥ size/2), bit1 = south half (z ≥ size/2). Mirrors `joinVisibilityData`'s camera key. */
function quadrantOfLocal(x: number, z: number, size: number): number {
  const half = size / 2;
  const east = x >= half ? 1 : 0;
  const south = z >= half ? 2 : 0;
  return east | south;
}

/**
 * Per-perspective sibling of `computeSectionVisibility` (OCC-2). Re-floods the SAME opacity grid into a
 * `Uint32Array(VIS_SETS_COUNT)`:
 *   - `[0]` = the symmetric union — BYTE-IDENTICAL to `computeSectionVisibility(isOpaque, size)`, so every
 *     `?.[0] ?? VIS_ALL` / off-path reader is unaffected.
 *   - `[1+q]` for q∈{0,1,2}… no — the 4 masks live at `[0..3]` where `[q]` accumulates the components whose
 *     air touches horizontal quadrant `q`. A component spanning two quadrants is recorded in BOTH (and its
 *     pairs also land in `[0]`), so a per-quadrant join is a conservative subset of the union (never a hole).
 *
 * Fast exits mirror the symmetric producer: all-air ⇒ every slot VIS_ALL; all-solid ⇒ every slot VIS_NONE.
 * Pure integer math, no allocation beyond the flood scratch. `computeSectionVisibility` is left UNTOUCHED as
 * the off-path producer (TRAP OCC-2.A — making the symmetric one asymmetric breaks every off-path reader).
 */
export function computeSectionVisibilitySets(
  isOpaque: (x: number, y: number, z: number) => boolean,
  size: number = SECTION_SIZE,
): Uint32Array {
  const sets = new Uint32Array(VIS_SETS_COUNT);
  const n = size * size * size;
  const opaque = new Uint8Array(n);
  let opaqueCount = 0;
  const idx = (x: number, y: number, z: number) => (y * size + z) * size + x;
  for (let y = 0; y < size; y++) {
    for (let z = 0; z < size; z++) {
      for (let x = 0; x < size; x++) {
        if (isOpaque(x, y, z)) {
          opaque[idx(x, y, z)] = 1;
          opaqueCount++;
        }
      }
    }
  }
  if (opaqueCount === 0) {
    sets.fill(VIS_ALL); // all-air: every quadrant fully traversable (matches VIS_ALL union)
    return sets;
  }
  if (opaqueCount === n) return sets; // all-solid: every slot VIS_NONE (already 0)

  const visited = new Uint8Array(n);
  const stack: number[] = [];
  for (let y0 = 0; y0 < size; y0++) {
    for (let z0 = 0; z0 < size; z0++) {
      for (let x0 = 0; x0 < size; x0++) {
        const start = idx(x0, y0, z0);
        if (opaque[start] || visited[start]) continue;
        // Flood this air component, accumulating its touched faces AND the horizontal quadrants it spans.
        let faces = 0;
        let quadMask = 0; // bit q ⇒ this component has a cell in horizontal quadrant q (conservative key).
        visited[start] = 1;
        stack.push(start);
        while (stack.length > 0) {
          const c = stack.pop()!;
          const x = c % size;
          const z = ((c / size) | 0) % size;
          const y = (c / (size * size)) | 0;
          faces |= faceMaskOf(x, y, z, size);
          quadMask |= 1 << quadrantOfLocal(x, z, size);
          if (x > 0) pushIf(c - 1, opaque, visited, stack);
          if (x < size - 1) pushIf(c + 1, opaque, visited, stack);
          if (z > 0) pushIf(c - size, opaque, visited, stack);
          if (z < size - 1) pushIf(c + size, opaque, visited, stack);
          if (y > 0) pushIf(c - size * size, opaque, visited, stack);
          if (y < size - 1) pushIf(c + size * size, opaque, visited, stack);
        }
        if (faces === 0) continue;
        // Every pair of touched faces is mutually visible through this component.
        let pairs = 0;
        for (let a = 0; a < DIRECTION_COUNT; a++) {
          if ((faces & (1 << a)) === 0) continue;
          for (let b = a + 1; b < DIRECTION_COUNT; b++) {
            if ((faces & (1 << b)) !== 0) pairs |= pairBit(a, b);
          }
        }
        // Element [0] is ALWAYS the symmetric UNION (every component, symmetric-compatible — the off-path
        // word). Elements [1..3] are the per-quadrant REFINEMENTS for camera quadrants 1,2,3: a component's
        // pairs land in slot q (q∈{1,2,3}) only if the component touches quadrant q (conservative — a camera in
        // quadrant q must keep every connection a component reaching its side carries; a component the quadrant
        // never reaches is the over-draw OCC-2 removes). Camera quadrant 0 (W-N) joins to [0] (the union) — it
        // is not separately refined, so it stays a conservative SUBSET (equal to the symmetric mask). A
        // component spanning a quadrant boundary is recorded in BOTH adjacent quadrant slots ⇒ no hole.
        sets[0] |= pairs;
        for (let q = 1; q < VIS_SETS_COUNT; q++) {
          if ((quadMask & (1 << q)) !== 0) sets[q] |= pairs;
        }
      }
    }
  }
  return sets;
}

/**
 * Collapse the per-perspective `sets` (from `computeSectionVisibilitySets`) to the single 15-bit mask the
 * camera at `cameraQuadrant` (0..3, from `joinQuadrantOf` / the BFS) should use, so the unchanged
 * `getConnections` consumes it exactly as the symmetric word. Returns the camera's quadrant slot, which is by
 * construction ⊆ `sets[0]` (every pair written to slot q was also written to the union `[0]`) → the join can
 * only ever REMOVE connections the symmetric mask had, never add one (conservative, never a hole). Quadrant 0
 * (West-North) is NOT separately refined — it maps to `[0]` (the union), i.e. equal to the symmetric mask (a
 * trivial subset). An undefined/short `sets` (legacy / off-path / mixed build) falls back to `[0] ?? VIS_ALL`
 * so it can never under-draw.
 */
export function joinVisibilityData(sets: Uint32Array | undefined, _cameraQuadrant: number): SectionVisibility {
  if (!sets || sets.length === 0) return VIS_ALL;
  // M1 / P2.2: return the SYMMETRIC UNION (slot [0]) regardless of camera quadrant. The previous per-
  // horizontal-quadrant refinement (`return sets[q]`) could DROP a connection a sight-line from the camera
  // could use — a verified HOLE — because this 4-horizontal-quadrant-slot model is NOT . keys
  // 4 OCTANT perspectives (`DirectionalVisGraph.DIRECTION_SETS`) and ORs EVERY set the camera's octant(s)
  // select (`OcclusionCuller.joinVisibilityData`) — a conservative UNION, never a single slot. Until that
  // octant model is ported, returning the union makes `perPerspectiveVisibility` SAFE to enable (it is then
  // byte-identical to the symmetric pass — no over-draw reduction yet, but no hole).
  // ⚠️ M2: a faithful octant port RE-INTRODUCES camera-octant dependence, so before the cached occlusion path
  // (`occBfsCache`) may be used with OCC-2 on, the cache key MUST fold in the camera octant (see
  // SchematicViewer.occludedTerrainDraws) — else a sub-section camera move would serve a stale visibility set.
  return sets[0] ?? VIS_ALL;
}

/** Horizontal-quadrant key (0..3) for a camera at world (cx,cz) relative to a section at coords (sx,sz).
 *  bit0 = camera east of / level with the section centre, bit1 = camera south of / level with it. For the
 *  HOME section pass the camera's SUB-section position (which of the 16³ corners it occupies); for any other
 *  section the sign of (camera − section centre) per axis. Conservative on a tie (level ⇒ both halves are
 *  recorded in the sets, so either choice keeps every needed connection). `worldToLocal` derives the
 *  sub-section offset for the home section from the world camera position (NOT just the floored coord). */
export function joinQuadrantOf(camWorldX: number, camWorldZ: number, sx: number, sz: number, size: number = SECTION_SIZE): number {
  // Section centre in world block coords.
  const cxCentre = sx * size + size / 2;
  const czCentre = sz * size + size / 2;
  const east = camWorldX >= cxCentre ? 1 : 0;
  const south = camWorldZ >= czCentre ? 2 : 0;
  return east | south;
}

/** Are faces `a` and `b` mutually visible through this section? (a===b ⇒ false: handled by the BFS.) */
export function isConnected(vis: SectionVisibility, a: Direction, b: Direction): boolean {
  if (a === b) return false;
  return (vis & pairBit(a, b)) !== 0;
}

/**
 * Given the set of directions the section was ENTERED from (`incoming`, a 6-bit Direction mask), return
 * the 6-bit mask of directions sight can EXIT through. This is `VisibilityEncoding.getConnections`.
 *
 * PARITY NOTE (audit P3 item — "getConnections O(36) fold"): the RESULT here is bit-for-bit.
 * stores a 48-bit DIRECTED `visibilityData` (`bit(from,to) = from*8 + to`) and computes the union with a
 * 64-bit `createMask`/`foldOutgoingDirections` shift-fold (`>> 32`/`16`/`8`). Vanilla's occlusion graph is
 * SYMMETRIC (visibilityBetween(a,b) == visibilityBetween(b,a) — both are flood-fill connectivity, identical
 * to `computeSectionVisibility`), so encode sets bit(from,to) AND bit(to,from); folding it yields
 * exactly "every face that shares a connected pair with any incoming face" — which is what this loop computes
 * over the equivalent 15-bit symmetric pack. The literal shift-fold is deliberately NOT ported: it needs fast
 * 64-bit bitwise ops, which JS lacks (`>> 32` truncates to 32 bits; `BigInt` would REGRESS this path). The
 * loop early-exits on the (usually 1–3 bit) `incoming` mask, and with `occBfsCache` on the BFS — hence this
 * call — runs only per home-section crossing, so the fold's micro-saving is moot. Same behavior, no risk.
 */
export function getConnections(vis: SectionVisibility, incoming: number): number {
  let out = 0;
  for (let from = 0; from < DIRECTION_COUNT; from++) {
    if ((incoming & (1 << from)) === 0) continue;
    for (let to = 0; to < DIRECTION_COUNT; to++) {
      if (to !== from && (vis & pairBit(from, to)) !== 0) out |= 1 << to;
    }
  }
  return out;
}

// ── OCC-2 angle / slope cone (default-off SECOND sub-flag `occlusionAngleMask`) ───────────────────────
// trims each BFS propagation step against an angle/slope cone seeded at the camera and intersected as
// the search steps outward, so a section only forwards sight in directions consistent with the camera angle.
// We model the cone as a per-frontier-node inclusive slope interval per axis-pair and AND a cone-derived
// 6-bit direction mask into `outgoing`. The cardinal rule (TRAP OCC-2.B): the cone is the OUTER (conservative)
// bound — `intersectSlopes` may only WIDEN a kept axis, never narrow past the true cone — so the angle mask
// can drop only directions the camera demonstrably cannot see down, never a grazing-but-visible one. Shipped
// behind its OWN sub-flag so a cone bug can never regress the DIRECTION_SETS win.

/** An inclusive slope interval `[lo, hi]` (a 1-D cone half-angle proxy). `lo ≤ hi`. The widest cone is
 *  `[-Infinity, Infinity]` (all directions). Intersection KEEPS the wider extreme on each side (outer bound). */
export interface SlopeCone {
  readonly lo: number;
  readonly hi: number;
}

/** The fully-open cone — every direction allowed (the conservative seed at the camera origin). */
export const SLOPE_CONE_OPEN: SlopeCone = { lo: -Infinity, hi: Infinity };

/**
 * Conservative (OUTER-bound) intersection of two slope cones: the result is at least as WIDE as the wider of
 * the two on each side — it never narrows past either input (TRAP OCC-2.B: an over-tight cone holes a grazing
 * section). Concretely `lo = min(a.lo, b.lo)`, `hi = max(a.hi, b.hi)` — a union/widen, NOT a true intersection,
 * deliberately, so the cone can only ever be the outer bound of the real sight cone. Returns a cone ⊇ both.
 */
export function intersectSlopes(a: SlopeCone, b: SlopeCone): SlopeCone {
  return { lo: Math.min(a.lo, b.lo), hi: Math.max(a.hi, b.hi) };
}

/**
 * The LOCAL (per-step, tighter) cone-derived 6-bit direction mask for a frontier node at section-delta
 * (dx,dy,dz) from the origin. Drops only the axis directions that point STRICTLY back toward the camera on a
 * dominated axis — a conservative subset of `outwardMask`. Returns ALL_DIRECTIONS when the node is the origin
 * (cone seeded fully open). Used when `occlusionAngleMask` is on; ANDed into `outgoing` AFTER the join.
 */
export function getAngleVisibilityMaskLocal(dx: number, dy: number, dz: number): number {
  // ⚠️ CONFIRMED NO-OP (audit P2.2/L): this only clears the single direction pointing back toward the camera
  // per axis — EXACTLY what `outwardMask` already removes — and the BFS ANDs `outwardMask` right after, so this
  // mask can never drop a direction that survives. The `occlusionAngleMask` sub-flag therefore provides ZERO
  // additional culling today. A real implementation would mirror MAGNITUDE-based perpendicular-axis
  // occlusion (e.g. `if (|dx| > |dy| || |dz| > |dy|) occlude UP/DOWN`), which removes directions BEYOND the
  // outward shell; until then this (and `getAngleVisibilityMaskWide`/`intersectSlopes`/`SlopeCone`) is dead
  // weight kept only so the sub-flag plumbing compiles — slated for deletion or a real cone.
  let m = ALL_DIRECTIONS_VIS;
  if (dx > 0) m &= ~(1 << 4); // far east of camera ⇒ cannot usefully forward WEST (toward camera)
  if (dx < 0) m &= ~(1 << 5);
  if (dy > 0) m &= ~(1 << 0);
  if (dy < 0) m &= ~(1 << 1);
  if (dz > 0) m &= ~(1 << 2);
  if (dz < 0) m &= ~(1 << 3);
  return m;
}

/** The WIDE (conservative, outer-bound) cone mask — ALWAYS all six directions, so ANDing it is a no-op. Used
 *  for grazing/near-origin nodes where the local cone would be too aggressive: widen-only fallback so the
 *  angle pass can never narrow past the true cone (TRAP OCC-2.B). `getAngleVisibilityMaskWide` ⊇ `…Local`. */
export function getAngleVisibilityMaskWide(): number {
  return ALL_DIRECTIONS_VIS;
}

const ALL_DIRECTIONS_VIS = (1 << DIRECTION_COUNT) - 1;

/** Which of the 6 boundary faces does cell (x,y,z) lie on? Returns a 6-bit Direction mask. */
function faceMaskOf(x: number, y: number, z: number, size: number): number {
  let m = 0;
  if (x === 0) m |= 1 << Direction.West;
  if (x === size - 1) m |= 1 << Direction.East;
  if (y === 0) m |= 1 << Direction.Down;
  if (y === size - 1) m |= 1 << Direction.Up;
  if (z === 0) m |= 1 << Direction.North;
  if (z === size - 1) m |= 1 << Direction.South;
  return m;
}

/**
 * Flood-fill a section's opacity grid into face-to-face connectivity (`DirectionalVisGraph`).
 * `isOpaque(x,y,z)` ⇒ the cell blocks sight (a full opaque block). Fast exit:
 *   - 0 opaque cells (all air)        → VIS_ALL  (fully traversable)
 *   - every cell opaque (solid)       → VIS_NONE (blocks everything)
 *   - otherwise → connected-component flood over the non-opaque cells; every pair of faces touched by
 *     the same air component is marked connected.
 */
export function computeSectionVisibility(
  isOpaque: (x: number, y: number, z: number) => boolean,
  size: number = SECTION_SIZE,
): SectionVisibility {
  const n = size * size * size;
  const opaque = new Uint8Array(n);
  let opaqueCount = 0;
  const idx = (x: number, y: number, z: number) => (y * size + z) * size + x;
  for (let y = 0; y < size; y++) {
    for (let z = 0; z < size; z++) {
      for (let x = 0; x < size; x++) {
        if (isOpaque(x, y, z)) {
          opaque[idx(x, y, z)] = 1;
          opaqueCount++;
        }
      }
    }
  }
  if (opaqueCount === 0) return VIS_ALL;
  if (opaqueCount === n) return VIS_NONE;

  let vis = VIS_NONE;
  const visited = new Uint8Array(n);
  const stack: number[] = []; // packed cell indices (DFS over 6-connected air cells)
  for (let y0 = 0; y0 < size; y0++) {
    for (let z0 = 0; z0 < size; z0++) {
      for (let x0 = 0; x0 < size; x0++) {
        const start = idx(x0, y0, z0);
        if (opaque[start] || visited[start]) continue;
        // Flood this air component, accumulating the faces it touches.
        let faces = 0;
        visited[start] = 1;
        stack.push(start);
        while (stack.length > 0) {
          const c = stack.pop()!;
          const x = c % size;
          const z = ((c / size) | 0) % size;
          const y = (c / (size * size)) | 0;
          faces |= faceMaskOf(x, y, z, size);
          // 6-connected neighbors within the section.
          if (x > 0) pushIf(c - 1, opaque, visited, stack);
          if (x < size - 1) pushIf(c + 1, opaque, visited, stack);
          if (z > 0) pushIf(c - size, opaque, visited, stack);
          if (z < size - 1) pushIf(c + size, opaque, visited, stack);
          if (y > 0) pushIf(c - size * size, opaque, visited, stack);
          if (y < size - 1) pushIf(c + size * size, opaque, visited, stack);
        }
        // Every pair of touched faces is mutually visible through this component.
        if (faces !== 0) {
          for (let a = 0; a < DIRECTION_COUNT; a++) {
            if ((faces & (1 << a)) === 0) continue;
            for (let b = a + 1; b < DIRECTION_COUNT; b++) {
              if ((faces & (1 << b)) !== 0) vis |= pairBit(a, b);
            }
          }
        }
      }
    }
  }
  return vis;
}

function pushIf(c: number, opaque: Uint8Array, visited: Uint8Array, stack: number[]): void {
  if (opaque[c] || visited[c]) return;
  visited[c] = 1;
  stack.push(c);
}

// `opposite` + the per-direction block step (the occlusion BFS's `SECTION_STEP`) are the canonical
// `DIRECTION_OFFSET`/`opposite` from types — re-exported here under the occlusion-domain name so the BFS
// keeps reading `SECTION_STEP` (single source of truth; no manual-sync copy).
export { DIRECTION_OFFSET as SECTION_STEP, opposite } from "../types";
