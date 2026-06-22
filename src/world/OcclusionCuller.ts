// Camera occlusion BFS over the section visibility graph (`OcclusionCuller`).
// RENDERER_PLAN.md §13; RENDERER_OPTIMIZATION_PLAN.md Phase 3 OCC-1; "Visibility And
// Culling".
//
// Breadth-first search outward from the camera's home section. A section is REACHED only by stepping
// through faces that the visibility graph says are connected (given how we entered), so a fully
// enclosed section halts the search and the geometry behind it is never drawn. Two restrictions on each
// step (both required — connectivity alone under-culls; direction-culling alone is just a frustum shell):
//
//   1. CONNECTIVITY — entering a section through face `opposite(step)`, only exit through faces the
//      section's `SectionVisibility` connects to that incoming face (getConnections). The camera
//      section itself exits all 6 (you're inside it).
//   2. OUTWARD DIRECTION-CULLING — never step back toward the camera on an axis. A section at/east of the
//      camera may step further EAST but not WEST, etc. This keeps the frontier a growing shell and bounds
//      the search; without it the BFS would flood the whole in-frustum volume.
//
// OCC-6 (empty-but-traversable): a section with no opacity data (absent / never-meshed / all-air) is
// fully connected (VIS_ALL) so sight passes straight through it — empty space must not block the search.
// Only sections that actually have geometry are returned for drawing; empty ones are still traversed.
//
// BOUNDED DOMAIN (parity): search walks a FINITE set of loaded section objects linked by
// adjacency pointers — it physically cannot flood into empty coordinate space, and a per-node distance test
// clamps it to the render distance (OcclusionCuller.java: `adjacent == null` dead-ends + `testDistance`).
// We have no per-air-section objects, so an unbounded flood would leak through OCC-6 air across the whole
// frustum (catastrophic on a sparse/empty editor world: O(frustum volume) per frame). `bounds` is our
// analog of that finite domain — the populated-section AABB (+margin). A neighbor outside it is a dead-end,
// exactly like a null adjacency. Inside the box, OCC-6 still holds (air gaps remain traversable).
//
// Pure integer math, no GPU/streaming dependency, worker-safe.

import { instrument } from "../core/Instrument";
import { DIRECTION_COUNT, SECTION_SIZE, type Vec3, type Vec3i } from "../types";
import {
  getAngleVisibilityMaskLocal,
  getConnections,
  joinQuadrantOf,
  joinVisibilityData,
  opposite,
  SECTION_STEP,
  VIS_ALL,
  type SectionVisibility,
} from "./SectionVisibility";

const ALL_DIRECTIONS = (1 << DIRECTION_COUNT) - 1; // 0b111111

/** Connectivity + drawability of one section — the single per-node lookup (reads one RenderSection
 *  per dequeued node, not two map gets). A live `RenderSection` satisfies this structurally. */
export interface SectionInfo {
  /** Face-pair connectivity of the presented geometry (VIS_ALL for empty/air). */
  readonly visibility: SectionVisibility;
  /** OCC-2 (optional): the 4-element per-perspective DIRECTION_SETS, `[0]` == `visibility`. Present only on a
   *  `perPerspectiveVisibility` build; when the query's `cameraPos` is also supplied the BFS joins these by
   *  camera quadrant (a strict subset of `visibility`). Absent ⇒ the BFS uses the symmetric `visibility`. */
  readonly visibilitySets?: Uint32Array;
  /** True iff the section carries drawable geometry (only these are returned; all are traversed). */
  readonly hasGeometry: boolean;
}

/** Inclusive section-coordinate AABB bounding the traversal domain (finite loaded-section set). */
export interface SectionAABB {
  readonly lo: Vec3i;
  readonly hi: Vec3i;
}

export interface OcclusionQuery {
  /** BFS seed section (section coords). The camera's home section, or — when the camera is outside the
   *  populated region — its CLAMP into `bounds` (initOutsideWorld: seed at the boundary nearest the
   *  camera instead of flooding the camera→world gap). Always visited; outward-culling is relative to it. */
  origin: Vec3i;
  /** Is this section within the view frustum + render distance? The origin is visited regardless. */
  inFrustum: (sx: number, sy: number, sz: number) => boolean;
  /** Optional LOOSER frustum test for the `addNearbySections` overhang pass (`isBoxVisibleLooser`):
   *  a section whose 16³ box is just out of frustum but whose model overhangs into view is still added if it
   *  passes this. Absent ⇒ the pass falls back to `inFrustum` (on the cached/reachability path that is the
   *  always-true predicate, so every geometry-bearing camera neighbor is added to the reachable set and the
   *  downstream per-section draw filter applies the real frustum — exactly intent). */
  inFrustumLoose?: (sx: number, sy: number, sz: number) => boolean;
  /** One lookup per section: connectivity + drawability, or null if absent/empty (⇒ VIS_ALL + no geometry,
   *  i.e. traversable but not drawn — OCC-6). Replaces the old visibilityAt+hasGeometry double get. */
  sectionAt: (sx: number, sy: number, sz: number) => SectionInfo | null;
  /** Inclusive section-coord box bounding traversal (the populated region +margin). A neighbor outside it
   *  is a dead-end — this is what stops the OCC-6 flood from leaking into empty space. Omit ⇒ unbounded. */
  bounds?: SectionAABB | null;
  /** OCC-2 (default-off): the WORLD camera position in block coords. When supplied AND a visited node carries
   *  `visibilitySets`, the BFS joins those sets by the camera's per-section quadrant (`joinQuadrantOf`) into a
   *  single mask ⊆ `visibility` before `getConnections` — drawing strictly FEWER sections, never more (a
   *  conservative subset of the symmetric pass, never a hole). Absent ⇒ the exact symmetric BFS (off-path).
   *  Use the world position (sub-section accuracy), NOT just the floored section coord (TRAP OCC-2 quadrant). */
  cameraPos?: Vec3;
  /** OCC-2 second sub-flag (default-off): also AND the angle/slope cone (`getAngleVisibilityMaskLocal`) into
   *  the joined `outgoing`. Outer-bound only (widen-only) so it can drop only directions the camera cannot see
   *  down, never a grazing-but-visible one (TRAP OCC-2.B). Requires `cameraPos`. */
  angleMask?: boolean;
  /** Safety bound on sections visited (runaway guard). Default 200_000. */
  maxVisited?: number;
}

// Numeric section key (avoids per-step string garbage). Bias keeps negatives non-negative; the cube
// 32768³ stays well under Number.MAX_SAFE_INTEGER (≈9.0e15 vs 3.5e13). EXPORTED so the occlusion-visible
// Set + the per-draw filter (app/scene.ts) key off the SAME packed int — no `"sx,sy,sz"` strings (S1).
const BIAS = 1 << 14; // ±16384 sections ≈ ±262144 blocks of headroom
const SPAN = 1 << 15;
export function packSectionCoord(x: number, y: number, z: number): number {
  return ((x + BIAS) * SPAN + (y + BIAS)) * SPAN + (z + BIAS);
}
const keyOf = packSectionCoord;

/** Per-axis directions that lead AWAY from (or stay level with) the camera — getOutwardDirections. */
function outwardMask(sx: number, sy: number, sz: number, o: Vec3i): number {
  let m = 0;
  if (sx <= o[0]) m |= 1 << 4; // West  (-x) allowed while at/ west of camera
  if (sx >= o[0]) m |= 1 << 5; // East  (+x)
  if (sy <= o[1]) m |= 1 << 0; // Down  (-y)
  if (sy >= o[1]) m |= 1 << 1; // Up    (+y)
  if (sz <= o[2]) m |= 1 << 2; // North (-z)
  if (sz >= o[2]) m |= 1 << 3; // South (+z)
  return m;
}

// ── OCC-1 persistent BFS scratch (zero per-frame allocation) ────────────────────────────────────────
// The Map + the three queue arrays are reused across every `computeVisibleSections` run instead of being
// freshly allocated each call (the old per-frame Map + 3 arrays + O(visited) churn). This is OUTPUT-
// IDENTICAL — buffer reuse only — so it ships unconditionally under the existing goldens; only the new
// reachability/frustum split (caller-side) is flag-gated. SINGLE-FLIGHT: the BFS is render-thread-only and
// never reentrant (no async/await inside), so sharing module-level scratch is safe. Every run resets the
// scratch up front (`incoming.clear()` + `length=0`), so a prior run can never leave a node marked
// already-visited (which would silently under-traverse → holes). A dev assertion guards reentrancy.
const scratchIncoming = new Map<number, number>(); // key → accumulated incoming-direction mask (6-bit)
const scratchQueueX: number[] = [];
const scratchQueueY: number[] = [];
const scratchQueueZ: number[] = [];
let bfsInFlight = false;

/**
 * Visible sections (those WITH geometry that survive the BFS), as `[sx,sy,sz]` triples. Traversal passes
 * through empty/absent sections but they aren't returned.
 *
 * Writes into a caller-owned `out` array when provided (drained per run, so even the result array is reuse-
 * able — zero steady-state alloc); otherwise allocates a fresh array (legacy callers / tests). The Map +
 * queue scratch is always reused module-level (see above). Bumps `instrument.job.occBfsRuns` (one per run)
 * + `occVisited` (nodes walked) so the OCC-1 cache win is measurable.
 */
export function computeVisibleSections(q: OcclusionQuery, out?: Vec3i[]): Vec3i[] {
  if (bfsInFlight) throw new Error("computeVisibleSections: reentrant call — BFS scratch is single-flight (render-thread only)");
  bfsInFlight = true;
  instrument.bumpJob("occBfsRuns");
  const maxVisited = q.maxVisited ?? 200_000;
  const bounds = q.bounds ?? null;
  const result = out ?? [];
  result.length = 0;
  // visited: key → accumulated incoming-direction mask (6-bit). Presence ⇒ enqueued. Reset per run.
  const incoming = scratchIncoming;
  const queueX = scratchQueueX;
  const queueY = scratchQueueY;
  const queueZ = scratchQueueZ;
  incoming.clear();
  queueX.length = 0;
  queueY.length = 0;
  queueZ.length = 0;

  const [ox, oy, oz] = q.origin;
  // OCC-2 (default-off): the camera world position drives the per-perspective DIRECTION_SETS join + angle cone.
  // Absent ⇒ every node below takes the exact symmetric branch (off-path, byte-identical visible set).
  const camPos = q.cameraPos;
  const angleMask = q.angleMask === true;
  const camX = camPos ? camPos[0] : 0;
  const camZ = camPos ? camPos[2] : 0;

  incoming.set(keyOf(ox, oy, oz), ALL_DIRECTIONS); // sentinel; origin exits all 6 regardless
  queueX.push(ox);
  queueY.push(oy);
  queueZ.push(oz);

  let head = 0;
  let visitedCount = 0;
  while (head < queueX.length) {
    const sx = queueX[head], sy = queueY[head], sz = queueZ[head];
    head++;
    if (++visitedCount > maxVisited) break;

    const info = q.sectionAt(sx, sy, sz); // one lookup: connectivity + drawability (null ⇒ empty/absent)
    if (info && info.hasGeometry) result.push([sx, sy, sz]);

    // parity (OcclusionCuller.initWithinWorld + processQueue): the visibility data is joined the SAME
    // way at the origin and at every other node — the only difference is the origin has NO incoming-face
    // restriction (the camera is inside it), so it folds the union of ALL outgoing paths
    // (`VisibilityEncoding.getConnections(visibilityData)`, the single-arg/all-incoming form). Crucially the
    // origin is NOT blindly opened to all 6: a solid or half-open camera section exits only the faces its own
    // geometry actually connects — so an embedded/walled-in camera prunes the directions it cannot see through
    // (conservative, never a hole; the old unconditional `ALL_DIRECTIONS` over-drew that degenerate case).
    const isOrigin = sx === ox && sy === oy && sz === oz;
    // OCC-2: when the camera position is supplied AND this node carries the per-perspective sets, join them by
    // the camera's section quadrant into a single mask ⊆ the symmetric `visibility` (conservative subset, never
    // a hole). Otherwise fall back to the symmetric word exactly as OCC-1 (off-path / mixed build).
    let vis: SectionVisibility;
    if (camPos && info && info.visibilitySets) {
      const quadrant = joinQuadrantOf(camX, camZ, sx, sz, SECTION_SIZE);
      vis = joinVisibilityData(info.visibilitySets, quadrant);
    } else {
      vis = info ? info.visibility : VIS_ALL; // OCC-6: absent/empty ⇒ traversable
    }
    // Origin: all 6 faces are potential exits (camera inside) ⇒ `getConnections(vis, ALL)` = 
    // single-arg fold. Every other node: only the faces sight ENTERED through.
    const inMask = isOrigin ? ALL_DIRECTIONS : incoming.get(keyOf(sx, sy, sz))!;
    let outgoing = getConnections(vis, inMask);
    // OCC-2 angle sub-flag: AND the OUTER-bound cone (widen-only) so the BFS forwards sight only in directions
    // consistent with the camera angle. Conservative — drops only directions the camera cannot see down (TRAP
    // OCC-2.B). Skipped at the origin (applies the angle mask only in processQueue, never in
    // initWithinWorld — the cone is fully open at the camera).
    if (angleMask && camPos && !isOrigin) outgoing &= getAngleVisibilityMaskLocal(sx - ox, sy - oy, sz - oz);
    outgoing &= outwardMask(sx, sy, sz, q.origin);
    if (outgoing === 0) continue;

    for (let dir = 0; dir < DIRECTION_COUNT; dir++) {
      if ((outgoing & (1 << dir)) === 0) continue;
      const step = SECTION_STEP[dir];
      const nx = sx + step[0], ny = sy + step[1], nz = sz + step[2];
      // Bounds dead-end: a neighbor outside the populated-region box is not part of the finite traversal
      // domain (`adjacent == null`). This is what halts the OCC-6 flood through empty space.
      if (bounds && (nx < bounds.lo[0] || nx > bounds.hi[0] || ny < bounds.lo[1] || ny > bounds.hi[1] || nz < bounds.lo[2] || nz > bounds.hi[2])) continue;
      if (!q.inFrustum(nx, ny, nz)) continue;
      const nkey = keyOf(nx, ny, nz);
      const incBit = 1 << opposite(dir); // we entered the neighbor through the opposite face
      const seen = incoming.get(nkey);
      if (seen === undefined) {
        incoming.set(nkey, incBit);
        queueX.push(nx);
        queueY.push(ny);
        queueZ.push(nz);
      } else {
        // Already discovered from another inward side — accumulate so it can expand correctly when
        // dequeued (BFS + outward-culling guarantees inward neighbors are processed first).
        incoming.set(nkey, seen | incBit);
      }
    }
  }
  // parity (OcclusionCuller.addNearbySections): after the main BFS, UNCONDITIONALLY add the (up to) 26
  // sections immediately around the camera — including diagonals — that the outward traversal skipped but whose
  // geometry may overhang their 16³ box into view (large/overhanging block models or block entities right next
  // to the camera). No connectivity or outward-direction test (does `visitAll`, not a graph step): just
  // within-bounds, has-geometry, in (looser) frustum, and not already visited by the BFS. This can only ADD
  // near-camera draws (closes overhang holes), never drop one — matching deliberate over-draw tradeoff.
  const nearbyInFrustum = q.inFrustumLoose ?? q.inFrustum;
  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0 && dz === 0) continue;
        const nx = ox + dx, ny = oy + dy, nz = oz + dz;
        if (bounds && (nx < bounds.lo[0] || nx > bounds.hi[0] || ny < bounds.lo[1] || ny > bounds.hi[1] || nz < bounds.lo[2] || nz > bounds.hi[2])) continue;
        if (incoming.get(keyOf(nx, ny, nz)) !== undefined) continue; // already visited by the BFS (face neighbor / reachable diagonal)
        const ninfo = q.sectionAt(nx, ny, nz);
        if (!ninfo || !ninfo.hasGeometry) continue; // only drawable sections are returned (empty ones need no overhang draw)
        if (!nearbyInFrustum(nx, ny, nz)) continue;
        result.push([nx, ny, nz]);
      }
    }
  }
  instrument.bumpJob("occVisited", visitedCount);
  bfsInFlight = false;
  return result;
}
