// Translucent re-sort TRIGGER + quad (de)serialization. RENDERER_OPTIMIZATION_PLAN.md Phase 5 (TR-1/5/6).
//
// Shared by the inline path (TerrainRenderer.resolveTranslucentIndex) AND the off-thread path (the Scheduler
// dispatches sort jobs; the worker computes them). The GFNI trigger (`crossedAnyPlane`) is — a
// DYNAMIC section only re-sorts when the camera crosses one of its quad PLANES (far fewer than its quads,
// TR-2). The quad (de)serialization moves a section's translucent geometry to a worker so the O(n²) topo
// sort runs off the render thread (the worker is stateless per job — Model A workers don't retain quads).

import { expandQuadOrder, sortIndicesByDistance, TFacing, type TQuad, type Vec3 } from "./TranslucentCollector";
import { bspSortOrder } from "./TranslucentBsp";
import { topoSortOrder, type TopoStatus } from "./TranslucentTopoSort";

/** True iff EVERY quad is aligned AND shares ONE facing axis (all ±X, or all ±Y, or all ±Z). This is the
 *  ONLY arrangement for which the BSP's centroid-side straddle resolution provably equals the topological
 *  (correct) back-to-front order: parallel planes never straddle one another, so the centroid approximation
 *  is exact and the painter's order is unique. ANY orthogonal/concave aligned mix — or any UNALIGNED quad —
 *  can carry a real mutual-visibility edge the BSP's centroid split orders wrongly (H4 — verified ~8% of
 *  random multi-facing aligned sets diverge from topo), so it must go through the topological sort. */
function singleAxisParallel(quads: readonly TQuad[]): boolean {
  let axis = -1;
  for (const q of quads) {
    if (q.facing === TFacing.Unaligned) return false;
    const a = q.facing % 3; // PosX/NegX→0, PosY/NegY→1, PosZ/NegZ→2 (TFacing PosX=0..NegZ=5)
    if (axis === -1) axis = a;
    else if (a !== axis) return false;
  }
  return true;
}

/** The DYNAMIC translucent index order for `camLocal` (INDICES ONLY, TRAP 12.A — never re-meshes vertices).
 *
 *  The BSP back-to-front sort (TR-3, O(n log n)) is a valid painter's order — and provably bit-identical to
 *  the topological (CORRECT) order — ONLY for PARALLEL panes (all quads on a single facing axis): there the
 *  centroid-side straddle resolution is exact. So BSP is reserved for `singleAxisParallel` sections (the
 *  common flat glass wall / water column); EVERY other section (orthogonal/concave aligned, or any unaligned
 *  quad) routes through the topological sort — true pairwise mutual-visibility — with a deterministic
 *  farthest-first distance sort as the fallback when the visibility graph cycles or exceeds the cap.
 *  (Previously every axis-aligned section took BSP, which silently diverged from topo for orthogonal/concave
 *  arrangements — H4. The single-axis case stays byte-identical to before; multi-axis now sorts correctly.)
 *
 *  H3: `bspSortOrder` recurses to depth O(n) on spatially-sorted parallel panes (a tall glass tower / water
 *  column), which overflows the stack at a few thousand quads. The BSP path is therefore CAPPED at
 *  `GATE_QUAD_THRESHOLD` — above it a parallel wall falls through to the distance sort (which is EXACT for
 *  parallel planes), bounding both the recursion depth (< GATE_QUAD_THRESHOLD) and the worst-case O(n²) work.
 *
 *  `budgetMs` caps the render-thread O(n²) DFS so a pathological section can't stall a frame; the count caps
 *  (not the timer) keep the common case deterministic — a section sitting exactly on the time boundary may
 *  briefly sort topo on one path and distance-fall-back on the other, self-healing on the next sort trigger.
 *
 *  PERSISTENT DISABLE (`DynamicTopoData`): pass a per-section `TopoSortState` and a section that keeps
 *  failing (cyclic graph) or times out stops re-attempting the expensive O(n²) topo every trigger — it
 *  permanently distance-sorts (the exact valid fallback) until a remesh resets the state. Omit `state` for the
 *  stateless behavior (the worker path + the static bake + tests): every call re-attempts topo as before. */
export function computeSortIndex(quads: TQuad[], camLocal: Vec3, budgetMs?: number, state?: TopoSortState): Uint32Array {
  // Single-axis PARALLEL panes: the BSP centroid order is provably == topo and can never cycle/fail, so it
  // bypasses the topo machinery (and its failure tracking) entirely. Capped at GATE_QUAD_THRESHOLD (H3 stack).
  if (singleAxisParallel(quads) && quads.length < GATE_QUAD_THRESHOLD) {
    return expandQuadOrder(bspSortOrder(quads, camLocal));
  }
  // parity: a section whose topo was disabled (repeated failure / timeout / over-cap) distance-sorts
  // directly — no more wasted topo attempts on the render thread.
  if (state?.gfniDisabled) return sortIndicesByDistance(quads, camLocal);
  const status: TopoStatus = { reason: "ok" };
  const t0 = state ? performance.now() : 0;
  const order = topoSortOrder(quads, camLocal, false, budgetMs, state ? status : undefined);
  if (state) {
    if (status.reason === "timeout" || status.reason === "overcap") {
      // Too slow even once, or too many quads (won't shrink without a remesh) ⇒ permanent distance fallback
      state.gfniDisabled = true;
    } else if (order) {
      state.failures = 0; // a success restores full patience (consecutiveTopoSortFailures = 0)
    } else {
      // A cycle (topo genuinely impossible from this angle). Give it a few tries — a later camera angle may
      // succeed — then give up. The patience scales with how fast the failing sort was (getAttemptsForTime).
      state.failures++;
      if (state.failures >= attemptsForTime(performance.now() - t0)) state.gfniDisabled = true;
    }
  }
  return order ? expandQuadOrder(order) : sortIndicesByDistance(quads, camLocal);
}

/** `DynamicTopoData` persistent topo-sort disable state, kept per section across re-sort triggers. A
 *  remesh mints a fresh state (the inline trigger map is keyed on the per-section `cpuVertex`, which changes
 *  on rebuild) — exactly "a new mesh re-enables GFNI triggering". */
export interface TopoSortState {
  /** Once true, topo is never re-attempted for this section — distance sort only (GFNITrigger=false). */
  gfniDisabled: boolean;
  /** Consecutive topo failures (cycles) since the last success (consecutiveTopoSortFailures). */
  failures: number;
}

/** A fresh (topo-enabled, no failures) state. */
export function newTopoSortState(): TopoSortState {
  return { gfniDisabled: false, failures: 0 };
}

// DynamicTopoData patience thresholds (ms ports of the ns constants): a FAST-but-cyclic sort gets more
// attempts (a different camera angle may yet topo-sort) before giving up than a SLOW one.
const TOPO_PATIENT_MS = 0.25; // MAX_TOPO_SORT_PATIENT_TIME_NS = 250_000ns
const TOPO_PATIENT_ATTEMPTS = 5; // PATIENT_TOPO_ATTEMPTS
const TOPO_REGULAR_ATTEMPTS = 2; // REGULAR_TOPO_ATTEMPTS
function attemptsForTime(ms: number): number {
  return ms <= TOPO_PATIENT_MS ? TOPO_PATIENT_ATTEMPTS : TOPO_REGULAR_ATTEMPTS;
}

/** Reduce a section's translucent quads to their UNIQUE (normal,dot) planes (TR-2), packed [nx,ny,nz,dot]
 *  per plane. Coplanar quads (a flat glass wall → 1 plane) collapse to one entry.
 *
 *  Dedup uses a NUMERIC key, not a `Set<string>` (no per-quad string garbage — closer to packed-int
 *  `NormI8` plane keys). The key bijectively packs the SAME `Math.round(·×256)` quantization the string key
 *  used (so the kept-plane set is bit-identical to the old version — `crossedAnyPlane` is unchanged): each
 *  normal component (a unit-vector axis ⇒ ×256 ∈ [-256,256], 513 values, offset to [0,512]) into a 513-radix
 *  int, then combined with the quantized plane offset (section-local ⇒ |dot·256| ≪ 2^20). The composite stays
 *  well under Number.MAX_SAFE_INTEGER (≈1.35e8 · 2^21 ≈ 2.8e14 < 9.0e15). */
const PLANE_NORM_RADIX = 513; // distinct ×256 normal-component values [-256,256] → [0,512]
const PLANE_DOT_SPAN = 1 << 21; // dot-offset multiplier; |round(dot·256)| < 2^20 (section-local) ⇒ injective
export function uniquePlanes(quads: readonly TQuad[]): Float32Array {
  const seen = new Set<number>();
  const out: number[] = [];
  for (const q of quads) {
    const n = q.normal;
    const nx = Math.round(n[0] * 256) + 256;
    const ny = Math.round(n[1] * 256) + 256;
    const nz = Math.round(n[2] * 256) + 256;
    const normKey = (nx * PLANE_NORM_RADIX + ny) * PLANE_NORM_RADIX + nz;
    const dq = Math.round(q.dot * 256) + (PLANE_DOT_SPAN >> 1); // shift negative offsets non-negative
    const key = normKey * PLANE_DOT_SPAN + dq;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(n[0], n[1], n[2], q.dot);
  }
  return new Float32Array(out);
}

/** Did the camera cross any of the section's unique planes between `a` and `b`? GFNI trigger.
 *  `planes` is the deduped [nx,ny,nz,dot]×k set — far fewer iterations than every quad. */
export function crossedAnyPlane(planes: Float32Array, a: Vec3, b: Vec3): boolean {
  for (let i = 0; i < planes.length; i += 4) {
    const nx = planes[i], ny = planes[i + 1], nz = planes[i + 2], dot = planes[i + 3];
    const sa = nx * a[0] + ny * a[1] + nz * a[2] - dot;
    const sb = nx * b[0] + ny * b[1] + nz * b[2] - dot;
    if (sa < 0 !== sb < 0) return true;
  }
  return false;
}

/** Reference distance (blocks, section-center → camera) at which `distEps` is the literal translation
 *  threshold. The translation gate scales INVERSELY with the section's distance (below), so this anchors the
 *  scale: at exactly `GATE_REF_DIST` the effective threshold equals `distEps`. ≈ one section span. */
export const GATE_REF_DIST = 16;

// ── TR-1 huge-section re-sort gate tuning (shared by the inline + off-thread trigger sites) ────────────────
/** Quad count at/above which a translucent section is "huge" and eligible for the distance/angle gate. Mirrors
 *  `MAX_TOPO_SORT_QUADS` (the existing huge-section cap). Below it the trigger is byte-for-byte today's
 *  `crossedAnyPlane`-only path even with the gate flag on (TRAP TR-1.B). */
export const GATE_QUAD_THRESHOLD = 1000;
/** World-space translation epsilon (at `GATE_REF_DIST`); the effective threshold scales ∝ 1/distance. */
export const GATE_DIST_EPS = 0.5;
/** Cosine of the view-angle epsilon (avoid `acos`): `dot(lastFwd,curFwd) < GATE_ANGLE_COS` ⇒ rotated past the
 *  bound. cos(2.6°) ≈ 0.99897 — a sub-block move at one-section distance AND a < ~2.6° rotation are suppressed. */
export const GATE_ANGLE_COS = 0.99897;
/** The section center in the section-LOCAL camera frame is always SECTION_CENTER (8) on every axis (camLocal =
 *  camPos − originBlocks; the center sits at +[8,8,8] from the origin) — a shared constant, no per-section alloc. */
export const GATE_CENTER_LOCAL: Vec3 = [8, 8, 8];

/** TR-1 distance/angle re-sort GATE for HUGE translucent sections. A SECOND damper layered on top of
 *  `crossedAnyPlane` (NEVER a replacement — TRAP TR-1.A): a barely-moved orbiting camera should not re-sort a
 *  far/huge glass/water volume whose painter's order barely changed. Returns true (re-sort ALLOWED) iff the
 *  camera moved past a distance threshold OR rotated past the angle bound — both referenced to the SECTION
 *  CENTER, so a FAR section clears the gate at a SMALLER world move than a near one for the same angular change.
 *
 *  Inputs are section-LOCAL (camera − section origin); `centerLocal` is the section center in that same local
 *  frame (SECTION_CENTER = [8,8,8]). `lastCam`/`curCam` are the last-sorted and current local camera positions;
 *  `lastFwd`/`curFwd` are the camera forward unit vectors at those two moments. `distEps` is a WORLD-space
 *  distance epsilon (the threshold AT `GATE_REF_DIST`); `angleCosEps` is a COSINE bound (avoid `acos` — TRAP:
 *  per-section per-frame trig over thousands of sections; `dot(lastFwd,curFwd) < angleCosEps` ⇒ "rotated past
 *  the bound"). Allocation-free (scalar reads only). Shared verbatim by the inline (TerrainRenderer) and
 *  off-thread (SchematicViewer) sites.
 *
 *  CENTER-RELATIVE distance: the same world move sweeps the section center through a LARGER angle when the
 *  section is near and a SMALLER angle when it is far — BUT a far/huge volume's painter's order is the more
 *  fragile (small parallax reorders deep stacks), so the GATE is biased the way DirectTriggers are:
 *  the effective translation threshold scales as `distEps · (GATE_REF_DIST / R)` where `R = |center − cam|`.
 *  A FAR section (large R) ⇒ a SMALLER effective threshold ⇒ clears at a smaller world move than a near one
 *  (which needs a larger move) for the same angular change. This is the monotonicity TR-1 requires. */
export function farAngleResortAllowed(
  centerLocal: Vec3,
  lastCam: Vec3,
  curCam: Vec3,
  lastFwd: Vec3,
  curFwd: Vec3,
  distEps: number,
  angleCosEps: number,
): boolean {
  // CENTER-RELATIVE translation gate. The effective threshold scales ∝ 1/R (R = section-center distance), so a
  // far section clears at a smaller move than a near one. Compare squared (no sqrt on the move) by scaling the
  // threshold: |Δcam|² > (distEps · GATE_REF_DIST / R)². R uses the last-sorted sightline (the deadband anchor).
  const dx = curCam[0] - lastCam[0];
  const dy = curCam[1] - lastCam[1];
  const dz = curCam[2] - lastCam[2];
  const move2 = dx * dx + dy * dy + dz * dz;
  const rx = centerLocal[0] - lastCam[0], ry = centerLocal[1] - lastCam[1], rz = centerLocal[2] - lastCam[2];
  const r = Math.sqrt(rx * rx + ry * ry + rz * rz);
  const effEps = distEps * (GATE_REF_DIST / (r > GATE_REF_DIST ? r : GATE_REF_DIST)); // clamp so very-near doesn't blow up
  if (move2 > effEps * effEps) return true;
  // Angular change of the view direction past the cosine bound — re-sort. `forward` rotation alone (camera
  // orbiting/panning in place) reorders a far volume even with negligible translation.
  const cos = lastFwd[0] * curFwd[0] + lastFwd[1] * curFwd[1] + lastFwd[2] * curFwd[2];
  if (cos < angleCosEps) return true;
  return false;
}

const QUAD_FLOATS = 26; // extents(6) + facing + dot + normal(3) + centroid(3) + positions(4×3=12)

/** Flatten a section's quads into one transferable ArrayBuffer (for an off-thread `sort` job). */
export function serializeQuads(quads: readonly TQuad[]): ArrayBuffer {
  const out = new Float32Array(1 + quads.length * QUAD_FLOATS);
  out[0] = quads.length;
  for (let i = 0; i < quads.length; i++) {
    const q = quads[i];
    let o = 1 + i * QUAD_FLOATS;
    out.set(q.extents.subarray(0, 6), o); o += 6;
    out[o++] = q.facing;
    out[o++] = q.dot;
    out[o++] = q.normal[0]; out[o++] = q.normal[1]; out[o++] = q.normal[2];
    out[o++] = q.centroid[0]; out[o++] = q.centroid[1]; out[o++] = q.centroid[2];
    for (let p = 0; p < 4; p++) { const v = q.positions[p]; out[o++] = v[0]; out[o++] = v[1]; out[o++] = v[2]; }
  }
  return out.buffer;
}

/** Reconstruct quads from `serializeQuads` (worker side) — round-trips every field the topo sort reads. */
export function deserializeQuads(buf: ArrayBuffer): TQuad[] {
  const a = new Float32Array(buf);
  const n = a[0];
  const quads: TQuad[] = [];
  for (let i = 0; i < n; i++) {
    let o = 1 + i * QUAD_FLOATS;
    const extents = a.slice(o, o + 6); o += 6;
    const facing = a[o++];
    const dot = a[o++];
    const normal: Vec3 = [a[o++], a[o++], a[o++]];
    const centroid: Vec3 = [a[o++], a[o++], a[o++]];
    const positions: [Vec3, Vec3, Vec3, Vec3] = [
      [a[o++], a[o++], a[o++]], [a[o++], a[o++], a[o++]], [a[o++], a[o++], a[o++]], [a[o++], a[o++], a[o++]],
    ];
    quads.push({ extents, facing, dot, normal, centroid, positions });
  }
  return quads;
}
