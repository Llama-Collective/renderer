// OCC-1 camera BFS occlusion culler (OcclusionCuller). RENDERER_OPTIMIZATION_PLAN Phase 3.

import { describe, it, expect } from "vitest";
import { type Vec3, type Vec3i } from "../types";
import { computeVisibleSections, type OcclusionQuery } from "./OcclusionCuller";
import {
  computeSectionVisibility,
  computeSectionVisibilitySets,
  VIS_ALL,
  VIS_NONE,
  type SectionVisibility,
} from "./SectionVisibility";

/** A section whose air is a single straight West–East tunnel (connects only those two faces). Built via
 *  the real flood so the bit layout is exactly what the BFS consumes. */
const WEST_EAST_TUNNEL: SectionVisibility = computeSectionVisibility((_x, y, z) => !(y === 8 && z === 8));

const key = (x: number, y: number, z: number) => `${x},${y},${z}`;
const has = (set: Vec3i[], x: number, y: number, z: number) =>
  set.some((c) => c[0] === x && c[1] === y && c[2] === z);

/** Build a query from explicit per-section maps; defaults: VIS_ALL connectivity, geometry everywhere. */
function query(opts: {
  origin?: Vec3i;
  box: { lo: Vec3i; hi: Vec3i }; // inclusive frustum box (section coords)
  vis?: Record<string, SectionVisibility>;
  noGeometry?: Set<string>;
}): OcclusionQuery {
  const { lo, hi } = opts.box;
  return {
    origin: opts.origin ?? [0, 0, 0],
    inFrustum: (x, y, z) => x >= lo[0] && x <= hi[0] && y >= lo[1] && y <= hi[1] && z >= lo[2] && z <= hi[2],
    // One lookup per section: connectivity (default VIS_ALL) + drawability (default true).
    sectionAt: (x, y, z) => ({
      visibility: opts.vis?.[key(x, y, z)] ?? VIS_ALL,
      hasGeometry: !(opts.noGeometry?.has(key(x, y, z)) ?? false),
    }),
  };
}

describe("computeVisibleSections — open space (no false culling)", () => {
  it("reveals every in-frustum section of a 3×3×3 open box", () => {
    const out = computeVisibleSections(query({ box: { lo: [-1, -1, -1], hi: [1, 1, 1] } }));
    expect(out.length).toBe(27);
    expect(has(out, 1, 1, 1)).toBe(true); // far corner reached by a monotonic-outward path
    expect(has(out, -1, -1, -1)).toBe(true);
    expect(has(out, 0, 0, 0)).toBe(true);
  });
});

describe("computeVisibleSections — occlusion actually culls", () => {
  it("a fully-opaque wall hides the section directly behind it", () => {
    // Camera at 0; solid section at (1,0,0); target at (2,0,0). Wide box so the only thing stopping the
    // BFS from reaching (2,0,0) is the wall + outward direction-culling (you can't loop back inward).
    const out = computeVisibleSections(
      query({ box: { lo: [0, -2, -2], hi: [3, 2, 2] }, vis: { [key(1, 0, 0)]: VIS_NONE } }),
    );
    expect(has(out, 0, 0, 0)).toBe(true); // camera section
    expect(has(out, 1, 0, 0)).toBe(true); // the wall itself is drawn
    expect(has(out, 2, 0, 0)).toBe(false); // …but what's directly behind it is culled
    expect(has(out, 3, 0, 0)).toBe(false);
  });

  it("a connected doorway in the wall lets sight (and the section behind) through", () => {
    // Same wall, but (1,0,0)'s visibility connects West↔East (a tunnel along X), so the BFS passes through.
    const out = computeVisibleSections(
      query({ box: { lo: [0, -2, -2], hi: [3, 2, 2] }, vis: { [key(1, 0, 0)]: WEST_EAST_TUNNEL } }),
    );
    expect(has(out, 1, 0, 0)).toBe(true);
    expect(has(out, 2, 0, 0)).toBe(true); // reachable through the doorway
  });
});

describe("computeVisibleSections — addNearbySections (overhang pass)", () => {
  it("draws a camera-adjacent diagonal section the outward BFS can't reach (overhang hole closed)", () => {
    // (1,1,0) is an EDGE neighbor of the camera; the only outward paths to it (E-then-N or N-then-E) pass
    // through (1,0,0) and (0,1,0), both made SOLID (VIS_NONE) so the BFS cannot reach it. 
    // addNearbySections still draws it (a model there could overhang into view right beside the camera).
    const out = computeVisibleSections(
      query({
        box: { lo: [-1, -1, -1], hi: [2, 2, 2] },
        vis: { [key(1, 0, 0)]: VIS_NONE, [key(0, 1, 0)]: VIS_NONE },
      }),
    );
    expect(has(out, 1, 1, 0)).toBe(true); // added by addNearbySections despite no graph path
    // The pass is the 26 NEIGHBORS only — a section directly behind a wall (distance 2) is still culled (the
    // existing "fully-opaque wall hides the section directly behind it" case proves (2,0,0) stays dropped).
  });
});

describe("computeVisibleSections — origin exits only its CONNECTED faces (initWithinWorld)", () => {
  it("a camera section that is a West–East tunnel does NOT forward sight upward", () => {
    // The camera's home section connects only West↔East. folds the origin's OWN visibility
    // (getConnections(visibilityData)), so it exits only W/E — NOT all 6. A section two up is therefore
    // unreachable (the old unconditional all-6 origin would have wrongly reached it through (0,1,0)).
    const out = computeVisibleSections(
      query({
        box: { lo: [-1, -1, -1], hi: [2, 2, 2] },
        vis: { [key(0, 0, 0)]: WEST_EAST_TUNNEL }, // the ORIGIN is the tunnel
      }),
    );
    expect(has(out, 2, 0, 0)).toBe(true); // east IS forwarded (the tunnel's open axis)
    expect(has(out, 0, 2, 0)).toBe(false); // up is NOT — and (0,2,0) is distance-2 so addNearbySections can't add it
  });
});

describe("computeVisibleSections — OCC-6 empty-but-traversable", () => {
  it("traverses an empty (no-geometry) section but does not draw it", () => {
    // (1,0,0) is empty: VIS_ALL connectivity (default null) but no geometry. (2,0,0) has geometry.
    const out = computeVisibleSections(
      query({
        box: { lo: [0, 0, 0], hi: [2, 0, 0] }, // a thin corridor along +x
        noGeometry: new Set([key(0, 0, 0), key(1, 0, 0)]),
      }),
    );
    expect(has(out, 1, 0, 0)).toBe(false); // traversed but empty ⇒ not drawn
    expect(has(out, 2, 0, 0)).toBe(true); // sight passed straight through the empty section
  });
});

describe("computeVisibleSections — bounds clamp the traversal domain (finite-set parity)", () => {
  it("halts the OCC-6 flood at the populated-region edge — geometry beyond `bounds` is never reached", () => {
    // The frustum is enormous and EVERY section is empty-but-traversable (OCC-6), with one geometry section
    // far out at (5,0,0). Without a bound the flood would walk all the way there (and across the whole
    // frustum — the empty-world rotation regression). `bounds` stops traversal at x=1, so it's unreachable.
    const out = computeVisibleSections({
      origin: [0, 0, 0],
      bounds: { lo: [-1, -1, -1], hi: [1, 1, 1] },
      inFrustum: (x, y, z) => x >= -50 && x <= 50 && y >= -50 && y <= 50 && z >= -50 && z <= 50,
      sectionAt: (x, y, z) => ({ visibility: VIS_ALL, hasGeometry: x === 5 && y === 0 && z === 0 }),
    });
    expect(has(out, 5, 0, 0)).toBe(false); // beyond the bound ⇒ not traversed, not drawn
    expect(out.length).toBe(0);
  });

  it("still reaches geometry that lies INSIDE the bound", () => {
    const out = computeVisibleSections({
      origin: [0, 0, 0],
      bounds: { lo: [-1, -1, -1], hi: [1, 1, 1] },
      inFrustum: () => true,
      sectionAt: (x, y, z) => ({ visibility: VIS_ALL, hasGeometry: x === 1 && y === 0 && z === 0 }),
    });
    expect(has(out, 1, 0, 0)).toBe(true);
  });

  it("a camera clamped onto the box boundary (viewer outside the region) still floods the whole box", () => {
    // Origin clamped to the box's west face (camera is far west, outside the populated region). The seed
    // exits all faces; outward culling then carries the search east across the box, reaching the far wall.
    const out = computeVisibleSections({
      origin: [-1, 0, 0], // = clamp(camera at x≪-1) onto bounds.lo[0]
      bounds: { lo: [-1, -1, -1], hi: [2, 1, 1] },
      inFrustum: () => true,
      sectionAt: () => ({ visibility: VIS_ALL, hasGeometry: true }),
    });
    expect(has(out, 2, 0, 0)).toBe(true); // far (east) wall reached from a west-boundary seed
    expect(has(out, -1, 0, 0)).toBe(true); // seed drawn
  });
});

// OCC-1: persistent BFS scratch (the Map + queue arrays are reused module-level instead of freshly allocated
// per call). This is OUTPUT-IDENTICAL — only buffer reuse — so it must never leak state between runs (a stale
// `incoming` Map would mark a node already-visited and silently under-traverse → holes). These cases prove it.
describe("computeVisibleSections — persistent scratch reuse (no cross-run contamination)", () => {
  const sorted = (out: Vec3i[]) => out.map((c) => key(c[0], c[1], c[2])).sort();

  it("two different queries against the shared scratch each match their from-scratch result", () => {
    // Query A: small 3×3×3 open box (origin 0). Query B: a different box, different origin, with a wall.
    const qA = query({ box: { lo: [-1, -1, -1], hi: [1, 1, 1] } });
    const qB = query({
      origin: [0, 0, 0],
      box: { lo: [0, -2, -2], hi: [3, 2, 2] },
      vis: { [key(1, 0, 0)]: VIS_NONE },
    });

    // Run interleaved so any leaked scratch from the previous run would corrupt the next.
    const a1 = sorted(computeVisibleSections(qA));
    const b1 = sorted(computeVisibleSections(qB));
    const a2 = sorted(computeVisibleSections(qA));
    const b2 = sorted(computeVisibleSections(qB));

    // Each query is internally consistent across repeated runs (no accumulation across the shared scratch).
    expect(a2).toEqual(a1);
    expect(b2).toEqual(b1);
    // …and matches the known goldens: A reveals all 27; B culls the section directly behind the wall.
    expect(a1.length).toBe(27);
    expect(b1).toContain(key(1, 0, 0)); // wall drawn
    expect(b1).not.toContain(key(2, 0, 0)); // behind-wall culled — no stale "visited" from query A let it through
  });

  it("a wall-culled run after an all-open run does NOT leak the open run's reachability (no holes-inverse)", () => {
    // A pathological order: first flood a wide-open box (marks many nodes visited in scratch), then run a
    // walled query whose correct answer culls (2,0,0). If scratch weren't cleared, the open run's visited
    // marks could either suppress enqueues (holes) or be invisible — assert the walled answer is exact.
    computeVisibleSections(query({ box: { lo: [-2, -2, -2], hi: [2, 2, 2] } })); // big open flood first
    const walled = computeVisibleSections(
      query({ box: { lo: [0, -2, -2], hi: [3, 2, 2] }, vis: { [key(1, 0, 0)]: VIS_NONE } }),
    );
    expect(has(walled, 1, 0, 0)).toBe(true);
    expect(has(walled, 2, 0, 0)).toBe(false);
  });

  it("writes into a caller-provided `out` array (drained, reused) producing the same result as a fresh array", () => {
    const q = query({ box: { lo: [-1, -1, -1], hi: [1, 1, 1] } });
    const reused: Vec3i[] = [[999, 999, 999]]; // pre-seeded junk triple; must be cleared on reuse
    const fromFresh = sorted(computeVisibleSections(q));
    const returned = computeVisibleSections(q, reused);
    expect(returned).toBe(reused); // same array object (zero-alloc drain)
    expect(sorted(returned)).toEqual(fromFresh); // identical contents, pre-seeded junk gone
  });
});

// OCC-1 reachability / frustum split: the cached BFS must produce an orientation-INDEPENDENT reachable set,
// and a per-frame frustum prune over it must yield the correct per-orientation visible subset. This proves
// the split is conservative — the reachable set is a superset of every orientation's visible set, so applying
// the live frustum downstream can never produce a hole (TRAP OCC-1.A).
describe("computeVisibleSections — reachability vs frustum split (conservative)", () => {
  const sortedKeys = (out: Vec3i[]) => out.map((c) => key(c[0], c[1], c[2])).sort();

  it("the same origin+graph yields the SAME reachable set under two different frustum orientations", () => {
    const box = { lo: [-2, -2, -2] as Vec3i, hi: [2, 2, 2] as Vec3i };
    // Reachability query: pass-through inFrustum (always true) → orientation-independent.
    const reach = sortedKeys(
      computeVisibleSections({
        origin: [0, 0, 0],
        bounds: box,
        inFrustum: () => true,
        sectionAt: () => ({ visibility: VIS_ALL, hasGeometry: true }),
      }),
    );
    // Run it again to confirm determinism (and scratch reuse) — same set.
    const reach2 = sortedKeys(
      computeVisibleSections({
        origin: [0, 0, 0],
        bounds: box,
        inFrustum: () => true,
        sectionAt: () => ({ visibility: VIS_ALL, hasGeometry: true }),
      }),
    );
    expect(reach2).toEqual(reach);

    // Two orientations = two frustum sub-boxes of the same domain. The per-orientation VISIBLE set is the
    // frustum-fused BFS; assert each is a SUBSET of the reachable set (so reapplying the frustum downstream
    // over the cached reachable set can only ever drop members — never add one → never a hole).
    const eastward = sortedKeys(
      computeVisibleSections({
        origin: [0, 0, 0],
        bounds: box,
        inFrustum: (x) => x >= 0, // looking east
        sectionAt: () => ({ visibility: VIS_ALL, hasGeometry: true }),
      }),
    );
    const upward = sortedKeys(
      computeVisibleSections({
        origin: [0, 0, 0],
        bounds: box,
        inFrustum: (_x, y) => y >= 0, // looking up
        sectionAt: () => ({ visibility: VIS_ALL, hasGeometry: true }),
      }),
    );
    const reachSet = new Set(reach);
    for (const k of eastward) expect(reachSet.has(k)).toBe(true);
    for (const k of upward) expect(reachSet.has(k)).toBe(true);
    // …and the two orientations differ (so the test is meaningful, not vacuously equal).
    expect(eastward).not.toEqual(upward);
  });

  it("frustum-pruning the cached reachable set equals the frustum-fused BFS (split is exact for these scenes)", () => {
    // For an open box where every section is reachable, a downstream frustum prune (filter reachable by the
    // same predicate) must reproduce the frustum-fused result exactly — confirming no over/under-cull.
    const box = { lo: [-2, 0, 0] as Vec3i, hi: [2, 0, 0] as Vec3i }; // 1-D corridor along x
    const sectionAt = () => ({ visibility: VIS_ALL, hasGeometry: true });
    const reachable = computeVisibleSections({ origin: [0, 0, 0], bounds: box, inFrustum: () => true, sectionAt });
    const frustum = (x: number) => x >= 0; // east half only
    const downstreamPruned = sortedKeys(reachable.filter((c) => frustum(c[0])));
    const fused = sortedKeys(
      computeVisibleSections({ origin: [0, 0, 0], bounds: box, inFrustum: (x) => frustum(x), sectionAt }),
    );
    expect(downstreamPruned).toEqual(fused);
  });
});

// ── OCC-2: per-perspective DIRECTION_SETS join in the BFS — CONSERVATIVE-ONLY (subset, never a hole) ──
describe("computeVisibleSections — per-perspective join is a SUBSET of the symmetric pass (OCC-2)", () => {
  const sortedKeys = (out: Vec3i[]) => out.map((c) => key(c[0], c[1], c[2])).sort();

  /** A query where every section carries BOTH the symmetric word and the 4-element DIRECTION_SETS, built from
   *  the SAME opacity predicate (so [0] === visibility). `cameraPos` + `angleMask` toggle the OCC-2 path. */
  function occ2Query(opts: {
    origin?: Vec3i;
    box: { lo: Vec3i; hi: Vec3i };
    opacityOf?: (sx: number, sy: number, sz: number) => ((x: number, y: number, z: number) => boolean) | null;
    cameraPos?: Vec3;
    angleMask?: boolean;
    noGeometry?: Set<string>;
  }): OcclusionQuery {
    const { lo, hi } = opts.box;
    return {
      origin: opts.origin ?? [0, 0, 0],
      cameraPos: opts.cameraPos,
      angleMask: opts.angleMask,
      inFrustum: (x, y, z) => x >= lo[0] && x <= hi[0] && y >= lo[1] && y <= hi[1] && z >= lo[2] && z <= hi[2],
      sectionAt: (x, y, z) => {
        const pred = opts.opacityOf?.(x, y, z) ?? null;
        const visibility = pred ? computeSectionVisibility(pred) : VIS_ALL;
        const visibilitySets = pred ? computeSectionVisibilitySets(pred) : undefined;
        return { visibility, visibilitySets, hasGeometry: !(opts.noGeometry?.has(key(x, y, z)) ?? false) };
      },
    };
  }

  it("open-box scene: per-perspective visible set ⊆ symmetric set (never adds a section)", () => {
    const box = { lo: [-2, -2, -2] as Vec3i, hi: [2, 2, 2] as Vec3i };
    // Every section is a diagonally-split solid-with-tunnels block, so the per-quadrant masks actually differ.
    const opacityOf = () => (x: number, y: number, z: number) => !((x === 3 && z === 3) || (y === 8 && x === 12));
    const symmetric = new Set(
      sortedKeys(computeVisibleSections(occ2Query({ box, opacityOf }))), // no cameraPos ⇒ symmetric path
    );
    for (const cam of [[-30, 0, -30], [30, 0, 30], [-30, 0, 30], [30, 0, -30]] as Vec3[]) {
      const perp = sortedKeys(computeVisibleSections(occ2Query({ box, opacityOf, cameraPos: cam })));
      for (const k of perp) expect(symmetric.has(k)).toBe(true); // subset — every member was also symmetric-visible
    }
  });

  it("open-box scene WITH the angle sub-flag still ⊆ symmetric (outer-bound cone never holes)", () => {
    const box = { lo: [-2, -2, -2] as Vec3i, hi: [2, 2, 2] as Vec3i };
    const opacityOf = () => (x: number, _y: number, z: number) => !(x === 8 && z === 8); // a straight tunnel each
    const symmetric = new Set(sortedKeys(computeVisibleSections(occ2Query({ box, opacityOf }))));
    const perp = sortedKeys(
      computeVisibleSections(occ2Query({ box, opacityOf, cameraPos: [30, 30, 30], angleMask: true })),
    );
    for (const k of perp) expect(symmetric.has(k)).toBe(true);
  });

  it("wall scene: still culls the section directly behind a solid wall, identically to the symmetric pass", () => {
    // Camera at 0; solid section at (1,0,0); target at (2,0,0). With OCC-2 on, behind-wall stays culled
    // (a solid section's sets are all VIS_NONE — the join cannot manufacture a connection through it).
    const opacityOf = (sx: number) => (sx === 1 ? () => true : () => false); // (1,0,0) solid, rest air
    const q = occ2Query({
      origin: [0, 0, 0],
      box: { lo: [0, -2, -2], hi: [3, 2, 2] },
      opacityOf: (sx) => opacityOf(sx),
      cameraPos: [8, 8, 8], // home section (0,0,0) world centre-ish
    });
    const out = computeVisibleSections(q);
    expect(has(out, 0, 0, 0)).toBe(true); // camera section drawn
    expect(has(out, 1, 0, 0)).toBe(true); // the wall itself drawn
    expect(has(out, 2, 0, 0)).toBe(false); // behind-wall culled — same as symmetric
    expect(has(out, 3, 0, 0)).toBe(false);
  });

  it("doorway scene: a West–East tunnel section still lets sight through under the per-perspective join", () => {
    // (1,0,0) is a straight West–East tunnel; the camera is west, so its quadrant join must keep W↔E.
    const opacityOf = (sx: number) => (sx === 1 ? (x: number, y: number, z: number) => !(y === 8 && z === 8) : () => false);
    const out = computeVisibleSections(
      occ2Query({
        origin: [0, 0, 0],
        box: { lo: [0, -2, -2], hi: [3, 2, 2] },
        opacityOf: (sx) => opacityOf(sx),
        cameraPos: [8, 8, 8],
      }),
    );
    expect(has(out, 1, 0, 0)).toBe(true);
    expect(has(out, 2, 0, 0)).toBe(true); // sight passes through the doorway — not over-culled
  });
});

