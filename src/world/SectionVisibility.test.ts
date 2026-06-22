// OCC-1 per-section directional visibility (DirectionalVisGraph). RENDERER_OPTIMIZATION_PLAN Phase 3.

import { describe, it, expect } from "vitest";
import { Direction } from "../types";
import {
  computeSectionVisibility,
  computeSectionVisibilitySets,
  getAngleVisibilityMaskLocal,
  getAngleVisibilityMaskWide,
  getConnections,
  intersectSlopes,
  isConnected,
  joinQuadrantOf,
  joinVisibilityData,
  opposite,
  SLOPE_CONE_OPEN,
  VIS_ALL,
  VIS_NONE,
  VIS_SETS_COUNT,
} from "./SectionVisibility";

const S = 16;

/** Build an isOpaque predicate from a fn over coords. */
const opaque = (f: (x: number, y: number, z: number) => boolean) => f;

describe("computeSectionVisibility — fast exits", () => {
  it("all-air section is fully connected (VIS_ALL)", () => {
    expect(computeSectionVisibility(opaque(() => false))).toBe(VIS_ALL);
  });

  it("fully solid section blocks everything (VIS_NONE)", () => {
    expect(computeSectionVisibility(opaque(() => true))).toBe(VIS_NONE);
  });
});

describe("computeSectionVisibility — flood connectivity", () => {
  it("a solid wall at z=8 disconnects North from South but keeps each side internally connected", () => {
    // Air everywhere except the full opaque plane z=8.
    const vis = computeSectionVisibility(opaque((_x, _y, z) => z === 8));
    expect(isConnected(vis, Direction.North, Direction.South)).toBe(false); // through the wall — blocked
    expect(isConnected(vis, Direction.West, Direction.East)).toBe(true); // within each half — open
    expect(isConnected(vis, Direction.Down, Direction.Up)).toBe(true);
    expect(isConnected(vis, Direction.North, Direction.Down)).toBe(true); // both touch the z<8 component
    expect(isConnected(vis, Direction.South, Direction.Up)).toBe(true); // both touch the z>8 component
  });

  it("a single straight tunnel connects only its two end faces", () => {
    // Solid everywhere except the column (x=8, z=8) running the full Y extent.
    const vis = computeSectionVisibility(opaque((x, _y, z) => !(x === 8 && z === 8)));
    expect(isConnected(vis, Direction.Down, Direction.Up)).toBe(true); // the tunnel's two ends
    expect(isConnected(vis, Direction.North, Direction.South)).toBe(false);
    expect(isConnected(vis, Direction.West, Direction.East)).toBe(false);
    expect(isConnected(vis, Direction.Down, Direction.North)).toBe(false);
  });

  it("two parallel disjoint tunnels do not cross-connect their faces", () => {
    // Tunnel A along Y at (4,*,4) connects Down–Up; Tunnel B along X at (*,12,12) connects West–East.
    const vis = computeSectionVisibility(
      opaque((x, y, z) => !((x === 4 && z === 4) || (y === 12 && z === 12))),
    );
    expect(isConnected(vis, Direction.Down, Direction.Up)).toBe(true);
    expect(isConnected(vis, Direction.West, Direction.East)).toBe(true);
    expect(isConnected(vis, Direction.Down, Direction.West)).toBe(false); // separate components
    expect(isConnected(vis, Direction.Up, Direction.East)).toBe(false);
  });
});

describe("getConnections / helpers", () => {
  it("getConnections(VIS_ALL, incoming) returns all faces except the incoming one", () => {
    const out = getConnections(VIS_ALL, 1 << Direction.North);
    expect(out & (1 << Direction.North)).toBe(0); // can't exit back the way you came (self-pair omitted)
    for (const d of [Direction.Down, Direction.Up, Direction.South, Direction.West, Direction.East]) {
      expect(out & (1 << d)).not.toBe(0);
    }
  });

  it("getConnections(VIS_NONE, …) is empty", () => {
    expect(getConnections(VIS_NONE, 0b111111)).toBe(0);
  });

  it("getConnections unions all incoming faces", () => {
    // Wall at z=8: entering from North reaches the z<8 faces; entering from South reaches z>8 faces.
    const vis = computeSectionVisibility(opaque((_x, _y, z) => z === 8));
    const fromNorth = getConnections(vis, 1 << Direction.North);
    const fromSouth = getConnections(vis, 1 << Direction.South);
    expect(fromNorth & (1 << Direction.South)).toBe(0); // North can't reach South
    const both = getConnections(vis, (1 << Direction.North) | (1 << Direction.South));
    expect(both).toBe(fromNorth | fromSouth);
  });

  it("opposite pairs each axis", () => {
    expect(opposite(Direction.Down)).toBe(Direction.Up);
    expect(opposite(Direction.North)).toBe(Direction.South);
    expect(opposite(Direction.East)).toBe(Direction.West);
    expect(opposite(opposite(Direction.West))).toBe(Direction.West);
  });
});

// ── OCC-2: per-perspective DIRECTION_SETS producer + camera-quadrant join + angle cone ──────────────
describe("computeSectionVisibilitySets — per-perspective DIRECTION_SETS (OCC-2)", () => {
  it("element [0] is the symmetric-compatible union (byte-identical to computeSectionVisibility)", () => {
    // A non-trivial section (wall at z=8) so the union is neither VIS_ALL nor VIS_NONE.
    const pred = opaque((_x, _y, z) => z === 8);
    const sets = computeSectionVisibilitySets(pred);
    expect(sets.length).toBe(VIS_SETS_COUNT);
    expect(sets[0]).toBe(computeSectionVisibility(pred)); // [0] === the off-path producer's word
  });

  it("fast exits: all-air ⇒ every slot VIS_ALL; all-solid ⇒ every slot VIS_NONE", () => {
    const air = computeSectionVisibilitySets(opaque(() => false));
    for (let q = 0; q < VIS_SETS_COUNT; q++) expect(air[q]).toBe(VIS_ALL);
    const solid = computeSectionVisibilitySets(opaque(() => true));
    for (let q = 0; q < VIS_SETS_COUNT; q++) expect(solid[q]).toBe(VIS_NONE);
  });

  it("OR of all 4 perspective masks equals the symmetric result (a refinement, not a contradiction)", () => {
    // Two disjoint tunnels in different quadrants so the per-quadrant sets actually differ.
    const pred = opaque((x, y, z) => !((x === 2 && z === 2) || (y === 8 && z === 12)));
    const sets = computeSectionVisibilitySets(pred);
    let union = 0;
    for (let q = 0; q < VIS_SETS_COUNT; q++) union |= sets[q];
    expect(union).toBe(computeSectionVisibility(pred));
    // …and [0] alone already holds the full union (the symmetric-compatible element).
    expect(sets[0]).toBe(union);
  });

  it("joinVisibilityData returns the SYMMETRIC UNION for EVERY camera quadrant (M1: no per-quadrant hole)", () => {
    // Two air tunnels, each confined to one horizontal quadrant, touching DIFFERENT boundary faces:
    //   East-North column (x=15,z=0) ⇒ touches East/North/Down/Up;  West-South column (x=0,z=15) ⇒ West/South/Down/Up.
    // The OLD per-quadrant refinement returned a single slot, which DROPPED the other quadrant's tunnel — a
    // verified HOLE (a camera could see through a tunnel its quadrant slot omitted). M1 neutralizes the
    // refinement to the symmetric union (slot [0]) — quadrant-INDEPENDENT — until a faithful OCTANT port
    // lands, so every camera quadrant keeps BOTH tunnels' connectivity (conservative, never a hole).
    const enTunnel = (x: number, z: number) => x === 15 && z === 0;
    const wsTunnel = (x: number, z: number) => x === 0 && z === 15;
    const pred = opaque((x, _y, z) => !(enTunnel(x, z) || wsTunnel(x, z)));
    const sets = computeSectionVisibilitySets(pred);

    // Every camera quadrant joins to the SAME union (slot [0]).
    for (const q of [0, 1, 2, 3]) expect(joinVisibilityData(sets, q)).toBe(sets[0]);
    // The union carries BOTH tunnels' connectivity — nothing is dropped for any camera position (no hole).
    const join = joinVisibilityData(sets, 2); // a camera quadrant whose OLD slot would have dropped the E-N tunnel
    expect(isConnected(join, Direction.Down, Direction.Up)).toBe(true);
    expect(isConnected(join, Direction.East, Direction.Up)).toBe(true); // E-N tunnel — KEPT (old code dropped it)
    expect(isConnected(join, Direction.West, Direction.Up)).toBe(true); // W-S tunnel — KEPT
    // The data path is intact: the per-quadrant slots still differ (for a future faithful octant join), and
    // each remains a subset of the union.
    expect(sets[1]).not.toBe(sets[2]);
    expect((sets[1] & ~sets[0])).toBe(0);
    expect((sets[2] & ~sets[0])).toBe(0);
  });

  it("every quadrant set is a SUBSET of the union [0] (conservative-only — a join never adds a connection)", () => {
    const pred = opaque((x, y, z) => !((x === 2 && z === 2) || (y === 8 && z === 12) || (x === 14 && z === 14)));
    const sets = computeSectionVisibilitySets(pred);
    for (let q = 0; q < VIS_SETS_COUNT; q++) {
      // q ⊆ [0]  ⟺  (q & ~[0]) === 0  (no bit in q that the union lacks).
      expect(sets[q] & ~sets[0]).toBe(0);
    }
  });
});

describe("joinVisibilityData — collapse to a plain 15-bit number (getConnections-consumable)", () => {
  it("returns a plain number the unchanged getConnections accepts", () => {
    const sets = computeSectionVisibilitySets(opaque((_x, _y, z) => z === 8));
    const joined = joinVisibilityData(sets, 1);
    expect(typeof joined).toBe("number");
    // getConnections must consume it exactly like the symmetric word (no throw, masked result).
    const out = getConnections(joined, 1 << Direction.North);
    expect(out & (1 << Direction.North)).toBe(0); // self-pair never re-exits
  });

  it("joining a VIS_NONE-everywhere set yields 0", () => {
    const solid = computeSectionVisibilitySets(opaque(() => true)); // every slot VIS_NONE
    for (let q = 0; q < VIS_SETS_COUNT; q++) expect(joinVisibilityData(solid, q)).toBe(0);
  });

  it("an undefined / legacy 1-element set falls back conservatively (never under-draws)", () => {
    expect(joinVisibilityData(undefined, 0)).toBe(VIS_ALL); // absent ⇒ fully traversable (no hole)
    const legacy = Uint32Array.of(0b101); // a mixed/legacy 1-element build
    expect(joinVisibilityData(legacy, 2)).toBe(0b101); // returns the symmetric union as-is
  });
});

describe("joinQuadrantOf — camera-quadrant key from the WORLD camera position", () => {
  it("derives the quadrant from the camera's side of the section centre (sub-section accurate)", () => {
    // Section (0,0,0) spans blocks [0,16); centre at 8. West-North camera (x<8, z<8) ⇒ quadrant 0.
    expect(joinQuadrantOf(2, 2, 0, 0)).toBe(0); // W-N
    expect(joinQuadrantOf(12, 2, 0, 0)).toBe(1); // E-N (east bit)
    expect(joinQuadrantOf(2, 12, 0, 0)).toBe(2); // W-S (south bit)
    expect(joinQuadrantOf(12, 12, 0, 0)).toBe(3); // E-S
    // A different section: the quadrant is relative to THAT section's centre, not the camera's own section.
    expect(joinQuadrantOf(2, 2, 5, 5)).toBe(0); // camera far west/north of section (5,*,5)
    expect(joinQuadrantOf(100, 100, 5, 5)).toBe(3); // camera far east/south of it
  });
});

describe("angle/slope cone helpers — OUTER-bound (widen-only), never narrows past the true cone (OCC-2.B)", () => {
  it("intersectSlopes widens: the result contains BOTH inputs on each side", () => {
    const a = { lo: -1, hi: 1 };
    const b = { lo: -3, hi: 0.5 };
    const r = intersectSlopes(a, b);
    expect(r.lo).toBeLessThanOrEqual(a.lo);
    expect(r.lo).toBeLessThanOrEqual(b.lo);
    expect(r.hi).toBeGreaterThanOrEqual(a.hi);
    expect(r.hi).toBeGreaterThanOrEqual(b.hi);
    expect(r).toEqual({ lo: -3, hi: 1 }); // widen, not narrow
  });

  it("intersecting with the open cone never narrows", () => {
    const c = { lo: -2, hi: 2 };
    expect(intersectSlopes(c, SLOPE_CONE_OPEN)).toEqual(SLOPE_CONE_OPEN); // stays fully open
  });

  it("getAngleVisibilityMaskWide is all six directions (a no-op AND) and ⊇ the local mask", () => {
    const wide = getAngleVisibilityMaskWide();
    expect(wide).toBe(0b111111);
    // The local mask is always a subset of the wide mask (conservative refinement, never wider).
    for (const [dx, dy, dz] of [[1, 0, 0], [-2, 3, -1], [0, 0, 5], [-4, -4, -4]] as const) {
      const local = getAngleVisibilityMaskLocal(dx, dy, dz);
      expect(local & ~wide).toBe(0); // local ⊆ wide
    }
  });

  it("the local mask only drops directions pointing back toward the camera (subset of the step direction)", () => {
    // Far east of camera (dx>0): WEST is dropped, EAST kept (you only forward away from the camera).
    const east = getAngleVisibilityMaskLocal(3, 0, 0);
    expect(east & (1 << 4)).toBe(0); // West dropped
    expect(east & (1 << 5)).not.toBe(0); // East kept
    // At the origin (no offset) the cone is fully open.
    expect(getAngleVisibilityMaskLocal(0, 0, 0)).toBe(0b111111);
  });
});
