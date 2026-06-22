import { describe, expect, it } from "vitest";
import { bakeModel, bakeModelInto, bakeModelPart, type BakeOptions, type ModelPartDef } from "./ModelPart";
import { mul, scaling, translation } from "./mat4";
import { TerrainPass, type SpriteUv } from "../../types";
import { PackedVertexSink, packVertices, type RawVertex } from "../VertexFormat";

const FULL_RECT: SpriteUv = { u: 0, v: 0, width: 1, height: 1 };
const OPTS = { texWidth: 64, texHeight: 64, uvRect: FULL_RECT } as const;

/** A unit cube (16px = 1 block) at the model origin, texOffs (0,0). */
const UNIT_CUBE: ModelPartDef = { cubes: [{ origin: [0, 0, 0], size: [16, 16, 16], uv: [0, 0] }] };

function faceVerts(v: RawVertex[], faceIndex: number): RawVertex[] {
  return v.slice(faceIndex * 4, faceIndex * 4 + 4);
}

const approx = (a: number, b: number) => expect(a).toBeCloseTo(b, 5);

describe("ModelPart bake", () => {
  it("emits 6 quads (24 verts) for one cube, all coords in [0,1]", () => {
    const v = bakeModelPart(UNIT_CUBE, OPTS);
    expect(v.length).toBe(24);
    for (const q of v) {
      for (const c of [q.x, q.y, q.z]) expect(c).toBeGreaterThanOrEqual(-1e-6);
      for (const c of [q.x, q.y, q.z]) expect(c).toBeLessThanOrEqual(1 + 1e-6);
    }
  });

  it("down face sits at y=0, up face at y=1", () => {
    const v = bakeModelPart(UNIT_CUBE, OPTS);
    // FACE_DEFS order: Down(0), Up(1), North(2), South(3), West(4), East(5).
    for (const q of faceVerts(v, 0)) approx(q.y, 0);
    for (const q of faceVerts(v, 1)) approx(q.y, 1);
  });

  it("box-unwraps the West face UVs into the standard MC cross layout", () => {
    const v = bakeModelPart(UNIT_CUBE, OPTS);
    // West (face index 4): texOffset (0,0), d=h=w=16, tex 64 → nl=0, nt=.25, nr=.25, nb=.5.
    // Non-up, non-mirror order: [[nr,nt],[nl,nt],[nl,nb],[nr,nb]].
    const west = faceVerts(v, 4).map((q) => [q.u, q.v]);
    expect(west).toEqual([
      [0.25, 0.25],
      [0, 0.25],
      [0, 0.5],
      [0.25, 0.5],
    ]);
  });

  it("remaps UVs into a sub-atlas rect", () => {
    const rect: SpriteUv = { u: 0.5, v: 0.25, width: 0.5, height: 0.5 };
    const v = bakeModelPart(UNIT_CUBE, { ...OPTS, uvRect: rect });
    const west = faceVerts(v, 4).map((q) => [q.u, q.v]);
    // Each [u,v] = rect.u + u*rect.width, rect.v + v*rect.height.
    expect(west[0][0]).toBeCloseTo(0.5 + 0.25 * 0.5, 5);
    expect(west[0][1]).toBeCloseTo(0.25 + 0.25 * 0.5, 5);
  });

  it("base transform translates the whole model", () => {
    const v = bakeModelPart(UNIT_CUBE, { ...OPTS, base: translation(10, 20, 30) });
    for (const q of faceVerts(v, 0)) approx(q.y, 20); // down face y was 0 → +20
  });

  it("part rotation rotates cube geometry about its pivot (180° about Z flips x,y)", () => {
    const part: ModelPartDef = { pivot: [8, 8, 8], rotation: [0, 0, Math.PI], children: [{ cubes: [{ origin: [-8, -8, -8], size: [16, 16, 16], uv: [0, 0] }] }] };
    const v = bakeModelPart(part, OPTS);
    // Cube spans [-0.5,0.5] about pivot (0.5,0.5,0.5); a 180° Z rotation maps it back onto itself,
    // so the AABB stays [0,1] — verifies the pose/pivot chain composes without drift.
    const xs = v.map((q) => q.x);
    approx(Math.min(...xs), 0);
    approx(Math.max(...xs), 1);
  });

  it("pose override replaces a named part's rotation", () => {
    const def: ModelPartDef = { name: "leg", pivot: [0, 16, 0], cubes: [{ origin: [-2, 0, -2], size: [4, 12, 4], uv: [0, 0] }] };
    const rest = bakeModelPart(def, OPTS);
    const posed = bakeModelPart(def, { ...OPTS, poses: { leg: { rotation: [Math.PI / 4, 0, 0] } } });
    // A non-zero pose changes geometry; the rest pose (no rotation) leaves it.
    let differs = false;
    for (let i = 0; i < rest.length; i++) {
      if (Math.abs(rest[i].z - posed[i].z) > 1e-4) differs = true;
    }
    expect(differs).toBe(true);
  });

  it("a flat (zero-thickness) plane emits ONE quad, not coincident faces (no z-fighting)", () => {
    // size y=0 → without the fix, Up+Down coincide (z-fight) + 4 zero-area sliver faces.
    const flat = bakeModelPart({ cubes: [{ origin: [0, 0, 0], size: [14, 0, 14], uv: [0, 0] }] }, OPTS);
    expect(flat.length).toBe(4); // exactly one quad
  });

  it("explicit `faces` restricts emission to the listed face", () => {
    const v = bakeModelPart({ cubes: [{ origin: [0, 0, 0], size: [14, 16, 0], uv: [0, 0], faces: [2 /* North */] }] }, OPTS);
    expect(v.length).toBe(4); // single North quad
  });

  it("mirror flips a side face's UV horizontally", () => {
    const plain = bakeModelPart(UNIT_CUBE, OPTS);
    const mir = bakeModelPart({ cubes: [{ origin: [0, 0, 0], size: [16, 16, 16], uv: [0, 0], mirror: true }] }, OPTS);
    const wp = faceVerts(plain, 4).map((q) => q.u);
    const wm = faceVerts(mir, 4).map((q) => q.u);
    expect(wm).not.toEqual(wp);
  });
});

// A4: baking STRAIGHT into a PackedVertexSink (the dynamic-mob path) must be byte-for-byte identical to
// packVertices(bakeModel(...)) — same vertex order/values; the sink's default light = full-bright = a
// RawVertex with no `light`. Exercises shade on/off, a non-Cutout pass + color, a flat (degenerate) face,
// mirror, inflate, nested children, a pose override, and a reused (reset) sink.
const SINK_ROOTS: ModelPartDef[] = [
  {
    name: "body",
    pivot: [1, 2, 3],
    rotation: [0.3, -0.5, 0.7],
    cubes: [
      { origin: [-4, -2, -3], size: [8, 4, 6], uv: [0, 0] },
      { origin: [0, 0, 0], size: [3, 0, 4], uv: [10, 5], mirror: true }, // flat (degenerate-Y) face
      { origin: [-1, -1, -1], size: [2, 2, 2], uv: [20, 10], inflate: 0.5 },
    ],
    children: [
      { name: "head", pivot: [0, 5, 0], rotation: [0, 1.1, 0], cubes: [{ origin: [-2, -2, -2], size: [4, 4, 4], uv: [30, 0] }] },
    ],
  },
];
const SINK_RECT: SpriteUv = { u: 0.1, v: 0.2, width: 0.5, height: 0.4 };
function sinkOpts(extra?: Partial<BakeOptions>): BakeOptions {
  return { texWidth: 64, texHeight: 32, uvRect: SINK_RECT, base: mul(scaling(-1, -1, 1), translation(0, -1.5, 0)), pass: TerrainPass.Cutout, shade: true, ...extra };
}

describe("bakeModelInto matches packVertices(bakeModel) byte-for-byte (A4 sink path)", () => {
  const cases: [string, BakeOptions][] = [
    ["shaded cutout", sinkOpts()],
    ["unshaded", sinkOpts({ shade: false })],
    ["solid pass + color", sinkOpts({ pass: TerrainPass.Solid, color: 0x8090a0ff })],
    ["posed", sinkOpts({ poses: { body: { rotation: [0.1, 0.2, 0.3], offsetPixels: [1, -2, 3] }, head: { rotation: [0.4, 0, 0] } } })],
  ];
  for (const [name, o] of cases) {
    it(name, () => {
      const refVerts = bakeModel(SINK_ROOTS, o);
      const ref = new Uint8Array(packVertices(refVerts));
      const sink = new PackedVertexSink();
      bakeModelInto(SINK_ROOTS, o, sink);
      expect(new Uint8Array(sink.finish())).toEqual(ref);
      expect(sink.quadCount).toBe(refVerts.length / 4);
    });
  }

  it("a reused sink (reset between bakes) produces the same bytes each time", () => {
    const o = sinkOpts();
    const ref = new Uint8Array(packVertices(bakeModel(SINK_ROOTS, o)));
    const sink = new PackedVertexSink();
    for (let i = 0; i < 3; i++) {
      sink.reset();
      bakeModelInto(SINK_ROOTS, o, sink);
      expect(new Uint8Array(sink.bytes())).toEqual(ref); // bytes() view is stable post-reset
    }
  });
});
