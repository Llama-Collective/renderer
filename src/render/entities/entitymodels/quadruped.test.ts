import { describe, expect, it } from "vitest";
import type { ModelPartDef } from "../../../mesh/entity/ModelPart";
import type { RenderEntity } from "../EntityScene";
import { quadrupedMesh, quadrupedWalk } from "./quadruped";

const HEAD: ModelPartDef = { name: "head", pivot: [0, 12, -6], cubes: [{ origin: [-4, -4, -8], size: [8, 8, 8], uv: [0, 0] }] };
const BODY: ModelPartDef = { name: "body", pivot: [0, 11, 2], cubes: [{ origin: [-5, -10, -7], size: [10, 16, 8], uv: [28, 8] }] };
const ent = (velocity: [number, number, number]): RenderEntity =>
  ({ id: "e", type: "minecraft:pig", position: [0, 0, 0], properties: {}, velocity }) as unknown as RenderEntity;

describe("QuadrupedModel family", () => {
  it("builds head + body + 4 named corner legs at the right offsets", () => {
    const parts = quadrupedMesh({ legSize: 6, head: HEAD, body: BODY });
    expect(parts.map((p) => p.name)).toEqual(["head", "body", "right_hind_leg", "left_hind_leg", "right_front_leg", "left_front_leg"]);
    const legY = 24 - 6; // vanilla pose Y
    expect(parts[2].pivot).toEqual([-3, legY, 7]); // right hind: (−legX, 24−legSize, zHind)
    expect(parts[5].pivot).toEqual([3, legY, -5]); // left front: (+legX, 24−legSize, zFront)
    expect(parts[2].cubes![0].size).toEqual([4, 6, 4]); // 4×legSize×4 leg box
  });

  it("honors legX / mirror overrides (cow)", () => {
    const parts = quadrupedMesh({ legSize: 12, legX: 4, mirrorLeftLeg: true, head: HEAD, body: BODY });
    expect(parts[2].pivot).toEqual([-4, 12, 7]); // legX = 4
    expect(parts[2].cubes![0].mirror).toBe(false); // right leg unmirrored
    expect(parts[3].cubes![0].mirror).toBe(true); // left leg mirrored
  });

  it("walks the legs in antiphase diagonal pairs when moving, idle when still", () => {
    expect(quadrupedWalk(ent([0, 0, 0]), 0)).toEqual({}); // no velocity → no pose
    const w = quadrupedWalk(ent([0.5, 0, 0]), 0); // clock 0 → cos(0)=1, cos(π)=−1
    expect(Object.keys(w).sort()).toEqual(["left_front_leg", "left_hind_leg", "right_front_leg", "right_hind_leg"]);
    // diagonal pairs share a phase: right_hind == left_front, left_hind == right_front, opposite sign.
    expect(w.right_hind_leg!.rotation![0]).toBeCloseTo(w.left_front_leg!.rotation![0], 6);
    expect(w.left_hind_leg!.rotation![0]).toBeCloseTo(w.right_front_leg!.rotation![0], 6);
    expect(w.right_hind_leg!.rotation![0]).toBeCloseTo(-w.left_hind_leg!.rotation![0], 6);
  });
});
