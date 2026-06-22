// Shulker dirRotation must equal vanilla Direction.getRotation() for every facing. AUDIT H5.

import { describe, it, expect } from "vitest";
import { dirRotation } from "./shulker";
import { transformDir } from "../../../mesh/entity/mat4";

describe("shulker dirRotation = vanilla Direction.getRotation()", () => {
  // Vanilla getRotation() (rotationXYZ = Rx·Ry·Rz, applied M·v) rotates the model's local +Y onto the
  // facing's unit normal. The earlier Rz·Rx order sent +Y to the wrong axis for north/west/east (and
  // the test only covered facing=up, so it slipped). Round to clean up sin/cos(π/2) ε; +0 kills −0.
  const NORMAL: Record<string, [number, number, number]> = {
    up: [0, 1, 0],
    down: [0, -1, 0],
    north: [0, 0, -1],
    south: [0, 0, 1],
    west: [-1, 0, 0],
    east: [1, 0, 0],
  };

  it("maps local +Y onto the facing normal for all six facings", () => {
    for (const [facing, normal] of Object.entries(NORMAL)) {
      const got = transformDir(dirRotation(facing), 0, 1, 0).map((v) => Math.round(v) + 0);
      expect(got, facing).toEqual(normal);
    }
  });
});
