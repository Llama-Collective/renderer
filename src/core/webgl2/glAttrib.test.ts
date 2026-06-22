// The asInt/normalize rule (TRAP 10.B / W2.A) is silent corruption when wrong, so pin it with a GPU-free
// unit test. The full attribute path (vertexAttribIPointer vs vertexAttribPointer on real GL) is proven
// by the webgl2 harness; this covers the pure decision the device makes per attribute.

import { describe, expect, it } from "vitest";
import { VertexScalarKind } from "../gpu-enums";
import { buildGlScalarTable, isNormalizedKind } from "./glAttrib";

describe("isNormalizedKind", () => {
  it("normalizes unorm*/snorm* (decoded floats), not uint*/float32", () => {
    expect(isNormalizedKind(VertexScalarKind.Unorm8)).toBe(true);
    expect(isNormalizedKind(VertexScalarKind.Unorm16)).toBe(true);
    expect(isNormalizedKind(VertexScalarKind.Snorm8)).toBe(true);
    expect(isNormalizedKind(VertexScalarKind.Snorm16)).toBe(true);
    expect(isNormalizedKind(VertexScalarKind.Uint8)).toBe(false);
    expect(isNormalizedKind(VertexScalarKind.Uint16)).toBe(false);
    expect(isNormalizedKind(VertexScalarKind.Uint32)).toBe(false);
    expect(isNormalizedKind(VertexScalarKind.Float32)).toBe(false);
  });
});

describe("buildGlScalarTable", () => {
  // Distinct sentinel constants so the mapping is verified by identity, not by real GL values.
  const gl = {
    UNSIGNED_BYTE: 0x1401,
    UNSIGNED_SHORT: 0x1403,
    UNSIGNED_INT: 0x1405,
    FLOAT: 0x1406,
    BYTE: 0x1400,
    SHORT: 0x1402,
  } as unknown as WebGL2RenderingContext;
  const t = buildGlScalarTable(gl);

  it("maps each scalar kind to its WebGL component type", () => {
    expect(t[VertexScalarKind.Uint8]).toBe(gl.UNSIGNED_BYTE);
    expect(t[VertexScalarKind.Uint16]).toBe(gl.UNSIGNED_SHORT);
    expect(t[VertexScalarKind.Uint32]).toBe(gl.UNSIGNED_INT);
    expect(t[VertexScalarKind.Float32]).toBe(gl.FLOAT);
    expect(t[VertexScalarKind.Snorm8]).toBe(gl.BYTE);
    expect(t[VertexScalarKind.Unorm8]).toBe(gl.UNSIGNED_BYTE);
    expect(t[VertexScalarKind.Snorm16]).toBe(gl.SHORT);
    expect(t[VertexScalarKind.Unorm16]).toBe(gl.UNSIGNED_SHORT);
  });

  it("covers every VertexScalarKind (exhaustive table)", () => {
    for (const k of Object.values(VertexScalarKind)) expect(typeof t[k]).toBe("number");
  });
});
