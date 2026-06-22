// Vertex-attribute setup helpers for the WebGL2 backend. WEBGL_PLAN.md (W2, TRAP W2.A);
// WEBGL_FINDINGS.md §5 (TRAP 10.B). Split out so the `asInt`/normalize rule — the one that silently
// corrupts data when wrong — is unit-testable without a GPU.
//
// The rule (total for the vertex formats this renderer uses):
//   asInt === true   → gl.vertexAttribIPointer (integer attribute, NO normalization) — uint*/int*.
//   asInt === false  → gl.vertexAttribPointer(..., normalized = isNormalizedKind(kind)) — floats,
//                      with unorm*/snorm* decoded to [0,1]/[-1,1] and float32 passed through.

import { VertexScalarKind } from "../gpu-enums";

/**
 * Whether a non-integer attribute of this scalar kind must be set up with `normalized = true` so the
 * stored integer decodes to a float (`unorm*`→[0,1], `snorm*`→[-1,1]). `Float32` passes through
 * (the flag is ignored for FLOAT); integer kinds go through `vertexAttribIPointer` instead, never here.
 */
export function isNormalizedKind(kind: VertexScalarKind): boolean {
  return (
    kind === VertexScalarKind.Unorm8 ||
    kind === VertexScalarKind.Unorm16 ||
    kind === VertexScalarKind.Snorm8 ||
    kind === VertexScalarKind.Snorm16
  );
}

/**
 * Build the neutral `VertexScalarKind` → WebGL component-type (GLenum) table from a live context
 * (STYLE_GUIDE §4: a declarative, exhaustive Record — a missing kind fails to compile). The `gl`
 * argument supplies the constants so the values stay readable and SSR-safe (no module-level globals).
 */
export function buildGlScalarTable(gl: WebGL2RenderingContext): Record<VertexScalarKind, number> {
  return {
    [VertexScalarKind.Uint8]: gl.UNSIGNED_BYTE,
    [VertexScalarKind.Uint16]: gl.UNSIGNED_SHORT,
    [VertexScalarKind.Uint32]: gl.UNSIGNED_INT,
    [VertexScalarKind.Float32]: gl.FLOAT,
    [VertexScalarKind.Snorm8]: gl.BYTE,
    [VertexScalarKind.Unorm8]: gl.UNSIGNED_BYTE,
    [VertexScalarKind.Snorm16]: gl.SHORT,
    [VertexScalarKind.Unorm16]: gl.UNSIGNED_SHORT,
  };
}
