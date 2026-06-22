// WebGPUDevice end-to-end smoke test. RENDERER_PLAN.md §18.5, GATE 0 (Phase 0 proof).
//
// This is the smallest program that exercises the WHOLE device path against a real GPU:
//   create() → createShaderModule(WGSL) → createPipeline(packed vertex layout + 3-pass state)
//   → createBuffer ×2 (vertex + index) → beginPass(clear) → setVertexBuffer/setIndexBuffer
//   → drawIndexed → end()(submit) → readCanvasPixels() → assert.
//
// It is SELF-VERIFYING: it reads the rendered pixels back and checks that the quad drew
// (center is non-black) over the clear (a corner stays black). That turns "open it and squint
// at a triangle" into an objective PASS/FAIL the page banner and the headless runner both read.
//
// vitest cannot run this (node has no GPU); it runs in a real browser via ../harness/serve.ts
// or the Playwright runner. Keep it backend-aware on purpose: it imports the concrete
// WebGPUDevice (not the neutral interface) because it tests THAT backend's wiring.

import { WebGPUDevice } from "../src/core/webgpu/WebGPUDevice";
import { BufferUsage, CullMode, IndexFormat, VertexScalarKind } from "../src/core/GraphicsDevice";
import { runWindingProbe } from "./windingProbe";

export interface SmokeResult {
  ok: boolean;
  backend: string;
  colorFormat: string;
  size: string;
  /** Human-readable explanation of the pass/fail decision. */
  detail: string;
}

// One shader, vertex + fragment entry points. The vertex carries a `unorm8x4` color — the exact
// auto-decoding format the terrain mesh uses — so this proves the packed-vertex mapping, not a
// toy float path. The color we output is treated as LINEAR and sRGB-encoded by the `-srgb`
// canvas view (the linear-color chain, WEBGPU_FINDINGS §17.2).
const WGSL = /* wgsl */ `
struct VsIn {
  @location(0) pos: vec2f,
  @location(1) color: vec4f,
};
struct VsOut {
  @builtin(position) clip: vec4f,
  @location(0) color: vec4f,
};

@vertex
fn vs(in: VsIn) -> VsOut {
  var out: VsOut;
  out.clip = vec4f(in.pos, 0.0, 1.0);
  out.color = in.color;
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  return in.color;
}
`;

const STRIDE = 12; // float32x2 pos (8) + unorm8x4 color (4)

/** Pack one quad (4 vertices, centered, leaving the canvas corners on the clear color). */
function buildQuad(r: number, g: number, b: number): ArrayBuffer {
  const buf = new ArrayBuffer(4 * STRIDE);
  const dv = new DataView(buf);
  // CCW or CW does not matter: the smoke pipeline disables culling (pass.cull === "none").
  const corners: Array<[number, number]> = [
    [-0.6, -0.6],
    [0.6, -0.6],
    [0.6, 0.6],
    [-0.6, 0.6],
  ];
  corners.forEach(([x, y], i) => {
    const o = i * STRIDE;
    dv.setFloat32(o + 0, x, true);
    dv.setFloat32(o + 4, y, true);
    dv.setUint8(o + 8, r);
    dv.setUint8(o + 9, g);
    dv.setUint8(o + 10, b);
    dv.setUint8(o + 11, 255);
  });
  return buf;
}

function pixelSum(data: Uint8Array, bytesPerRow: number, x: number, y: number): number {
  const o = y * bytesPerRow + x * 4;
  return data[o] + data[o + 1] + data[o + 2];
}

/** Run the smoke test against `canvas`. Resolves with a structured PASS/FAIL result. */
export async function runSmoke(canvas: HTMLCanvasElement): Promise<SmokeResult> {
  const device = await WebGPUDevice.create(canvas);
  try {
    const width = canvas.width;
    const height = canvas.height;

    const shader = device.createShaderModule(WGSL, "smoke");
    const pipeline = device.createPipeline({
      label: "smoke",
      shader,
      vertexEntry: "vs",
      fragmentEntry: "fs",
      vertexLayout: {
        strideBytes: STRIDE,
        attributes: [
          { location: 0, kind: VertexScalarKind.Float32, components: 2, offsetBytes: 0, asInt: false },
          { location: 1, kind: VertexScalarKind.Unorm8, components: 4, offsetBytes: 8, asInt: false },
        ],
      },
      // Solid-pass state (depth test + write, no blend); cull off so winding is irrelevant.
      pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None },
      colorFormat: device.colorFormat,
      depthFormat: device.depthFormat,
    });

    const QR = 255;
    const QG = 90;
    const QB = 30;
    const vbuf = device.createBuffer({ sizeBytes: 4 * STRIDE, usage: BufferUsage.Vertex, label: "smoke-vtx" });
    device.writeBuffer(vbuf, 0, new Uint8Array(buildQuad(QR, QG, QB)));

    const indices = new Uint16Array([0, 1, 2, 0, 2, 3]); // two triangles, 12 bytes (4-aligned)
    const ibuf = device.createBuffer({ sizeBytes: indices.byteLength, usage: BufferUsage.Index, label: "smoke-idx" });
    device.writeBuffer(ibuf, 0, indices);

    const pass = device.beginPass(
      { id: null, width, height },
      { color: [0, 0, 0, 1], depth: 1 },
    );
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, vbuf);
    pass.setIndexBuffer(ibuf, IndexFormat.Uint16);
    pass.drawIndexed(6, 0, 0);
    pass.end();

    const shot = await device.readCanvasPixels();
    const center = pixelSum(shot.data, shot.bytesPerRow, Math.floor(shot.width / 2), Math.floor(shot.height / 2));
    const corner = pixelSum(shot.data, shot.bytesPerRow, 2, 2); // top-left, outside the quad

    // Center must show the quad (clearly non-black); the corner must keep the clear color
    // (near-black). Order/encoding-agnostic, so it holds whether the swapchain is BGRA or RGBA.
    const drew = center > 120;
    const cleared = corner < 24;
    const ok = drew && cleared;

    device.destroyBuffer(vbuf);
    device.destroyBuffer(ibuf);

    // W5 winding probe (the WebGPU REFERENCE verdict): does a CCW-in-NDC quad survive cull:Back here?
    // Compare this against the webgl2 page's verdict to settle the gl.frontFace CCW-vs-CW question.
    const wind = await runWindingProbe(device, width, height);

    return {
      ok,
      backend: device.backend,
      colorFormat: device.colorFormat,
      size: `${shot.width}×${shot.height}`,
      detail:
        (ok
          ? `quad drew (centerSum=${center}) over clear (cornerSum=${corner})`
          : `unexpected pixels — centerSum=${center} (want >120, drew=${drew}), cornerSum=${corner} (want <24, cleared=${cleared})`) +
        ` | winding-cullback(CCW-in-NDC)=${wind.visible ? "VISIBLE(front)" : "CULLED(back)"} sum=${wind.sum}`,
    };
  } finally {
    device.destroy();
  }
}
