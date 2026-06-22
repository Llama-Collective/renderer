// Cross-backend front-face WINDING probe. WEBGL_PLAN.md W5.
//
// WebGPU (Vulkan/Metal-like) uses a +Y-DOWN framebuffer while WebGL2 (OpenGL) uses +Y-UP, with an
// identical Y-up projection feeding both — so it is NOT obvious whether matching WebGPU's frontFace:"ccw"
// back-face culling requires GL_CCW or GL_CW on WebGL2 (the classic winding gotcha). Rather than reason
// about spec internals, settle it EMPIRICALLY: render the SAME CCW-in-NDC quad with cull:Back through the
// neutral GraphicsDevice on each backend and read whether it survived (front) or was culled (back). Run on
// real WebGPU (smoke page, Metal) AND WebGL2 (webgl2 page, SwiftShader) and compare the two verdicts:
//   - SAME verdict  → the device's current gl.frontFace matches WebGPU (correct).
//   - DIFFERENT     → the device's gl.frontFace is inverted vs WebGPU (must flip CCW↔CW).
// Uses the cross-backend "overlay" color-vertex shader so the exact same pipeline runs on both.

import type { CanvasDevice } from "../src/core/createDevice";
import { BufferUsage, CullMode, IndexFormat, VertexScalarKind } from "../src/core/GraphicsDevice";
import { COLOR_VERTEX_WGSL, COLOR_VERTEX_STRIDE } from "../src/render/colorVertex";

/** Render a CCW-in-NDC quad with cull:Back; report whether the center survived (front-facing) or was culled. */
export async function runWindingProbe(device: CanvasDevice, width: number, height: number): Promise<{ visible: boolean; sum: number }> {
  const shader = device.createShaderModule(COLOR_VERTEX_WGSL, "overlay");
  const pipe = device.createPipeline({
    label: "winding-probe",
    shader,
    vertexEntry: "vs",
    fragmentEntry: "fs",
    vertexLayout: {
      strideBytes: COLOR_VERTEX_STRIDE,
      attributes: [
        { location: 0, kind: VertexScalarKind.Float32, components: 3, offsetBytes: 0, asInt: false },
        { location: 1, kind: VertexScalarKind.Unorm8, components: 4, offsetBytes: 12, asInt: false },
      ],
    },
    pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.Back },
    colorFormat: device.colorFormat,
    depthFormat: device.depthFormat,
  });

  const ident = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  const ubo = device.createBuffer({ sizeBytes: 64, usage: BufferUsage.Uniform, label: "wp-ubo" });
  device.writeBuffer(ubo, 0, ident);
  const bind = device.createBindings({ pipeline: pipe, group: 0, entries: [{ binding: 0, resource: { buffer: ubo, offset: 0, size: 64 } }] });

  // Corners BL,BR,TR,TL with indices [0,1,2,0,2,3] → triangle (-,-)→(+,-)→(+,+): signed area > 0 = CCW-in-NDC.
  const corners: [number, number][] = [[-0.6, -0.6], [0.6, -0.6], [0.6, 0.6], [-0.6, 0.6]];
  const vtx = new ArrayBuffer(corners.length * COLOR_VERTEX_STRIDE);
  const dv = new DataView(vtx);
  corners.forEach(([x, y], i) => {
    const o = i * COLOR_VERTEX_STRIDE;
    dv.setFloat32(o, x, true);
    dv.setFloat32(o + 4, y, true);
    dv.setFloat32(o + 8, 0, true);
    dv.setUint8(o + 12, 255); dv.setUint8(o + 13, 255); dv.setUint8(o + 14, 255); dv.setUint8(o + 15, 255);
  });
  const vb = device.createBuffer({ sizeBytes: vtx.byteLength, usage: BufferUsage.Vertex, label: "wp-vb" });
  device.writeBuffer(vb, 0, new Uint8Array(vtx));
  const ib = device.createBuffer({ sizeBytes: 24, usage: BufferUsage.Index, label: "wp-ib" });
  device.writeBuffer(ib, 0, new Uint32Array([0, 1, 2, 0, 2, 3]));

  const pass = device.beginPass({ id: null, width, height }, { color: [0, 0, 0, 1], depth: 1 });
  pass.setPipeline(pipe);
  pass.setBindings(bind);
  pass.setVertexBuffer(0, vb);
  pass.setIndexBuffer(ib, IndexFormat.Uint32);
  pass.drawIndexed(6, 0, 0);
  pass.end();

  const shot = await device.readCanvasPixels();
  const cx = width >> 1;
  const cy = height >> 1;
  const o = cy * shot.bytesPerRow + cx * 4;
  const sum = shot.data[o] + shot.data[o + 1] + shot.data[o + 2];
  return { visible: sum > 120, sum };
}
