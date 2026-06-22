// WebGL2Device end-to-end smoke test. WEBGL_PLAN.md (REVIEW GATE W0.1 / W1.1).
//
// The WebGL2 fallback is built phase-by-phase; this proves the IMPLEMENTED surface (W0 clear + W1
// buffers/copy/textures/samplers) against a REAL WebGL2 context — the thing vitest can't do (node
// has no GL). It is SELF-VERIFYING via GPU readback:
//   * W0  clear the canvas, gl.readPixels the center, assert it ≈ the clear color.
//   * W1  buffer round-trip (write → getBufferSubData) for a vertex AND an index buffer; the index
//         buffer is additionally bound to ELEMENT_ARRAY_BUFFER to prove it was typed "element array"
//         and not poisoned "other data" (TRAP W1.A); GPU-to-GPU copyBuffer + copyBufferBatch
//         (arena-compaction analogues) checked byte-for-byte; a src!==dst guard; a texture full +
//         sub-rect upload read back through an FBO; createSampler; and a final no-GL-error backstop.
//
// Like ../harness/smoke.ts it is BACKEND-AWARE on purpose: it imports the concrete WebGL2Device
// (not the neutral interface) because it tests THAT backend's wiring, and to read resources back it
// unwraps the opaque handle to the GL object the device stored in it (handleOf returns the *Rec as
// the handle). If a Rec field is renamed this fails loudly — correct for a test.
//
//   * W2  draw a colored triangle (color-vertex program), prove #define injection links two constant
//         sets, and draw a line-list box — float attrs, topology, state.
//   * W3  draw a terrain quad through the REAL terrain twin: integer attrs (uint16x4 pos / uint8x4
//         light), the atlas + light-LUT sampler2D pairs, and divergence (A) — the per-section origin
//         UBO indexed by the per-draw u_BaseInstance (firstInstance). Verified two ways: the quad's
//         center samples the atlas color, AND drawing with the OTHER origin slot (whose origin pushes
//         the quad off-screen) leaves the center cleared — proving u_BaseInstance selects.
//   * W4  divergence (C), the linear-sRGB chain: every pass now renders into an offscreen SRGB8_ALPHA8
//         FBO presented via a re-encoding shader, so (a) the clear/terrain colors are sRGB-ENCODED
//         (the W0/W3 readbacks above now expect the bright presented bytes), and (b) `linear-blend`
//         proves a 50%-alpha "over" blends in LINEAR space (~188), not the gamma-space bug (~128).
//   * W5  the rest of the roster: `entity-draw`/`entity-dynoffset` (the entity twin + its 256-byte
//         dynamic-offset multi-mat4 UBO selecting per-draw state), `portal-draw` (the end-portal twin —
//         matrices, the per-layer loop, and one sampler shared across two texture units),
//         `readback-shape`/`readback-orient` (readCanvasPixels returns the right shape and a TOP-first,
//         Y-flipped image), and `winding-cullback` (a CCW-in-NDC quad survives cull:Back as FRONT —
//         front-face parity with WebGPU, cross-validated against the smoke page's real-Metal verdict).
//   * W6  `bundle-replay` (a recorded render-bundle replays pixel-identically to direct encode) +
//         `bundle-firstinstance` (firstInstance captured by value — the off-slot bundle culls off-screen)
//         + `multidraw` (multiDrawIndexed draws a batched run); the detail reports which feature-detected
//         fast-paths (clip_control / drawingBufferStorage / multi_draw) are active on this context.
//
// This file grows with the backend.

import { WebGL2Device } from "../src/core/webgl2/WebGL2Device";
import { AddressMode, type BindingLayoutEntry, BufferUsage, CompareFn, CullMode, FilterMode, type GpuBufferHandle, type GpuTextureHandle, IndexFormat, PrimitiveTopology, SamplerBindingType, ShaderStage, TextureFormat, TextureSampleType, VertexScalarKind } from "../src/core/GraphicsDevice";
import { terrainVertexLayout } from "../src/mesh/VertexFormat";
import { runWindingProbe } from "./windingProbe";

export interface WebGL2SmokeResult {
  ok: boolean;
  backend: string;
  colorFormat: string;
  size: string;
  /** Human-readable explanation of the pass/fail decision (lists the checks). */
  detail: string;
}

/** White-box views of the backend's opaque handles (the runtime *Rec shape — see file header). */
type BufferPeek = { buffer: WebGLBuffer };
type TexturePeek = { texture: WebGLTexture };

/** createShaderModule resolves the GLSL twin by LABEL and ignores the `code` arg — any string suffices. */
const TERRAIN_PLACEHOLDER = "/* terrain (resolved by label) */";
const ENTITY_PLACEHOLDER = "/* entity (resolved by label) */";
const PORTAL_PLACEHOLDER = "/* portal (resolved by label) */";
const OVERLAY_PLACEHOLDER = "/* overlay (resolved by label) */";

function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Read a device buffer back into `out` via the type-neutral COPY_READ_BUFFER (no residual-state reliance). */
function readBuffer(gl: WebGL2RenderingContext, handle: GpuBufferHandle, out: Uint8Array): void {
  gl.bindBuffer(gl.COPY_READ_BUFFER, (handle as unknown as BufferPeek).buffer);
  gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, out);
  gl.bindBuffer(gl.COPY_READ_BUFFER, null);
}

/** Read a device texture's level 0 back by attaching it to a throwaway FBO and reading pixels. */
function readTexture(gl: WebGL2RenderingContext, handle: GpuTextureHandle, w: number, h: number): Uint8Array {
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, (handle as unknown as TexturePeek).texture, 0);
  const out = new Uint8Array(w * h * 4);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE) {
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out);
  } else {
    out.fill(0xff); // not readable → force the texture check to fail loudly
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(fbo);
  return out;
}

/** Run the WebGL2 smoke against `canvas`. Resolves with a structured PASS/FAIL result. */
export async function runWebGL2Smoke(canvas: HTMLCanvasElement): Promise<WebGL2SmokeResult> {
  const device = WebGL2Device.create(canvas);
  const gl = device.raw;
  const passed: string[] = [];
  const failed: string[] = [];
  const check = (name: string, ok: boolean, info = ""): void => {
    (ok ? passed : failed).push(ok ? name : `${name}${info ? ` (${info})` : ""}`);
  };

  try {
    const width = canvas.width;
    const height = canvas.height;

    // --- W0/W4: clear, read the center back (now through the sRGB color chain) ----------------
    // W4 routes every pass through the offscreen SRGB8_ALPHA8 FBO + the present shader, so the clear color
    // (taken as LINEAR) is sRGB-ENCODED on store/present: clearColor(0.2,0.4,0.8) presents as ~(124,170,231),
    // NOT the raw-linear (51,102,204) the pre-W4 default-framebuffer path produced. That shift IS the chain.
    const CR = 0.2;
    const CG = 0.4;
    const CB = 0.8;
    const lin2srgb8 = (c: number): number => Math.round((c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055) * 255);
    device.beginPass({ id: null, width, height }, { color: [CR, CG, CB, 1], depth: 1 }).end();
    const px = new Uint8Array(4);
    gl.readPixels(width >> 1, height >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const near = (got: number, want: number): boolean => Math.abs(got - want) <= 3; // ±3 LSB rounding slack
    const clearOk = near(px[0], lin2srgb8(CR)) && near(px[1], lin2srgb8(CG)) && near(px[2], lin2srgb8(CB));
    check("clear", clearOk, `center=[${px[0]},${px[1]},${px[2]}] want≈[${lin2srgb8(CR)},${lin2srgb8(CG)},${lin2srgb8(CB)}] (sRGB-encoded)`);

    // --- W1: vertex buffer round-trip -------------------------------------------------------
    const vsrc = new Uint8Array(32);
    for (let i = 0; i < vsrc.length; i++) vsrc[i] = (i * 7 + 11) & 0xff;
    const vbuf = device.createBuffer({ sizeBytes: vsrc.byteLength, usage: BufferUsage.Vertex, label: "smoke-vtx" });
    device.writeBuffer(vbuf, 0, vsrc);
    const vout = new Uint8Array(vsrc.length);
    readBuffer(gl, vbuf, vout);
    check("vertex-buffer", bytesEqual(vout, vsrc));

    // --- W1: index buffer round-trip + element-array typing (TRAP W1.A) ----------------------
    const isrc = new Uint32Array([0, 1, 2, 0, 2, 3, 4, 5]);
    const ibuf = device.createBuffer({ sizeBytes: isrc.byteLength, usage: BufferUsage.Index, label: "smoke-idx" });
    device.writeBuffer(ibuf, 0, isrc);
    const iout = new Uint8Array(isrc.byteLength);
    readBuffer(gl, ibuf, iout);
    check("index-buffer", bytesEqual(iout, new Uint8Array(isrc.buffer.slice(0))));
    // If createBuffer had wrongly typed the index buffer "other data" (e.g. by allocating it through a
    // COPY target), binding it to ELEMENT_ARRAY_BUFFER is INVALID_OPERATION → caught here.
    gl.getError(); // isolate: drain any pending error before the typing probe
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, (ibuf as unknown as BufferPeek).buffer);
    check("index-typing", gl.getError() === gl.NO_ERROR, "index buffer not bindable as ELEMENT_ARRAY_BUFFER");
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    // --- W1: GPU-to-GPU copy (arena compaction analogues) -----------------------------------
    const dst = device.createBuffer({ sizeBytes: 32, usage: BufferUsage.Vertex, label: "smoke-copy" });
    device.copyBuffer(vbuf, { offsetBytes: 0, sizeBytes: 32 }, dst, 0);
    const cout = new Uint8Array(32);
    readBuffer(gl, dst, cout);
    check("copyBuffer", bytesEqual(cout, vsrc));

    // copyBufferBatch: relocate two live blocks into a fresh buffer (swap the two halves), like compaction.
    const dst2 = device.createBuffer({ sizeBytes: 32, usage: BufferUsage.Vertex, label: "smoke-batch" });
    device.copyBufferBatch(
      vbuf,
      [
        { srcOffset: 16, size: 16, dstOffset: 0 },
        { srcOffset: 0, size: 16, dstOffset: 16 },
      ],
      dst2,
    );
    const bout = new Uint8Array(32);
    readBuffer(gl, dst2, bout);
    const bexp = new Uint8Array(32);
    bexp.set(vsrc.subarray(16, 32), 0);
    bexp.set(vsrc.subarray(0, 16), 16);
    check("copyBufferBatch", bytesEqual(bout, bexp));

    // src !== dst guard (the WebGPU contract; the arena always compacts into a fresh buffer).
    let guarded = false;
    try {
      device.copyBuffer(vbuf, { offsetBytes: 0, sizeBytes: 4 }, vbuf, 8);
    } catch {
      guarded = true;
    }
    check("copy-guard", guarded, "copyBuffer(src===dst) should throw");

    // --- W1: texture full + sub-rect upload, read back via an FBO ----------------------------
    const TW = 4;
    const TH = 4;
    const full = new Uint8Array(TW * TH * 4);
    for (let i = 0; i < TW * TH; i++) {
      full[i * 4] = (i * 16) & 0xff;
      full[i * 4 + 1] = (i * 8) & 0xff;
      full[i * 4 + 2] = (i * 4) & 0xff;
      full[i * 4 + 3] = 255;
    }
    const tex = device.createTexture({ width: TW, height: TH, format: TextureFormat.Rgba8, label: "smoke-tex" });
    device.writeTexture(tex, full);
    // Overwrite a 2×2 sub-rect at (1,1) with a solid color (exercises the texSubImage2D region path +
    // UNPACK_ALIGNMENT=1 for the 2-texel = 8-byte rows).
    const sub = new Uint8Array(2 * 2 * 4);
    for (let i = 0; i < 4; i++) {
      sub[i * 4] = 200;
      sub[i * 4 + 1] = 30;
      sub[i * 4 + 2] = 90;
      sub[i * 4 + 3] = 255;
    }
    device.writeTexture(tex, sub, 0, { x: 1, y: 1, width: 2, height: 2 });
    const texOut = readTexture(gl, tex, TW, TH);
    const texExp = full.slice();
    for (let y = 1; y <= 2; y++) {
      for (let x = 1; x <= 2; x++) {
        const o = (y * TW + x) * 4;
        texExp[o] = 200;
        texExp[o + 1] = 30;
        texExp[o + 2] = 90;
        texExp[o + 3] = 255;
      }
    }
    check("texture", bytesEqual(texOut, texExp));

    // --- W1: sampler ------------------------------------------------------------------------
    let samplerOk = true;
    try {
      device.createSampler({ mag: FilterMode.Nearest, min: FilterMode.Nearest, address: AddressMode.Clamp });
    } catch {
      samplerOk = false;
    }
    check("sampler", samplerOk);

    // --- W2: shaders, pipelines, the VAO draw path ------------------------------------------
    const COLOR_STRIDE = 16; // pos vec3 (12) + unorm8x4 color (4) — render/colorVertex.ts
    const colorLayout = {
      strideBytes: COLOR_STRIDE,
      attributes: [
        { location: 0, kind: VertexScalarKind.Float32, components: 3, offsetBytes: 0, asInt: false },
        { location: 1, kind: VertexScalarKind.Unorm8, components: 4, offsetBytes: 12, asInt: false },
      ],
    };
    const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

    // (a) draw a colored triangle with the color-vertex program (resolved by the "overlay" label).
    const triShader = device.createShaderModule("/* color-vertex (resolved by label) */", "overlay");
    const triPipe = device.createPipeline({
      label: "smoke-tri",
      shader: triShader,
      vertexLayout: colorLayout,
      pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None },
      colorFormat: device.colorFormat,
      depthFormat: device.depthFormat,
    });
    const triUbo = device.createBuffer({ sizeBytes: 64, usage: BufferUsage.Uniform, label: "smoke-tri-ubo" });
    device.writeBuffer(triUbo, 0, IDENTITY);
    const triBind = device.createBindings({ pipeline: triPipe, group: 0, entries: [{ binding: 0, resource: { buffer: triUbo, offset: 0, size: 64 } }] });
    const tri = new ArrayBuffer(3 * COLOR_STRIDE);
    const tdv = new DataView(tri);
    const TR = 255;
    const TG = 90;
    const TB = 30;
    ([[-0.6, -0.6], [0.6, -0.6], [0.0, 0.6]] as [number, number][]).forEach(([x, y], i) => {
      const o = i * COLOR_STRIDE;
      tdv.setFloat32(o, x, true);
      tdv.setFloat32(o + 4, y, true);
      tdv.setFloat32(o + 8, 0, true);
      tdv.setUint8(o + 12, TR);
      tdv.setUint8(o + 13, TG);
      tdv.setUint8(o + 14, TB);
      tdv.setUint8(o + 15, 255);
    });
    const triVb = device.createBuffer({ sizeBytes: tri.byteLength, usage: BufferUsage.Vertex, label: "smoke-tri-vb" });
    device.writeBuffer(triVb, 0, new Uint8Array(tri));
    const triIb = device.createBuffer({ sizeBytes: 12, usage: BufferUsage.Index, label: "smoke-tri-ib" });
    device.writeBuffer(triIb, 0, new Uint32Array([0, 1, 2]));

    const triPass = device.beginPass({ id: null, width, height }, { color: [0, 0, 0, 1], depth: 1 });
    triPass.setPipeline(triPipe);
    triPass.setBindings(triBind);
    triPass.setVertexBuffer(0, triVb);
    triPass.setIndexBuffer(triIb, IndexFormat.Uint32);
    triPass.drawIndexed(3, 0, 0);
    triPass.end();
    const tc = new Uint8Array(4);
    gl.readPixels(width >> 1, height >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, tc); // center: inside the triangle
    const cn = new Uint8Array(4);
    gl.readPixels(2, 2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, cn); // corner: outside the triangle (cleared)
    check("draw-triangle", tc[0] + tc[1] + tc[2] > 120 && cn[0] + cn[1] + cn[2] < 24, `center=${tc[0] + tc[1] + tc[2]} corner=${cn[0] + cn[1] + cn[2]}`);

    // (b) #define injection — two pipelines from one shader with different override constants both link.
    let definePair = false;
    try {
      const ps = { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None } as const;
      const p0 = device.createPipeline({ label: "smoke-d0", shader: triShader, vertexLayout: colorLayout, pass: ps, colorFormat: device.colorFormat, depthFormat: device.depthFormat, constants: { ALPHA_TEST: 0 } });
      const p1 = device.createPipeline({ label: "smoke-d1", shader: triShader, vertexLayout: colorLayout, pass: ps, colorFormat: device.colorFormat, depthFormat: device.depthFormat, constants: { ALPHA_TEST: 1 } });
      definePair = (p0 as object) !== (p1 as object);
    } catch {
      definePair = false;
    }
    check("define-injection", definePair, "two constant sets → two linkable programs");

    // (c) line-list topology — the box shader links + draws a GPU line without error.
    let lineOk = true;
    try {
      const boxShader = device.createShaderModule("/* box (resolved by label) */", "linebox");
      const boxPipe = device.createPipeline({
        label: "smoke-box",
        shader: boxShader,
        vertexLayout: colorLayout,
        pass: { depthTest: false, depthWrite: false, blend: false, cull: CullMode.None, topology: PrimitiveTopology.LineList },
        colorFormat: device.colorFormat,
        depthFormat: device.depthFormat,
      });
      const boxU = new Float32Array(20); // std140: mat4 viewProj (identity) + time/resX/resY + pad = 80B
      boxU[0] = 1;
      boxU[5] = 1;
      boxU[10] = 1;
      boxU[15] = 1;
      const boxUbo = device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Uniform, label: "smoke-box-ubo" });
      device.writeBuffer(boxUbo, 0, boxU);
      const boxBind = device.createBindings({ pipeline: boxPipe, group: 0, entries: [{ binding: 0, resource: { buffer: boxUbo, offset: 0, size: 80 } }] });
      const ln = new ArrayBuffer(2 * COLOR_STRIDE);
      const ldv = new DataView(ln);
      ([[-0.5, 0.4], [0.5, 0.4]] as [number, number][]).forEach(([x, y], i) => {
        const o = i * COLOR_STRIDE;
        ldv.setFloat32(o, x, true);
        ldv.setFloat32(o + 4, y, true);
        ldv.setFloat32(o + 8, 0, true);
        ldv.setUint8(o + 12, 255);
        ldv.setUint8(o + 13, 255);
        ldv.setUint8(o + 14, 255);
        ldv.setUint8(o + 15, 255);
      });
      const lnVb = device.createBuffer({ sizeBytes: ln.byteLength, usage: BufferUsage.Vertex, label: "smoke-ln-vb" });
      device.writeBuffer(lnVb, 0, new Uint8Array(ln));
      const lnIb = device.createBuffer({ sizeBytes: 8, usage: BufferUsage.Index, label: "smoke-ln-ib" });
      device.writeBuffer(lnIb, 0, new Uint32Array([0, 1]));
      for (let k = 0; k < 8 && gl.getError() !== gl.NO_ERROR; k++) { /* drain before the isolated line draw */ }
      const lnPass = device.beginPass({ id: null, width, height }, {});
      lnPass.setPipeline(boxPipe);
      lnPass.setBindings(boxBind);
      lnPass.setVertexBuffer(0, lnVb);
      lnPass.setIndexBuffer(lnIb, IndexFormat.Uint32);
      lnPass.drawIndexed(2, 0, 0);
      lnPass.end();
      lineOk = gl.getError() === gl.NO_ERROR;
    } catch {
      lineOk = false;
    }
    check("draw-line", lineOk, "line-list box pipeline draws without error");

    // --- W3: terrain vertical slice (integer attrs + origin UBO + depth fixup) ----------------
    // Build a unit quad through the REAL terrain twin/layout. viewProj = identity (orthographic-ish), so a
    // section-local position decodes straight to clip space: a vertex at local (x,y,0) lands at clip (x,y).
    // The quad spans local [-0.5,0.5]² (centered) so it fills the viewport center and leaves the corners clear.
    const t3 = (() => {
      try {
        const terrainShader = device.createShaderModule(TERRAIN_PLACEHOLDER, "terrain");
        const terrainPipe = device.createPipeline({
          label: "smoke-terrain",
          shader: terrainShader,
          vertexLayout: terrainVertexLayout(),
          // Solid pass state: depth-test+write on, no blend, no cull (winding parity is a W7 cross-diff concern).
          pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None },
          colorFormat: device.colorFormat,
          depthFormat: device.depthFormat,
        });

        // Frame UBO: identity viewProj @0 (16 floats), camOrigin=(0,0,0,0) @64 → 80 bytes.
        const frame = new Float32Array(20);
        frame[0] = 1;
        frame[5] = 1;
        frame[10] = 1;
        frame[15] = 1;
        const frameUbo = device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Uniform, label: "smoke-frame" });
        device.writeBuffer(frameUbo, 0, frame);

        // Origins storage buffer, TWO slots: [0] pushes the quad far off-screen, [1] is the real (0,0,0) origin.
        // Drawing with firstInstance=1 must select origins[1] → on-screen; firstInstance=0 → off-screen.
        const origins = new Float32Array([10, 10, 10, 0, /* slot 1 */ 0, 0, 0, 0]);
        const originBuf = device.createBuffer({ sizeBytes: origins.byteLength, usage: BufferUsage.Storage, label: "smoke-origins" });
        device.writeBuffer(originBuf, 0, origins);

        // Atlas: 2×2 SRGB8_ALPHA8, every texel (200,100,50) — samples back as that color LINEARIZED.
        const atlas = device.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8Srgb, label: "smoke-atlas" });
        const atlasPx = new Uint8Array(2 * 2 * 4);
        for (let i = 0; i < 4; i++) {
          atlasPx[i * 4] = 200;
          atlasPx[i * 4 + 1] = 100;
          atlasPx[i * 4 + 2] = 50;
          atlasPx[i * 4 + 3] = 255;
        }
        device.writeTexture(atlas, atlasPx);
        const atlasSamp = device.createSampler({ mag: FilterMode.Nearest, min: FilterMode.Nearest });

        // Light LUT: 2×2 RGBA8 white → the linear light multiplier is 1 (full-bright), so it doesn't tint.
        const lut = device.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8, label: "smoke-lut" });
        device.writeTexture(lut, new Uint8Array(2 * 2 * 4).fill(255));
        const lutSamp = device.createSampler({ mag: FilterMode.Linear, min: FilterMode.Linear, address: AddressMode.Clamp });

        const terrainBind = device.createBindings({
          pipeline: terrainPipe,
          group: 0,
          entries: [
            { binding: 0, resource: { buffer: frameUbo, offset: 0, size: 80 } },
            { binding: 1, resource: { texture: atlas } },
            { binding: 2, resource: { sampler: atlasSamp } },
            { binding: 3, resource: { buffer: originBuf } },
            { binding: 4, resource: { texture: lut } },
            { binding: 5, resource: { sampler: lutSamp } },
          ],
        });

        // Pack the quad in the terrain vertex format (20B stride). packedPos.xyz = (local + 8) * 2048 (the
        // inverse of the shader's decode); w = normalIndex 1 (Up → SHADE 1.0, no directional darkening).
        const enc = (local: number): number => Math.round((local + 8) * 2048); // inverse of decode
        const Z = enc(0);
        const corners: [number, number][] = [
          [-0.5, -0.5],
          [0.5, -0.5],
          [0.5, 0.5],
          [-0.5, 0.5],
        ];
        const vtx = new ArrayBuffer(corners.length * 20);
        const dv = new DataView(vtx);
        corners.forEach(([lx, ly], i) => {
          const o = i * 20;
          dv.setUint16(o + 0, enc(lx), true); // packedPos.x
          dv.setUint16(o + 2, enc(ly), true); // packedPos.y
          dv.setUint16(o + 4, Z, true); // packedPos.z
          dv.setUint16(o + 6, 1, true); // packedPos.w = normalIndex 1 (Up)
          dv.setUint8(o + 8, 255); // color.r (white sRGB tint)
          dv.setUint8(o + 9, 255);
          dv.setUint8(o + 10, 255);
          dv.setUint8(o + 11, 255);
          dv.setUint16(o + 12, 32768, true); // uv.x ≈ 0.5 (unorm16)
          dv.setUint16(o + 14, 32768, true); // uv.y ≈ 0.5
          dv.setUint8(o + 16, 8); // light.x = blockCoord
          dv.setUint8(o + 17, 248); // light.y = skyCoord (sky 15 texel center)
          dv.setUint8(o + 18, 255); // light.z = ao 255 → multiplier 1.0
          dv.setUint8(o + 19, 0); // light.w = flags
        });
        const tVb = device.createBuffer({ sizeBytes: vtx.byteLength, usage: BufferUsage.Vertex, label: "smoke-terrain-vb" });
        device.writeBuffer(tVb, 0, new Uint8Array(vtx));
        const tIb = device.createBuffer({ sizeBytes: 24, usage: BufferUsage.Index, label: "smoke-terrain-ib" });
        device.writeBuffer(tIb, 0, new Uint32Array([0, 1, 2, 0, 2, 3]));

        const drawQuad = (firstInstance: number): Uint8Array => {
          const pass = device.beginPass({ id: null, width, height }, { color: [0, 0, 0, 1], depth: 1 });
          pass.setPipeline(terrainPipe);
          pass.setBindings(terrainBind);
          pass.setVertexBuffer(0, tVb);
          pass.setIndexBuffer(tIb, IndexFormat.Uint32);
          pass.drawIndexed(6, 0, 0, firstInstance);
          pass.end();
          const out = new Uint8Array(4);
          gl.readPixels(width >> 1, height >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
          return out;
        };

        // firstInstance=0 → origins[0]=(10,10,10) pushes the quad off the viewport → center stays cleared.
        const off = drawQuad(0);
        // firstInstance=1 → origins[1]=(0,0,0) → quad fills the center, sampling the (linearized) atlas color.
        const on = drawQuad(1);
        const corner = new Uint8Array(4);
        gl.readPixels(2, 2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, corner);

        // The atlas (200,100,50) keeps its R>G>B ordering through linearization; center is bright, corners clear.
        const onSum = on[0] + on[1] + on[2];
        const offSum = off[0] + off[1] + off[2];
        const cornerSum = corner[0] + corner[1] + corner[2];
        const drewColor = onSum > 60 && on[0] > on[1] && on[1] >= on[2] && on[0] > 80;
        check("terrain-draw", drewColor && cornerSum < 24, `center=[${on[0]},${on[1]},${on[2]}] corner=${cornerSum}`);
        // The origin-selector proof: SAME geometry, different firstInstance → on-screen vs off-screen.
        check("origin-select", onSum > 60 && offSum < 24, `slot0(off)=${offSum} slot1(on)=${onSum}`);

        return { ok: true };
      } catch (e) {
        check("terrain-draw", false, String(e));
        return { ok: false };
      }
    })();
    void t3;

    // --- W4: the linear-sRGB color chain — blending must run in LINEAR space ------------------
    // The whole reason for the offscreen SRGB8_ALPHA8 FBO: a translucent "over" must blend in linear space.
    // Draw a 50%-alpha WHITE quad over a BLACK clear. Linear "over" = 0.5·white + 0.5·black = 0.5 linear,
    // which sRGB-encodes to ~188. If blending ran in the WRONG (sRGB/gamma) space — i.e. straight to a plain
    // RGBA8 buffer — it would be 0.5·255 = ~128. So ~188 proves linear blend; ~128 would be the bug.
    const t4 = (() => {
      try {
        const terrainShader = device.createShaderModule(TERRAIN_PLACEHOLDER, "terrain");
        // Translucent terrain pass state: blend on, depth-test on, depth-WRITE OFF (TRAP 12.C), EMIT_ALPHA.
        const translucentPipe = device.createPipeline({
          label: "smoke-translucent",
          shader: terrainShader,
          vertexLayout: terrainVertexLayout(),
          pass: { depthTest: true, depthWrite: false, blend: true, cull: CullMode.None },
          colorFormat: device.colorFormat,
          depthFormat: device.depthFormat,
          constants: { EMIT_ALPHA: 1 },
        });

        const frame = new Float32Array(20);
        frame[0] = 1;
        frame[5] = 1;
        frame[10] = 1;
        frame[15] = 1;
        const frameUbo = device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Uniform, label: "smoke-t4-frame" });
        device.writeBuffer(frameUbo, 0, frame);
        const originBuf = device.createBuffer({ sizeBytes: 16, usage: BufferUsage.Storage, label: "smoke-t4-origins" });
        device.writeBuffer(originBuf, 0, new Float32Array([0, 0, 0, 0]));

        // White SRGB8 atlas (255,255,255) → decodes to linear 1.0; white light LUT → light multiplier 1.
        const atlas = device.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8Srgb, label: "smoke-t4-atlas" });
        device.writeTexture(atlas, new Uint8Array(2 * 2 * 4).fill(255));
        const atlasSamp = device.createSampler({ mag: FilterMode.Nearest, min: FilterMode.Nearest });
        const lut = device.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8, label: "smoke-t4-lut" });
        device.writeTexture(lut, new Uint8Array(2 * 2 * 4).fill(255));
        const lutSamp = device.createSampler({ mag: FilterMode.Linear, min: FilterMode.Linear, address: AddressMode.Clamp });

        const bind = device.createBindings({
          pipeline: translucentPipe,
          group: 0,
          entries: [
            { binding: 0, resource: { buffer: frameUbo, offset: 0, size: 80 } },
            { binding: 1, resource: { texture: atlas } },
            { binding: 2, resource: { sampler: atlasSamp } },
            { binding: 3, resource: { buffer: originBuf } },
            { binding: 4, resource: { texture: lut } },
            { binding: 5, resource: { sampler: lutSamp } },
          ],
        });

        // The quad: white tint (255,255,255) at HALF alpha (color.a = 128 → v_tintA ≈ 0.5 → EMIT_ALPHA emits
        // tex.a·0.5 = 0.5). normalIndex 1 (Up → SHADE 1.0). ao 255 → 1.0. Same center-covering geometry as W3.
        const enc = (local: number): number => Math.round((local + 8) * 2048);
        const Z = enc(0);
        const corners: [number, number][] = [
          [-0.5, -0.5],
          [0.5, -0.5],
          [0.5, 0.5],
          [-0.5, 0.5],
        ];
        const vtx = new ArrayBuffer(corners.length * 20);
        const dv = new DataView(vtx);
        corners.forEach(([lx, ly], i) => {
          const o = i * 20;
          dv.setUint16(o + 0, enc(lx), true);
          dv.setUint16(o + 2, enc(ly), true);
          dv.setUint16(o + 4, Z, true);
          dv.setUint16(o + 6, 1, true); // normalIndex 1 (Up)
          dv.setUint8(o + 8, 255); // white tint
          dv.setUint8(o + 9, 255);
          dv.setUint8(o + 10, 255);
          dv.setUint8(o + 11, 128); // tint ALPHA = 0.5 (the translucent factor)
          dv.setUint16(o + 12, 32768, true); // uv ≈ 0.5
          dv.setUint16(o + 14, 32768, true);
          dv.setUint8(o + 16, 8); // light blockCoord
          dv.setUint8(o + 17, 248); // light skyCoord
          dv.setUint8(o + 18, 255); // ao → 1.0
          dv.setUint8(o + 19, 0);
        });
        const vb = device.createBuffer({ sizeBytes: vtx.byteLength, usage: BufferUsage.Vertex, label: "smoke-t4-vb" });
        device.writeBuffer(vb, 0, new Uint8Array(vtx));
        const ib = device.createBuffer({ sizeBytes: 24, usage: BufferUsage.Index, label: "smoke-t4-ib" });
        device.writeBuffer(ib, 0, new Uint32Array([0, 1, 2, 0, 2, 3]));

        const pass = device.beginPass({ id: null, width, height }, { color: [0, 0, 0, 1], depth: 1 });
        pass.setPipeline(translucentPipe);
        pass.setBindings(bind);
        pass.setVertexBuffer(0, vb);
        pass.setIndexBuffer(ib, IndexFormat.Uint32);
        pass.drawIndexed(6, 0, 0, 0);
        pass.end();
        const lb = new Uint8Array(4);
        gl.readPixels(width >> 1, height >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, lb);
        // Linear blend ⇒ ~188 (sRGB of 0.5). The sRGB-space (wrong) blend would be ~128, far below 168.
        const linearBlend = lb[0] >= 168 && lb[0] <= 205 && lb[1] >= 168 && lb[2] >= 168;
        check("linear-blend", linearBlend, `center=[${lb[0]},${lb[1]},${lb[2]}] want≈188 (linear); sRGB-space bug would be ≈128`);
        return { ok: true };
      } catch (e) {
        check("linear-blend", false, String(e));
        return { ok: false };
      }
    })();
    void t4;

    // --- W5: entity twin — dynamic-offset multi-mat4 UBO + gl_FrontFacing lighting ------------
    // Draw an entity quad whose per-draw state (viewProj/model/tint/…) lives in a 256-byte dynamic-offset
    // UBO slot. Slot 0's model translates the quad off-screen; slot 1 is identity. Selecting the slot via the
    // setBindings dynamic offset (i·256) proves the dynamic-offset path; the tint (1,0.3,0.1) proves the UBO.
    const t5e = (() => {
      try {
        const entityShader = device.createShaderModule(ENTITY_PLACEHOLDER, "entity");
        const entityLayout: BindingLayoutEntry[] = [
          { binding: 0, visibility: [ShaderStage.Vertex, ShaderStage.Fragment], type: { kind: "uniform-buffer", dynamicOffset: true } },
          { binding: 1, visibility: [ShaderStage.Fragment], type: { kind: "texture", sampleType: TextureSampleType.Float } },
          { binding: 2, visibility: [ShaderStage.Fragment], type: { kind: "sampler", sampler: SamplerBindingType.Filtering } },
        ];
        const entityPipe = device.createPipeline({
          label: "smoke-entity",
          shader: entityShader,
          vertexLayout: terrainVertexLayout(), // entities reuse it; the program skips loc 3 (TRAP 5.1)
          pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None },
          colorFormat: device.colorFormat,
          depthFormat: device.depthFormat,
          bindingLayout: entityLayout,
          constants: { ALPHA_TEST: 0 },
        });
        const atlas = device.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8Srgb, label: "smoke-e-atlas" });
        device.writeTexture(atlas, new Uint8Array(2 * 2 * 4).fill(255));
        const samp = device.createSampler({ mag: FilterMode.Nearest, min: FilterMode.Nearest });

        // Two 256-byte slots (64 floats). Layout: viewProj@0, model@16, tint@32, flash@36, light0@40, light1@44.
        const SLOT_F = 64;
        const slots = new Float32Array(2 * SLOT_F);
        const setSlot = (s: number, modelTx: number): void => {
          const o = s * SLOT_F;
          slots[o + 0] = 1; slots[o + 5] = 1; slots[o + 10] = 1; slots[o + 15] = 1; // viewProj identity
          slots[o + 16] = 1; slots[o + 21] = 1; slots[o + 26] = 1; slots[o + 31] = 1; // model identity
          slots[o + 28] = modelTx; slots[o + 29] = modelTx; slots[o + 30] = modelTx; // model translation (col 3)
          slots[o + 32] = 1; slots[o + 33] = 0.3; slots[o + 34] = 0.1; slots[o + 35] = 1; // tint (linear) + alpha
          // flash.a (o+39)=0 → no flash; light0/light1 (o+40..47)=0 → axisShade, no per-face. Already zero.
        };
        setSlot(0, 10); // off-screen
        setSlot(1, 0); // on-screen
        const ubo = device.createBuffer({ sizeBytes: 2 * 256, usage: BufferUsage.Uniform, label: "smoke-e-ubo" });
        device.writeBuffer(ubo, 0, slots);
        const bind = device.createBindings({
          pipeline: entityPipe,
          group: 0,
          entries: [
            { binding: 0, resource: { buffer: ubo, offset: 0, size: 192 } },
            { binding: 1, resource: { texture: atlas } },
            { binding: 2, resource: { sampler: samp } },
          ],
        });
        const enc = (local: number): number => Math.round((local + 8) * 2048);
        const Z = enc(0);
        const corners: [number, number][] = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        const vtx = new ArrayBuffer(corners.length * 20);
        const dv = new DataView(vtx);
        corners.forEach(([lx, ly], i) => {
          const o = i * 20;
          dv.setUint16(o + 0, enc(lx), true);
          dv.setUint16(o + 2, enc(ly), true);
          dv.setUint16(o + 4, Z, true);
          dv.setUint16(o + 6, 1, true); // normalIndex 1 (Up → axisShade 1.0)
          dv.setUint8(o + 8, 255); dv.setUint8(o + 9, 255); dv.setUint8(o + 10, 255); dv.setUint8(o + 11, 255);
          dv.setUint16(o + 12, 32768, true); dv.setUint16(o + 14, 32768, true);
          // loc 3 (light) bytes 16..19 unused by the entity program (activeLocations skip).
        });
        const vb = device.createBuffer({ sizeBytes: vtx.byteLength, usage: BufferUsage.Vertex, label: "smoke-e-vb" });
        device.writeBuffer(vb, 0, new Uint8Array(vtx));
        const ib = device.createBuffer({ sizeBytes: 24, usage: BufferUsage.Index, label: "smoke-e-ib" });
        device.writeBuffer(ib, 0, new Uint32Array([0, 1, 2, 0, 2, 3]));
        const drawSlot = (slot: number): Uint8Array => {
          const pass = device.beginPass({ id: null, width, height }, { color: [0, 0, 0, 1], depth: 1 });
          pass.setPipeline(entityPipe);
          pass.setBindings(bind, [slot * 256]); // dynamic offset selects the slot
          pass.setVertexBuffer(0, vb);
          pass.setIndexBuffer(ib, IndexFormat.Uint32);
          pass.drawIndexed(6, 0, 0, 0);
          pass.end();
          const out = new Uint8Array(4);
          gl.readPixels(width >> 1, height >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
          return out;
        };
        const off = drawSlot(0);
        const on = drawSlot(1);
        // tint (1,0.3,0.1) linear → presented sRGB ≈ (255,149,89): R≫G>B proves the per-draw UBO applied.
        const tinted = on[0] > 200 && on[0] > on[1] && on[1] > on[2] && on[2] > 40;
        const offDark = off[0] + off[1] + off[2] < 24;
        check("entity-draw", tinted && offDark, `on=[${on[0]},${on[1]},${on[2]}] off=${off[0] + off[1] + off[2]}`);
        check("entity-dynoffset", on[0] > 200 && offDark, `slot0(off)=${off[0] + off[1] + off[2]} slot1(on)=${on[0]}`);
        return { ok: true };
      } catch (e) {
        check("entity-draw", false, String(e));
        return { ok: false };
      }
    })();
    void t5e;

    // --- W5: portal twin — dual-sampler binding + per-layer matrices + dynamic loop -----------
    // The portal's single sampler@1 feeds BOTH skyTex(unit 0) and portalTex(unit 1). With white textures the
    // 15-layer COLORS sum saturates green; if the shared sampler weren't bound to unit 1 (the createBindings
    // .filter fix), portalTex would sample an incomplete (black) texture and green would collapse to just the
    // sky term (~88). So center.g > 180 proves the shared-sampler binding AND that the matrices/loop/textureLod
    // all compile + run on real GL.
    const t5p = (() => {
      try {
        const portalShader = device.createShaderModule(PORTAL_PLACEHOLDER, "portal");
        const portalLayout = {
          strideBytes: 16,
          attributes: [
            { location: 0, kind: VertexScalarKind.Float32, components: 3, offsetBytes: 0, asInt: false },
            { location: 1, kind: VertexScalarKind.Float32, components: 1, offsetBytes: 12, asInt: false },
          ],
        };
        const portalPipe = device.createPipeline({
          label: "smoke-portal",
          shader: portalShader,
          vertexLayout: portalLayout,
          pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None },
          depthCompare: CompareFn.LessEqual,
          colorFormat: device.colorFormat,
          depthFormat: device.depthFormat,
        });
        const u = new Float32Array(20); // viewProj identity (16) + time + resX + resY + pad
        u[0] = 1; u[5] = 1; u[10] = 1; u[15] = 1;
        u[16] = 0; u[17] = width; u[18] = height;
        const uBuf = device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Uniform, label: "smoke-p-ubo" });
        device.writeBuffer(uBuf, 0, u);
        const sky = device.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8, label: "smoke-p-sky" });
        device.writeTexture(sky, new Uint8Array(2 * 2 * 4).fill(255));
        const portalTex = device.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8, label: "smoke-p-tex" });
        device.writeTexture(portalTex, new Uint8Array(2 * 2 * 4).fill(255));
        const portalSamp = device.createSampler({ mag: FilterMode.Linear, min: FilterMode.Linear, address: AddressMode.Repeat });
        const bind = device.createBindings({
          pipeline: portalPipe,
          group: 0,
          entries: [
            { binding: 0, resource: { buffer: uBuf, offset: 0, size: 80 } },
            { binding: 1, resource: { sampler: portalSamp } },
            { binding: 2, resource: { texture: sky } },
            { binding: 3, resource: { texture: portalTex } },
          ],
        });
        const corners: [number, number][] = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        const vtx = new ArrayBuffer(corners.length * 16);
        const dv = new DataView(vtx);
        corners.forEach(([x, y], i) => {
          const o = i * 16;
          dv.setFloat32(o + 0, x, true);
          dv.setFloat32(o + 4, y, true);
          dv.setFloat32(o + 8, 0, true);
          dv.setFloat32(o + 12, 15, true); // layers (end portal)
        });
        const vb = device.createBuffer({ sizeBytes: vtx.byteLength, usage: BufferUsage.Vertex, label: "smoke-p-vb" });
        device.writeBuffer(vb, 0, new Uint8Array(vtx));
        const ib = device.createBuffer({ sizeBytes: 24, usage: BufferUsage.Index, label: "smoke-p-ib" });
        device.writeBuffer(ib, 0, new Uint32Array([0, 1, 2, 0, 2, 3]));
        const pass = device.beginPass({ id: null, width, height }, { color: [0, 0, 0, 1], depth: 1 });
        pass.setPipeline(portalPipe);
        pass.setBindings(bind);
        pass.setVertexBuffer(0, vb);
        pass.setIndexBuffer(ib, IndexFormat.Uint32);
        pass.drawIndexed(6, 0, 0, 0);
        pass.end();
        const c = new Uint8Array(4);
        gl.readPixels(width >> 1, height >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, c);
        check("portal-draw", c[1] > 180 && gl.getError() === gl.NO_ERROR, `center=[${c[0]},${c[1]},${c[2]}]`);
        return { ok: true };
      } catch (e) {
        check("portal-draw", false, String(e));
        return { ok: false };
      }
    })();
    void t5p;

    // --- W5: readCanvasPixels — readback shape + Y-flip orientation ---------------------------
    // Render a quad in the TOP of NDC (y∈[0.1,0.9]); the readback must return row 0 = TOP (WebGPU contract),
    // so a top row is the (white) quad and a bottom row is cleared. gl.readPixels is bottom-up, so this only
    // holds if the Y-flip fired (without it, the quad would land in the BOTTOM rows — the opposite).
    const t5r = await (async () => {
      try {
        const terrainShader = device.createShaderModule(TERRAIN_PLACEHOLDER, "terrain");
        const pipe = device.createPipeline({ label: "smoke-rb", shader: terrainShader, vertexLayout: terrainVertexLayout(), pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None }, colorFormat: device.colorFormat, depthFormat: device.depthFormat });
        const frame = new Float32Array(20); frame[0] = 1; frame[5] = 1; frame[10] = 1; frame[15] = 1;
        const frameUbo = device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Uniform, label: "smoke-rb-frame" }); device.writeBuffer(frameUbo, 0, frame);
        const originBuf = device.createBuffer({ sizeBytes: 16, usage: BufferUsage.Storage, label: "smoke-rb-origins" }); device.writeBuffer(originBuf, 0, new Float32Array([0, 0, 0, 0]));
        const atlas = device.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8Srgb, label: "smoke-rb-atlas" }); device.writeTexture(atlas, new Uint8Array(2 * 2 * 4).fill(255));
        const atlasSamp = device.createSampler({ mag: FilterMode.Nearest, min: FilterMode.Nearest });
        const lut = device.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8, label: "smoke-rb-lut" }); device.writeTexture(lut, new Uint8Array(2 * 2 * 4).fill(255));
        const lutSamp = device.createSampler({ mag: FilterMode.Linear, min: FilterMode.Linear, address: AddressMode.Clamp });
        const bind = device.createBindings({ pipeline: pipe, group: 0, entries: [
          { binding: 0, resource: { buffer: frameUbo, offset: 0, size: 80 } },
          { binding: 1, resource: { texture: atlas } },
          { binding: 2, resource: { sampler: atlasSamp } },
          { binding: 3, resource: { buffer: originBuf } },
          { binding: 4, resource: { texture: lut } },
          { binding: 5, resource: { sampler: lutSamp } },
        ] });
        const enc = (local: number): number => Math.round((local + 8) * 2048);
        const Z = enc(0);
        const corners: [number, number][] = [[-0.9, 0.1], [0.9, 0.1], [0.9, 0.9], [-0.9, 0.9]]; // TOP band of NDC
        const vtx = new ArrayBuffer(corners.length * 20); const dv = new DataView(vtx);
        corners.forEach(([lx, ly], i) => {
          const o = i * 20;
          dv.setUint16(o + 0, enc(lx), true); dv.setUint16(o + 2, enc(ly), true); dv.setUint16(o + 4, Z, true); dv.setUint16(o + 6, 1, true);
          dv.setUint8(o + 8, 255); dv.setUint8(o + 9, 255); dv.setUint8(o + 10, 255); dv.setUint8(o + 11, 255);
          dv.setUint16(o + 12, 32768, true); dv.setUint16(o + 14, 32768, true);
          dv.setUint8(o + 16, 8); dv.setUint8(o + 17, 248); dv.setUint8(o + 18, 255); dv.setUint8(o + 19, 0);
        });
        const vb = device.createBuffer({ sizeBytes: vtx.byteLength, usage: BufferUsage.Vertex, label: "smoke-rb-vb" }); device.writeBuffer(vb, 0, new Uint8Array(vtx));
        const ib = device.createBuffer({ sizeBytes: 24, usage: BufferUsage.Index, label: "smoke-rb-ib" }); device.writeBuffer(ib, 0, new Uint32Array([0, 1, 2, 0, 2, 3]));
        const pass = device.beginPass({ id: null, width, height }, { color: [0, 0, 0, 1], depth: 1 });
        pass.setPipeline(pipe); pass.setBindings(bind); pass.setVertexBuffer(0, vb); pass.setIndexBuffer(ib, IndexFormat.Uint32); pass.drawIndexed(6, 0, 0, 0); pass.end();

        const shot = await device.readCanvasPixels();
        const shapeOk = shot.width === width && shot.height === height && shot.bytesPerRow === width * 4 && shot.data.length === width * 4 * height;
        const cx = width >> 1;
        const at = (x: number, y: number): number => { const o = y * shot.bytesPerRow + x * 4; return shot.data[o] + shot.data[o + 1] + shot.data[o + 2]; };
        const top = at(cx, (height / 6) | 0); // inside the top-band quad
        const bottom = at(cx, height - 1 - ((height / 6) | 0)); // below the quad → cleared
        check("readback-shape", shapeOk, `${shot.width}×${shot.height} bpr=${shot.bytesPerRow} len=${shot.data.length}`);
        check("readback-orient", top > 300 && bottom < 24, `top=${top} bottom=${bottom}`);
        return { ok: true };
      } catch (e) {
        check("readback-shape", false, String(e));
        return { ok: false };
      }
    })();
    void t5r;

    // --- W6: render-bundle replay — captured-and-replayed thunks are pixel-identical to direct encode ---
    const t6b = (() => {
      try {
        const terrainShader = device.createShaderModule(TERRAIN_PLACEHOLDER, "terrain");
        const pipe = device.createPipeline({ label: "smoke-bundle", shader: terrainShader, vertexLayout: terrainVertexLayout(), pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None }, colorFormat: device.colorFormat, depthFormat: device.depthFormat });
        const frame = new Float32Array(20); frame[0] = 1; frame[5] = 1; frame[10] = 1; frame[15] = 1;
        const frameUbo = device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Uniform, label: "b-frame" }); device.writeBuffer(frameUbo, 0, frame);
        const originBuf = device.createBuffer({ sizeBytes: 32, usage: BufferUsage.Storage, label: "b-origins" }); device.writeBuffer(originBuf, 0, new Float32Array([10, 10, 10, 0, 0, 0, 0, 0]));
        const atlas = device.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8Srgb, label: "b-atlas" });
        const apx = new Uint8Array(2 * 2 * 4); for (let i = 0; i < 4; i++) { apx[i * 4] = 200; apx[i * 4 + 1] = 100; apx[i * 4 + 2] = 50; apx[i * 4 + 3] = 255; } device.writeTexture(atlas, apx);
        const atlasSamp = device.createSampler({ mag: FilterMode.Nearest, min: FilterMode.Nearest });
        const lut = device.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8, label: "b-lut" }); device.writeTexture(lut, new Uint8Array(2 * 2 * 4).fill(255));
        const lutSamp = device.createSampler({ mag: FilterMode.Linear, min: FilterMode.Linear, address: AddressMode.Clamp });
        const bind = device.createBindings({ pipeline: pipe, group: 0, entries: [
          { binding: 0, resource: { buffer: frameUbo, offset: 0, size: 80 } },
          { binding: 1, resource: { texture: atlas } },
          { binding: 2, resource: { sampler: atlasSamp } },
          { binding: 3, resource: { buffer: originBuf } },
          { binding: 4, resource: { texture: lut } },
          { binding: 5, resource: { sampler: lutSamp } },
        ] });
        const enc = (local: number): number => Math.round((local + 8) * 2048); const Z = enc(0);
        const corners: [number, number][] = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        const vtx = new ArrayBuffer(corners.length * 20); const dv = new DataView(vtx);
        corners.forEach(([lx, ly], i) => { const o = i * 20; dv.setUint16(o, enc(lx), true); dv.setUint16(o + 2, enc(ly), true); dv.setUint16(o + 4, Z, true); dv.setUint16(o + 6, 1, true); dv.setUint8(o + 8, 255); dv.setUint8(o + 9, 255); dv.setUint8(o + 10, 255); dv.setUint8(o + 11, 255); dv.setUint16(o + 12, 32768, true); dv.setUint16(o + 14, 32768, true); dv.setUint8(o + 16, 8); dv.setUint8(o + 17, 248); dv.setUint8(o + 18, 255); dv.setUint8(o + 19, 0); });
        const vb = device.createBuffer({ sizeBytes: vtx.byteLength, usage: BufferUsage.Vertex, label: "b-vb" }); device.writeBuffer(vb, 0, new Uint8Array(vtx));
        const ib = device.createBuffer({ sizeBytes: 24, usage: BufferUsage.Index, label: "b-ib" }); device.writeBuffer(ib, 0, new Uint32Array([0, 1, 2, 0, 2, 3]));
        const readCenter = (): number => { const o = new Uint8Array(4); gl.readPixels(width >> 1, height >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, o); return o[0] + o[1] + o[2]; };

        // Direct encode (firstInstance=1 → on-screen origin slot).
        const dpass = device.beginPass({ id: null, width, height }, { color: [0, 0, 0, 1], depth: 1 });
        dpass.setBindings(bind); dpass.setPipeline(pipe); dpass.setVertexBuffer(0, vb); dpass.setIndexBuffer(ib, IndexFormat.Uint32); dpass.drawIndexed(6, 0, 0, 1); dpass.end();
        const directSum = readCenter();

        // Record the SAME command stream as a bundle, then replay it — must be pixel-identical.
        const bundle = device.createRenderBundle({ colorFormat: device.colorFormat, depthFormat: device.depthFormat, label: "smoke-bundle" }, (b) => {
          b.setBindings(bind); b.setPipeline(pipe); b.setVertexBuffer(0, vb); b.setIndexBuffer(ib, IndexFormat.Uint32); b.drawIndexed(6, 0, 0, 1);
        });
        const bpass = device.beginPass({ id: null, width, height }, { color: [0, 0, 0, 1], depth: 1 });
        bpass.executeBundles([bundle]); bpass.end();
        const bundleSum = readCenter();

        // A second bundle with firstInstance=0 → off-screen origin: proves firstInstance is captured BY VALUE.
        const offBundle = device.createRenderBundle({ colorFormat: device.colorFormat, depthFormat: device.depthFormat, label: "smoke-bundle-off" }, (b) => {
          b.setBindings(bind); b.setPipeline(pipe); b.setVertexBuffer(0, vb); b.setIndexBuffer(ib, IndexFormat.Uint32); b.drawIndexed(6, 0, 0, 0);
        });
        const opass = device.beginPass({ id: null, width, height }, { color: [0, 0, 0, 1], depth: 1 });
        opass.executeBundles([offBundle]); opass.end();
        const offSum = readCenter();

        check("bundle-replay", directSum > 60 && Math.abs(directSum - bundleSum) <= 6, `direct=${directSum} bundle=${bundleSum}`);
        check("bundle-firstinstance", bundleSum > 60 && offSum < 24, `slot1(on)=${bundleSum} slot0(off)=${offSum}`);
        return { ok: true };
      } catch (e) {
        check("bundle-replay", false, String(e));
        return { ok: false };
      }
    })();
    void t6b;

    // --- W6: multiDrawIndexed — one batched run (WEBGL_multi_draw when present, else the loop) ----------
    const t6m = (() => {
      try {
        const shader = device.createShaderModule(OVERLAY_PLACEHOLDER, "overlay");
        const layout = {
          strideBytes: 16,
          attributes: [
            { location: 0, kind: VertexScalarKind.Float32, components: 3, offsetBytes: 0, asInt: false },
            { location: 1, kind: VertexScalarKind.Unorm8, components: 4, offsetBytes: 12, asInt: false },
          ],
        };
        const pipe = device.createPipeline({ label: "smoke-multidraw", shader, vertexLayout: layout, pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None }, colorFormat: device.colorFormat, depthFormat: device.depthFormat });
        const ident = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        const ubo = device.createBuffer({ sizeBytes: 64, usage: BufferUsage.Uniform, label: "md-ubo" }); device.writeBuffer(ubo, 0, ident);
        const bind = device.createBindings({ pipeline: pipe, group: 0, entries: [{ binding: 0, resource: { buffer: ubo, offset: 0, size: 64 } }] });
        // Two white quads: A left (x −0.8..−0.2), B right (x 0.2..0.8), y −0.4..0.4. 8 verts → 12 indices.
        const quads: [number, number][][] = [
          [[-0.8, -0.4], [-0.2, -0.4], [-0.2, 0.4], [-0.8, 0.4]],
          [[0.2, -0.4], [0.8, -0.4], [0.8, 0.4], [0.2, 0.4]],
        ];
        const vtx = new ArrayBuffer(8 * 16); const dv = new DataView(vtx); let vi = 0;
        for (const q of quads) for (const [x, y] of q) { const o = vi * 16; dv.setFloat32(o, x, true); dv.setFloat32(o + 4, y, true); dv.setFloat32(o + 8, 0, true); dv.setUint8(o + 12, 255); dv.setUint8(o + 13, 255); dv.setUint8(o + 14, 255); dv.setUint8(o + 15, 255); vi++; }
        const vb = device.createBuffer({ sizeBytes: vtx.byteLength, usage: BufferUsage.Vertex, label: "md-vb" }); device.writeBuffer(vb, 0, new Uint8Array(vtx));
        const ib = device.createBuffer({ sizeBytes: 48, usage: BufferUsage.Index, label: "md-ib" }); device.writeBuffer(ib, 0, new Uint32Array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]));
        const pass = device.beginPass({ id: null, width, height }, { color: [0, 0, 0, 1], depth: 1 });
        pass.setPipeline(pipe); pass.setBindings(bind); pass.setVertexBuffer(0, vb); pass.setIndexBuffer(ib, IndexFormat.Uint32);
        pass.multiDrawIndexed(new Int32Array([6, 6]), new Int32Array([0, 6]), new Int32Array([0, 0]));
        pass.end();
        const at = (x: number): number => { const o = new Uint8Array(4); gl.readPixels(x, height >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, o); return o[0] + o[1] + o[2]; };
        const left = at((width * 0.25) | 0); const right = at((width * 0.75) | 0);
        check("multidraw", left > 120 && right > 120 && gl.getError() === gl.NO_ERROR, `left=${left} right=${right} (batch=${device.fastPaths.multiDraw})`);
        return { ok: true };
      } catch (e) {
        check("multidraw", false, String(e));
        return { ok: false };
      }
    })();
    void t6m;

    // --- W6.1: fast-path vs core-fallback pixel parity (GATE W6.1 — toggle each fast-path on/off) --------
    // SwiftShader has all three fast-paths, so the run above used them (clip_control omits the depth fixup,
    // drawingBufferStorage makes present a plain blit, multi_draw batches). Verify the CORE fallbacks (z*2-w
    // fixup + present shader + drawIndexed loop) render the SAME terrain color by drawing it on a SECOND
    // offscreen device created with preferCoreFallbacks, and comparing to the fast-path device.
    const t6p = (() => {
      try {
        if (typeof document === "undefined") { check("fastpath-parity", true, "no DOM — skipped"); return { ok: true }; }
        const drawTerrainCenter = (dev: WebGL2Device): number => {
          const shader = dev.createShaderModule(TERRAIN_PLACEHOLDER, "terrain");
          const pipe = dev.createPipeline({ label: "fp-terrain", shader, vertexLayout: terrainVertexLayout(), pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None }, colorFormat: dev.colorFormat, depthFormat: dev.depthFormat });
          const frame = new Float32Array(20); frame[0] = 1; frame[5] = 1; frame[10] = 1; frame[15] = 1;
          const frameUbo = dev.createBuffer({ sizeBytes: 80, usage: BufferUsage.Uniform, label: "fp-frame" }); dev.writeBuffer(frameUbo, 0, frame);
          const originBuf = dev.createBuffer({ sizeBytes: 16, usage: BufferUsage.Storage, label: "fp-origins" }); dev.writeBuffer(originBuf, 0, new Float32Array([0, 0, 0, 0]));
          const atlas = dev.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8Srgb, label: "fp-atlas" });
          const apx = new Uint8Array(2 * 2 * 4); for (let i = 0; i < 4; i++) { apx[i * 4] = 200; apx[i * 4 + 1] = 100; apx[i * 4 + 2] = 50; apx[i * 4 + 3] = 255; } dev.writeTexture(atlas, apx);
          const atlasSamp = dev.createSampler({ mag: FilterMode.Nearest, min: FilterMode.Nearest });
          const lut = dev.createTexture({ width: 2, height: 2, format: TextureFormat.Rgba8, label: "fp-lut" }); dev.writeTexture(lut, new Uint8Array(2 * 2 * 4).fill(255));
          const lutSamp = dev.createSampler({ mag: FilterMode.Linear, min: FilterMode.Linear, address: AddressMode.Clamp });
          const bind = dev.createBindings({ pipeline: pipe, group: 0, entries: [
            { binding: 0, resource: { buffer: frameUbo, offset: 0, size: 80 } },
            { binding: 1, resource: { texture: atlas } },
            { binding: 2, resource: { sampler: atlasSamp } },
            { binding: 3, resource: { buffer: originBuf } },
            { binding: 4, resource: { texture: lut } },
            { binding: 5, resource: { sampler: lutSamp } },
          ] });
          const e = (local: number): number => Math.round((local + 8) * 2048); const Z = e(0);
          const corners: [number, number][] = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
          const vtx = new ArrayBuffer(corners.length * 20); const dv = new DataView(vtx);
          corners.forEach(([lx, ly], i) => { const o = i * 20; dv.setUint16(o, e(lx), true); dv.setUint16(o + 2, e(ly), true); dv.setUint16(o + 4, Z, true); dv.setUint16(o + 6, 1, true); dv.setUint8(o + 8, 255); dv.setUint8(o + 9, 255); dv.setUint8(o + 10, 255); dv.setUint8(o + 11, 255); dv.setUint16(o + 12, 32768, true); dv.setUint16(o + 14, 32768, true); dv.setUint8(o + 16, 8); dv.setUint8(o + 17, 248); dv.setUint8(o + 18, 255); dv.setUint8(o + 19, 0); });
          const vb = dev.createBuffer({ sizeBytes: vtx.byteLength, usage: BufferUsage.Vertex, label: "fp-vb" }); dev.writeBuffer(vb, 0, new Uint8Array(vtx));
          const ib = dev.createBuffer({ sizeBytes: 24, usage: BufferUsage.Index, label: "fp-ib" }); dev.writeBuffer(ib, 0, new Uint32Array([0, 1, 2, 0, 2, 3]));
          const pass = dev.beginPass({ id: null, width, height }, { color: [0, 0, 0, 1], depth: 1 });
          pass.setBindings(bind); pass.setPipeline(pipe); pass.setVertexBuffer(0, vb); pass.setIndexBuffer(ib, IndexFormat.Uint32); pass.drawIndexed(6, 0, 0, 0); pass.end();
          const out = new Uint8Array(4); dev.raw.readPixels(width >> 1, height >> 1, 1, 1, dev.raw.RGBA, dev.raw.UNSIGNED_BYTE, out);
          return out[0] * 1000000 + out[1] * 1000 + out[2]; // pack R,G,B for an exact-ish compare
        };
        const fastPacked = drawTerrainCenter(device); // main device — fast-paths ON
        const c2 = document.createElement("canvas"); c2.width = width; c2.height = height;
        const core = WebGL2Device.create(c2, { preferCoreFallbacks: true }); // fixup + present-shader + loop
        const corePacked = drawTerrainCenter(core);
        const fr = Math.floor(fastPacked / 1000000), fg = Math.floor((fastPacked % 1000000) / 1000), fb = fastPacked % 1000;
        const cr = Math.floor(corePacked / 1000000), cg = Math.floor((corePacked % 1000000) / 1000), cb = corePacked % 1000;
        core.destroy();
        // Pixel-identical within ±2 LSB/channel (the present-shader's extra 8-bit round trip vs the blit).
        const parity = fr > 80 && Math.abs(fr - cr) <= 2 && Math.abs(fg - cg) <= 2 && Math.abs(fb - cb) <= 2;
        check("fastpath-parity", parity, `fast=[${fr},${fg},${fb}] core=[${cr},${cg},${cb}]`);
        return { ok: true };
      } catch (e) {
        check("fastpath-parity", false, String(e));
        return { ok: false };
      }
    })();
    void t6p;

    // --- W5 winding probe: front-face parity with WebGPU (the gl.frontFace CCW-vs-CW question) ----
    // Empirically measured on real Metal WebGPU (smoke page): a CCW-in-NDC quad SURVIVES cull:Back, i.e.
    // WebGPU classifies it FRONT-facing. WebGL2 must agree, which it does with gl.frontFace(gl.CCW). This
    // guards against a regression to gl.frontFace(gl.CW) (which would invert culling → inside-out terrain).
    const wind = await runWindingProbe(device, width, height);
    check("winding-cullback", wind.visible, `CCW-in-NDC must survive cull:Back (front), matching WebGPU; sum=${wind.sum}`);

    // --- backstop: no GL errors slipped through ---------------------------------------------
    check("no-gl-errors", gl.getError() === gl.NO_ERROR);

    device.destroyBuffer(vbuf);
    device.destroyBuffer(ibuf);
    device.destroyBuffer(dst);
    device.destroyBuffer(dst2);
    device.destroyTexture(tex);

    const ok = failed.length === 0;
    return {
      ok,
      backend: device.backend,
      colorFormat: device.colorFormat,
      size: `${width}×${height}`,
      detail:
        (ok ? `W0–W6 checks passed: ${passed.join(", ")}` : `FAILED: ${failed.join("; ")}`) +
        ` | winding-cullback(CCW-in-NDC)=${wind.visible ? "VISIBLE(front)" : "CULLED(back)"} sum=${wind.sum}` +
        ` | fast-paths active: ${JSON.stringify(device.fastPaths)}`,
    };
  } finally {
    // Intentionally NOT device.destroy() here: it loseContext()s, which blanks the canvas before the
    // harness/Playwright screenshots it. This is a one-shot page — the context is freed on unload.
    // (The WebGPU smoke can destroy() because a WebGPU canvas keeps its last presented frame regardless.)
  }
}
