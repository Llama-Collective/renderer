// Backend selection — the single neutral entry point the app uses instead of naming a concrete
// device. RENDERER_PLAN.md §2; WEBGL_PLAN.md (Phase W0). Feature-detects WebGPU and falls back to
// WebGL2, so `app/` never imports a `WebGPUDevice`/`WebGL2Device` directly.
//
// `CanvasDevice` is the SUPER-interface the app actually programs against: the pure command-encoding
// `GraphicsDevice` plus the swapchain/canvas concerns that are deliberately kept OFF the neutral
// interface (color/depth target formats, resize, screenshot readback) and the Phase-P HUD counters.
// Both backends satisfy it structurally; the WebGL2 backend also `implements CanvasDevice` explicitly.

import type { GraphicsDevice, TextureFormat } from "./GraphicsDevice";
import { WebGPUDevice } from "./webgpu/WebGPUDevice";
import { WebGL2Device } from "./webgl2/WebGL2Device";

/** A `GraphicsDevice` bound to a presentable canvas: adds the swapchain surface the app reads off the
 *  concrete device (formats, resize, readback) and the HUD metric counters. Kept separate from
 *  `GraphicsDevice` so the pure command-encoding interface stays free of canvas concerns. */
export interface CanvasDevice extends GraphicsDevice {
  /** Color-target format pipelines/bundles must be built against (an `*-srgb` format — §17.2 linear chain). */
  readonly colorFormat: TextureFormat;
  /** Depth-target format pipelines must be built against. */
  readonly depthFormat: TextureFormat;
  /** Recreate size-dependent attachments + viewport on a canvas resize. */
  resize(width: number, height: number): void;
  /** Screenshot the canvas (headless verification / pixel-diff harness). */
  readCanvasPixels(): Promise<{ width: number; height: number; bytesPerRow: number; data: Uint8Array }>;
  // Phase-P HUD metrics — read as per-frame deltas by SchematicViewer.sampleStats.
  readonly drawCalls: number;
  readonly passCount: number;
  readonly liveBuffers: number;
  readonly createBufferCalls: number;
  readonly poolHits: number;
}

/**
 * Create the best available device for `canvas`: WebGPU when present, else the WebGL2 fallback.
 * Uniformly async so the caller's `ready`-promise buffering is identical for both backends (WebGPU's
 * adapter/device requests are async; WebGL2's `getContext` is synchronous and is simply awaited).
 *
 * @param opts.preferWebGL2 force the fallback (testing / diagnostics).
 */
export async function createDevice(
  canvas: HTMLCanvasElement,
  opts: { bufferPool?: boolean; preferWebGL2?: boolean } = {},
): Promise<CanvasDevice> {
  if (!opts.preferWebGL2 && typeof navigator !== "undefined" && navigator.gpu) {
    try {
      return await WebGPUDevice.create(canvas, { bufferPool: opts.bufferPool });
    } catch (e) {
      console.warn("[renderer] WebGPU init failed — falling back to WebGL2", e);
    }
  }
  return WebGL2Device.create(canvas, { bufferPool: opts.bufferPool });
}
