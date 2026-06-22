# WebGPU harness

In-browser proof that the renderer works, since vitest (node) has no GPU. Two pages:

- **`index.html` (smoke)** — the smallest program that drives `WebGPUDevice` end-to-end. Tangible
  **Phase 0 / GATE 0** evidence.
- **`demo.html` (Phase-1 terrain)** — the full slice: a heightmap world → `SnapshotSource` →
  `meshSection` (with face culling) → `AtlasManager` → invariant-enforcing `SectionStore` commit →
  `TerrainRenderer` with an orbiting camera. Tangible **GATE 1** evidence.

Both pages self-verify via a pixel readback and publish a verdict (`window.__SMOKE__` /
`window.__DEMO__`) that the Playwright runner reads.

> WebGPU is exposed only on a **secure context**, so the pages must be served over
> `http://localhost` (what `serve.ts`/`verify.ts` do) — opening the `.html` from `file://` or
> `about:blank` won't get `navigator.gpu`.

What it exercises, in one frame:

```
WebGPUDevice.create(canvas)        device + adapter + the linear -srgb color chain
  → createShaderModule(WGSL)        hand-written vertex+fragment
  → createPipeline(...)             packed vertex layout (float32x2 + unorm8x4), solid-pass state
  → createBuffer ×2                 vertex + index
  → beginPass(clear) → draw → end   indexed draw, submit
  → readCanvasPixels()              screenshot the swapchain
  → assert                          center shows the quad, a corner keeps the clear color
```

It is **self-verifying**: the pixel readback makes it an objective PASS/FAIL, not an eyeball
check. The page shows a banner; the headless runner exits 0/1 on the same verdict.

## Run it (manual)

```sh
npm run harness:webgpu
```

Open the printed URLs in a WebGPU-capable browser (Chrome/Edge 113+, or Safari Technology
Preview): `/` is the smoke triangle (orange quad on black), `/demo.html` is the orbiting voxel
terrain. Each shows a green **PASS** banner.

## Run it (automated — Playwright on real Chrome)

```sh
npm run harness:webgpu:verify            # both pages
npm run harness:webgpu:verify -- demo    # just the terrain demo
```

Serves the pages over `http://localhost` and drives installed Chrome (Apple/Metal WebGPU), reading
each page's self-check and saving `verify-<name>.png`. Exit codes: `0` PASS, `1` FAIL, `2` SKIP
(no WebGPU adapter reachable).

## Files

- `smoke.ts` / `main.ts` / `index.html` — the device smoke test + page.
- `demo.ts` / `demoTextures.ts` / `demo.html` — the Phase-1 terrain scene + procedural textures.
- `serve.ts` — esbuild dev server (in-memory bundle; nothing written to disk).
- `verify.ts` — Playwright runner (serves localhost, screenshots, exits 0/1/2).

> Backend-specific by design: unlike `mesh/`/`world/`, the harness imports the concrete
> `WebGPUDevice` because it tests *that backend's* wiring. The neutral data-model + mesher tests
> live in `src/**/*.test.ts`.
