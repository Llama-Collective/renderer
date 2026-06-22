import { defineConfig } from "vite";

// Static-site build for the WebGPU schematic viewer page (src/web).
//
// • `base: "./"` emits relative asset URLs so the build works from a GitHub Pages
//   project subpath (https://<user>.github.io/<repo>/) without knowing the repo name.
// • `outDir: "docs"` + `emptyOutDir` writes the deployable site into docs/, which
//   GitHub Pages can serve directly (Settings → Pages → "Deploy from branch", /docs).
// • The mesher worker is bundled automatically from the `new URL("../workers/mesher.worker.ts",
//   import.meta.url)` pattern in SchematicViewer — no extra config needed. (Cross-origin
//   isolation for the SharedArrayBuffer "Model B" fast path isn't available on plain GitHub
//   Pages, so the renderer transparently degrades to the postMessage "Model A" path.)
export default defineConfig({
  base: "./",
  build: {
    outDir: "docs",
    emptyOutDir: true,
    target: "esnext", // top-level await / modern worker output — matches the renderer's ES2020+ source
  },
  // The `nucleation` schematic parser is a wasm-bindgen ESM module that resolves its `.wasm` via
  // `new URL("nucleation_bg.wasm", import.meta.url)`. Vite's native asset handling rewrites that in
  // the build, but the dev-time esbuild pre-bundle rewrites `import.meta.url` incorrectly — exclude
  // it so the dev server serves the module (and its wasm) untouched.
  optimizeDeps: {
    exclude: ["nucleation"],
  },
});
