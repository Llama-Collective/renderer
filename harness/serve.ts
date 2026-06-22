// Dev server for the WebGPU smoke harness. Run from the repo root:
//
//   npm run harness:webgpu        (alias for: npx tsx schematic-renderer-webgpu/harness/serve.ts)
//
// then open the printed URL in a WebGPU-capable browser (Chrome/Edge 113+, or Safari Tech
// Preview). esbuild bundles harness/main.ts → /dist/main.js in memory and serves index.html
// next to it; no build artifacts touch disk. Uses only esbuild, which is already a root dep.

import * as esbuild from "esbuild";
import { fileURLToPath } from "node:url";
import { copyFileSync, existsSync } from "node:fs";
import * as path from "node:path";

const dir = path.dirname(fileURLToPath(import.meta.url));

/** The Phase-2 demo fetches /pack.zip; mirror the repo's public/pack.zip into the served dir. */
function stagePackZip(): void {
  const src = path.resolve(dir, "../../public/pack.zip");
  if (existsSync(src)) copyFileSync(src, path.join(dir, "pack.zip"));
}

async function main(): Promise<void> {
  stagePackZip();
  // Bundle EVERY harness page (one per *.html). Anything omitted here is served as a STALE on-disk
  // dist/*.js — source edits silently don't take effect (this bit glass/stress/fluid/piston before).
  const pages = ["main", "webgl2", "crossbackend", "demo", "entities", "blockentities", "glass", "fluid", "stress", "piston", "viewer", "integration", "simstorm", "lighting", "farworld"];
  const ctx = await esbuild.context({
    // Pages + the mesher-worker entry: the SchematicViewer's worker is loaded as a prebuilt module bundle
    // (`dist/mesher.worker.entry.js`) because esbuild does NOT bundle the viewer's `new URL(...worker.ts)` (it
    // leaves the raw .ts). Building it here (vs a stale on-disk copy) keeps it in sync with the worker source.
    entryPoints: [...pages.map((p) => path.join(dir, `${p}.ts`)), path.join(dir, "mesher.worker.entry.ts")],
    bundle: true,
    format: "esm",
    target: "es2022",
    sourcemap: true,
    outdir: path.join(dir, "dist"),
  });

  const { port } = await ctx.serve({ servedir: dir, host: "127.0.0.1", port: 5273 });
  // eslint-disable-next-line no-console
  console.log(
    `\n  WebGPU harness running (Ctrl-C to stop):\n` +
      `    smoke (triangle)  → http://localhost:${port}/\n` +
      pages.filter((p) => p !== "main").map((p) => `    ${p.padEnd(16)}→ http://localhost:${port}/${p}.html`).join("\n") +
      `\n`,
  );
}

void main();
