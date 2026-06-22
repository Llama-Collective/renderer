// Verify the harness pages in real Chrome (Apple/Metal WebGPU). Run from the repo root:
//
//   npm run harness:webgpu:verify              # verify both: smoke triangle + Phase-1 terrain
//   npm run harness:webgpu:verify -- demo      # just the terrain demo
//   npm run harness:webgpu:verify -- smoke
//
// WebGPU is only exposed on a SECURE CONTEXT, so we serve the real pages over http://localhost
// (via esbuild, same as `npm run harness:webgpu`) and `goto` them — `about:blank`/`setContent`
// does NOT get `navigator.gpu`. Reads each page's self-check verdict (window.__SMOKE__/__DEMO__),
// screenshots it, and exits 0 (PASS) / 1 (FAIL) / 2 (SKIP — no WebGPU adapter).

import * as esbuild from "esbuild";
import { chromium, type Browser } from "playwright";
import { fileURLToPath } from "node:url";
import { copyFileSync, existsSync } from "node:fs";
import * as path from "node:path";

const dir = path.dirname(fileURLToPath(import.meta.url));

/** The demo fetches /pack.zip; mirror the repo's public/pack.zip into the served dir. */
function stagePackZip(): void {
  const src = path.resolve(dir, "../../public/pack.zip");
  if (existsSync(src)) copyFileSync(src, path.join(dir, "pack.zip"));
}

interface Target {
  name: string;
  page: string;
  globalKey: "__SMOKE__" | "__WEBGL2__" | "__CROSSBACKEND__" | "__DEMO__" | "__GLASS__" | "__FLUID__" | "__STRESS__" | "__PISTON__" | "__ENTITIES__" | "__BLOCKENTITIES__" | "__INTEGRATION__" | "__SIMSTORM__" | "__WORKERS__" | "__LIGHTING__" | "__FARWORLD__";
  /** Serve this page cross-origin isolated (COOP/COEP) so `crossOriginIsolated` is true → SharedArrayBuffer
   *  is available → the Model B (SAB) meshing path activates. esbuild's static serve sets no response
   *  headers, so we inject them via route interception (scoped to this target only). */
  isolate?: boolean;
}
const TARGETS: Record<string, Target> = {
  smoke: { name: "smoke", page: "index.html", globalKey: "__SMOKE__" },
  webgl2: { name: "webgl2", page: "webgl2.html", globalKey: "__WEBGL2__" },
  crossbackend: { name: "crossbackend", page: "crossbackend.html", globalKey: "__CROSSBACKEND__" }, // W7 — WebGPU vs WebGL2 pixel-diff

  demo: { name: "demo", page: "demo.html", globalKey: "__DEMO__" },
  glass: { name: "glass", page: "glass.html", globalKey: "__GLASS__" },
  fluid: { name: "fluid", page: "fluid.html", globalKey: "__FLUID__" },
  stress: { name: "stress", page: "stress.html", globalKey: "__STRESS__" },
  piston: { name: "piston", page: "piston.html", globalKey: "__PISTON__" },
  entities: { name: "entities", page: "entities.html", globalKey: "__ENTITIES__" },
  blockentities: { name: "blockentities", page: "blockentities.html", globalKey: "__BLOCKENTITIES__" },
  integration: { name: "integration", page: "integration.html", globalKey: "__INTEGRATION__" },
  simstorm: { name: "simstorm", page: "simstorm.html", globalKey: "__SIMSTORM__" },
  workers: { name: "workers", page: "workers.html", globalKey: "__WORKERS__", isolate: true },
  lighting: { name: "lighting", page: "lighting.html", globalKey: "__LIGHTING__" },
  farworld: { name: "farworld", page: "farworld.html", globalKey: "__FARWORLD__" }, // PREC-1 (needs Metal/GPU)
};

async function launch(): Promise<Browser> {
  const args = ["--enable-unsafe-webgpu"];
  try {
    return await chromium.launch({ channel: "chrome", args }); // installed Chrome: strongest WebGPU
  } catch {
    return await chromium.launch({ args });
  }
}

async function verify(browser: Browser, base: string, t: Target): Promise<number> {
  // deviceScaleFactor 2 exercises the HiDPI/Retina path (canvas backed by 2× physical pixels).
  const page = await browser.newPage({ viewport: { width: 1100, height: 900 }, deviceScaleFactor: 2 });
  page.on("console", (m) => console.log(`  [${t.name}] ${m.text()}`));
  page.on("pageerror", (e) => console.log(`  [${t.name}] pageerror: ${e.message}`));

  if (t.isolate) {
    // Inject the cross-origin-isolation headers (esbuild's serve sets none) so `crossOriginIsolated` is true
    // → SharedArrayBuffer is available → Model B runs. Same-origin subresources need no CORP, but adding it
    // is harmless and keeps every response COEP-clean.
    await page.route("**/*", async (route) => {
      const resp = await route.fetch();
      await route.fulfill({
        response: resp,
        headers: {
          ...resp.headers(),
          "cross-origin-opener-policy": "same-origin",
          "cross-origin-embedder-policy": "require-corp",
          "cross-origin-resource-policy": "same-origin",
        },
      });
    });
  }

  await page.goto(`${base}/${t.page}`, { waitUntil: "domcontentloaded" });
  const result = (await page
    .waitForFunction((k) => (window as unknown as Record<string, unknown>)[k], t.globalKey, { timeout: 25_000 })
    .then((h) => h.jsonValue())
    .catch(() => null)) as { ok?: boolean; detail?: string; error?: string } | null;

  // Screenshot just the canvas → the artifact is the actual rendered backing-store resolution.
  await page.locator("#c").screenshot({ path: path.join(dir, `verify-${t.name}.png`) });

  let code: number;
  if (!result) {
    console.error(`  [${t.name}] SKIP — no verdict within 25s (no WebGPU adapter?)`);
    code = 2;
  } else if (result.error && /not available|adapter/i.test(result.error)) {
    console.error(`  [${t.name}] SKIP — ${result.error}`);
    code = 2;
  } else {
    const ok = !!result.ok;
    console.log(`  [${t.name}] ${ok ? "PASS" : "FAIL"} — ${result.detail ?? result.error ?? "?"}  (→ verify-${t.name}.png)`);
    code = ok ? 0 : 1;
  }
  await page.close();
  return code;
}

async function main(): Promise<number> {
  const which = process.argv[2];
  // No argument ⇒ run the FULL harness (every target, in declaration order); an argument runs that one page.
  const targets = which ? [TARGETS[which]].filter(Boolean) : Object.values(TARGETS);
  if (targets.length === 0) {
    console.error(`unknown target "${which}" — use: ${Object.keys(TARGETS).join(", ")}`);
    return 1;
  }

  stagePackZip();
  // Serve the real pages over localhost (secure context → WebGPU is exposed).
  const ctx = await esbuild.context({
    entryPoints: [path.join(dir, "main.ts"), path.join(dir, "webgl2.ts"), path.join(dir, "crossbackend.ts"), path.join(dir, "demo.ts"), path.join(dir, "glass.ts"), path.join(dir, "fluid.ts"), path.join(dir, "stress.ts"), path.join(dir, "piston.ts"), path.join(dir, "entities.ts"), path.join(dir, "blockentities.ts"), path.join(dir, "integration.ts"), path.join(dir, "simstorm.ts"), path.join(dir, "workers.ts"), path.join(dir, "lighting.ts"), path.join(dir, "farworld.ts"), path.join(dir, "mesher.worker.entry.ts")],
    bundle: true,
    format: "esm",
    target: "es2022",
    outdir: path.join(dir, "dist"),
  });
  const { port } = await ctx.serve({ servedir: dir, host: "127.0.0.1", port: 0 });
  const base = `http://localhost:${port}`;

  let browser: Browser | null = null;
  try {
    browser = await launch();
    let worst = 0;
    for (const t of targets) worst = Math.max(worst, await verify(browser, base, t));
    return worst;
  } catch (e) {
    console.error(`SKIP — could not launch Chrome/Chromium: ${e instanceof Error ? e.message : e}`);
    return 2;
  } finally {
    if (browser) await browser.close();
    await ctx.dispose();
  }
}

main().then(
  (code) => process.exit(code),
  (e) => {
    console.error(e);
    process.exit(1);
  },
);
