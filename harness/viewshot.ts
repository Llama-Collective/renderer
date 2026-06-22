// Screenshot the single-subject viewer for a set of query strings. Run from repo root:
//   tsx schematic-renderer-webgpu/harness/viewshot.ts "block=minecraft:hopper" "entity=hopper_minecart"
// Each arg is the viewer.html query (no leading ?); the shot is saved as view-<sanitized>.png.
import * as esbuild from "esbuild";
import { chromium, type Browser } from "playwright";
import { fileURLToPath } from "node:url";
import { copyFileSync, existsSync } from "node:fs";
import * as path from "node:path";

const dir = path.dirname(fileURLToPath(import.meta.url));
const SHOTS = process.argv.slice(2);
if (SHOTS.length === 0) SHOTS.push("block=minecraft:hopper[facing=down]", "entity=hopper_minecart");

async function main(): Promise<void> {
  const src = path.resolve(dir, "../../public/pack.zip");
  if (existsSync(src)) copyFileSync(src, path.join(dir, "pack.zip"));
  const ctx = await esbuild.context({
    entryPoints: [path.join(dir, "viewer.ts")],
    bundle: true, format: "esm", target: "es2022", outdir: path.join(dir, "dist"),
  });
  const { port } = await ctx.serve({ servedir: dir, host: "127.0.0.1", port: 0 });
  const base = `http://localhost:${port}`;
  let browser: Browser | null = null;
  try {
    browser = await chromium.launch({ channel: "chrome", args: ["--enable-unsafe-webgpu"] }).catch(() =>
      chromium.launch({ args: ["--enable-unsafe-webgpu"] }));
    for (const q of SHOTS) {
      const name = q.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").slice(0, 60);
      const page = await browser.newPage({ viewport: { width: 640, height: 600 }, deviceScaleFactor: 2 });
      page.on("pageerror", (e) => console.log(`  [${name}] pageerror: ${e.message}`));
      await page.goto(`${base}/viewer.html?${q}`, { waitUntil: "domcontentloaded" });
      await page.waitForFunction(() => (window as unknown as Record<string, unknown>).__VIEWER__, null, { timeout: 25_000 }).catch(() => {});
      const detail = await page.evaluate(() => (window as unknown as { __VIEWER__?: { detail?: string; error?: string } }).__VIEWER__);
      await page.waitForTimeout(900);
      await page.locator("#c").screenshot({ path: path.join(dir, `view-${name}.png`) });
      console.log(`  [${name}] → view-${name}.png  ${detail?.detail ?? detail?.error ?? ""}`);
      await page.close();
    }
  } finally {
    if (browser) await browser.close();
    await ctx.dispose();
  }
}
main().then(() => process.exit(0), (e) => { console.error(e); process.exit(1); });
