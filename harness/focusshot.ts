// One-off visual inspector: serve blockentities.html and screenshot specific BE cells up close via the
// `?focus=<type>` hook. Run from repo root: tsx schematic-renderer-webgpu/harness/focusshot.ts portal
import * as esbuild from "esbuild";
import { chromium, type Browser } from "playwright";
import { fileURLToPath } from "node:url";
import { copyFileSync, existsSync } from "node:fs";
import * as path from "node:path";

const dir = path.dirname(fileURLToPath(import.meta.url));
// page defaults to blockentities.html; set it to target the entity demo.
const SHOTS: { name: string; focus: string; page?: string }[] = [
  { name: "item-sword", focus: "item:sword", page: "entities.html" },
  { name: "item-apple", focus: "item:apple", page: "entities.html" },
  { name: "item-stick", focus: "item:stick", page: "entities.html" },
];

async function main(): Promise<void> {
  const src = path.resolve(dir, "../../public/pack.zip");
  if (existsSync(src)) copyFileSync(src, path.join(dir, "pack.zip"));
  const ctx = await esbuild.context({
    entryPoints: [path.join(dir, "blockentities.ts"), path.join(dir, "entities.ts")],
    bundle: true, format: "esm", target: "es2022", outdir: path.join(dir, "dist"),
  });
  const { port } = await ctx.serve({ servedir: dir, host: "127.0.0.1", port: 0 });
  const base = `http://localhost:${port}`;
  let browser: Browser | null = null;
  try {
    browser = await chromium.launch({ channel: "chrome", args: ["--enable-unsafe-webgpu"] }).catch(() =>
      chromium.launch({ args: ["--enable-unsafe-webgpu"] }));
    for (const s of SHOTS) {
      const pageName = s.page ?? "blockentities.html";
      const globalKey = pageName.startsWith("entities") ? "__ENTITIES__" : "__BLOCKENTITIES__";
      const page = await browser.newPage({ viewport: { width: 700, height: 600 }, deviceScaleFactor: 2 });
      page.on("pageerror", (e) => console.log(`  [${s.name}] pageerror: ${e.message}`));
      await page.goto(`${base}/${pageName}?focus=${encodeURIComponent(s.focus)}`, { waitUntil: "domcontentloaded" });
      await page.waitForFunction((k) => (window as unknown as Record<string, unknown>)[k], globalKey, { timeout: 25_000 }).catch(() => {});
      await page.waitForTimeout(1500); // let the animation run a beat
      await page.locator("#c").screenshot({ path: path.join(dir, `focus-${s.name}.png`) });
      console.log(`  [${s.name}] → focus-${s.name}.png`);
      await page.close();
    }
  } finally {
    if (browser) await browser.close();
    await ctx.dispose();
  }
}
main().then(() => process.exit(0), (e) => { console.error(e); process.exit(1); });
