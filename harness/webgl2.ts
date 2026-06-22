// Browser entry for the WebGL2 device smoke harness. Bundled by ../harness/serve.ts (manual) and the
// Playwright runner (headless). Drives the implemented WebGL2 surface (W0 clear + W1 buffers/copy/
// textures/samplers) and reports PASS/FAIL into the page so a human (banner) and automation
// (window.__WEBGL2__ + #status text) read the same verdict.

import { runWebGL2Smoke, type WebGL2SmokeResult } from "./webgl2smoke";

declare global {
  interface Window {
    /** Set when the smoke settles, so the Playwright runner can poll a single value. */
    __WEBGL2__?: { ok: boolean; detail: string } | { ok: false; error: string };
  }
}

function setBanner(cls: "pass" | "fail" | "run", text: string): void {
  const el = document.getElementById("status");
  if (!el) return;
  el.className = cls;
  el.textContent = text;
}

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement | null;
  if (!canvas) {
    setBanner("fail", 'FAIL — no <canvas id="c">');
    window.__WEBGL2__ = { ok: false, error: "no canvas" };
    return;
  }

  // HiDPI: back the canvas with physical pixels so the result is sharp on Retina screens.
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));

  setBanner("run", "running…");
  try {
    const r: WebGL2SmokeResult = await runWebGL2Smoke(canvas);
    setBanner(r.ok ? "pass" : "fail", `${r.ok ? "PASS" : "FAIL"} — ${r.backend} · ${r.colorFormat} · ${r.size} — ${r.detail}`);
    window.__WEBGL2__ = { ok: r.ok, detail: r.detail };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    setBanner("fail", `FAIL — ${msg}`);
    window.__WEBGL2__ = { ok: false, error: msg };
  }
}

void main();
