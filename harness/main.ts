// Browser entry for the WebGPU smoke harness. Bundled by ../harness/serve.ts (manual) and the
// Playwright runner (headless). Renders the quad and reports PASS/FAIL into the page so a human
// (banner) and automation (window.__SMOKE__ + #status text) read the same verdict.

import { runSmoke, type SmokeResult } from "./smoke";

declare global {
  interface Window {
    /** Set when the smoke test settles, so the Playwright runner can poll a single value. */
    __SMOKE__?: { ok: boolean; detail: string } | { ok: false; error: string };
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
    setBanner("fail", "FAIL — no <canvas id=\"c\">");
    window.__SMOKE__ = { ok: false, error: "no canvas" };
    return;
  }

  // HiDPI: back the canvas with physical pixels so the result is sharp on Retina screens.
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));

  setBanner("run", "running…");
  try {
    const r: SmokeResult = await runSmoke(canvas);
    setBanner(
      r.ok ? "pass" : "fail",
      `${r.ok ? "PASS" : "FAIL"} — ${r.backend} · ${r.colorFormat} · ${r.size} — ${r.detail}`,
    );
    window.__SMOKE__ = { ok: r.ok, detail: r.detail };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    setBanner("fail", `FAIL — ${msg}`);
    window.__SMOKE__ = { ok: false, error: msg };
  }
}

void main();
