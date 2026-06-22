// Static web page for the WebGPU SchematicViewer — the drop-in replacement for the old React
// `App.tsx` (../../old/src/App.tsx). Same URL-load behaviour:
//
//   index.html?url=<schematic url>
//
// fetches the resource pack + the schematic, parses it, and renders it. The old app drove the
// `schematic-renderer` npm package (which fetched + parsed internally); the new `SchematicViewer`
// takes an already-parsed `EditableSchematic`, so this page does the fetch + parse itself via the
// shared `schematic-io` parser and then hands the result to `viewer.loadSchematic`.

import "./style.css";
import { SchematicViewer } from "../app/SchematicViewer";
import { initParser, parseSchematic } from "./parseSchematic";
import type { BackendKind } from "../core/gpu-enums";

// ── DOM handles ────────────────────────────────────────────────────────────────
const canvas = document.getElementById("renderer-canvas") as HTMLCanvasElement;
const dotEl = document.getElementById("status-dot") as HTMLSpanElement;
const labelEl = document.getElementById("status-label") as HTMLDivElement;
const subtextEl = document.getElementById("status-subtext") as HTMLDivElement;
const progressEl = document.getElementById("progress") as HTMLDivElement;
const progressBarEl = document.getElementById("progress-bar") as HTMLDivElement;
const helperCard = document.getElementById("helper-card") as HTMLDivElement;
const errorCard = document.getElementById("error-card") as HTMLDivElement;
const errorMessage = document.getElementById("error-message") as HTMLParagraphElement;

type Status = "initializing" | "idle" | "loading" | "ready" | "error";

const STATUS_LABELS: Record<Status, string> = {
  initializing: "Initializing renderer",
  idle: "Waiting for a schematic",
  loading: "Loading schematic",
  ready: "Renderer ready",
  error: "Something went wrong",
};

// ── Status / progress UI ─────────────────────────────────────────────────────────
function setStatus(status: Status, message?: string): void {
  dotEl.className = `status-dot status-${status}`;
  labelEl.textContent = STATUS_LABELS[status];
  if (message !== undefined) subtextEl.textContent = message;
  helperCard.hidden = status !== "idle";
  errorCard.hidden = status !== "error";
}

function setMessage(message: string): void {
  subtextEl.textContent = message;
}

/** Show the progress bar at `fraction` (0..1), or `null` to report indeterminate (bar hidden). */
function setProgress(fraction: number | null): void {
  if (fraction === null) {
    progressEl.hidden = true;
    return;
  }
  progressEl.hidden = false;
  progressBarEl.style.width = `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
}

function fail(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  console.error("[viewer]", error);
  setStatus("error", message);
  setProgress(null);
  errorMessage.textContent = message;
}

/** Friendly schematic name from a URL's last path segment (matches the old app). */
function nameFromUrl(url: string): string {
  try {
    const last = new URL(url).pathname.split("/").filter(Boolean).pop();
    return last || "schematic";
  } catch {
    return url;
  }
}

/** Fetch `url` into an ArrayBuffer, reporting download progress (0..1) when the server sends a
 *  Content-Length — the analogue of the old app's `onProgress` callback. */
async function fetchWithProgress(
  url: string,
  onProgress: (fraction: number | null) => void,
): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Fetch failed: ${response.status} ${response.statusText} (${url})`);

  const total = Number(response.headers.get("Content-Length")) || 0;
  if (!response.body || !total) {
    onProgress(null); // no stream / unknown length → indeterminate
    return response.arrayBuffer();
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received / total);
  }

  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out.buffer;
}

// ── Resource-pack location ────────────────────────────────────────────────────────
// pack.zip ships in public/ and is copied to the build root; resolve it against the page's base
// URL so it works both at the dev-server root and under a GitHub Pages project subpath.
const PACK_URL = new URL("pack.zip", document.baseURI).toString();

async function main(): Promise<void> {
  setStatus("initializing", "Preparing renderer…");

  // Resolve the ?url param up front (so an invalid URL is reported before we touch the GPU).
  const rawUrl = new URLSearchParams(window.location.search).get("url");
  let schematicUrl: string | null = null;
  if (rawUrl) {
    try {
      schematicUrl = new URL(rawUrl, window.location.href).toString();
    } catch {
      setStatus("error", "The ?url parameter is not a valid URL.");
      errorMessage.textContent = "The ?url parameter is not a valid URL.";
      return;
    }
  }

  // Start the parser's WASM download now (when we'll need it) so it overlaps the pack + schematic
  // fetches below — by parse time it's usually already initialised.
  if (schematicUrl) initParser();

  const viewer = new SchematicViewer(canvas);
  viewer.showGrid = false; // no ground-reference floor grid in the standalone viewer

  // Surface a fatal "no GPU backend" instead of a silent black canvas (W7).
  viewer.onDeviceError = (message) => {
    setStatus("error", message);
    errorMessage.textContent = message;
  };
  viewer.onBackendReady = (backend: BackendKind) => {
    console.log(`[viewer] backend: ${backend}`);
  };

  // 1) Resource pack.
  setStatus("initializing", "Fetching resource pack…");
  setProgress(0);
  let packBuffer: ArrayBuffer;
  try {
    packBuffer = await fetchWithProgress(PACK_URL, (f) => setProgress(f));
  } catch (error) {
    fail(new Error(`Could not load resource pack (pack.zip): ${error instanceof Error ? error.message : error}`));
    return;
  }
  setMessage("Loading resource pack…");
  setProgress(null);
  await viewer.loadResourcePack(packBuffer);

  // 2) No URL → idle: render the empty framed scene and show the hint card.
  if (!schematicUrl) {
    await viewer.whenReady();
    setStatus("idle", "Append ?url=<schematic url> to load one.");
    return;
  }

  // 3) Fetch + parse + load the schematic.
  setStatus("loading", "Fetching schematic…");
  setProgress(0);
  try {
    const name = nameFromUrl(schematicUrl);
    const buffer = await fetchWithProgress(schematicUrl, (f) => setProgress(f));
    setMessage("Parsing schematic…");
    setProgress(null);
    const schematic = await parseSchematic(new Uint8Array(buffer), name, schematicUrl);
    setMessage("Building scene…");
    viewer.loadSchematic(schematic);
    await viewer.whenReady();
    setStatus("ready", `Showing ${name} · ${schematic.blocks.length.toLocaleString()} blocks`);
    setProgress(null);
  } catch (error) {
    fail(error);
  }
}

main().catch(fail);
