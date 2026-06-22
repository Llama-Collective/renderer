// Bitmap font for sign text (legacy `font/ascii.png`: 128×128, a 16×16 grid of 8×8 glyph cells).
// RENDERER_PLAN.md §18, Phase 4.5d. Per-glyph advance widths are SCANNED from the decoded sheet once at
// load (`setFontWidths`) — exactly how vanilla derives them from the bitmap — so layout is version-correct
// (no hard-coded width table). Glyphs render as flat quads whose front face box-unwraps the glyph cell.

import { Direction } from "../../../types";
import type { ModelPartDef } from "../../../mesh/entity/ModelPart";

export const FONT_TEX = "font/ascii";
export const FONT_SIZE = 128; // texWidth/texHeight of ascii.png
export const LINE_HEIGHT = 10; // vanilla SignBlockEntity.getTextLineHeight
const CELL = 8; // glyph cell px

let GLYPH_W: Uint8Array | null = null; // visual width per char code (0 = empty / space)

/** Scan per-glyph visual widths (rightmost non-transparent column + 1) from a decoded RGBA ascii sheet. */
export function scanFontWidths(rgba: Uint8Array, w: number, h: number): Uint8Array {
  const widths = new Uint8Array(256);
  const cols = Math.floor(w / CELL); // 16
  for (let c = 0; c < 256; c++) {
    const cx = (c % cols) * CELL, cy = Math.floor(c / cols) * CELL;
    if (cy + CELL > h) break;
    let maxCol = -1;
    for (let gx = CELL - 1; gx >= 0; gx--) {
      let any = false;
      for (let gy = 0; gy < CELL; gy++) {
        if (rgba[((cy + gy) * w + (cx + gx)) * 4 + 3] > 16) { any = true; break; }
      }
      if (any) { maxCol = gx; break; }
    }
    widths[c] = maxCol + 1; // 0 if the cell is empty
  }
  return widths;
}

export function setFontWidths(widths: Uint8Array): void {
  GLYPH_W = widths;
}
export function fontReady(): boolean {
  return GLYPH_W !== null;
}

/** Pen advance for a char: visual width + 1px gap; an empty cell (space) advances 4 (vanilla). */
function advance(code: number): number {
  const w = GLYPH_W?.[code] ?? 5;
  return w > 0 ? w + 1 : 4;
}
function visualWidth(code: number): number {
  return GLYPH_W?.[code] ?? 5;
}

/** Total advance width of a line in font px (vanilla `Font.width`). */
export function lineWidth(line: string): number {
  let x = 0;
  for (let i = 0; i < line.length; i++) x += advance(line.charCodeAt(i) & 0xff);
  return x;
}

/**
 * Glyph quads for up to 4 lines, as flat `ModelPartDef` cubes in FONT-PIXEL text space (origin = the text
 * block centre). Each line is centred (`x = −width/2`) and stacked by LINE_HEIGHT, the block centred
 * vertically (vanilla `y = i·lh − 2·lh`). A glyph's front face box-unwraps its cell — `origin [penX,y,0]`,
 * `size [w,8,0]`, `uv [col·8,row·8]` on a 128×128 sheet. `mirror` flips X (for a back-facing board side).
 */
export function signTextParts(lines: readonly string[], mirror: boolean, face: Direction): ModelPartDef[] {
  if (!GLYPH_W) return [];
  const parts: ModelPartDef[] = [];
  const n = Math.min(lines.length, 4);
  for (let i = 0; i < n; i++) {
    const line = lines[i] ?? "";
    if (!line) continue;
    const y = i * LINE_HEIGHT - 2 * LINE_HEIGHT;
    let penX = -lineWidth(line) / 2;
    for (let j = 0; j < line.length; j++) {
      const code = line.charCodeAt(j) & 0xff;
      const w = visualWidth(code);
      if (w > 0) {
        const col = code % 16, row = Math.floor(code / 16);
        const x0 = mirror ? -(penX + w) : penX;
        parts.push({
          name: `g${i}_${j}`,
          pivot: [0, 0, 0],
          cubes: [{ origin: [x0, y, 0], size: [w, CELL, 0], uv: [col * CELL, row * CELL], faces: [face] }],
        });
      }
      penX += advance(code);
    }
  }
  return parts;
}
