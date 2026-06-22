import { describe, expect, it } from "vitest";
import { Direction } from "../../../types";
import { FONT_SIZE, lineWidth, scanFontWidths, setFontWidths, signTextParts } from "./font";

/** A synthetic 128×128 ascii sheet with two glyphs of known width (the rest empty). */
function sheet(): Uint8Array {
  const rgba = new Uint8Array(FONT_SIZE * FONT_SIZE * 4);
  const setGlyph = (code: number, width: number): void => {
    const cx = (code % 16) * 8, cy = Math.floor(code / 16) * 8;
    for (let gx = 0; gx < width; gx++) for (let gy = 0; gy < 7; gy++) rgba[((cy + gy) * FONT_SIZE + (cx + gx)) * 4 + 3] = 255;
  };
  setGlyph(65, 5); // 'A' visual width 5
  setGlyph(66, 3); // 'B' visual width 3
  return rgba; // ' ' (32) left empty
}

describe("bitmap font (sign text)", () => {
  it("scans per-glyph visual widths (rightmost non-transparent column + 1)", () => {
    const w = scanFontWidths(sheet(), FONT_SIZE, FONT_SIZE);
    expect(w[65]).toBe(5);
    expect(w[66]).toBe(3);
    expect(w[32]).toBe(0); // empty cell → width 0
  });

  it("lays out centered glyph quads (advance = width+1, empty = 4)", () => {
    setFontWidths(scanFontWidths(sheet(), FONT_SIZE, FONT_SIZE));
    expect(lineWidth("AB")).toBe(10); // A=6, B=4
    expect(lineWidth("A B")).toBe(14); // A=6, space=4, B=4

    const parts = signTextParts(["AB"], false, Direction.North);
    expect(parts.length).toBe(2); // one quad per non-empty glyph
    const a = parts[0].cubes![0];
    expect(a.origin[0]).toBe(-5); // centered: pen starts at -width/2
    expect(a.size).toEqual([5, 8, 0]); // glyph width × cell height, flat
    expect(a.uv).toEqual([8, 32]); // 'A' → cell (col 1, row 4) → (8, 32)
    expect(a.faces).toEqual([Direction.North]);
  });

  it("emits no quads for blank lines and caps at 4 lines", () => {
    setFontWidths(scanFontWidths(sheet(), FONT_SIZE, FONT_SIZE));
    expect(signTextParts(["", "  "], false, Direction.North).length).toBe(0); // spaces have no quads
    expect(signTextParts(["A", "A", "A", "A", "A", "A"], false, Direction.North).length).toBe(4); // ≤4 lines
  });
});
