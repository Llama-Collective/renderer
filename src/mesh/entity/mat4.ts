// Minimal column-major 4x4 matrix builders for entity / block-entity model baking + posing.
// RENDERER_PLAN.md §18 (Phase 4.5).
//
// COLUMN-MAJOR, identical convention to camera/math.ts and the WGSL shaders: element (row r, col c)
// is m[c*4 + r]; `M * v` treats v as a column vector. Kept in mesh/ (not imported from camera/) so the
// GPU-free mesher stays self-contained and worker-safe. Only the ops the part-tree bake + per-entity
// model matrix need — no inverse/projection here.

export type Mat4 = Float32Array; // length 16, column-major

/** Radians per degree — the shared rotation-unit constant (entity/item/block-entity transforms). */
export const DEG = Math.PI / 180;

export function identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

/** o = a * b (column-major), `o` aliases neither input. */
export function multiply(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  multiplyInto(o, a, b);
  return o;
}

/** o = a * b into a CALLER-OWNED buffer (zero-alloc — `o` must alias neither input). Bit-identical to
 *  `multiply`; used by the memoized camera so the per-frame view-projection allocates nothing. */
export function multiplyInto(o: Mat4, a: Mat4, b: Mat4): void {
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
}

/** Chain-multiply left→right: mul(A, B, C) = A·B·C. */
export function mul(...ms: Mat4[]): Mat4 {
  let acc = ms[0] ?? identity();
  for (let i = 1; i < ms.length; i++) acc = multiply(acc, ms[i]);
  return acc;
}

/** o = ms[0]·ms[1]·…·ms[n-1] into a CALLER-OWNED `out`, bit-identical to `mul(...ms)` (the same left-fold
 *  of `multiplyInto`). `tmp` is a scratch buffer that must alias NEITHER `out` nor any `ms` entry; the two
 *  ping-pong so the final product lands in `out` (one final copy only for an odd product count). Zero-alloc
 *  — the entity build path drives this through `Mat4Frame.mul`. */
export function mulChainInto(out: Mat4, tmp: Mat4, ms: readonly Mat4[]): void {
  const n = ms.length;
  if (n === 0) { identityInto(out); return; }
  if (n === 1) { out.set(ms[0]); return; }
  let dst = out, src = tmp;             // multiply into `dst`, reading the prior accumulator from `src`
  multiplyInto(dst, ms[0], ms[1]);      // acc_1 → out
  for (let i = 2; i < n; i++) {
    const t = dst; dst = src; src = t;  // swap: read previous result, write the other buffer
    multiplyInto(dst, src, ms[i]);      // dst aliases neither src nor ms[i] (both distinct scratch/inputs)
  }
  if (dst !== out) out.set(dst);        // odd #products → result ended in `tmp`; copy to `out`
}

// ── In-place builders (zero-alloc; each FULLY overwrites all 16 elements, bit-identical to the allocating
//    sibling above so a reused arena buffer carries no stale state). ───────────────────────────────────

export function identityInto(o: Mat4): void {
  o.fill(0);
  o[0] = o[5] = o[10] = o[15] = 1;
}

export function translationInto(o: Mat4, x: number, y: number, z: number): void {
  o.fill(0);
  o[0] = o[5] = o[10] = o[15] = 1;
  o[12] = x; o[13] = y; o[14] = z;
}

export function scalingInto(o: Mat4, x: number, y: number, z: number): void {
  o.fill(0);
  o[0] = x; o[5] = y; o[10] = z; o[15] = 1;
}

export function rotationXInto(o: Mat4, rad: number): void {
  const s = Math.sin(rad), c = Math.cos(rad);
  o.fill(0);
  o[0] = 1; o[15] = 1;
  o[5] = c; o[6] = s; o[9] = -s; o[10] = c;
}

export function rotationYInto(o: Mat4, rad: number): void {
  const s = Math.sin(rad), c = Math.cos(rad);
  o.fill(0);
  o[5] = 1; o[15] = 1;
  o[0] = c; o[2] = -s; o[8] = s; o[10] = c;
}

export function rotationZInto(o: Mat4, rad: number): void {
  const s = Math.sin(rad), c = Math.cos(rad);
  o.fill(0);
  o[10] = 1; o[15] = 1;
  o[0] = c; o[1] = s; o[4] = -s; o[5] = c;
}

/**
 * A per-frame pool of reused mat4 buffers for the entity build path (A3). `reset()` (once per frame, before
 * building draws) rewinds the cursor; each builder hands out the next slot. The pool grows to the frame's
 * high-water mark then reuses those buffers forever — so a steady-state frame allocates nothing. SAFE
 * because every matrix it hands out is consumed (copied into the entity uniform buffer by EntityRenderer)
 * within the SAME frame, before the next `reset()`. Builders (`T`/`S`/`Rx`/`Ry`/`Rz`/`mul`) mirror the
 * free-function builders bit-for-bit, so swapping `translation(…)`→`F.T(…)` is byte-identical. `onGrow`
 * fires only when a NEW buffer is actually allocated (X5 regression gauge — 0 in steady state).
 */
export class Mat4Frame {
  private readonly pool: Mat4[] = [];
  private cursor = 0;
  /** Dedicated scratch for `mul`'s ping-pong tmp — allocated once, never handed out, so it can't alias an
   *  argument; `mulChainInto` fully consumes it before returning, so nested `mul` calls share it safely. */
  private readonly tmp = new Float32Array(16);
  /** PREC-1 / M6: the camera anchor subtracted by `Tw` (the OUTERMOST world-placement translation). Built in
   *  DOUBLE before the f32 store, so an entity at extreme world coords lands near the camera with no f32
   *  precision loss (the renderer then uses a translation-free viewProjRel). [0,0,0] ⇒ `Tw` === `T` (byte-
   *  identical, the camera-relative-OFF path). Set per frame by `setAnchor`. */
  private ax = 0; private ay = 0; private az = 0;

  constructor(private readonly onGrow?: () => void) {}

  /** Rewind for a new frame — every previously handed-out matrix is now free to overwrite. */
  reset(): void {
    this.cursor = 0;
  }

  /** M6: set the per-frame world anchor `Tw` subtracts (the rounded/exact camera position when camera-relative
   *  is on, else 0). Doubles, so `world − anchor` is exact even at 30M before the f32 truncation. */
  setAnchor(ax: number, ay: number, az: number): void {
    this.ax = ax; this.ay = ay; this.az = az;
  }

  /** Next free (uninitialized) slot; the caller fills all 16 elements. */
  next(): Mat4 {
    let m = this.pool[this.cursor];
    if (m === undefined) {
      m = new Float32Array(16);
      this.pool[this.cursor] = m;
      this.onGrow?.();
    }
    this.cursor++;
    return m;
  }

  T(x: number, y: number, z: number): Mat4 { const m = this.next(); translationInto(m, x, y, z); return m; }
  /** World-placement translation (M6): `T(x − anchor)`, the subtraction done in DOUBLE before the f32 store.
   *  Use for the OUTERMOST entity-placement translation so the whole model is built camera-relative (every
   *  downstream part offset then stays small → f32-exact). With anchor [0,0,0] it is byte-identical to `T`. */
  Tw(x: number, y: number, z: number): Mat4 { const m = this.next(); translationInto(m, x - this.ax, y - this.ay, z - this.az); return m; }
  S(x: number, y: number, z: number): Mat4 { const m = this.next(); scalingInto(m, x, y, z); return m; }
  Rx(rad: number): Mat4 { const m = this.next(); rotationXInto(m, rad); return m; }
  Ry(rad: number): Mat4 { const m = this.next(); rotationYInto(m, rad); return m; }
  Rz(rad: number): Mat4 { const m = this.next(); rotationZInto(m, rad); return m; }

  /** Chain product into a fresh slot, bit-identical to `mul(...ms)`. */
  mul(...ms: Mat4[]): Mat4 {
    const out = this.next();
    mulChainInto(out, this.tmp, ms);
    return out;
  }
}

export function translation(x: number, y: number, z: number): Mat4 {
  const m = identity();
  m[12] = x;
  m[13] = y;
  m[14] = z;
  return m;
}

export function scaling(x: number, y: number, z: number): Mat4 {
  const m = new Float32Array(16);
  m[0] = x;
  m[5] = y;
  m[10] = z;
  m[15] = 1;
  return m;
}

export function rotationX(rad: number): Mat4 {
  const s = Math.sin(rad);
  const c = Math.cos(rad);
  const m = identity();
  m[5] = c; m[6] = s; m[9] = -s; m[10] = c;
  return m;
}

export function rotationY(rad: number): Mat4 {
  const s = Math.sin(rad);
  const c = Math.cos(rad);
  const m = identity();
  m[0] = c; m[2] = -s; m[8] = s; m[10] = c;
  return m;
}

export function rotationZ(rad: number): Mat4 {
  const s = Math.sin(rad);
  const c = Math.cos(rad);
  const m = identity();
  m[0] = c; m[1] = s; m[4] = -s; m[5] = c;
  return m;
}

/** Rotation by `rad` about an arbitrary axis (Rodrigues; axis normalized here). Column-major. */
export function rotationAxis(rad: number, ax: number, ay: number, az: number): Mat4 {
  const len = Math.hypot(ax, ay, az) || 1;
  const x = ax / len, y = ay / len, z = az / len;
  const s = Math.sin(rad), c = Math.cos(rad), t = 1 - c;
  const m = identity();
  m[0] = c + x * x * t;     m[1] = y * x * t + z * s; m[2] = z * x * t - y * s;
  m[4] = x * y * t - z * s; m[5] = c + y * y * t;     m[6] = z * y * t + x * s;
  m[8] = x * z * t + y * s; m[9] = y * z * t - x * s; m[10] = c + z * z * t;
  return m;
}

/** Transform a point (w=1): returns [x,y,z]. */
export function transformPoint(m: Mat4, x: number, y: number, z: number): [number, number, number] {
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ];
}

/** Transform a direction (w=0; ignores translation). NOT renormalized — caller decides. */
export function transformDir(m: Mat4, x: number, y: number, z: number): [number, number, number] {
  return [
    m[0] * x + m[4] * y + m[8] * z,
    m[1] * x + m[5] * y + m[9] * z,
    m[2] * x + m[6] * y + m[10] * z,
  ];
}
