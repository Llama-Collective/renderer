// 3×3 matrix helpers for model baking. RENDERER_PLAN §24.5.
//
// Row-major `Mat3` (m[row*3 + col]); `transformPoint(m,v) = m·v` (column vector). Model space is
// vanilla's: +X east, +Y up, +Z south. Blockstate rotations are exact 90° integer matrices (no
// float fuzz); element rotations use a general axis-angle. Used for element rotation (+rescale),
// blockstate x/y/z rotation, cullface remap, and the uvlock face transform.

export type Vec3 = [number, number, number];
export type Mat3 = [number, number, number, number, number, number, number, number, number];

export function identity3(): Mat3 {
  return [1, 0, 0, 0, 1, 0, 0, 0, 1];
}

export function mul3(a: Mat3, b: Mat3): Mat3 {
  const o = identity3();
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    }
  }
  return o;
}

export function transformPoint3(m: Mat3, v: readonly [number, number, number]): Vec3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

export function transpose3(m: Mat3): Mat3 {
  return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}

export function isIdentity3(m: Mat3): boolean {
  const id = identity3();
  for (let i = 0; i < 9; i++) if (Math.abs(m[i] - id[i]) > 1e-9) return false;
  return true;
}

// --- Exact 90° rotations (n = number of 90° CCW steps; handles any integer degrees /90) ----------
function steps(degrees: number): number {
  return (((Math.round(degrees / 90) % 4) + 4) % 4);
}
/** Rotation about +X by `degrees` (multiple of 90), exact integer entries. */
export function rotX90(degrees: number): Mat3 {
  const [c, s] = cs(steps(degrees));
  return [1, 0, 0, 0, c, -s, 0, s, c];
}
export function rotY90(degrees: number): Mat3 {
  const [c, s] = cs(steps(degrees));
  return [c, 0, s, 0, 1, 0, -s, 0, c];
}
export function rotZ90(degrees: number): Mat3 {
  const [c, s] = cs(steps(degrees));
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}
function cs(n: number): [number, number] {
  return [[1, 0], [0, 1], [-1, 0], [0, -1]][n] as [number, number];
}

/**
 * Blockstate rotation: apply X then Y then Z (matrix Z·Y·X), exact for 90° steps. §24.5.
 *
 * The angles are NEGATED — this matches vanilla `BlockModelRotation` (`new Vector3f(-x, -y, 0)`) and the
 * old three.js renderer (Cubane: `rotateX(-angle)`/`rotateY(-angle)`). Without the negation directional
 * blocks render mirrored on the horizontal axes (east↔west swapped, since 90°↔270°) and wall/ceiling
 * attachments (levers, buttons) flip — the simulation stays correct, only the render was mirrored.
 */
export function blockstateMatrix(x: number, y: number, z: number): Mat3 {
  return mul3(rotZ90(-z), mul3(rotY90(-y), rotX90(-x)));
}

// --- General axis-angle (element rotation) + rescale --------------------------------------------
/** Rotation about a cardinal axis ("x"|"y"|"z") by `angle` radians (Rodrigues, but axis is unit). */
export function axisRotation(axis: "x" | "y" | "z", angle: number): Mat3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  if (axis === "x") return [1, 0, 0, 0, c, -s, 0, s, c];
  if (axis === "y") return [c, 0, s, 0, 1, 0, -s, 0, c];
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}

/**
 * Element-rotation matrix incl. vanilla `rescale`: scale each axis by `1 / max|column|` so the
 * rotated box still touches its original bounds (≈1/cos(22.5°) for a 22.5° rotation). §24.5.
 */
export function elementMatrix(axis: "x" | "y" | "z", angleDeg: number, rescale: boolean): Mat3 {
  const r = axisRotation(axis, (angleDeg * Math.PI) / 180);
  if (!rescale) return r;
  const sx = 1 / Math.max(Math.abs(r[0]), Math.abs(r[3]), Math.abs(r[6]));
  const sy = 1 / Math.max(Math.abs(r[1]), Math.abs(r[4]), Math.abs(r[7]));
  const sz = 1 / Math.max(Math.abs(r[2]), Math.abs(r[5]), Math.abs(r[8]));
  // r · diag(sx,sy,sz): scale the columns.
  return [r[0] * sx, r[1] * sy, r[2] * sz, r[3] * sx, r[4] * sy, r[5] * sz, r[6] * sx, r[7] * sy, r[8] * sz];
}
