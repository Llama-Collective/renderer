// Camera/frustum view+projection math. RENDERER_PLAN.md §3, §14.
//
// The column-major `Mat4` + `identity`/`multiply` primitives are the GPU-free ones from
// mesh/entity/mat4.ts (single source of truth — column-major, `M*v` column-vector, matches the WGSL
// shaders); this module re-exports them and layers the camera-only ops (lookAt, perspectiveZO, vec3)
// on top. Right-handed view space (camera looks down -Z); projection maps depth to [0,1] (WebGPU clip),
// NOT GL's [-1,1] — hence the zero-to-one perspective + clear-to-1.0 with a `Less` test.

import type { Vec3 } from "../types";
import { identity, multiply, multiplyInto, type Mat4 } from "../mesh/entity/mat4";

export { identity, multiply, multiplyInto };
export type { Mat4 };

export function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
export function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

/** Right-handed look-at view matrix (camera at `eye` looking at `center`). */
export function lookAt(eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  const m = identity();
  lookAtInto(m, eye, center, up);
  return m;
}

/** Right-handed look-at into a CALLER-OWNED buffer (zero-alloc — inlines the basis math, no Vec3 temps).
 *  Bit-identical to `lookAt` (same float ops, same order); used by the memoized camera. */
export function lookAtInto(out: Mat4, eye: Vec3, center: Vec3, up: Vec3): void {
  // forward = normalize(center - eye)
  let fx = center[0] - eye[0], fy = center[1] - eye[1], fz = center[2] - eye[2];
  const fl = Math.hypot(fx, fy, fz) || 1; fx /= fl; fy /= fl; fz /= fl;
  // right = normalize(cross(forward, up))
  let sx = fy * up[2] - fz * up[1], sy = fz * up[0] - fx * up[2], sz = fx * up[1] - fy * up[0];
  const sl = Math.hypot(sx, sy, sz) || 1; sx /= sl; sy /= sl; sz /= sl;
  // u = cross(right, forward)
  const ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;
  // Rows are [right, up, -forward]; translation is -(basis · eye).
  out[0] = sx; out[4] = sy; out[8] = sz; out[12] = -(sx * eye[0] + sy * eye[1] + sz * eye[2]);
  out[1] = ux; out[5] = uy; out[9] = uz; out[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
  out[2] = -fx; out[6] = -fy; out[10] = -fz; out[14] = fx * eye[0] + fy * eye[1] + fz * eye[2];
  out[3] = 0; out[7] = 0; out[11] = 0; out[15] = 1;
}

/**
 * Right-handed perspective with a zero-to-one depth range (WebGPU). `fovY` in radians.
 * Mirrors glam's `perspective_rh` / wgpu conventions.
 */
export function perspectiveZO(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const m = new Float32Array(16);
  perspectiveZOInto(m, fovY, aspect, near, far);
  return m;
}

/** `perspectiveZO` into a CALLER-OWNED buffer (zero-alloc, bit-identical). */
export function perspectiveZOInto(out: Mat4, fovY: number, aspect: number, near: number, far: number): void {
  out.fill(0);
  const f = 1 / Math.tan(fovY / 2);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far / (near - far);
  out[11] = -1;
  out[14] = (far * near) / (near - far);
}
