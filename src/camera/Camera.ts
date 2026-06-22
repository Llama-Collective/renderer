// Orbit camera + matrices. RENDERER_PLAN.md §3 (frame loop), §14.
//
// A simple orbit camera (target + yaw/pitch/distance) is enough for the editor and the Phase-1
// demo. The matrices it produces are column-major and WebGPU-ready (see ./math) — upload
// `viewProjection()` straight into the camera uniform.

import type { Vec3 } from "../types";
import { identity, lookAtInto, multiplyInto, perspectiveZOInto, type Mat4 } from "./math";

export type { Mat4 } from "./math";

const WORLD_UP: Vec3 = [0, 1, 0];

/** The minimal camera surface the renderers consume (orbit `Camera` or free-fly `FreeCamera`). */
export interface CameraView {
  readonly position: Vec3;
  aspect: number;
  viewProjection(): Mat4;
  /** TR-1: the right/up/forward unit basis (memoized, zero-alloc on `Camera`). The huge-section re-sort gate
   *  reads `.forward` as its angle reference. Optional so callers that never touch the gate need not provide it
   *  — `TerrainRenderer.render` reads it via `viewBasis?.()` and falls back to the world-up axis when absent. */
  viewBasis?(): { right: Vec3; up: Vec3; forward: Vec3 };
  /** PREC-1: the view + projection matrices SEPARATELY (the world-space `viewProjection()` stays shared by the
   *  frustum/occlusion BFS/overlay — TRAP PREC-1.B). The camera-relative path reads these to build a
   *  TRANSLATION-FREE `viewProjRel = proj · viewRel` for the terrain/entity vertex shader WITHOUT disturbing
   *  the world matrix. Optional: a camera that omits them disables the precision path (renderer falls back to
   *  the world-space vp + camOrigin=0, i.e. today's exact transform). Reused buffers (read within the frame). */
  viewMatrix?(): Mat4;
  projectionMatrix?(): Mat4;
}

export class Camera implements CameraView {
  /** Point the camera orbits / looks at (world space). */
  target: Vec3 = [0, 0, 0];
  /** Orbit distance from `target`. */
  distance = 32;
  /** Azimuth (radians, around +Y). */
  yaw = Math.PI * 0.25;
  /** Elevation (radians; keep within (-π/2, π/2) to avoid a degenerate up vector). */
  pitch = Math.PI * 0.2;

  fovY = (60 * Math.PI) / 180;
  aspect = 1;
  near = 0.1;
  far = 1000;

  // Memoized outputs (reused buffers) + the input snapshot that produced them, so the 3–4 per-frame reads
  // of position/viewProjection allocate NOTHING in steady state — recomputed only when a pose/projection
  // input actually changes (the render loop's zero-alloc convention; F1 bundle keys on viewProj bits, which
  // a stable cached buffer preserves). Consumers read these synchronously within a frame (never retained by
  // reference across frames — verified), so returning the reused buffers is safe.
  private readonly _pos: [number, number, number] = [0, 0, 0];
  private readonly _view = identity();
  private readonly _proj = identity();
  private readonly _vp = identity();
  private readonly _basis = { right: [0, 0, 0] as [number, number, number], up: [0, 0, 0] as [number, number, number], forward: [0, 0, 0] as [number, number, number] };
  // EXACT input snapshots (no hashing → no collision/stale-matrix risk) + monotonic version counters that
  // invalidate the derived view-projection / basis only when their inputs actually changed.
  private _st0 = NaN; private _st1 = NaN; private _st2 = NaN; private _syaw = NaN; private _spitch = NaN; private _sdist = NaN;
  private _sfov = NaN; private _saspect = NaN; private _snear = NaN; private _sfar = NaN;
  private _poseVer = 0;
  private _projVer = 0;
  private _vpPoseVer = -1; private _vpProjVer = -1;
  private _basisVer = -1;

  private syncPose(): void {
    const t = this.target;
    if (this._st0 === t[0] && this._st1 === t[1] && this._st2 === t[2] && this._syaw === this.yaw && this._spitch === this.pitch && this._sdist === this.distance) return;
    this._st0 = t[0]; this._st1 = t[1]; this._st2 = t[2]; this._syaw = this.yaw; this._spitch = this.pitch; this._sdist = this.distance;
    const cp = Math.cos(this.pitch);
    this._pos[0] = t[0] + this.distance * cp * Math.sin(this.yaw);
    this._pos[1] = t[1] + this.distance * Math.sin(this.pitch);
    this._pos[2] = t[2] + this.distance * cp * Math.cos(this.yaw);
    lookAtInto(this._view, this._pos, t, WORLD_UP);
    this._poseVer++;
  }
  private syncProj(): void {
    if (this._sfov === this.fovY && this._saspect === this.aspect && this._snear === this.near && this._sfar === this.far) return;
    this._sfov = this.fovY; this._saspect = this.aspect; this._snear = this.near; this._sfar = this.far;
    perspectiveZOInto(this._proj, this.fovY, this.aspect, this.near, this.far);
    this._projVer++;
  }

  /** Eye position derived from target + yaw/pitch/distance (memoized; reused buffer). */
  get position(): Vec3 {
    this.syncPose();
    return this._pos;
  }

  /** Monotonic token that changes whenever ANY pose (target/yaw/pitch/distance) or projection
   *  (fov/aspect/near/far) input changes — the render-on-demand loop reads it to detect camera movement
   *  without diffing matrices. Both sub-counters are strictly increasing, so their sum is too; an equality
   *  check against a stored value therefore catches any change. Ensures the lazy sync ran first (cheap —
   *  a no-op when nothing changed). */
  get changeVersion(): number {
    this.syncPose();
    this.syncProj();
    return this._poseVer + this._projVer;
  }

  viewMatrix(): Mat4 {
    this.syncPose();
    return this._view;
  }

  projectionMatrix(): Mat4 {
    this.syncProj();
    return this._proj;
  }

  /** proj * view — upload verbatim as the camera uniform (memoized; reused buffer). */
  viewProjection(): Mat4 {
    this.syncPose();
    this.syncProj();
    if (this._poseVer !== this._vpPoseVer || this._projVer !== this._vpProjVer) {
      multiplyInto(this._vp, this._proj, this._view);
      this._vpPoseVer = this._poseVer;
      this._vpProjVer = this._projVer;
    }
    return this._vp;
  }

  /** Right/up/forward unit basis from the current pose (for the on-screen compass; memoized). */
  viewBasis(): { right: Vec3; up: Vec3; forward: Vec3 } {
    this.syncPose();
    if (this._poseVer !== this._basisVer) {
      this._basisVer = this._poseVer;
      const eye = this._pos, t = this.target, b = this._basis;
      let fx = t[0] - eye[0], fy = t[1] - eye[1], fz = t[2] - eye[2];
      const fl = Math.hypot(fx, fy, fz) || 1; fx /= fl; fy /= fl; fz /= fl;
      let rx = fy * WORLD_UP[2] - fz * WORLD_UP[1], ry = fz * WORLD_UP[0] - fx * WORLD_UP[2], rz = fx * WORLD_UP[1] - fy * WORLD_UP[0];
      const rl = Math.hypot(rx, ry, rz) || 1; rx /= rl; ry /= rl; rz /= rl;
      b.forward[0] = fx; b.forward[1] = fy; b.forward[2] = fz;
      b.right[0] = rx; b.right[1] = ry; b.right[2] = rz;
      b.up[0] = ry * fz - rz * fy; b.up[1] = rz * fx - rx * fz; b.up[2] = rx * fy - ry * fx;
    }
    return this._basis;
  }

  /** Frame an AABB: center the target and back off so the bounding sphere fits the view. */
  frameBounds(min: Vec3, max: Vec3): void {
    this.target = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    const r = 0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
    const hfov = 2 * Math.atan(Math.tan(this.fovY / 2) * Math.max(this.aspect, 0.0001));
    const fit = Math.max(0.1, Math.min(this.fovY, hfov) / 2);
    this.distance = r / Math.sin(Math.min(fit, Math.PI / 2 - 0.01)) + r;
  }
}
