// Free-fly camera (WASD + mouse-drag look). RENDERER_PLAN.md §14. For interactive demos: position +
// yaw/pitch, move relative to facing, rotate by drag. Same `CameraView` surface the renderers consume.

import { identity, lookAtInto, multiplyInto, perspectiveZOInto, type Mat4 } from "./math";
import type { CameraView } from "./Camera";
import type { Vec3 } from "../types";

const HALF_PI = Math.PI / 2 - 0.01;
const WORLD_UP: Vec3 = [0, 1, 0];

export class FreeCamera implements CameraView {
  pos: Vec3 = [0, 2, 8];
  /** Yaw (radians); 0 looks toward -Z, increasing turns toward +X. */
  yaw = 0;
  /** Pitch (radians); clamped to avoid gimbal flip at the poles. */
  pitch = 0;

  fovY = (60 * Math.PI) / 180;
  aspect = 1;
  near = 0.05;
  far = 2000;

  // P1.3: memoized outputs (reused buffers) + the input snapshot that produced them — mirrors `Camera`, so the
  // 6–8 per-frame reads of forward/viewBasis/viewMatrix/projectionMatrix/viewProjection allocate NOTHING in
  // steady state (they re-derive only when a pose/projection input actually changes). The F1 bundle keys on the
  // viewProjection bits, which a stable cached buffer preserves. Consumers read these synchronously within a
  // frame (never retained by reference across frames), so returning the reused buffers is safe.
  private readonly _fwd: [number, number, number] = [0, 0, 0];
  private readonly _center: [number, number, number] = [0, 0, 0];
  private readonly _view = identity();
  private readonly _proj = identity();
  private readonly _vp = identity();
  private readonly _basis = { right: [0, 0, 0] as [number, number, number], up: [0, 0, 0] as [number, number, number], forward: [0, 0, 0] as [number, number, number] };
  private _spx = NaN; private _spy = NaN; private _spz = NaN; private _syaw = NaN; private _spitch = NaN;
  private _sfov = NaN; private _saspect = NaN; private _snear = NaN; private _sfar = NaN;
  private _poseVer = 0;
  private _projVer = 0;
  private _vpPoseVer = -1; private _vpProjVer = -1;
  private _basisVer = -1;

  private syncPose(): void {
    const p = this.pos;
    if (this._spx === p[0] && this._spy === p[1] && this._spz === p[2] && this._syaw === this.yaw && this._spitch === this.pitch) return;
    this._spx = p[0]; this._spy = p[1]; this._spz = p[2]; this._syaw = this.yaw; this._spitch = this.pitch;
    const cp = Math.cos(this.pitch);
    this._fwd[0] = Math.sin(this.yaw) * cp;
    this._fwd[1] = Math.sin(this.pitch);
    this._fwd[2] = -Math.cos(this.yaw) * cp;
    this._center[0] = p[0] + this._fwd[0]; this._center[1] = p[1] + this._fwd[1]; this._center[2] = p[2] + this._fwd[2];
    lookAtInto(this._view, p, this._center, WORLD_UP);
    this._poseVer++;
  }

  private syncProj(): void {
    if (this._sfov === this.fovY && this._saspect === this.aspect && this._snear === this.near && this._sfar === this.far) return;
    this._sfov = this.fovY; this._saspect = this.aspect; this._snear = this.near; this._sfar = this.far;
    perspectiveZOInto(this._proj, this.fovY, this.aspect, this.near, this.far);
    this._projVer++;
  }

  get position(): Vec3 {
    return this.pos;
  }

  /** Unit forward direction from yaw/pitch (memoized; reused buffer). */
  forward(): Vec3 {
    this.syncPose();
    return this._fwd;
  }

  /** TR-1: right/up/forward unit basis (the `CameraView` surface the huge-section re-sort gate reads). */
  viewBasis(): { right: Vec3; up: Vec3; forward: Vec3 } {
    this.syncPose();
    if (this._poseVer !== this._basisVer) {
      this._basisVer = this._poseVer;
      const f = this._fwd, b = this._basis;
      // right = normalize(forward × worldUp) with worldUp = [0,1,0] ⇒ (-fz, 0, fx); up = right × forward.
      let rx = -f[2], ry = 0, rz = f[0];
      const rl = Math.hypot(rx, ry, rz) || 1; rx /= rl; ry /= rl; rz /= rl;
      b.right[0] = rx; b.right[1] = ry; b.right[2] = rz;
      b.up[0] = ry * f[2] - rz * f[1]; b.up[1] = rz * f[0] - rx * f[2]; b.up[2] = rx * f[1] - ry * f[0];
      b.forward[0] = f[0]; b.forward[1] = f[1]; b.forward[2] = f[2];
    }
    return this._basis;
  }

  /** PREC-1: the look-at view matrix (full eye translation baked into the translation column; memoized). */
  viewMatrix(): Mat4 {
    this.syncPose();
    return this._view;
  }

  /** PREC-1: the perspective projection matrix (memoized; reused buffer). */
  projectionMatrix(): Mat4 {
    this.syncProj();
    return this._proj;
  }

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

  /** Move relative to facing: +forward (W), +right (D), +up (space). */
  moveLocal(dForward: number, dRight: number, dUp: number): void {
    const f = this.forward();
    // right = normalize(forward × up); on the ground plane (ignore pitch for strafing feel).
    const rx = -f[2], rz = f[0];
    const rl = Math.hypot(rx, rz) || 1;
    this.pos = [
      this.pos[0] + f[0] * dForward + (rx / rl) * dRight,
      this.pos[1] + f[1] * dForward + dUp,
      this.pos[2] + f[2] * dForward + (rz / rl) * dRight,
    ];
  }

  /** Add to yaw/pitch (mouse drag); pitch clamped. */
  rotate(dYaw: number, dPitch: number): void {
    this.yaw += dYaw;
    this.pitch = Math.max(-HALF_PI, Math.min(HALF_PI, this.pitch + dPitch));
  }

  /** Point the camera at a world target from its current position (initial framing). */
  lookAt(target: Vec3): void {
    const dx = target[0] - this.pos[0];
    const dy = target[1] - this.pos[1];
    const dz = target[2] - this.pos[2];
    this.yaw = Math.atan2(dx, -dz);
    this.pitch = Math.atan2(dy, Math.hypot(dx, dz));
  }
}
