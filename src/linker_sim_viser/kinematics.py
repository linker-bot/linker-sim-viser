"""Arm forward and inverse kinematics shared by conversion and the viewer."""

from __future__ import annotations

import numpy as np
import yourdfpy

from .pose_math import (
    inverse_quaternion_wxyz,
    multiply_quaternions_wxyz,
    normalize_quaternion_wxyz,
    quaternion_wxyz_from_matrix,
)


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for a unit axis and angle in radians."""
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _orientation_error_rotvec(
    target_wxyz: np.ndarray, current_rotation: np.ndarray
) -> np.ndarray:
    """Return the shortest target-minus-current error as a rotation vector.

    Quaternion composition avoids Euler-angle singularities. The resulting
    rotation vector supplies the three angular rows of the DLS residual.
    """
    current_wxyz = quaternion_wxyz_from_matrix(current_rotation)
    error = multiply_quaternions_wxyz(
        normalize_quaternion_wxyz(target_wxyz),
        inverse_quaternion_wxyz(current_wxyz),
    )
    if error[0] < 0.0:
        error = -error

    sin_half = float(np.linalg.norm(error[1:]))
    if sin_half <= 1e-12:
        return 2.0 * error[1:]
    angle = 2.0 * np.arctan2(sin_half, max(float(error[0]), 0.0))
    return error[1:] * (angle / sin_half)


class ArmChain:
    """DLS FK/IK for one 7-DoF arm chain in a workstation URDF.

    This is the solver previously embedded in ``umi_mcap_to_episode.py``.
    It preserves the geometric Jacobian, joint-limit clipping, and frame-to-
    frame warm start while accepting an explicit end-effector link.
    """

    def __init__(
        self, urdf: yourdfpy.URDF, ee_link: str, damping: float = 0.05
    ) -> None:
        if not np.isfinite(damping) or damping <= 0.0:
            raise ValueError("damping must be a positive finite value")
        self.damping = float(damping)

        joints_by_child = {joint.child: joint for joint in urdf.robot.joints}
        if ee_link not in joints_by_child:
            raise ValueError(f"end-effector link {ee_link!r} is not in the URDF chain")

        chain = []
        link = ee_link
        while link in joints_by_child:
            joint = joints_by_child[link]
            chain.append(joint)
            link = joint.parent
        chain.reverse()

        self._steps: list[tuple[np.ndarray, np.ndarray | None]] = []
        joint_names: list[str] = []
        lower: list[float] = []
        upper: list[float] = []
        for joint in chain:
            axis = None
            if joint.type in ("revolute", "continuous"):
                axis = np.asarray(joint.axis, dtype=np.float64)
                axis_norm = float(np.linalg.norm(axis))
                if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
                    raise ValueError(f"joint {joint.name!r} has an invalid rotation axis")
                axis = axis / axis_norm
                joint_names.append(joint.name)
                if joint.limit is None:
                    lo, hi = -np.pi, np.pi
                else:
                    lo, hi = joint.limit.lower, joint.limit.upper
                    lo = -np.pi if lo is None else float(lo)
                    hi = np.pi if hi is None else float(hi)
                lower.append(lo)
                upper.append(hi)
            self._steps.append((np.asarray(joint.origin, dtype=np.float64), axis))

        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.joint_names = tuple(joint_names)
        self.n = len(self.lower)
        if self.n != 7:
            raise ValueError(
                f"end-effector link {ee_link!r} must have 7 revolute joints; found {self.n}"
            )

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return EE transform plus world joint axes and origins."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.n,) or not np.all(np.isfinite(q)):
            raise ValueError(f"q must contain {self.n} finite joint values")

        transform = np.eye(4)
        axes = np.empty((self.n, 3))
        origins = np.empty((self.n, 3))
        actuated_index = 0
        for origin, axis in self._steps:
            transform = transform @ origin
            if axis is not None:
                axes[actuated_index] = transform[:3, :3] @ axis
                origins[actuated_index] = transform[:3, 3]
                rotation = np.eye(4)
                rotation[:3, :3] = _rodrigues(axis, q[actuated_index])
                transform = transform @ rotation
                actuated_index += 1
        return transform, axes, origins

    def tool0_default(self) -> np.ndarray:
        """Return the end-effector transform at zero joint angles."""
        return self.fk(np.zeros(self.n))[0]

    def solve(
        self,
        target: np.ndarray,
        q_init: np.ndarray,
        max_iters: int,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
        max_step: float = 0.3,
    ) -> tuple[np.ndarray, float, float]:
        """Solve one wxyz target pose; return joints and position/angle errors."""
        q, position_error, rotation_error, _ = self.solve_detailed(
            target,
            q_init,
            max_iters,
            tol_pos=tol_pos,
            tol_rot=tol_rot,
            max_step=max_step,
        )
        return q, position_error, rotation_error

    def solve_detailed(
        self,
        target: np.ndarray,
        q_init: np.ndarray,
        max_iters: int,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
        max_step: float = 0.3,
    ) -> tuple[np.ndarray, float, float, bool]:
        """Solve one pose and report whether the final residual meets tolerances."""
        target = np.asarray(target, dtype=np.float64)
        q_init = np.asarray(q_init, dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("target must contain finite xyz and wxyz values")
        if q_init.shape != (self.n,) or not np.all(np.isfinite(q_init)):
            raise ValueError(f"q_init must contain {self.n} finite joint values")
        if max_iters < 1:
            raise ValueError("max_iters must be at least 1")
        tolerances = np.asarray([tol_pos, tol_rot, max_step], dtype=np.float64)
        if not np.all(np.isfinite(tolerances)) or np.any(tolerances <= 0.0):
            raise ValueError("IK tolerances and max_step must be positive and finite")

        target_position = target[:3]
        target_quaternion = normalize_quaternion_wxyz(target[3:7])
        q = np.clip(q_init.copy(), self.lower, self.upper)
        pos_error_norm = rot_error_norm = 0.0
        for _ in range(max_iters):
            transform, axes, origins = self.fk(q)
            position = transform[:3, 3]
            pos_error = target_position - position
            rot_error = _orientation_error_rotvec(
                target_quaternion, transform[:3, :3]
            )
            pos_error_norm = float(np.linalg.norm(pos_error))
            rot_error_norm = float(np.linalg.norm(rot_error))
            if pos_error_norm < tol_pos and rot_error_norm < tol_rot:
                break

            jacobian = np.empty((6, self.n))
            jacobian[:3, :] = np.cross(axes, position - origins).T
            jacobian[3:, :] = axes.T
            residual = np.concatenate([pos_error, rot_error])
            delta_q = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + self.damping**2 * np.eye(6), residual
            )
            step_norm = float(np.linalg.norm(delta_q))
            if step_norm > max_step:
                delta_q *= max_step / step_norm
            q = np.clip(q + delta_q, self.lower, self.upper)

        # Re-evaluate after the final update so returned residuals describe the
        # returned q, including when convergence happens on the last iteration.
        transform, _, _ = self.fk(q)
        final_position_error = target_position - transform[:3, 3]
        final_rotation_error = _orientation_error_rotvec(
            target_quaternion, transform[:3, :3]
        )
        pos_error_norm = float(np.linalg.norm(final_position_error))
        rot_error_norm = float(np.linalg.norm(final_rotation_error))
        converged = pos_error_norm <= tol_pos and rot_error_norm <= tol_rot
        return q, pos_error_norm, rot_error_norm, converged

    def track(
        self,
        arm_pose: np.ndarray,
        warmup: int,
        iters: int,
        q_init: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve a pose sequence with each frame warm-started by the previous."""
        q, position_errors, rotation_errors, _ = self.track_detailed(
            arm_pose, warmup, iters, q_init=q_init
        )
        return q, position_errors, rotation_errors

    def track_detailed(
        self,
        arm_pose: np.ndarray,
        warmup: int,
        iters: int,
        q_init: np.ndarray | None = None,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Track a pose sequence and return a per-frame convergence flag."""
        arm_pose = np.asarray(arm_pose, dtype=np.float64)
        if arm_pose.ndim != 2 or arm_pose.shape[1] != 7:
            raise ValueError("arm_pose must have shape (frames, 7) with wxyz quaternions")
        if len(arm_pose) == 0:
            raise ValueError("arm_pose must contain at least one frame")
        if warmup < 1 or iters < 1:
            raise ValueError("warmup and iters must be at least 1")

        n_frames = len(arm_pose)
        arm_q = np.zeros((n_frames, self.n))
        pos_errors = np.zeros(n_frames)
        rot_errors = np.zeros(n_frames)
        converged = np.zeros(n_frames, dtype=bool)
        q = np.zeros(self.n) if q_init is None else np.asarray(q_init, dtype=np.float64)
        if q.shape != (self.n,) or not np.all(np.isfinite(q)):
            raise ValueError(f"q_init must contain {self.n} finite joint values")
        for frame in range(n_frames):
            q, pos_errors[frame], rot_errors[frame], converged[frame] = self.solve_detailed(
                arm_pose[frame],
                q,
                warmup if frame == 0 else iters,
                tol_pos=tol_pos,
                tol_rot=tol_rot,
            )
            arm_q[frame] = q
        return arm_q, pos_errors, rot_errors, converged
