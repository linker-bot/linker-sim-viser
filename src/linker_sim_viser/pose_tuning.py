"""Prepare and solve the arm target poses used by Pose Tuning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yourdfpy

from .config import PoseTuningConfig, RobotConfig
from .episode import Episode
from .kinematics import ArmChain
from .pose_math import (
    apply_transform_to_pose,
    inverse_quaternion_wxyz,
    matrix_to_pose,
    multiply_quaternions_wxyz,
    normalize_quaternion_wxyz,
    pose_to_matrix,
    quaternion_wxyz_to_matrix,
)


@dataclass(frozen=True)
class TrajectoryIKValidation:
    """Threshold summary; listed frame indices are zero-based."""

    mean_position_error_mm: float
    max_position_error_mm: float
    mean_angle_error_deg: float
    max_angle_error_deg: float
    position_limit_frames: tuple[int, ...]
    angle_limit_frames: tuple[int, ...]
    unconverged_frames: tuple[int, ...]
    passed: bool


@dataclass(frozen=True)
class TrajectoryIKResult:
    """Solved joint trajectory plus the residuals used for validation."""

    joint_positions: np.ndarray
    position_errors_m: np.ndarray
    angle_errors_rad: np.ndarray
    converged: np.ndarray
    validation: TrajectoryIKValidation


@dataclass
class PoseTuningModel:
    """Episode-specific target trajectory and reusable arm IK chain."""

    side: str
    chain: ArmChain
    baseline_poses: np.ndarray
    baseline_joint_positions: np.ndarray
    target_source: str
    config: PoseTuningConfig

    @classmethod
    def from_episode(
        cls,
        robot: RobotConfig,
        episode: Episode,
        urdf: yourdfpy.URDF,
        side: str,
        config: PoseTuningConfig | None = None,
    ) -> PoseTuningModel:
        """Use stored absolute arm poses, or derive them from qpos with FK."""
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        ee_link = robot.ik_frames.get(side)
        if not ee_link:
            raise ValueError(f"robot config does not define an IK frame for {side!r}")

        config = config or PoseTuningConfig()
        chain = ArmChain(urdf, ee_link, damping=config.damping)
        missing = [name for name in chain.joint_names if name not in episode.joint_positions]
        if missing:
            raise ValueError(
                f"episode is missing {side} arm joints required by IK: {', '.join(missing)}"
            )

        joint_positions = np.stack(
            [episode.joint_positions[name] for name in chain.joint_names], axis=1
        ).astype(np.float64, copy=False)
        if joint_positions.shape != (episode.n_frames, chain.n):
            raise ValueError(
                f"{side} arm joint data must have shape "
                f"({episode.n_frames}, {chain.n})"
            )
        if not np.all(np.isfinite(joint_positions)):
            raise ValueError(f"{side} arm joint data contains non-finite values")

        if side in episode.arm_poses:
            poses = np.asarray(episode.arm_poses[side], dtype=np.float64)
            if poses.shape != (episode.n_frames, 7) or not np.all(np.isfinite(poses)):
                raise ValueError(
                    f"stored {side} arm target poses must have shape "
                    f"({episode.n_frames}, 7) and finite values"
                )
            poses = poses.copy()
            for frame in range(episode.n_frames):
                poses[frame, 3:] = normalize_quaternion_wxyz(poses[frame, 3:])
            source = f"arm_{side}"
        else:
            poses = np.empty((episode.n_frames, 7), dtype=np.float64)
            for frame, q in enumerate(joint_positions):
                position, quaternion = matrix_to_pose(chain.fk(q)[0])
                poses[frame, :3] = position
                poses[frame, 3:] = quaternion
            source = "qpos FK"

        return cls(
            side=side,
            chain=chain,
            baseline_poses=poses,
            baseline_joint_positions=joint_positions.copy(),
            target_source=source,
            config=config,
        )

    def reanchor(
        self, new_initial_position: np.ndarray, new_initial_quaternion: np.ndarray
    ) -> np.ndarray:
        """Move the whole target path so its first pose equals the requested pose."""
        if len(self.baseline_poses) == 0:
            raise ValueError("cannot reanchor an empty pose trajectory")
        new_position = np.asarray(new_initial_position, dtype=np.float64)
        if new_position.shape != (3,) or not np.all(np.isfinite(new_position)):
            raise ValueError("new initial position must contain three finite values")
        new_quaternion = normalize_quaternion_wxyz(new_initial_quaternion)

        old_position = self.baseline_poses[0, :3]
        old_quaternion = self.baseline_poses[0, 3:]
        delta_quaternion = multiply_quaternions_wxyz(
            new_quaternion, inverse_quaternion_wxyz(old_quaternion)
        )
        delta_rotation = quaternion_wxyz_to_matrix(delta_quaternion)
        delta_position = new_position - delta_rotation @ old_position
        return self._apply_world_delta(delta_position, delta_quaternion)

    def reanchor_at_frame(
        self,
        frame: int,
        new_position: np.ndarray,
        new_quaternion: np.ndarray,
        source_poses: np.ndarray | None = None,
        reference_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reanchor the full path at ``frame`` using an FK or path reference.

        On the first edit, the reference is the recorded frame's actual FK pose
        and the transformed path is the original episode target. Later edits
        can pass the already adjusted path as ``source_poses`` so edits remain
        relative to the path currently shown in the viewer.
        """
        if frame < 0 or frame >= len(self.baseline_poses):
            raise IndexError(f"frame {frame} is outside the episode")
        position = np.asarray(new_position, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("new frame position must contain three finite values")
        quaternion = normalize_quaternion_wxyz(new_quaternion)
        if source_poses is None:
            source = self.baseline_poses
        else:
            source = np.asarray(source_poses, dtype=np.float64)
            if source.shape != self.baseline_poses.shape or not np.all(np.isfinite(source)):
                raise ValueError("source pose array must match the baseline trajectory")
        if reference_pose is None:
            reference = (
                self.forward_pose(self.baseline_joint_positions[frame])
                if source_poses is None
                else source[frame]
            )
        else:
            reference = np.asarray(reference_pose, dtype=np.float64)
            if reference.shape != (7,) or not np.all(np.isfinite(reference)):
                raise ValueError("reference pose must contain seven finite values")
        old_position = reference[:3]
        old_quaternion = normalize_quaternion_wxyz(reference[3:])
        delta_quaternion = multiply_quaternions_wxyz(
            quaternion, inverse_quaternion_wxyz(old_quaternion)
        )
        delta_rotation = quaternion_wxyz_to_matrix(delta_quaternion)
        delta_position = position - delta_rotation @ old_position
        return self._apply_world_delta(delta_position, delta_quaternion, source)

    def forward_pose(self, joint_positions: np.ndarray) -> np.ndarray:
        """Return the actual FK end-effector pose as ``xyz + wxyz``."""
        position, quaternion = matrix_to_pose(self.chain.fk(joint_positions)[0])
        return np.concatenate((position, quaternion))

    def _apply_world_delta(
        self,
        delta_position: np.ndarray,
        delta_quaternion: np.ndarray,
        source_poses: np.ndarray | None = None,
    ) -> np.ndarray:
        delta_transform = pose_to_matrix(delta_position, delta_quaternion)
        source = self.baseline_poses if source_poses is None else source_poses

        adjusted = np.empty_like(source)
        for frame, pose in enumerate(source):
            position, quaternion = apply_transform_to_pose(
                delta_transform, pose[:3], pose[3:]
            )
            adjusted[frame, :3] = position
            adjusted[frame, 3:] = quaternion
        return adjusted

    def preview(
        self, target_poses: np.ndarray, frame: int, max_iters: int | None = None
    ) -> tuple[np.ndarray, float, float]:
        """Solve one frame, seeded from that frame's original arm joints."""
        targets = np.asarray(target_poses, dtype=np.float64)
        if targets.shape != self.baseline_poses.shape:
            raise ValueError("target pose array must match the baseline trajectory shape")
        if frame < 0 or frame >= len(targets):
            raise IndexError(f"frame {frame} is outside the episode")
        return self.chain.solve(
            targets[frame],
            self.baseline_joint_positions[frame],
            max_iters=(
                self.config.preview_iterations if max_iters is None else max_iters
            ),
        )

    def recompute(
        self,
        target_poses: np.ndarray,
    ) -> TrajectoryIKResult:
        """Solve and validate every pose, warm-starting from the previous frame."""
        config = self.config
        targets = np.asarray(target_poses, dtype=np.float64)
        if targets.shape != self.baseline_poses.shape:
            raise ValueError("target pose array must match the baseline trajectory shape")
        if not np.all(np.isfinite(targets)):
            raise ValueError("target pose array contains non-finite values")

        (
            joint_positions,
            position_errors_m,
            angle_errors_rad,
            converged,
        ) = self.chain.track_detailed(
            targets,
            warmup=config.first_frame_iterations,
            iters=config.frame_iterations,
            q_init=self.baseline_joint_positions[0],
            tol_pos=config.solve_position_tolerance_mm / 1000.0,
            tol_rot=np.deg2rad(config.solve_angle_tolerance_deg),
        )
        if not all(
            np.all(np.isfinite(values))
            for values in (joint_positions, position_errors_m, angle_errors_rad)
        ):
            raise FloatingPointError("IK returned non-finite joints or residuals")
        position_errors_mm = position_errors_m * 1000.0
        angle_errors_deg = np.rad2deg(angle_errors_rad)
        mean_position_error_mm = float(np.mean(position_errors_mm))
        max_position_error_mm = float(np.max(position_errors_mm))
        mean_angle_error_deg = float(np.mean(angle_errors_deg))
        max_angle_error_deg = float(np.max(angle_errors_deg))
        validation = TrajectoryIKValidation(
            mean_position_error_mm=mean_position_error_mm,
            max_position_error_mm=max_position_error_mm,
            mean_angle_error_deg=mean_angle_error_deg,
            max_angle_error_deg=max_angle_error_deg,
            position_limit_frames=tuple(
                int(frame)
                for frame in np.flatnonzero(
                    position_errors_mm > config.max_position_limit_mm
                )
            ),
            angle_limit_frames=tuple(
                int(frame)
                for frame in np.flatnonzero(
                    angle_errors_deg > config.max_angle_limit_deg
                )
            ),
            unconverged_frames=tuple(
                int(frame) for frame in np.flatnonzero(~converged)
            ),
            passed=(
                mean_position_error_mm <= config.mean_position_limit_mm
                and max_position_error_mm <= config.max_position_limit_mm
                and mean_angle_error_deg <= config.mean_angle_limit_deg
                and max_angle_error_deg <= config.max_angle_limit_deg
                and bool(np.all(converged))
            ),
        )
        return TrajectoryIKResult(
            joint_positions=joint_positions,
            position_errors_m=position_errors_m,
            angle_errors_rad=angle_errors_rad,
            converged=converged,
            validation=validation,
        )
