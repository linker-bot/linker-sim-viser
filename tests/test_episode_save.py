"""Pose-tuned episodes update only the selected arm and preserve source data."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from linker_sim_viser.config import PoseTuningConfig, RobotConfig, StreamSpec
from linker_sim_viser.episode_save import save_pose_tuned_episode
from linker_sim_viser.pose_tuning import (
    TrajectoryIKResult,
    TrajectoryIKValidation,
)


def _inputs(tmp_path, *, passed=True):
    source = tmp_path / "source_episode"
    source.mkdir()
    frame_count = 3
    qpos = np.arange(frame_count * 12, dtype=np.float32).reshape(frame_count, 12)
    joint_names = tuple(f"joint{i}" for i in range(7))
    stream = StreamSpec(
        key="qpos",
        slice=(2, 9),
        joints=list(reversed(joint_names)),
    )
    robot = RobotConfig(urdf_path=Path("/unused"), streams=[stream])
    arm_joints = np.arange(frame_count * 7, dtype=np.float32).reshape(frame_count, 7)
    targets = np.zeros((frame_count, 7), dtype=np.float32)
    targets[:, 0] = np.arange(frame_count)
    targets[:, 3] = 1.0
    extra = np.array([[4, 5], [6, 7], [8, 9]], dtype=np.int16)
    np.savez(
        source / "telemetry.npz",
        qpos=qpos,
        arm_left_qpos=np.full((frame_count, 7), -1.0, dtype=np.float32),
        arm_left=np.full((frame_count, 7), -2.0, dtype=np.float32),
        arm_right=np.full((frame_count, 7), -3.0, dtype=np.float32),
        extra=extra,
    )
    (source / "metadata.json").write_text(
        json.dumps({"recording_id": "keep-me", "pose_tuning": {"old": True}}),
        encoding="utf-8",
    )
    model = SimpleNamespace(
        side="left",
        chain=SimpleNamespace(n=7, joint_names=joint_names),
        baseline_poses=np.zeros((frame_count, 7), dtype=np.float64),
        target_source="qpos FK",
        config=PoseTuningConfig(),
    )
    request = SimpleNamespace(
        side="left",
        frame=1,
        position_mm=np.array([10.0, 20.0, 30.0]),
        rpy_deg=np.array([1.0, 2.0, 3.0]),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    validation = TrajectoryIKValidation(
        mean_position_error_mm=0.1,
        max_position_error_mm=0.2,
        mean_angle_error_deg=0.01,
        max_angle_error_deg=0.02,
        position_limit_frames=(),
        angle_limit_frames=(),
        unconverged_frames=(),
        passed=passed,
    )
    result = TrajectoryIKResult(
        joint_positions=arm_joints,
        position_errors_m=np.zeros(frame_count),
        angle_errors_rad=np.zeros(frame_count),
        converged=np.full(frame_count, passed),
        validation=validation,
    )
    return source, robot, model, request, targets, result, qpos, arm_joints, extra


def test_save_creates_new_episode_and_preserves_unselected_data(tmp_path):
    (
        source,
        robot,
        model,
        request,
        targets,
        result,
        old_qpos,
        arm_joints,
        extra,
    ) = _inputs(tmp_path)

    output = save_pose_tuned_episode(
        source, robot, model, request, targets, result
    )

    assert output != source
    assert output.parent == source.parent
    with np.load(output / "telemetry.npz") as saved:
        expected_stream_order = arm_joints[:, ::-1]
        np.testing.assert_array_equal(saved["qpos"][:, 2:9], expected_stream_order)
        np.testing.assert_array_equal(saved["qpos"][:, :2], old_qpos[:, :2])
        np.testing.assert_array_equal(saved["qpos"][:, 9:], old_qpos[:, 9:])
        np.testing.assert_array_equal(saved["arm_left_qpos"], arm_joints)
        np.testing.assert_array_equal(saved["arm_left"], targets)
        np.testing.assert_array_equal(saved["arm_right"], -3.0)
        np.testing.assert_array_equal(saved["extra"], extra)

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["recording_id"] == "keep-me"
    assert metadata["pose_tuning"]["old"] is True
    assert metadata["pose_tuning"]["side"] == "left"
    assert metadata["pose_tuning"]["anchor_pose"]["frame"] == 1
    assert metadata["pose_tuning"]["validation"]["passed"] is True
    assert (source / "metadata.json").is_file()


def test_failed_full_ik_result_cannot_be_saved(tmp_path):
    source, robot, model, request, targets, result, *_ = _inputs(
        tmp_path, passed=False
    )

    with pytest.raises(ValueError, match="only a passing"):
        save_pose_tuned_episode(source, robot, model, request, targets, result)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["source_episode"]
