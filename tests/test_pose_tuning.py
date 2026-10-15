"""Current-frame trajectory anchoring and full-IK validation rules."""

import numpy as np

from linker_sim_viser.config import PoseTuningConfig
from linker_sim_viser.pose_tuning import (
    PoseTuningModel,
    TrajectoryIKValidation,
)
from linker_sim_viser.pose_math import (
    quaternion_wxyz_from_rpy_xyz,
    quaternion_wxyz_to_matrix,
)


class StubChain:
    n = 7
    joint_names = tuple(f"joint{i}" for i in range(7))

    def __init__(self):
        self.response = None

    def track_detailed(self, targets, **_kwargs):
        return self.response


def _model(frames=3):
    chain = StubChain()
    poses = np.zeros((frames, 7), dtype=np.float64)
    poses[:, 0] = np.arange(frames)
    poses[:, 3] = 1.0
    model = PoseTuningModel(
        side="left",
        chain=chain,
        baseline_poses=poses,
        baseline_joint_positions=np.zeros((frames, 7), dtype=np.float64),
        target_source="test",
        config=PoseTuningConfig(),
    )
    return model, chain


def test_reanchor_at_frame_applies_one_world_transform_to_whole_path():
    model, _ = _model()
    target_quaternion = quaternion_wxyz_from_rpy_xyz(
        np.array([0.0, 0.0, np.pi / 2])
    )

    adjusted = model.reanchor_at_frame(
        1,
        np.array([1.0, 1.0, 0.0]),
        target_quaternion,
        reference_pose=model.baseline_poses[1],
    )

    np.testing.assert_allclose(
        adjusted[:, :3],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
        atol=1e-12,
    )
    assert np.allclose(
        quaternion_wxyz_to_matrix(adjusted[1, 3:]),
        quaternion_wxyz_to_matrix(target_quaternion),
    )
    for pose in adjusted:
        np.testing.assert_allclose(
            quaternion_wxyz_to_matrix(pose[3:]),
            quaternion_wxyz_to_matrix(target_quaternion),
            atol=1e-12,
        )


def test_full_ik_passes_when_all_frames_converge_within_error_limits():
    model, chain = _model()
    position_errors_m = np.array([0.001, 0.005, 0.010])
    angle_errors_rad = np.deg2rad([0.5, 1.0, 2.0])
    chain.response = (
        np.zeros((3, 7)),
        position_errors_m,
        angle_errors_rad,
        np.array([True, True, True]),
    )

    result = model.recompute(model.baseline_poses)

    assert result.validation.passed
    assert result.validation.mean_position_error_mm == 16 / 3
    assert result.validation.max_position_error_mm == 10.0
    assert result.validation.position_limit_frames == ()
    assert result.validation.angle_limit_frames == ()
    assert result.validation.unconverged_frames == ()


def test_one_unconverged_frame_fails_even_when_residuals_are_small():
    model, chain = _model()
    chain.response = (
        np.zeros((3, 7)),
        np.array([0.0, 0.00001, 0.00002]),
        np.deg2rad([0.0, 0.001, 0.002]),
        np.array([True, False, True]),
    )

    result = model.recompute(model.baseline_poses)

    assert not result.validation.passed
    assert result.validation.unconverged_frames == (1,)


def test_position_limit_frames_and_overall_failure_are_reported():
    model, chain = _model()
    chain.response = (
        np.zeros((3, 7)),
        np.array([0.0, 0.0, 0.031]),
        np.deg2rad([0.0, 0.0, 0.0]),
        np.array([True, True, True]),
    )

    result = model.recompute(model.baseline_poses)

    assert not result.validation.passed
    assert result.validation.position_limit_frames == (2,)
    assert result.validation.unconverged_frames == ()


def test_non_finite_target_path_is_rejected():
    model, _ = _model()
    targets = model.baseline_poses.copy()
    targets[1, 0] = np.nan

    try:
        model.recompute(targets)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite targets must be rejected")
