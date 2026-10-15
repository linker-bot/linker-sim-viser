"""Quaternion pose math used by MCAP conversion and Pose Tuning."""

import numpy as np
import pytest

from linker_sim_viser.pose_math import (
    apply_transform_to_pose,
    multiply_quaternions_wxyz,
    pose_to_matrix,
    quaternion_wxyz_from_rpy_xyz,
    quaternion_wxyz_to_matrix,
    quaternion_wxyz_to_rpy_xyz,
)


def test_rpy_quaternion_round_trip_away_from_singularity():
    rpy = np.array([0.37, -0.81, 1.12])

    recovered = quaternion_wxyz_to_rpy_xyz(
        quaternion_wxyz_from_rpy_xyz(rpy)
    )

    np.testing.assert_allclose(recovered, rpy, atol=1e-12)


def test_quaternion_composition_matches_xyz_euler_rotation_order():
    rpy = np.array([0.4, -0.3, 0.7])
    roll, pitch, yaw = rpy
    rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)],
         [0, np.sin(roll), np.cos(roll)]]
    )
    ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
         [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0],
         [0, 0, 1]]
    )

    actual = quaternion_wxyz_to_matrix(quaternion_wxyz_from_rpy_xyz(rpy))

    np.testing.assert_allclose(actual, rz @ ry @ rx, atol=1e-12)


def test_rpy_singularity_round_trip_preserves_orientation():
    quaternion = quaternion_wxyz_from_rpy_xyz(
        np.array([0.4, np.pi / 2, -1.2])
    )
    recovered = quaternion_wxyz_from_rpy_xyz(
        quaternion_wxyz_to_rpy_xyz(quaternion)
    )

    np.testing.assert_allclose(
        quaternion_wxyz_to_matrix(recovered),
        quaternion_wxyz_to_matrix(quaternion),
        atol=1e-10,
    )


def test_world_transform_moves_position_and_orientation_together():
    delta = pose_to_matrix(
        np.array([1.0, 2.0, 3.0]),
        quaternion_wxyz_from_rpy_xyz(np.array([0.0, 0.0, np.pi / 2])),
    )
    position, quaternion = apply_transform_to_pose(
        delta,
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(position, [1.0, 3.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(
        quaternion_wxyz_to_matrix(quaternion),
        delta[:3, :3],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "quaternion",
    [
        np.zeros(4),
        np.array([1.0, np.nan, 0.0, 0.0]),
        np.ones(3),
    ],
)
def test_invalid_quaternion_is_rejected(quaternion):
    with pytest.raises(ValueError):
        quaternion_wxyz_to_matrix(quaternion)
