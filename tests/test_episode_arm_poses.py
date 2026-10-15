"""Optional arm target poses are loaded without changing qpos streams."""

import json
from pathlib import Path

import numpy as np

from linker_sim_viser.config import RobotConfig, StreamSpec
from linker_sim_viser.episode import load_episode


def _robot():
    return RobotConfig(
        urdf_path=Path("/unused"),
        streams=[
            StreamSpec(key="qpos", slice=(0, 1), joints=["arm_joint"]),
        ],
    )


def test_valid_arm_pose_array_is_exposed_separately_from_joint_positions(tmp_path):
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    qpos = np.array([[0.1], [0.2]], dtype=np.float32)
    poses = np.array(
        [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
         [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    np.savez(episode_dir / "telemetry.npz", qpos=qpos, arm_left=poses)
    (episode_dir / "metadata.json").write_text(
        json.dumps({"frame_count": 2, "dt": 1 / 30}),
        encoding="utf-8",
    )

    episode = load_episode(episode_dir, _robot())

    np.testing.assert_array_equal(episode.joint_positions["arm_joint"], qpos[:, 0])
    np.testing.assert_array_equal(episode.arm_poses["left"], poses)
    assert "right" not in episode.arm_poses


def test_invalid_shape_or_non_finite_arm_pose_is_ignored(tmp_path):
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    qpos = np.zeros((2, 1), dtype=np.float32)
    malformed = np.zeros((2, 6), dtype=np.float32)
    non_finite = np.zeros((2, 7), dtype=np.float32)
    non_finite[1, 0] = np.nan
    np.savez(
        episode_dir / "telemetry.npz",
        qpos=qpos,
        arm_left=malformed,
        arm_right=non_finite,
    )
    (episode_dir / "metadata.json").write_text(
        json.dumps({"frame_count": 2}),
        encoding="utf-8",
    )

    episode = load_episode(episode_dir, _robot())

    assert episode.arm_poses == {}
