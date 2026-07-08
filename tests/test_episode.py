"""Real loader tests against data/episode_000000/.

`data/` is not tracked in git — these tests are skipped when the sample
episode is missing (e.g., on a fresh clone before someone drops recordings in).
"""

from pathlib import Path

import numpy as np
import pytest

from linker_sim_viser.config import load_robot_config
from linker_sim_viser.decoders import _urdf_joint_limits
from linker_sim_viser.episode import load_episode

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_EPISODE = REPO_ROOT / "data" / "episode_000000"
ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "a7_lite_l6_dc.yaml"

if not (SAMPLE_EPISODE / "telemetry.npz").is_file():
    pytest.skip(
        f"sample episode missing at {SAMPLE_EPISODE}; drop a recording there to run these tests",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def robot():
    return load_robot_config(ROBOT_CONFIG)


@pytest.fixture(scope="module")
def episode(robot):
    return load_episode(SAMPLE_EPISODE, robot)


def test_episode_shape(episode):
    assert episode.name == "episode_000000"
    assert episode.n_frames == 411
    assert 0.03 < episode.dt < 0.04              # ~30 fps


def test_all_expected_joints_present(episode):
    expected = {
        "arm_left_L1_Joint", "arm_left_L7_Joint",
        "arm_right_R1_Joint", "arm_right_R7_Joint",
        "hand_left_lh_thumb_cmc_pitch", "hand_left_lh_pinky_mcp_pitch",
        "hand_right_rh_thumb_cmc_pitch", "hand_right_rh_pinky_mcp_pitch",
    }
    assert expected <= set(episode.joint_positions)


def test_arm_pass_through(episode):
    # From manual inspection: left-arm col 4 (L5_Joint) mean ~1.302 rad.
    q = episode.joint_positions["arm_left_L5_Joint"]
    assert q.shape == (411,)
    assert 1.25 < q.mean() < 1.35


def test_hand_decoded_within_urdf_limits(episode, robot):
    limits = _urdf_joint_limits(str(robot.urdf_path))
    for j, q in episode.joint_positions.items():
        if j.startswith("hand_"):
            lo, hi = limits[j]
            # linear-fit-v0 clips SDK to [0, 100] before the map, so output
            # must sit inside URDF limits (with a tiny fp epsilon).
            assert q.min() >= lo - 1e-5, f"{j}: {q.min()} < {lo}"
            assert q.max() <= hi + 1e-5, f"{j}: {q.max()} > {hi}"
            assert np.all(np.isfinite(q))


def test_hand_sdk_raw_preserved(episode):
    for side in ("left", "right"):
        raw = episode.hand_sdk[side]
        assert raw.shape == (411, 6)
        assert 0.0 <= raw.min() and raw.max() <= 255.0


def test_ee_frames_configured(robot):
    labels = {e.label for e in robot.ee_frames}
    assert labels == {"left_tcp", "right_tcp"}
    links = {e.link for e in robot.ee_frames}
    assert links == {"hand_left_lh_hand_base_link", "hand_right_rh_hand_base_link"}


def test_linear_fit_endpoints(robot):
    """SDK 100% → URDF lower, SDK 0% → URDF upper (locks the contract)."""
    from linker_sim_viser.decoders import decode_hand_sdk

    joints = [
        "hand_left_lh_thumb_cmc_pitch",
        "hand_left_lh_thumb_cmc_roll",
        "hand_left_lh_index_mcp_pitch",
        "hand_left_lh_middle_mcp_pitch",
        "hand_left_lh_ring_mcp_pitch",
        "hand_left_lh_pinky_mcp_pitch",
    ]
    limits = _urdf_joint_limits(str(robot.urdf_path))
    lo = np.array([limits[j][0] for j in joints], dtype=np.float32)
    hi = np.array([limits[j][1] for j in joints], dtype=np.float32)

    at_100 = decode_hand_sdk(np.full(6, 100.0), "linkerhand_l6", joints, robot.urdf_path)
    at_0 = decode_hand_sdk(np.zeros(6), "linkerhand_l6", joints, robot.urdf_path)
    np.testing.assert_allclose(at_100, lo, atol=1e-5)
    np.testing.assert_allclose(at_0, hi, atol=1e-5)
