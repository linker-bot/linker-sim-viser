"""Config↔recording width-mismatch diagnostics for `load_episode`.

Regression for issue #7: replaying a 26-column L6 episode with the L25 config
(which slices qpos out to column 46) used to fail deep in the per-stream count
check with a cryptic "produced 12 cols, but joints has 16" — because numpy
silently truncates an over-long slice. `load_episode` now detects the overrun
up front and explains the mismatch.

These tests build synthetic episodes + configs, so they run on any clone
regardless of whether `data/episode_000000/` (or robot assets) are present.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from linker_sim_viser.config import RobotConfig, StreamSpec
from linker_sim_viser.episode import load_episode


def _write_episode(tmp_path: Path, qpos: np.ndarray) -> Path:
    """Minimal viewer episode: telemetry.npz (qpos only) + empty metadata.json."""
    ep = tmp_path / "episode_test"
    ep.mkdir()
    np.savez(ep / "telemetry.npz", qpos=qpos.astype(np.float32))
    (ep / "metadata.json").write_text(json.dumps({}))
    return ep


def _config(streams: list[StreamSpec]) -> RobotConfig:
    # load_episode reads only `robot.streams`; urdf_path/ee_frames are unused here.
    return RobotConfig(urdf_path=Path("/unused"), streams=streams)


def test_l25_config_on_l6_episode_raises_clear_error(tmp_path):
    """The reported case: 26-col (L6) qpos driven by an L25 config wanting 46."""
    ep = _write_episode(tmp_path, np.zeros((5, 26)))
    robot = _config([
        StreamSpec(key="qpos", slice=(0, 7), joints=[f"la{i}" for i in range(7)]),
        StreamSpec(key="qpos", slice=(7, 14), joints=[f"ra{i}" for i in range(7)]),
        StreamSpec(key="qpos", slice=(14, 30), joints=[f"lh{i}" for i in range(16)]),
        StreamSpec(key="qpos", slice=(30, 46), joints=[f"rh{i}" for i in range(16)]),
    ])

    with pytest.raises(ValueError) as excinfo:
        load_episode(ep, robot)

    msg = str(excinfo.value)
    # Names both widths and points at the config↔recording mismatch, not a
    # bare column count.
    assert "46" in msg and "26" in msg
    assert "does not match" in msg
    # And crucially NOT the old cryptic per-stream count message.
    assert "produced" not in msg


def test_matching_width_loads(tmp_path):
    """A config whose slices fit the qpos loads without the mismatch guard firing."""
    qpos = np.arange(5 * 26, dtype=np.float32).reshape(5, 26)
    ep = _write_episode(tmp_path, qpos)
    robot = _config([
        StreamSpec(key="qpos", slice=(0, 7), joints=[f"la{i}" for i in range(7)]),
        StreamSpec(key="qpos", slice=(7, 14), joints=[f"ra{i}" for i in range(7)]),
        StreamSpec(key="qpos", slice=(14, 20), joints=[f"lh{i}" for i in range(6)]),
        StreamSpec(key="qpos", slice=(20, 26), joints=[f"rh{i}" for i in range(6)]),
    ])

    episode = load_episode(ep, robot)

    assert episode.n_frames == 5
    assert "la0" in episode.joint_positions and "rh5" in episode.joint_positions
    # Column mapping is preserved: rh5 is qpos column 25.
    np.testing.assert_array_equal(
        episode.joint_positions["rh5"], qpos[:, 25]
    )


def test_wider_episode_than_config_is_fine(tmp_path):
    """An episode with extra unused columns is not a mismatch — only under-width is."""
    ep = _write_episode(tmp_path, np.zeros((5, 46)))
    robot = _config([
        StreamSpec(key="qpos", slice=(0, 26), joints=[f"j{i}" for i in range(26)]),
    ])

    episode = load_episode(ep, robot)  # must not raise

    assert len(episode.joint_positions) == 26
