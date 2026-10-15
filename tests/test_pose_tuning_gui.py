"""Pose Tuning input ownership and FK feedback synchronization."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import viser

from linker_sim_viser.config import PoseTuningConfig
from linker_sim_viser.pose_tuning import PoseTuningModel
from linker_sim_viser.pose_tuning_gui import PoseTuningGUI


class StubChain:
    n = 7
    joint_names = tuple(f"joint{i}" for i in range(7))

    def fk(self, q):
        transform = np.eye(4)
        transform[:3, 3] = np.array([0.2, 0.3, 0.4]) + np.asarray(q[:3])
        return transform, np.zeros((7, 3)), np.zeros((7, 3))


@pytest.fixture
def pose_gui():
    server = viser.ViserServer(port=0, verbose=False)
    baseline_pose = np.array([[0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]])
    model = PoseTuningModel(
        side="left",
        chain=StubChain(),
        baseline_poses=baseline_pose,
        baseline_joint_positions=np.zeros((1, 7)),
        target_source="test",
        config=PoseTuningConfig(),
    )
    try:
        yield PoseTuningGUI(server, {"left": model})
    finally:
        server.stop()


def _event(phase, *, x_m=0.2, y_m=0.3, z_m=0.4):
    return SimpleNamespace(
        phase=phase,
        target=SimpleNamespace(
            position=(x_m, y_m, z_m),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def test_numeric_edit_is_detected_and_not_overwritten_by_same_frame_fk(pose_gui):
    initial = pose_gui.snapshot()
    pose_gui._widgets["x"].value += 25.0

    assert pose_gui.poll_numeric_values()
    edited = pose_gui.snapshot()
    assert edited.revision == initial.revision + 1
    assert edited.dirty
    assert edited.position_mm[0] == pytest.approx(initial.position_mm[0] + 25.0)
    np.testing.assert_array_equal(edited.reference_pose, initial.reference_pose)

    pose_gui.set_playback_frame(edited.frame, initial.reference_pose)

    assert pose_gui.snapshot().position_mm[0] == pytest.approx(
        initial.position_mm[0] + 25.0
    )


def test_controller_owns_pose_until_release_and_commits_start_reference(pose_gui):
    initial = pose_gui.snapshot()
    asyncio.run(pose_gui._on_transform_control_update(_event("start")))
    started = pose_gui.snapshot()
    assert started.controller_active

    pose_gui._widgets["x"].value += 100.0
    assert not pose_gui.poll_numeric_values()

    asyncio.run(
        pose_gui._on_transform_control_update(_event("update", x_m=0.23))
    )
    updated = pose_gui.snapshot()
    assert updated.controller_active
    assert updated.controller_revision == started.controller_revision + 1
    assert updated.position_mm[0] == pytest.approx(230.0)

    asyncio.run(
        pose_gui._on_transform_control_update(_event("end", x_m=0.24))
    )
    ended = pose_gui.snapshot()
    assert not ended.controller_active
    assert ended.revision == initial.revision + 1
    assert ended.dirty
    assert ended.position_mm[0] == pytest.approx(240.0)
    np.testing.assert_allclose(ended.reference_pose, initial.reference_pose)
