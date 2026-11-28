"""Reachable and unreachable target behavior for the DLS arm solver."""

from types import SimpleNamespace

import numpy as np

from linker_sim_viser.kinematics import ArmChain
from linker_sim_viser.pose_math import matrix_to_pose


def _synthetic_urdf():
    axes = (
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (0, 0, 1),
    )
    joints = []
    for index, axis in enumerate(axes):
        origin = np.eye(4)
        origin[0, 3] = 0.1
        joints.append(
            SimpleNamespace(
                name=f"joint{index}",
                parent=f"link{index}",
                child=f"link{index + 1}",
                type="revolute",
                axis=np.asarray(axis, dtype=np.float64),
                origin=origin,
                limit=SimpleNamespace(lower=-0.8, upper=0.8),
            )
        )
    return SimpleNamespace(robot=SimpleNamespace(joints=joints))


def test_dls_converges_to_a_reachable_nearby_pose():
    chain = ArmChain(_synthetic_urdf(), "link7", damping=0.03)
    expected_q = np.array([0.08, -0.12, 0.1, 0.04, -0.08, 0.12, 0.05])
    position, quaternion = matrix_to_pose(chain.fk(expected_q)[0])
    target = np.concatenate((position, quaternion))

    solved_q, pos_error, rot_error, converged = chain.solve_detailed(
        target,
        np.zeros(7),
        max_iters=200,
        tol_pos=1e-4,
        tol_rot=1e-3,
    )

    assert converged
    assert pos_error <= 1e-4
    assert rot_error <= 1e-3
    assert np.all(solved_q >= chain.lower)
    assert np.all(solved_q <= chain.upper)


def test_unreachable_target_returns_best_effort_solution_and_residual():
    chain = ArmChain(_synthetic_urdf(), "link7", damping=0.05)
    target = np.array([10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    solved_q, pos_error, rot_error, converged = chain.solve_detailed(
        target, np.zeros(7), max_iters=40
    )

    assert not converged
    assert pos_error > 1.0
    assert np.isfinite(rot_error)
    assert np.all(solved_q >= chain.lower)
    assert np.all(solved_q <= chain.upper)
