"""Quaternion-based rigid-pose helpers.

Quaternions use the scalar-first ``(w, x, y, z)`` convention used by the
converted episode ``arm_{side}`` arrays. RPY is accepted only at the UI/input
boundary; pose composition itself stays in quaternion form.
"""

from __future__ import annotations

import numpy as np

_QUAT_EPS = 1e-12


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return a finite unit quaternion in scalar-first order."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must be four finite values in wxyz order")
    norm = float(np.linalg.norm(q))
    if norm <= _QUAT_EPS:
        raise ValueError("quaternion norm is too small")
    return q / norm


def multiply_quaternions_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose rotations as ``left * right`` using Hamilton products."""
    a = normalize_quaternion_wxyz(left)
    b = normalize_quaternion_wxyz(right)
    vector = a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])
    scalar = a[0] * b[0] - float(np.dot(a[1:], b[1:]))
    return normalize_quaternion_wxyz(np.concatenate(([scalar], vector)))


def inverse_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return the inverse of a unit rotation quaternion."""
    q = normalize_quaternion_wxyz(quaternion)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_wxyz_from_rpy_xyz(rpy_rad: np.ndarray) -> np.ndarray:
    """Convert XYZ roll/pitch/yaw input to a unit wxyz quaternion.

    The multiplication order matches ``scipy Rotation.from_euler('xyz', ...)``.
    No Euler extraction is used, so orientations can be composed without an
    intermediate gimbal-lock singularity.
    """
    rpy = np.asarray(rpy_rad, dtype=np.float64)
    if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
        raise ValueError("rpy_rad must contain three finite values")

    half = rpy * 0.5
    cr, cp, cy = np.cos(half)
    sr, sp, sy = np.sin(half)
    return normalize_quaternion_wxyz(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
    )


def quaternion_wxyz_to_rpy_xyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert wxyz orientation to XYZ RPY for display/input widgets only.

    Pose composition and IK never convert back to Euler angles. At the Euler
    singularity this returns one valid representative of the orientation.
    """
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    sin_pitch = 2.0 * (w * y - z * x)
    if abs(sin_pitch) >= 1.0 - 1e-10:
        # At pitch = +/-90 degrees, roll and yaw are coupled. Choose yaw=0
        # and preserve the observable roll/yaw combination from the rotation
        # matrix instead of letting both atan2 calls collapse to zero.
        rotation = quaternion_wxyz_to_matrix([w, x, y, z])
        roll = np.arctan2(np.sign(sin_pitch) * rotation[0, 1], rotation[1, 1])
        return np.array(
            [roll, np.copysign(np.pi / 2.0, sin_pitch), 0.0],
            dtype=np.float64,
        )
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_wxyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a canonical wxyz quaternion."""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(r), 1.0, atol=1e-6
    ):
        raise ValueError("rotation matrix must be orthonormal with determinant +1")

    trace = float(np.trace(r))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(max(trace + 1.0, 0.0))
        q = np.array(
            [
                0.25 * scale,
                (r[2, 1] - r[1, 2]) / scale,
                (r[0, 2] - r[2, 0]) / scale,
                (r[1, 0] - r[0, 1]) / scale,
            ]
        )
    elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
        scale = 2.0 * np.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 0.0))
        q = np.array(
            [
                (r[2, 1] - r[1, 2]) / scale,
                0.25 * scale,
                (r[0, 1] + r[1, 0]) / scale,
                (r[0, 2] + r[2, 0]) / scale,
            ]
        )
    elif r[1, 1] >= r[2, 2]:
        scale = 2.0 * np.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 0.0))
        q = np.array(
            [
                (r[0, 2] - r[2, 0]) / scale,
                (r[0, 1] + r[1, 0]) / scale,
                0.25 * scale,
                (r[1, 2] + r[2, 1]) / scale,
            ]
        )
    else:
        scale = 2.0 * np.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 0.0))
        q = np.array(
            [
                (r[1, 0] - r[0, 1]) / scale,
                (r[0, 2] + r[2, 0]) / scale,
                (r[1, 2] + r[2, 1]) / scale,
                0.25 * scale,
            ]
        )

    q = normalize_quaternion_wxyz(q)
    # q and -q encode the same rotation; choose one stable representative.
    if q[0] < 0.0:
        q = -q
    elif abs(q[0]) <= _QUAT_EPS:
        dominant = 1 + int(np.argmax(np.abs(q[1:])))
        if q[dominant] < 0.0:
            q = -q
    return q


def pose_to_matrix(position_xyz: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from a position and wxyz orientation."""
    position = np.asarray(position_xyz, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_xyz must contain three finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(quaternion_wxyz)
    transform[:3, 3] = position
    return transform


def matrix_to_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and canonical wxyz orientation from a rigid transform."""
    t = np.asarray(transform, dtype=np.float64)
    if t.shape != (4, 4) or not np.all(np.isfinite(t)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(t[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("transform must be a rigid homogeneous transform")
    return t[:3, 3].copy(), quaternion_wxyz_from_matrix(t[:3, :3])


def apply_transform_to_pose(
    transform: np.ndarray,
    position_xyz: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a world-frame 4x4 transform to a position and wxyz orientation."""
    t = np.asarray(transform, dtype=np.float64)
    p = np.asarray(position_xyz, dtype=np.float64)
    if t.shape != (4, 4) or not np.all(np.isfinite(t)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError("position_xyz must contain three finite values")
    if not np.allclose(t[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("transform must be a rigid homogeneous transform")

    transform_q = quaternion_wxyz_from_matrix(t[:3, :3])
    position_out = t[:3, :3] @ p + t[:3, 3]
    orientation_out = multiply_quaternions_wxyz(transform_q, quaternion_wxyz)
    return position_out, orientation_out
