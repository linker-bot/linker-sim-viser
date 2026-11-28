"""Write a validated Pose Tuning result as a new episode directory."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING
import uuid
import zipfile

import numpy as np

from .config import RobotConfig, StreamSpec

if TYPE_CHECKING:
    from .pose_tuning import PoseTuningModel, TrajectoryIKResult
    from .pose_tuning_gui import PoseTuningSnapshot


def save_pose_tuned_episode(
    source_dir: Path | str,
    robot: RobotConfig,
    model: PoseTuningModel,
    request: PoseTuningSnapshot,
    target_poses: np.ndarray,
    result: TrajectoryIKResult,
) -> Path:
    """Save a passing arm trajectory to a new sibling directory atomically.

    Existing NPZ members are copied without loading unknown arrays. Only the
    selected arm's qpos slice and arm-specific IK fields are decoded/replaced.
    """
    source_dir = Path(source_dir).resolve()
    if request.side != model.side:
        raise ValueError("save request arm does not match the IK model")
    if not result.validation.passed:
        raise ValueError("only a passing full IK result can be saved")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"episode directory does not exist: {source_dir}")

    source_npz = source_dir / "telemetry.npz"
    source_metadata = source_dir / "metadata.json"
    if not source_npz.is_file() or not source_metadata.is_file():
        raise FileNotFoundError("episode must contain telemetry.npz and metadata.json")

    joint_values = np.asarray(result.joint_positions, dtype=np.float64)
    targets = np.asarray(target_poses, dtype=np.float64)
    expected_frames = len(model.baseline_poses)
    if joint_values.shape != (expected_frames, model.chain.n):
        raise ValueError("IK joint trajectory has an unexpected shape")
    if targets.shape != (expected_frames, 7):
        raise ValueError("target pose trajectory has an unexpected shape")
    if not np.all(np.isfinite(joint_values)) or not np.all(np.isfinite(targets)):
        raise ValueError("IK result contains non-finite values")

    arm_stream = _find_arm_stream(robot, model.chain.joint_names)
    lo, hi = arm_stream.columns()
    qpos_key = arm_stream.key
    arm_qpos_key = f"arm_{model.side}_qpos"
    arm_pose_key = f"arm_{model.side}"

    with zipfile.ZipFile(source_npz, "r") as source_archive:
        member_names = [info.filename for info in source_archive.infolist()]
        qpos_member = f"{qpos_key}.npy"
        if member_names.count(qpos_member) != 1:
            raise ValueError(f"NPZ must contain exactly one {qpos_key!r} array")
        for key in (arm_qpos_key, arm_pose_key):
            if member_names.count(f"{key}.npy") > 1:
                raise ValueError(f"NPZ contains duplicate {key!r} arrays")
        qpos = _read_npz_array(source_archive, qpos_member)
        if qpos.ndim != 2 or qpos.shape[0] != expected_frames:
            raise ValueError(f"{qpos_key!r} must have shape (frames, columns)")
        if lo < 0 or hi > qpos.shape[1] or hi - lo != len(arm_stream.joints):
            raise ValueError("configured arm qpos slice is outside the stored array")

        qpos_dtype = qpos.dtype if qpos.dtype.kind == "f" else np.dtype(np.float32)
        qpos_replacement = qpos.astype(qpos_dtype, copy=True)
        model_joint_index = {
            name: index for index, name in enumerate(model.chain.joint_names)
        }
        stream_order = [model_joint_index[name] for name in arm_stream.joints]
        qpos_replacement[:, lo:hi] = joint_values[:, stream_order].astype(
            qpos_dtype, copy=False
        )

        metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("episode metadata.json must contain a JSON object")
        metadata = dict(metadata)
        validation = result.validation
        pose_metadata = metadata.get("pose_tuning", {})
        if not isinstance(pose_metadata, dict):
            raise ValueError("metadata field 'pose_tuning' must be a JSON object")
        pose_metadata = dict(pose_metadata)
        pose_metadata.update(
            {
                "side": model.side,
                "source_episode": source_dir.name,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "target_source": model.target_source,
                "anchor_pose": {
                    "frame": request.frame,
                    "position_mm": request.position_mm.tolist(),
                    "rpy_xyz_deg": request.rpy_deg.tolist(),
                    "quaternion_wxyz": request.quaternion_wxyz.tolist(),
                },
                "config": {
                    "mean_position_limit_mm": model.config.mean_position_limit_mm,
                    "max_position_limit_mm": model.config.max_position_limit_mm,
                    "mean_angle_limit_deg": model.config.mean_angle_limit_deg,
                    "max_angle_limit_deg": model.config.max_angle_limit_deg,
                    "solve_position_tolerance_mm": (
                        model.config.solve_position_tolerance_mm
                    ),
                    "solve_angle_tolerance_deg": (
                        model.config.solve_angle_tolerance_deg
                    ),
                    "preview_iterations": model.config.preview_iterations,
                    "first_frame_iterations": model.config.first_frame_iterations,
                    "frame_iterations": model.config.frame_iterations,
                    "damping": model.config.damping,
                },
                "validation": {
                    "passed": validation.passed,
                    "mean_position_error_mm": validation.mean_position_error_mm,
                    "max_position_error_mm": validation.max_position_error_mm,
                    "mean_angle_error_deg": validation.mean_angle_error_deg,
                    "max_angle_error_deg": validation.max_angle_error_deg,
                    "position_limit_frames": list(validation.position_limit_frames),
                    "angle_limit_frames": list(validation.angle_limit_frames),
                    "unconverged_frames": list(validation.unconverged_frames),
                },
            }
        )
        metadata["pose_tuning"] = pose_metadata

        destination = _new_destination(source_dir)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{source_dir.name}.pose-tuning-", dir=source_dir.parent
            )
        )
        try:
            new_npz = staging / "telemetry.npz"
            with zipfile.ZipFile(new_npz, "w") as output_archive:
                output_archive.comment = source_archive.comment
                for info in source_archive.infolist():
                    data = source_archive.read(info)
                    key = _npz_key(info.filename)
                    if key == qpos_key:
                        data = _write_npy_array(qpos_replacement)
                    elif key == arm_qpos_key:
                        old = _read_npy_bytes(data, key)
                        if old.ndim != 2 or old.shape != joint_values.shape:
                            raise ValueError(
                                f"{key!r} must have shape {joint_values.shape}"
                            )
                        data = _write_npy_array(_cast_like(joint_values, old.dtype))
                    elif key == arm_pose_key:
                        old = _read_npy_bytes(data, key)
                        if old.shape != targets.shape:
                            raise ValueError(f"{key!r} must have shape {targets.shape}")
                        data = _write_npy_array(_cast_like(targets, old.dtype))
                    output_archive.writestr(info, data)

                written_keys = {_npz_key(name) for name in member_names}
                if arm_qpos_key not in written_keys:
                    output_archive.writestr(
                        f"{arm_qpos_key}.npy",
                        _write_npy_array(np.asarray(joint_values, dtype=np.float32)),
                    )

            with zipfile.ZipFile(new_npz, "r") as check_archive:
                corrupt_member = check_archive.testzip()
                if corrupt_member is not None:
                    raise OSError(f"written NPZ failed CRC check: {corrupt_member}")

            (staging / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            staging.rename(destination)
            return destination
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)


def _find_arm_stream(
    robot: RobotConfig, joint_names: tuple[str, ...]
) -> StreamSpec:
    expected = set(joint_names)
    matches = [
        stream
        for stream in robot.streams
        if stream.decoder is None
        and len(stream.joints) == len(joint_names)
        and set(stream.joints) == expected
    ]
    if len(matches) != 1:
        raise ValueError(
            f"robot config must map the {len(joint_names)} {joint_names[0]} arm "
            "joints "
            f"to exactly one undecoded stream; found {len(matches)}"
        )
    return matches[0]


def _new_destination(source_dir: Path) -> Path:
    for _ in range(100):
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix += f"_{uuid.uuid4().hex[:8]}"
        candidate = source_dir.with_name(f"{source_dir.name}_pose_tuned_{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError("could not choose an unused output episode directory")


def _npz_key(filename: str) -> str | None:
    return filename[:-4] if filename.endswith(".npy") else None


def _read_npz_array(archive: zipfile.ZipFile, member_name: str) -> np.ndarray:
    return _read_npy_bytes(archive.read(member_name), member_name[:-4])


def _read_npy_bytes(data: bytes, key: str) -> np.ndarray:
    try:
        return np.lib.format.read_array(BytesIO(data), allow_pickle=False)
    except ValueError as exc:
        raise ValueError(f"NPZ member {key!r} is not a safe numeric array") from exc


def _write_npy_array(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def _cast_like(values: np.ndarray, old_dtype: np.dtype) -> np.ndarray:
    dtype = old_dtype if old_dtype.kind == "f" else np.dtype(np.float32)
    return np.asarray(values, dtype=dtype)
