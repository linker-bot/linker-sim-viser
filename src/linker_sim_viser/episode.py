"""Episode loader — one recorded episode from `data/episode_XXXXXX/`.

Reads `metadata.json` for dt/fps and `telemetry.npz` for tensors. Streams
declared in the robot config are sliced from the npz and hand streams are
decoded (SDK raw → 0-100 → URDF radians).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import RobotConfig
from .decoders import decode_hand_sdk


@dataclass
class Episode:
    name: str
    dt: float
    n_frames: int
    joint_positions: dict[str, np.ndarray]     # URDF joint name -> (T,) rad
    hand_sdk: dict[str, np.ndarray] = field(default_factory=dict)   # side -> (T, 6) raw


def load_episode(episode_dir: Path | str, robot: RobotConfig) -> Episode:
    ep = Path(episode_dir)
    meta = json.loads((ep / "metadata.json").read_text())
    npz = np.load(ep / "telemetry.npz")

    fps = meta.get("fps")
    dt = float(meta.get("dt") or (1.0 / fps if fps else 1.0 / 30.0))
    n_frames = int(meta.get("frame_count") or next(iter(npz.values())).shape[0])

    joint_positions: dict[str, np.ndarray] = {}
    hand_sdk: dict[str, np.ndarray] = {}

    for s in robot.streams:
        arr = np.asarray(npz[s.key])
        if s.slice is not None:
            arr = arr[:, s.slice[0]:s.slice[1]]
        if arr.shape[1] != len(s.joints):
            raise ValueError(
                f"stream key={s.key!r} slice={s.slice} produced {arr.shape[1]} "
                f"cols, but joints has {len(s.joints)}"
            )

        if s.decoder is not None:
            lo, hi = s.decoder.sdk_range
            sdk_percent = (arr.astype(np.float32) - lo) * (100.0 / (hi - lo))
            hand_sdk[s.decoder.side] = arr.astype(np.float32)
            arr = decode_hand_sdk(
                sdk_0_100=sdk_percent,
                hand_kind=s.decoder.kind,
                joint_names=s.joints,
                urdf_path=robot.urdf_path,
            )

        for j, joint in enumerate(s.joints):
            joint_positions[joint] = arr[:, j].astype(np.float32)

    return Episode(
        name=ep.name,
        dt=dt,
        n_frames=n_frames,
        joint_positions=joint_positions,
        hand_sdk=hand_sdk,
    )
