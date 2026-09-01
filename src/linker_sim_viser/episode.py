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

from linker_robot_assets.decoders import decode_hand

from .config import RobotConfig


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

    # Fail fast on a config↔recording mismatch. A robot config declares column
    # ranges into each npz key, and numpy silently *truncates* an over-long slice
    # (`arr[:, 14:30]` on a 26-wide array yields 12 cols, not an error), so the
    # real cause otherwise only surfaces downstream as a cryptic per-stream count
    # mismatch. Check the declared ranges against the actual array width here and
    # say why: almost always the --robot config was recorded with a different
    # hand/arm than this episode (e.g. an L6 recording — 26-col qpos — replayed
    # with an L25 config that wants 46).
    needed: dict[str, int] = {}
    for s in robot.streams:
        if s.slice is not None:
            needed[s.key] = max(needed.get(s.key, 0), s.slice[1])
    for key, need in needed.items():
        have = np.asarray(npz[key]).shape[1]
        if have < need:
            raise ValueError(
                f"robot config expects a {need}-column {key!r} array, but this "
                f"episode's {key!r} is only {have} columns wide. The --robot config "
                f"likely does not match the recording (e.g. replaying a 6-DoF "
                f"LinkerHand L6 episode, whose qpos is 26 columns, with an L25 "
                f"config that expects 46). Use the robot config matching the "
                f"hand/arm this episode was recorded with."
            )

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
            # decode_hand takes SDK-order columns and returns radians in the
            # hand's URDF actuated-joint order (== manifest joints[role], which
            # is this stream's `joints`), reordering internally.
            arr = decode_hand(
                name=s.decoder.kind,
                side=s.decoder.side,
                sdk_0_100=sdk_percent,
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
