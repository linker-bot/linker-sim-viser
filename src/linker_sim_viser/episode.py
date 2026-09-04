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


def _layout_note(robot: RobotConfig, key: str) -> str:
    """Point at `slots` when this key's hand blocks are SDK-layout-derived.

    A hand stores its SDK packet either verbatim (reserved slots included) or
    pre-stripped, and picking wrong is issue #9. It surfaces only as a total
    width mismatch, because each stream's slice is derived from the same layout
    it decodes against — so name the knob rather than leave it to be found.
    """
    decoded = [
        s
        for s in robot.streams
        if s.key == key and s.decoder is not None and s.offset is not None
    ]
    if not decoded:
        return ""
    modes = sorted({s.decoder.slots for s in decoded})
    return (
        f" Hand block widths here come from the SDK layout of "
        f"{sorted({s.decoder.kind for s in decoded})} with `slots: {'/'.join(modes)}`; "
        f"a recording that stores pre-stripped hand columns instead needs "
        f"`slots: active` and offsets repacked to match."
    )


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
    # ranges into each npz key, and numpy silently *truncates* an over-long
    # slice (`arr[:, 14:30]` on a 26-wide array yields 12 cols, not an error),
    # so the real cause otherwise only surfaces downstream as a cryptic
    # per-stream count mismatch.
    #
    # Extra columns stay legal: a config may drive part of a recording on
    # purpose (a7_lite_l6_umi_left reads 20 of 26). They do get a note, since
    # unaccounted columns are the early smell of a layout mismatch like #9.
    declared: dict[str, int] = {}
    for s in robot.streams:
        declared[s.key] = max(declared.get(s.key, 0), s.columns()[1])
    for key, need in declared.items():
        have = np.asarray(npz[key]).shape[1]
        if have < need:
            raise ValueError(
                f"robot config expects a {need}-column {key!r} array, but this "
                f"episode's {key!r} is only {have} columns wide. The --robot config "
                f"likely does not match the recording (e.g. replaying a 6-DoF "
                f"LinkerHand L6 episode, whose qpos is 26 columns, with an L25 "
                f"config that expects 46). Use the robot config matching the "
                f"hand/arm this episode was recorded with."
                + _layout_note(robot, key)
            )
        if have > need:
            print(
                f"[linker-sim-viser] note: {key!r} is {have} columns wide but this "
                f"config reads {need}; {have - need} column(s) unused"
                + _layout_note(robot, key)
            )

    for s in robot.streams:
        arr = np.asarray(npz[s.key])
        lo, hi = s.columns()
        arr = arr[:, lo:hi]
        expect = len(s.joints) if s.decoder is None else hi - lo
        if arr.shape[1] != expect:
            raise ValueError(
                f"stream key={s.key!r} columns={(lo, hi)} produced {arr.shape[1]} "
                f"cols, but {expect} were expected"
            )

        if s.decoder is not None:
            lo_r, hi_r = s.decoder.sdk_range
            sdk_percent = (arr.astype(np.float32) - lo_r) * (100.0 / (hi_r - lo_r))
            hand_sdk[s.decoder.side] = arr.astype(np.float32)
            # decode_hand takes SDK-order columns and returns radians in the
            # hand's URDF actuated-joint order (== manifest joints[role], which
            # is this stream's `joints`), reordering internally. With
            # `slots="raw"` it also drops the SDK's reserved slots, so the
            # returned width is the actuated-joint count, not the input width.
            arr = decode_hand(
                name=s.decoder.kind,
                side=s.decoder.side,
                sdk_0_100=sdk_percent,
                slots=s.decoder.slots,
            )
            if arr.shape[1] != len(s.joints):
                raise ValueError(
                    f"stream key={s.key!r} decoder {s.decoder.kind}/"
                    f"{s.decoder.side} returned {arr.shape[1]} joints, but "
                    f"`joints` lists {len(s.joints)}"
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
