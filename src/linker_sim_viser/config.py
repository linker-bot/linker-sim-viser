"""Typed YAML config for robots and the viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import yaml


@dataclass
class DecoderSpec:
    kind: str                                  # e.g. "linkerhand_l6"
    side: str                                  # "left" | "right"
    sdk_range: tuple[float, float] = (0.0, 255.0)
    # Which form the recording stores: "raw" is the SDK's full wire vector
    # (reserved slots included), "active" is one column per actuated joint.
    # The decoder owns the slot layout; see linker_robot_assets.decoders.
    slots: str = "active"


@dataclass
class StreamSpec:
    """One npz key (optionally sliced) mapped to a list of URDF joints.

    `joints` order is authoritative — column i of the streamed array feeds
    `joints[i]`. If `decoder` is set, values are treated as raw SDK numbers
    in `decoder.sdk_range` and decoded to URDF radians.

    Give either `slice` (explicit half-open column range) or, for a decoded
    stream, `offset` — the block's first column, whose width then comes from
    the hand's declared SDK layout rather than a hand-copied number. Writing
    widths out by hand is what produced issues #7 and #9.
    """

    key: str
    joints: list[str]
    slice: tuple[int, int] | None = None
    offset: int | None = None
    decoder: DecoderSpec | None = None

    def columns(self) -> tuple[int, int]:
        """Half-open column range this stream reads from its npz key."""
        if self.slice is not None:
            return self.slice
        if self.offset is None:
            raise ValueError(
                f"stream key={self.key!r} declares neither 'slice' nor 'offset'"
            )
        if self.decoder is None:
            raise ValueError(
                f"stream key={self.key!r} uses 'offset' without a decoder; "
                "only a decoded stream can derive its width from a SDK layout"
            )
        from linker_robot_assets.decoders import sdk_channel_width

        width = sdk_channel_width(self.decoder.kind, slots=self.decoder.slots)
        return self.offset, self.offset + width


@dataclass
class EEFrameSpec:
    label: str
    link: str                                  # URDF link name to trail via FK
    color: tuple[int, int, int] = (100, 200, 255)


@dataclass
class RobotConfig:
    urdf_path: Path
    streams: list[StreamSpec]
    ee_frames: list[EEFrameSpec] = field(default_factory=list)
    # End-effector links used for arm IK, keyed by "left" / "right".
    ik_frames: dict[str, str] = field(default_factory=dict)


@dataclass
class TrailsConfig:
    # Off by default: comet EE trails are an annotation, not core replay.
    enabled: bool = False
    max_points: int = 500


@dataclass
class KeyposesConfig:
    # Off by default: keypose stamps are data-driven annotations that draw a
    # persistent 3D label at each detected grasp/release. Opt in per config.
    enabled: bool = False


@dataclass
class PoseTuningConfig:
    """IK convergence tolerances and full-trajectory validation thresholds."""

    enabled: bool = True
    mean_position_limit_mm: float = 10.0
    max_position_limit_mm: float = 30.0
    mean_angle_limit_deg: float = 5.0
    max_angle_limit_deg: float = 10.0
    solve_position_tolerance_mm: float = 0.1
    solve_angle_tolerance_deg: float = 0.0572957795
    preview_iterations: int = 100
    first_frame_iterations: int = 200
    frame_iterations: int = 100
    damping: float = 0.05

    def __post_init__(self) -> None:
        limits = (
            self.mean_position_limit_mm,
            self.max_position_limit_mm,
            self.mean_angle_limit_deg,
            self.max_angle_limit_deg,
            self.solve_position_tolerance_mm,
            self.solve_angle_tolerance_deg,
            self.damping,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in limits):
            raise ValueError("Pose Tuning thresholds and tolerances must be positive")
        if self.mean_position_limit_mm > self.max_position_limit_mm:
            raise ValueError("mean position limit cannot exceed max position limit")
        if self.mean_angle_limit_deg > self.max_angle_limit_deg:
            raise ValueError("mean angle limit cannot exceed max angle limit")
        if min(
            self.preview_iterations, self.first_frame_iterations, self.frame_iterations
        ) < 1:
            raise ValueError("Pose Tuning iteration counts must be at least 1")


@dataclass
class ViewerConfig:
    port: int = 8080
    loop: bool = False
    default_speed: float = 1.0
    speed_presets: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    trails: TrailsConfig = field(default_factory=TrailsConfig)
    keyposes: KeyposesConfig = field(default_factory=KeyposesConfig)
    pose_tuning: PoseTuningConfig = field(default_factory=PoseTuningConfig)


def load_robot_config(path: Path | str) -> RobotConfig:
    """Load a per-robot YAML. URDF path is resolved relative to the yaml file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    streams: list[StreamSpec] = []
    for s in raw["streams"]:
        dec = s.get("decoder")
        decoder = (
            DecoderSpec(
                kind=dec["kind"],
                side=dec["side"],
                sdk_range=tuple(dec.get("sdk_range", (0.0, 255.0))),
                slots=dec.get("slots", "active"),
            )
            if dec
            else None
        )
        streams.append(
            StreamSpec(
                key=s["key"],
                joints=list(s["joints"]),
                slice=tuple(s["slice"]) if s.get("slice") else None,
                offset=s.get("offset"),
                decoder=decoder,
            )
        )

    ee_frames = [
        EEFrameSpec(
            label=e["label"],
            link=e["link"],
            color=tuple(e.get("color", (100, 200, 255))),
        )
        for e in raw.get("ee_frames", [])
    ]
    ik_frames = {
        str(side): str(link) for side, link in raw.get("ik_frames", {}).items()
    }

    raw_urdf = raw["urdf_path"]
    if isinstance(raw_urdf, str) and raw_urdf.startswith("pkg://"):
        # Resolve against the installed linker-robot-assets asset tree
        # (single source of truth), e.g. pkg://workstations/<name>/workstation.urdf.
        from linker_robot_assets import asset_root

        urdf_path = (asset_root() / raw_urdf[len("pkg://") :]).resolve()
    else:
        urdf_path = (path.parent / raw_urdf).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"urdf_path in {path} resolves to {urdf_path}, which does not exist. "
            "A `pkg://` path needs linker-robot-assets installed; a relative path "
            "points at the sibling `linker-sim/` checkout — clone it alongside "
            "this repo or edit `urdf_path` in the robot config."
        )
    return RobotConfig(
        urdf_path=urdf_path,
        streams=streams,
        ee_frames=ee_frames,
        ik_frames=ik_frames,
    )


def load_viewer_config(path: Path | str) -> ViewerConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    trails = TrailsConfig(**raw.get("trails", {}))
    keyposes = KeyposesConfig(**raw.get("keyposes", {}))
    pose_tuning = PoseTuningConfig(**raw.get("pose_tuning", {}))
    return ViewerConfig(
        port=raw.get("port", 8080),
        loop=raw.get("loop", False),
        default_speed=raw.get("default_speed", 1.0),
        speed_presets=tuple(raw.get("speed_presets", (0.25, 0.5, 1.0, 2.0, 4.0))),
        trails=trails,
        keyposes=keyposes,
        pose_tuning=pose_tuning,
    )
