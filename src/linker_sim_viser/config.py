"""Typed YAML config for robots and the viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DecoderSpec:
    kind: str                                  # e.g. "linkerhand_l6"
    side: str                                  # "left" | "right"
    sdk_range: tuple[float, float] = (0.0, 255.0)


@dataclass
class StreamSpec:
    """One npz key (optionally sliced) mapped to a list of URDF joints.

    `joints` order is authoritative — column i of the streamed array feeds
    `joints[i]`. If `decoder` is set, values are treated as raw SDK numbers
    in `decoder.sdk_range` and decoded to URDF radians.
    """

    key: str
    joints: list[str]
    slice: tuple[int, int] | None = None
    decoder: DecoderSpec | None = None


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


@dataclass
class TrailsConfig:
    enabled: bool = True
    max_points: int = 500


@dataclass
class KeyposesConfig:
    # Off by default: keypose stamps are data-driven annotations that draw a
    # persistent 3D label at each detected grasp/release. Opt in per config.
    enabled: bool = False


@dataclass
class ViewerConfig:
    port: int = 8080
    loop: bool = False
    default_speed: float = 1.0
    speed_presets: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    trails: TrailsConfig = field(default_factory=TrailsConfig)
    keyposes: KeyposesConfig = field(default_factory=KeyposesConfig)


def load_robot_config(path: Path | str) -> RobotConfig:
    """Load a per-robot YAML. URDF path is resolved relative to the yaml file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    streams: list[StreamSpec] = []
    for s in raw["streams"]:
        dec = s.get("decoder")
        decoder = (
            DecoderSpec(
                kind=dec["kind"],
                side=dec["side"],
                sdk_range=tuple(dec.get("sdk_range", (0.0, 255.0))),
            )
            if dec
            else None
        )
        streams.append(
            StreamSpec(
                key=s["key"],
                joints=list(s["joints"]),
                slice=tuple(s["slice"]) if s.get("slice") else None,
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
    return RobotConfig(urdf_path=urdf_path, streams=streams, ee_frames=ee_frames)


def load_viewer_config(path: Path | str) -> ViewerConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    trails = TrailsConfig(**raw.get("trails", {}))
    keyposes = KeyposesConfig(**raw.get("keyposes", {}))
    return ViewerConfig(
        port=raw.get("port", 8080),
        loop=raw.get("loop", False),
        default_speed=raw.get("default_speed", 1.0),
        speed_presets=tuple(raw.get("speed_presets", (0.25, 0.5, 1.0, 2.0, 4.0))),
        trails=trails,
        keyposes=keyposes,
    )
