"""Hand SDK → URDF-radian decoders.

VENDORED (partial) from `linker_robot_assets.decoders.hand`. See
`hand_decoders/<variant>/decoder.yaml` for the vendored sidecars. When
`linker-robot-assets` ships as an installable dep, delete both this file's
decoder logic and the `hand_decoders/` tree and import from that package.

Convention `linear-fit-v0`:
    SDK 100 -> URDF lower limit (rest / open)
    SDK   0 -> URDF upper limit (full travel)
    q = lower + (100 - sdk)/100 * (upper - lower)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

CONVENTION = "linear-fit-v0"

_HAND_DECODERS_DIR = Path(__file__).parent / "hand_decoders"


@lru_cache(maxsize=8)
def _urdf_joint_limits(urdf_path: str) -> dict[str, tuple[float, float]]:
    tree = ET.parse(urdf_path)
    out: dict[str, tuple[float, float]] = {}
    for joint in tree.getroot().findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        out[name] = (float(limit.get("lower", "0")), float(limit.get("upper", "0")))
    return out


def _load_sidecar(hand_kind: str) -> dict:
    path = _HAND_DECODERS_DIR / hand_kind / "decoder.yaml"
    if not path.is_file():
        available = [p.name for p in _HAND_DECODERS_DIR.iterdir() if p.is_dir()]
        raise FileNotFoundError(
            f"vendored decoder sidecar missing: {path} (available: {available})"
        )
    spec = yaml.safe_load(path.read_text()) or {}
    if spec.get("convention") != CONVENTION:
        raise ValueError(
            f"{path}: convention {spec.get('convention')!r} != {CONVENTION!r}"
        )
    return spec


def decode_hand_sdk(
    sdk_0_100: np.ndarray,
    hand_kind: str,
    joint_names: list[str],
    urdf_path: Path | str,
) -> np.ndarray:
    """SDK 0-100 → URDF radians via linear-fit-v0.

    `joint_names` must be in SDK column order (matching `sdk_0_100`'s last axis).
    URDF `[lower, upper]` is read from `urdf_path` for those joints.
    """
    _load_sidecar(hand_kind)  # validates convention tag
    limits = _urdf_joint_limits(str(urdf_path))
    missing = [j for j in joint_names if j not in limits]
    if missing:
        raise KeyError(f"joints not found in {urdf_path}: {missing}")
    lo = np.array([limits[j][0] for j in joint_names], dtype=np.float32)
    hi = np.array([limits[j][1] for j in joint_names], dtype=np.float32)
    sdk = np.clip(np.asarray(sdk_0_100, dtype=np.float32), 0.0, 100.0)
    return (lo + ((100.0 - sdk) / 100.0) * (hi - lo)).astype(np.float32)
