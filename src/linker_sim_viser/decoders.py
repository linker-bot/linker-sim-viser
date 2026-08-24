"""URDF joint-limit lookup used by the replay / live sources.

Hand SDK→radian decoding now lives in the installed ``linker-robot-assets``
package (``linker_robot_assets.decoders.decode_hand``), which self-reorders SDK
channels into URDF actuated-joint order from each hand's bundled
``decoder.yaml`` and reads limits from the hand's own URDF. This module keeps
only the small URDF-limit reader that the sources still use for validation /
assertions.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache


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
