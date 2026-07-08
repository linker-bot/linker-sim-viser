"""Viser scene wrapper: URDF via `viser.extras.ViserUrdf` (yourdfpy under the hood).

`yourdfpy` resolves URDF `<mimic>` natively, so LinkerHand tendon coupling
works with no extra code. We load the URDF ourselves and hand it to
`ViserUrdf`, keeping direct access to the yourdfpy handle for FK.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf


class SceneRobot:
    """One URDF-backed robot mounted under a scene subtree."""

    def __init__(
        self,
        server: viser.ViserServer,
        urdf_path: Path,
        root: str = "/robot",
    ) -> None:
        urdf_path = Path(urdf_path)
        self._urdf: yourdfpy.URDF = yourdfpy.URDF.load(
            str(urdf_path),
            build_scene_graph=True,
            build_collision_scene_graph=False,
            load_meshes=True,
            load_collision_meshes=False,
            filename_handler=partial(
                yourdfpy.filename_handler_magic, dir=urdf_path.parent
            ),
        )
        self._viser_urdf = ViserUrdf(
            server, urdf_or_path=self._urdf, root_node_name=root
        )
        self._joint_names: tuple[str, ...] = self._viser_urdf.get_actuated_joint_names()
        self._name_to_idx: dict[str, int] = {n: i for i, n in enumerate(self._joint_names)}
        self._cfg = np.zeros(len(self._joint_names), dtype=np.float32)

    @property
    def actuated_joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def urdf(self) -> yourdfpy.URDF:
        return self._urdf

    def set_q_by_name(self, q_by_name: dict[str, float] | dict[str, np.floating]) -> None:
        for name, value in q_by_name.items():
            idx = self._name_to_idx.get(name)
            if idx is not None:
                self._cfg[idx] = float(value)
        self._viser_urdf.update_cfg(self._cfg)
