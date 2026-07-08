"""End-effector trails and keypose stamps.

v0 renders each EE as a growing "comet" trail:
    * a `line_segments` handle covering all T-1 potential segments, of which
      the first `frame` are shown with a dim-to-bright gradient and the rest
      are collapsed to the current head position (zero-length → invisible);
    * an icosphere handle marking the current head.

Both handles are created once and their buffers mutated per-frame, so there
is no scene-node churn.

Keypose stamps are static markers (icosphere + label) drawn once at load
time. Events are detected from the SDK hand stream after a rolling median
that suppresses single-frame sensor dropouts (see the `sdk_collection_data_format`
memory for the dropout pattern).

Positions are computed via FK on the same workstation URDF we render, since
the npz's precomputed `ee_poses_*` are in the arm's J2 frame — see the
`sdk_collection_data_format` memory for the gotcha.
"""

from __future__ import annotations

import numpy as np
import viser
import yourdfpy


def compute_ee_positions(
    urdf: yourdfpy.URDF,
    joint_positions: dict[str, np.ndarray],
    link_name: str,
) -> np.ndarray:
    """Run FK for `link_name` at every frame. Returns (T, 3) in URDF world frame."""
    actuated = [j.name for j in urdf.actuated_joints]
    T = len(next(iter(joint_positions.values())))
    q = np.stack(
        [joint_positions.get(n, np.zeros(T, dtype=np.float32)) for n in actuated],
        axis=1,
    )
    xyz = np.empty((T, 3), dtype=np.float32)
    for t in range(T):
        urdf.update_cfg(q[t])
        xyz[t] = urdf.get_transform(link_name)[:3, 3]
    return xyz


class GrowingTrail:
    """A comet-style EE trail that reveals itself as playback advances."""

    def __init__(
        self,
        server: viser.ViserServer,
        positions: np.ndarray,
        name: str,
        color: tuple[int, int, int] = (100, 200, 255),
        line_width: float = 3.5,
        head_radius: float = 0.012,
        tail_intensity: float = 0.15,
    ) -> None:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (T, 3); got {positions.shape}")
        if positions.shape[0] < 2:
            raise ValueError(f"need at least 2 frames; got {positions.shape[0]}")

        self._xyz = positions.astype(np.float32)
        self._T = self._xyz.shape[0]
        self._color = np.array(color, dtype=np.float32)
        self._tail = tail_intensity
        self._last_frame = -1

        # Preallocate all T-1 segments collapsed at the start position.
        n = self._T - 1
        pts0 = np.broadcast_to(self._xyz[0], (n, 2, 3)).astype(np.float32).copy()
        cols0 = np.zeros((n, 2, 3), dtype=np.uint8)
        self._segments = server.scene.add_line_segments(
            f"{name}/line", points=pts0, colors=cols0, line_width=line_width
        )
        self._head = server.scene.add_icosphere(
            f"{name}/head",
            radius=head_radius,
            color=color,
            position=tuple(self._xyz[0].tolist()),
        )

    def update(self, frame: int) -> None:
        """Redraw with segments [0, frame] visible."""
        if frame == self._last_frame:
            return
        self._last_frame = frame

        T, n = self._T, self._T - 1
        f = max(0, min(frame, T - 1))

        pts = np.empty((n, 2, 3), dtype=np.float32)
        cols = np.zeros((n, 2, 3), dtype=np.uint8)

        if f > 0:
            pts[:f, 0] = self._xyz[:f]
            pts[:f, 1] = self._xyz[1:f + 1]
            # Vertex-space fade: f+1 vertices along the visible portion.
            fade = np.linspace(self._tail, 1.0, f + 1)
            vcols = np.clip(self._color[None, :] * fade[:, None], 0, 255).astype(np.uint8)
            cols[:f, 0] = vcols[:-1]
            cols[:f, 1] = vcols[1:]

        # Collapse hidden segments to the head so they render as zero-length.
        head_xyz = self._xyz[f]
        pts[f:] = head_xyz

        self._segments.points = pts
        self._segments.colors = cols
        self._head.position = tuple(head_xyz.tolist())


def keypose_events_from_hand(
    hand_sdk: np.ndarray,
    side: str,
    sdk_range: tuple[float, float] = (0.0, 255.0),
    threshold_percent: float = 50.0,
    median_window: int = 5,
    curl_channels: tuple[int, ...] = (2, 3, 4, 5),
) -> list[dict]:
    """Detect grasp_close / release events from an SDK hand stream.

    Aggregates the four curl channels (index/middle/ring/pinky by default —
    skips thumb DoFs which are noisier), inverts to a `closedness` signal in
    [0, 100] (100 → fully closed, per linear-fit-v0 with SDK 0 mapping to
    URDF upper limit), applies a rolling median to reject sensor dropouts up
    to `median_window // 2` frames long, then reports each threshold crossing.

    Returns a list of ``{"frame": int, "kind": str, "side": str}``.
    """
    lo, hi = sdk_range
    pct = (hand_sdk.astype(np.float32) - lo) * (100.0 / (hi - lo))
    curl_pct = pct[:, list(curl_channels)].mean(axis=1)
    closed = 100.0 - curl_pct

    if median_window > 1 and len(closed) >= median_window:
        pad = median_window // 2
        padded = np.pad(closed, pad, mode="edge")
        idx = np.arange(median_window)[None, :] + np.arange(len(closed))[:, None]
        closed = np.median(padded[idx], axis=1)

    above = closed > threshold_percent
    events: list[dict] = []
    for i in range(1, len(above)):
        if not above[i - 1] and above[i]:
            events.append({"frame": int(i), "kind": "grasp_close", "side": side})
        elif above[i - 1] and not above[i]:
            events.append({"frame": int(i), "kind": "release", "side": side})
    return events


def add_keypose_stamps(
    server: viser.ViserServer,
    events: list[dict],
    ee_xyz_by_side: dict[str, np.ndarray],
    color_by_side: dict[str, tuple[int, int, int]],
    root: str = "/keyposes",
    radius: float = 0.018,
) -> None:
    """Place a labelled marker at each event's EE position (static, drawn once)."""
    for ev in events:
        side, kind, frame = ev["side"], ev["kind"], ev["frame"]
        xyz = ee_xyz_by_side[side][frame]
        # grasp = side colour, release = grey (visually distinguishable).
        marker_color = color_by_side[side] if kind == "grasp_close" else (180, 180, 180)
        base_name = f"{root}/{side}_{kind}_{frame}"
        server.scene.add_icosphere(
            base_name,
            radius=radius,
            color=marker_color,
            position=tuple(float(x) for x in xyz),
        )
        server.scene.add_label(
            f"{base_name}_label",
            text=f"{kind}@{frame}",
            position=(float(xyz[0]), float(xyz[1]), float(xyz[2]) + 0.04),
        )
