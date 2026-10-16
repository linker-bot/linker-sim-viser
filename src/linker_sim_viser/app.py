"""Main entry: launch ViserServer, load one episode, drive playback."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import viser

from .config import load_robot_config, load_viewer_config
from .episode import load_episode
from .timeline import PlaybackGUI
from .trails import GrowingTrail, add_keypose_stamps, compute_ee_positions, keypose_events_from_hand
from .viewer import SceneRobot


def run(
    robot_config_path: str | Path,
    episode_path: str | Path,
    viewer_config_path: str | Path | None = None,
    port: int | None = None,
) -> None:
    """Bring up the viewer for a single episode."""
    robot = load_robot_config(robot_config_path)
    episode = load_episode(episode_path, robot)

    viewer_cfg = (
        load_viewer_config(viewer_config_path)
        if viewer_config_path
        else load_viewer_config(
            Path(__file__).resolve().parents[2] / "configs" / "viewer.yaml"
        )
    )
    resolved_port = port if port is not None else viewer_cfg.port

    server = viser.ViserServer(port=resolved_port)
    scene = SceneRobot(server, urdf_path=robot.urdf_path)
    gui = PlaybackGUI(
        server,
        n_frames=episode.n_frames,
        dt=episode.dt,
        speed_presets=viewer_cfg.speed_presets,
        default_speed=viewer_cfg.default_speed,
        default_loop=viewer_cfg.loop,
    )

    @server.on_client_connect
    def _init_camera(client: viser.ClientHandle) -> None:
        # Workstation extent is ~0.5x0.6x1.6m centered near (0, 0, 0.7).
        client.camera.position = (1.6, -1.6, 1.2)
        client.camera.look_at = (0.0, 0.0, 0.9)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    trails: list[GrowingTrail] = []
    ee_xyz_by_label: dict[str, np.ndarray] = {}
    ee_color_by_label: dict[str, tuple[int, int, int]] = {}
    # EE FK positions feed both trails and keypose stamps; compute them if
    # either is enabled (cheap), but only build trail geometry when trails are on.
    if viewer_cfg.trails.enabled or viewer_cfg.keyposes.enabled:
        for ee in robot.ee_frames:
            xyz = compute_ee_positions(scene.urdf, episode.joint_positions, ee.link)
            ee_xyz_by_label[ee.label] = xyz
            ee_color_by_label[ee.label] = ee.color
            if viewer_cfg.trails.enabled:
                trails.append(
                    GrowingTrail(
                        server,
                        positions=xyz,
                        name=f"/trails/{ee.label}",
                        color=ee.color,
                    )
                )

    # Keypose stamps (opt-in): detect from each hand's SDK stream, stamp on the
    # matching EE trail. Sides are keyed by "left"/"right" in hand_sdk but
    # by label ("left_tcp"/"right_tcp") in ee_xyz_by_label — bridge here.
    if viewer_cfg.keyposes.enabled:
        label_by_side = {"left": "left_tcp", "right": "right_tcp"}
        for side, sdk in episode.hand_sdk.items():
            label = label_by_side.get(side)
            if label is None or label not in ee_xyz_by_label:
                continue
            events = keypose_events_from_hand(sdk, side=side)
            print(f"[linker-sim-viser] keyposes {side}-hand: {len(events)} event(s)")
            if events:
                add_keypose_stamps(
                    server,
                    events=events,
                    ee_xyz_by_side={side: ee_xyz_by_label[label]},
                    color_by_side={side: ee_color_by_label[label]},
                )

    joint_names = tuple(episode.joint_positions.keys())
    joint_stack = _stack_joints(episode.joint_positions, joint_names)   # (T, J)

    print(f"[linker-sim-viser] episode={episode.name} frames={episode.n_frames} "
          f"dt={episode.dt:.4f}s -> http://localhost:{resolved_port}")

    # Pace the loop to the next frame's deadline. Sleeping a fixed interval
    # instead quantizes frame *presentation* to the loop period (~12ms here:
    # sleep + render), which has no relation to the frame period -- at 1x that
    # scatters frames 29-39ms apart instead of every 33.3ms, and at 4x the loop
    # is slower than the frame rate, so frames drop in bursts (issue #10).
    # The ceiling only bounds how long a pause/speed click can sit unhandled.
    poll_ceiling = 1.0 / 60.0
    last_rendered = -1
    while True:
        gui.tick()
        f = gui.frame
        if f != last_rendered:
            scene.set_q_by_name(dict(zip(joint_names, joint_stack[f])))
            for tr in trails:
                tr.update(f)
            last_rendered = f
        time.sleep(min(gui.seconds_until_next_frame(), poll_ceiling))


def _stack_joints(joint_positions, joint_names):
    return np.stack([joint_positions[n] for n in joint_names], axis=1)
