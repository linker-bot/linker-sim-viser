"""Main entry: launch ViserServer, load one episode, drive playback."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import viser

from .config import load_robot_config, load_viewer_config
from .episode import load_episode
from .episode_save import save_pose_tuned_episode
from .pose_tuning import PoseTuningModel, TrajectoryIKResult
from .pose_tuning_gui import PoseTuningGUI, PoseTuningSnapshot
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

    pose_models: dict[str, PoseTuningModel] = {}
    if viewer_cfg.pose_tuning.enabled:
        for side in ("left", "right"):
            if side not in robot.ik_frames:
                continue
            try:
                pose_models[side] = PoseTuningModel.from_episode(
                    robot, episode, scene.urdf, side, viewer_cfg.pose_tuning
                )
            except (ValueError, KeyError) as exc:
                # A bad optional IK mapping must not disable ordinary replay.
                print(f"[linker-sim-viser] Pose Tuning unavailable for {side}: {exc}")
    pose_gui = PoseTuningGUI(server, pose_models) if pose_models else None

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

    last_rendered = -1
    pose_revision = -1
    pose_side = ""
    pose_targets: np.ndarray | None = None
    pose_preview_q: np.ndarray | None = None
    pose_preview_frame = -1
    pose_preview_side = ""
    pose_controller_active = False
    pose_controller_revision = -1
    pose_controller_source: np.ndarray | None = None
    pose_controller_reference: np.ndarray | None = None
    pose_controller_side = ""
    pose_full_q: np.ndarray | None = None
    pose_full_side = ""
    pose_full_revision = -1
    pose_compute_future: Future[TrajectoryIKResult] | None = None
    pose_compute_request: PoseTuningSnapshot | None = None
    pose_compute_targets: np.ndarray | None = None
    pose_validated_result: TrajectoryIKResult | None = None
    pose_validated_request: PoseTuningSnapshot | None = None
    pose_validated_targets: np.ndarray | None = None
    pose_save_future: Future[Path] | None = None
    pose_save_request: PoseTuningSnapshot | None = None
    pose_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose-tuning-ik")
        if pose_gui is not None
        else None
    )
    pose_render_dirty = False
    while True:
        gui.tick()
        f = gui.frame
        if pose_gui is not None:
            pose_gui.poll_numeric_values()
            snapshot = pose_gui.snapshot()
            if snapshot.controller_active:
                gui.pause()
                f = gui.frame
                if not pose_controller_active:
                    pose_controller_source = (
                        pose_targets.copy() if pose_targets is not None else None
                    )
                    pose_controller_reference = snapshot.reference_pose.copy()
                    pose_controller_side = snapshot.side
                    pose_controller_revision = -1
                pose_controller_active = True
                if snapshot.controller_revision != pose_controller_revision:
                    pose_controller_revision = snapshot.controller_revision
                    model = pose_models[snapshot.side]
                    try:
                        drag_targets = model.reanchor_at_frame(
                            f,
                            snapshot.position_m,
                            snapshot.quaternion_wxyz,
                            source_poses=(
                                pose_controller_source
                                if pose_controller_side == snapshot.side
                                else None
                            ),
                            reference_pose=pose_controller_reference,
                        )
                        pose_preview_q, pos_error, rot_error = model.preview(
                            drag_targets,
                            f,
                            viewer_cfg.pose_tuning.preview_iterations,
                        )
                        pose_preview_frame = f
                        pose_preview_side = snapshot.side
                        pose_gui.set_status(
                            f"Controller preview frame {f}: "
                            f"{pos_error * 1000.0:.2f} mm, "
                            f"{np.degrees(rot_error):.2f} deg"
                        )
                    except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
                        pose_preview_q = None
                        pose_preview_frame = f
                        pose_preview_side = snapshot.side
                        pose_gui.set_status(f"Controller preview error: {exc}")
                    pose_render_dirty = True
            else:
                pose_controller_active = False
                pose_controller_revision = -1
                pose_controller_source = None
                pose_controller_reference = None
                pose_controller_side = ""
            if snapshot.revision != pose_revision:
                # A numeric edit or transform drag is defined at the frame
                # currently shown. Stop playback before solving that pose.
                gui.pause()
                f = gui.frame
                previous_side = pose_side
                previous_targets = pose_targets
                pose_gui.invalidate_result()
                pose_revision = snapshot.revision
                pose_side = snapshot.side
                pose_preview_q = None
                pose_preview_frame = -1
                pose_preview_side = ""
                pose_full_q = None
                pose_full_side = ""
                pose_full_revision = -1
                pose_validated_result = None
                pose_validated_request = None
                pose_validated_targets = None
                if snapshot.dirty:
                    model = pose_models[pose_side]
                    try:
                        pose_targets = model.reanchor_at_frame(
                            f,
                            snapshot.position_m,
                            snapshot.quaternion_wxyz,
                            source_poses=(
                                previous_targets
                                if previous_side == pose_side
                                else None
                            ),
                            reference_pose=snapshot.reference_pose,
                        )
                        pose_preview_q, pos_error, rot_error = model.preview(
                            pose_targets,
                            f,
                            viewer_cfg.pose_tuning.preview_iterations,
                        )
                        actual_pose = model.forward_pose(pose_preview_q)
                        # Echo the achieved FK pose and make that exact pose
                        # the full-trajectory anchor. This keeps the gizmo on
                        # the rendered hand even when IK has a small residual.
                        pose_targets = model.reanchor_at_frame(
                            f,
                            actual_pose[:3],
                            actual_pose[3:],
                            source_poses=pose_targets,
                        )
                        pose_preview_frame = f
                        pose_preview_side = pose_side
                        pose_gui.set_playback_frame(
                            f,
                            actual_pose,
                            force=True,
                            expected_revision=snapshot.revision,
                        )
                        snapshot = pose_gui.snapshot()
                        pose_gui.set_status(
                            f"Preview frame {f}: {pos_error * 1000.0:.2f} mm, "
                            f"{np.degrees(rot_error):.2f} deg. Recompute required"
                        )
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        pose_targets = None
                        fallback_pose = model.forward_pose(
                            model.baseline_joint_positions[f]
                        )
                        pose_gui.set_playback_frame(
                            f,
                            fallback_pose,
                            force=True,
                            expected_revision=pose_revision,
                        )
                        snapshot = pose_gui.snapshot()
                        pose_gui.set_status(f"Pose preview error: {exc}")
                else:
                    pose_targets = None
                    model = pose_models[pose_side]
                    actual_pose = model.forward_pose(
                        model.baseline_joint_positions[f]
                    )
                    pose_gui.set_playback_frame(
                        f,
                        actual_pose,
                        force=True,
                        expected_revision=pose_revision,
                    )
                    snapshot = pose_gui.snapshot()
                    pose_gui.set_status("Ready")
                pose_render_dirty = True

            has_full_solution = (
                not snapshot.controller_active
                and pose_full_q is not None
                and pose_full_side == pose_side
                and pose_full_revision == snapshot.revision
            )
            if (
                snapshot.dirty
                and not snapshot.computing
                and not has_full_solution
                and pose_targets is not None
                and f != pose_preview_frame
            ):
                model = pose_models[pose_side]
                try:
                    pose_preview_q, pos_error, rot_error = model.preview(
                        pose_targets, f, viewer_cfg.pose_tuning.preview_iterations
                    )
                    pose_preview_frame = f
                    pose_preview_side = pose_side
                    pose_gui.set_status(
                        f"Preview frame {f}: {pos_error * 1000.0:.2f} mm, "
                        f"{np.degrees(rot_error):.2f} deg. Recompute required"
                    )
                except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
                    pose_preview_q = None
                    pose_preview_frame = f
                    pose_gui.set_status(f"Pose preview error: {exc}")
                pose_render_dirty = True

            # Keep the controls tied to the FK pose that is actually being
            # rendered at this playback frame. Feedback does not increment the
            # edit revision or invalidate a passing full-trajectory result.
            model = pose_models[pose_side]
            if has_full_solution:
                displayed_q = pose_full_q[f]
            elif (
                pose_preview_q is not None
                and pose_preview_frame == f
                and pose_preview_side == pose_side
            ):
                displayed_q = pose_preview_q
            else:
                displayed_q = model.baseline_joint_positions[f]
            pose_gui.set_playback_frame(f, model.forward_pose(displayed_q))
            snapshot = pose_gui.snapshot()

            request = pose_gui.consume_recompute_request()
            if request is not None and pose_compute_future is None:
                try:
                    gui.pause()
                    f = gui.frame
                    if request.revision != snapshot.revision or request.side != snapshot.side:
                        raise ValueError("Pose Tuning inputs changed before recompute")
                    model = pose_models[request.side]
                    pose_full_q = None
                    pose_full_side = ""
                    pose_full_revision = -1
                    pose_validated_result = None
                    pose_validated_request = None
                    pose_validated_targets = None
                    # The request is based on the latest actual FK feedback,
                    # including the selected frame if playback was running.
                    snapshot = pose_gui.snapshot()
                    pose_targets = model.reanchor_at_frame(
                        f,
                        snapshot.position_m,
                        snapshot.quaternion_wxyz,
                        source_poses=(
                            pose_targets if pose_targets is not None else None
                        ),
                        reference_pose=snapshot.reference_pose,
                    )
                    preview_q, _, _ = model.preview(
                        pose_targets,
                        f,
                        viewer_cfg.pose_tuning.preview_iterations,
                    )
                    actual_pose = model.forward_pose(preview_q)
                    pose_targets = model.reanchor_at_frame(
                        f,
                        actual_pose[:3],
                        actual_pose[3:],
                        source_poses=pose_targets,
                    )
                    pose_preview_q = preview_q
                    pose_preview_frame = f
                    pose_preview_side = request.side
                    pose_gui.set_playback_frame(
                        f,
                        actual_pose,
                        force=True,
                        expected_revision=snapshot.revision,
                    )
                    snapshot = pose_gui.snapshot()
                    pose_compute_request = snapshot
                    pose_compute_targets = pose_targets.copy()
                    assert pose_executor is not None
                    pose_compute_future = pose_executor.submit(
                        model.recompute, pose_compute_targets
                    )
                    pose_render_dirty = True
                except Exception as exc:
                    pose_gui.finish_recompute_error(request, str(exc))
                    pose_render_dirty = True

            if pose_compute_future is not None and pose_compute_future.done():
                request = pose_compute_request
                try:
                    result = pose_compute_future.result()
                    if request is None:
                        raise RuntimeError("IK completion has no matching request")
                    if request.revision != snapshot.revision or request.side != snapshot.side:
                        raise RuntimeError("Pose Tuning inputs changed during full IK")
                    if result.validation.passed:
                        pose_full_q = result.joint_positions
                        pose_full_side = request.side
                        pose_full_revision = request.revision
                        pose_validated_result = result
                        pose_validated_request = request
                        if pose_compute_targets is None:
                            raise RuntimeError("full IK has no matching target trajectory")
                        pose_validated_targets = pose_compute_targets.copy()
                        active_model = pose_models[request.side]
                        pose_gui.set_playback_frame(
                            f,
                            active_model.forward_pose(pose_full_q[f]),
                            force=True,
                            expected_revision=request.revision,
                        )
                        snapshot = pose_gui.snapshot()
                    else:
                        pose_validated_result = None
                        pose_validated_request = None
                        pose_validated_targets = None
                    pose_gui.finish_validation(request, result, episode.n_frames)
                except Exception as exc:
                    if request is not None:
                        pose_gui.finish_recompute_error(request, str(exc))
                finally:
                    pose_compute_future = None
                    pose_compute_request = None
                    pose_compute_targets = None
                    pose_render_dirty = True

            save_request = pose_gui.consume_save_request()
            if save_request is not None and pose_save_future is None:
                try:
                    if (
                        pose_validated_result is None
                        or pose_validated_request is None
                        or pose_validated_targets is None
                        or save_request.revision != pose_validated_request.revision
                        or save_request.side != pose_validated_request.side
                        or save_request.revision != snapshot.revision
                        or save_request.side != snapshot.side
                    ):
                        raise ValueError(
                            "the passing IK result is stale; recompute before saving"
                        )
                    pose_save_request = save_request
                    assert pose_executor is not None
                    pose_save_future = pose_executor.submit(
                        save_pose_tuned_episode,
                        episode_path,
                        robot,
                        pose_models[save_request.side],
                        pose_validated_request,
                        pose_validated_targets,
                        pose_validated_result,
                    )
                except Exception as exc:
                    pose_gui.finish_save_error(save_request, str(exc))

            if pose_save_future is not None and pose_save_future.done():
                request = pose_save_request
                try:
                    output_dir = pose_save_future.result()
                    if request is None:
                        raise RuntimeError("save completion has no matching request")
                    if request.revision != snapshot.revision or request.side != snapshot.side:
                        raise RuntimeError("Pose Tuning inputs changed during save")
                    pose_gui.finish_saved(request, output_dir)
                except Exception as exc:
                    if request is not None:
                        pose_gui.finish_save_error(request, str(exc))
                finally:
                    pose_save_future = None
                    pose_save_request = None
                    pose_render_dirty = True

        if f != last_rendered or pose_render_dirty:
            q_by_name = dict(zip(joint_names, joint_stack[f]))
            if (
                pose_full_q is not None
                and pose_full_side == pose_side
                and pose_full_revision == pose_revision
                and not pose_controller_active
            ):
                model = pose_models[pose_full_side]
                q_by_name.update(zip(model.chain.joint_names, pose_full_q[f]))
            elif (
                pose_preview_q is not None
                and pose_preview_frame == f
                and pose_preview_side == pose_side
            ):
                model = pose_models[pose_preview_side]
                q_by_name.update(zip(model.chain.joint_names, pose_preview_q))
            scene.set_q_by_name(q_by_name)
            for tr in trails:
                if f != last_rendered:
                    tr.update(f)
            last_rendered = f
            pose_render_dirty = False
        time.sleep(1.0 / 120.0)                # 120 Hz outer loop cap


def _stack_joints(joint_positions, joint_names):
    return np.stack([joint_positions[n] for n in joint_names], axis=1)
