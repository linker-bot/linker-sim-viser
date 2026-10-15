"""Viser controls for arm selection and current-frame pose preview."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

import numpy as np
import viser

from .pose_math import (
    quaternion_wxyz_from_rpy_xyz,
    quaternion_wxyz_to_rpy_xyz,
)
from .pose_tuning import PoseTuningModel, TrajectoryIKResult


@dataclass(frozen=True)
class PoseTuningSnapshot:
    revision: int
    side: str
    frame: int
    reference_pose: np.ndarray
    controller_active: bool
    controller_revision: int
    dirty: bool
    computing: bool
    position_mm: np.ndarray
    rpy_deg: np.ndarray
    position_m: np.ndarray
    quaternion_wxyz: np.ndarray


class PoseTuningGUI:
    """Owns Viser widgets and provides thread-safe snapshots to the app loop."""

    _POSITION_MIN_MM = -5000.0
    _POSITION_MAX_MM = 5000.0
    _ANGLE_MIN_DEG = -180.0
    _ANGLE_MAX_DEG = 180.0

    def __init__(self, server: viser.ViserServer, models: dict[str, PoseTuningModel]):
        if not models:
            raise ValueError("Pose Tuning needs at least one configured arm")
        self._server = server
        self._models = models
        self._lock = threading.Lock()
        self._revision = 0
        self._frame = 0
        self._side = "left" if "left" in models else next(iter(models))
        self._dirty = False
        self._computing = False
        self._recompute_request: PoseTuningSnapshot | None = None
        self._save_request: PoseTuningSnapshot | None = None
        self._can_save = False
        self._validated_revision = -1
        self._validated_side = ""
        self._result_modal: Any | None = None
        self._syncing = False
        self._user_edit_pending = False
        self._controller_active = False
        self._controller_revision = 0
        self._controller_reference_pose: np.ndarray | None = None
        self._controller_drag_changed = False
        self._reference_pose = np.zeros(7, dtype=np.float64)
        self._values: dict[str, float] = {}
        self._load_side_values(self._side, self._frame)

        with server.gui.add_folder("Pose Tuning"):
            server.gui.add_markdown(
                "While dragging, the scene controller owns the pose and updates "
                "the numbers live. Otherwise numeric changes are polled and "
                "previewed with IK; the FK result is echoed afterward."
            )
            self._side_widget = server.gui.add_dropdown(
                "Arm",
                options=tuple(side.title() for side in models),
                initial_value=self._side.title(),
            )
            self._widgets = {
                "x": server.gui.add_slider(
                    "X (mm)", min=self._POSITION_MIN_MM, max=self._POSITION_MAX_MM,
                    step=1.0, initial_value=self._values["x"]
                ),
                "y": server.gui.add_slider(
                    "Y (mm)", min=self._POSITION_MIN_MM, max=self._POSITION_MAX_MM,
                    step=1.0, initial_value=self._values["y"]
                ),
                "z": server.gui.add_slider(
                    "Z (mm)", min=self._POSITION_MIN_MM, max=self._POSITION_MAX_MM,
                    step=1.0, initial_value=self._values["z"]
                ),
                "roll": server.gui.add_slider(
                    "Roll (deg)", min=self._ANGLE_MIN_DEG, max=self._ANGLE_MAX_DEG,
                    step=1.0, initial_value=self._values["roll"]
                ),
                "pitch": server.gui.add_slider(
                    "Pitch (deg)", min=self._ANGLE_MIN_DEG, max=self._ANGLE_MAX_DEG,
                    step=1.0, initial_value=self._values["pitch"]
                ),
                "yaw": server.gui.add_slider(
                    "Yaw (deg)", min=self._ANGLE_MIN_DEG, max=self._ANGLE_MAX_DEG,
                    step=1.0, initial_value=self._values["yaw"]
                ),
            }
            self._status = server.gui.add_text(
                "Status", initial_value="Ready", disabled=True
            )
            self._recompute_button = server.gui.add_button("Recompute full IK")
            self._save_button = server.gui.add_button(
                "Save as new episode", disabled=True
            )

        self._transform_control = server.scene.add_transform_controls(
            "/pose_tuning/start_ee_pose",
            scale=0.15,
            translation_limits=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)),
            position=self._position_from_values(self._values),
            wxyz=self._quaternion_from_values(self._values),
        )

        self._side_widget.on_update(self._on_side_update)
        self._transform_control.on_update(self._on_transform_control_update)
        self._recompute_button.on_click(self._on_recompute_click)
        self._save_button.on_click(self._on_save_click)

    def snapshot(self) -> PoseTuningSnapshot:
        """Return a consistent copy of the latest UI values."""
        with self._lock:
            values = self._values.copy()
            revision = self._revision
            side = self._side
            frame = self._frame
            reference_pose = self._reference_pose.copy()
            controller_active = self._controller_active
            controller_revision = self._controller_revision
            dirty = self._dirty
            computing = self._computing
        position_mm = np.array([values["x"], values["y"], values["z"]])
        rpy_deg = np.array([values["roll"], values["pitch"], values["yaw"]])
        quaternion = quaternion_wxyz_from_rpy_xyz(np.deg2rad(rpy_deg))
        return PoseTuningSnapshot(
            revision=revision,
            side=side,
            frame=frame,
            reference_pose=reference_pose,
            controller_active=controller_active,
            controller_revision=controller_revision,
            dirty=dirty,
            computing=computing,
            position_mm=position_mm,
            rpy_deg=rpy_deg,
            position_m=position_mm / 1000.0,
            quaternion_wxyz=quaternion,
        )

    def set_playback_frame(
        self,
        frame: int,
        actual_pose: np.ndarray,
        *,
        force: bool = False,
        expected_revision: int | None = None,
    ) -> None:
        """Echo FK when the frame changes or a newly solved pose is ready.

        Avoid pushing the same frame's pose every render tick: that can race a
        browser edit and immediately overwrite the value the operator entered.
        """
        pose = np.asarray(actual_pose, dtype=np.float64)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("actual pose must contain seven finite xyz+wxyz values")
        if force:
            # Catch an input that arrived after the app's last poll before
            # applying an older IK result to the widgets.
            self.poll_numeric_values()
        rpy_deg = np.rad2deg(quaternion_wxyz_to_rpy_xyz(pose[3:]))
        position_mm = pose[:3] * 1000.0
        values = {
            "x": float(position_mm[0]),
            "y": float(position_mm[1]),
            "z": float(position_mm[2]),
            "roll": float(rpy_deg[0]),
            "pitch": float(rpy_deg[1]),
            "yaw": float(rpy_deg[2]),
        }
        with self._lock:
            if (
                expected_revision is not None
                and self._revision != expected_revision
            ):
                return
            frame_changed = int(frame) != self._frame
            self._frame = int(frame)
            if self._controller_active:
                return
            if self._user_edit_pending and not force:
                return
            if not frame_changed and not force:
                return
            self._syncing = True
            self._values = values
            self._reference_pose = pose.copy()
        try:
            for name, widget in self._widgets.items():
                widget.value = values[name]
            self._set_transform_control_pose(values)
        finally:
            with self._lock:
                self._syncing = False
                if force and (
                    expected_revision is None
                    or self._revision == expected_revision
                ):
                    self._user_edit_pending = False

    def poll_numeric_values(self) -> bool:
        """Treat changed numeric widgets as input when the gizmo is idle."""
        with self._lock:
            if self._controller_active or self._computing or self._syncing:
                return False
            raw_values = {
                name: float(widget.value) for name, widget in self._widgets.items()
            }
            values = {
                name: float(
                    np.clip(
                        value,
                        self._POSITION_MIN_MM,
                        self._POSITION_MAX_MM,
                    )
                    if name in ("x", "y", "z")
                    else np.clip(value, self._ANGLE_MIN_DEG, self._ANGLE_MAX_DEG)
                )
                for name, value in raw_values.items()
            }
            changed = any(
                not np.isclose(values[name], self._values[name], atol=1e-6)
                for name in values
            )
            if not changed:
                return False
            # _reference_pose remains the last actual FK pose until IK returns.
            # Keep it stable while multiple numeric edits are coalesced.
            self._values = values
            self._dirty = True
            self._user_edit_pending = True
            self._revision += 1
            self._can_save = False
            self._validated_revision = -1
            self._validated_side = ""
            self._syncing = True
        try:
            for name, widget in self._widgets.items():
                widget.value = values[name]
            self._set_transform_control_pose(values)
        finally:
            with self._lock:
                self._syncing = False
        self._save_button.disabled = True
        self.set_status("Recompute required")
        return True

    def set_status(self, value: str) -> None:
        self._status.value = value

    def consume_recompute_request(self) -> PoseTuningSnapshot | None:
        """Take the queued full-trajectory request, if a button was pressed."""
        with self._lock:
            request = self._recompute_request
            self._recompute_request = None
            return request

    def consume_save_request(self) -> PoseTuningSnapshot | None:
        """Take a save request created by a validated-result click."""
        with self._lock:
            request = self._save_request
            self._save_request = None
            return request

    def finish_recompute_error(
        self,
        request: PoseTuningSnapshot,
        error: str,
    ) -> None:
        """Unlock the controls and show the result on the GUI thread."""
        with self._lock:
            self._computing = False
            self._can_save = False
            self._validated_revision = -1
            self._validated_side = ""
        self._set_controls_disabled(False)
        self.set_status(f"Full IK failed: {error}")
        self._show_result_modal(
            "Full IK failed",
            f"**Arm:** {request.side.title()}\n\n**Error:** {error}",
        )

    def finish_validation(
        self,
        request: PoseTuningSnapshot,
        result: TrajectoryIKResult,
        frame_count: int,
    ) -> None:
        """Unlock controls and display the full IK validation summary."""
        validation = result.validation
        with self._lock:
            self._computing = False
            self._can_save = validation.passed
            self._validated_revision = request.revision if validation.passed else -1
            self._validated_side = request.side if validation.passed else ""
        self._set_controls_disabled(False)
        outcome = "PASS" if validation.passed else "FAIL"
        self.set_status(f"Full IK {outcome} ({frame_count} frames)")
        config = self._models[request.side].config
        details = [
            f"**Arm:** {request.side.title()}",
            f"**Frames:** {frame_count}",
            f"**Result:** {outcome}",
            (
                f"**Anchor frame {request.frame} position (mm):** "
                f"{request.position_mm[0]:.1f}, {request.position_mm[1]:.1f}, "
                f"{request.position_mm[2]:.1f}"
            ),
            (
                f"**Anchor frame {request.frame} RPY (deg):** "
                f"{request.rpy_deg[0]:.1f}, {request.rpy_deg[1]:.1f}, "
                f"{request.rpy_deg[2]:.1f}"
            ),
            "",
            "| Error | Mean | Maximum | Mean / maximum limit |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| Position | {validation.mean_position_error_mm:.3f} mm | "
                f"{validation.max_position_error_mm:.3f} mm | "
                f"{config.mean_position_limit_mm:.3f} / "
                f"{config.max_position_limit_mm:.3f} mm |"
            ),
            (
                f"| Angle | {validation.mean_angle_error_deg:.3f}° | "
                f"{validation.max_angle_error_deg:.3f}° | "
                f"{config.mean_angle_limit_deg:.3f} / "
                f"{config.max_angle_limit_deg:.3f}° |"
            ),
            "",
            f"**Position-limit frames (0-based):** {self._format_frames(validation.position_limit_frames)}",
            f"**Angle-limit frames (0-based):** {self._format_frames(validation.angle_limit_frames)}",
            f"**Unconverged frames (0-based):** {self._format_frames(validation.unconverged_frames)}",
            "",
            (
                "The solved trajectory is now used for playback."
                if validation.passed
                else "The full IK trajectory is not applied because validation failed."
            ),
            (
                "Use **Save as new episode** to create a separate copy."
                if validation.passed
                else "Adjust the current-frame pose and recompute to make the result saveable."
            ),
        ]
        self._show_result_modal(
            "Full IK validation", "\n".join(details)
        )

    def finish_save_error(self, request: PoseTuningSnapshot, error: str) -> None:
        """Unlock controls after a failed save and keep a valid result retryable."""
        with self._lock:
            self._computing = False
            self._can_save = (
                request.revision == self._validated_revision
                and request.side == self._validated_side
            )
        self._set_controls_disabled(False)
        self.set_status(f"Save failed: {error}")
        self._show_result_modal(
            "Episode save failed",
            f"**Arm:** {request.side.title()}\n\n**Error:** {error}",
        )

    def finish_saved(self, request: PoseTuningSnapshot, output_dir: Path) -> None:
        """Mark the validated result saved and show the new episode path."""
        with self._lock:
            self._computing = False
            self._can_save = False
        self._set_controls_disabled(False)
        self.set_status(f"Saved episode: {output_dir}")
        self._show_result_modal(
            "Episode saved",
            f"Saved the validated {request.side} arm trajectory to:\n\n`{output_dir}`",
        )

    def invalidate_result(self) -> None:
        """Close a result dialog when new input makes its result stale."""
        modal = self._result_modal
        self._result_modal = None
        if modal is not None and not modal.closed:
            modal.close()

    def _show_result_modal(self, title: str, markdown: str) -> None:
        self.invalidate_result()
        modal = self._server.gui.add_modal(title)
        self._result_modal = modal
        with modal:
            self._server.gui.add_markdown(markdown)
            close_button = self._server.gui.add_button("Close")
        close_button.on_click(lambda _: self._close_result_modal(modal))

    def _close_result_modal(self, modal: Any) -> None:
        if not modal.closed:
            modal.close()
        if self._result_modal is modal:
            self._result_modal = None

    @staticmethod
    def _format_frames(frames: tuple[int, ...], limit: int = 20) -> str:
        if not frames:
            return "none"
        shown = ", ".join(str(frame) for frame in frames[:limit])
        remainder = len(frames) - limit
        if remainder > 0:
            shown += f", … ({remainder} more)"
        return shown

    def _set_controls_disabled(self, disabled: bool) -> None:
        with self._lock:
            self._side_widget.disabled = disabled
            self._recompute_button.disabled = disabled
            self._save_button.disabled = disabled or not self._can_save
            for widget in self._widgets.values():
                widget.disabled = disabled
            self._transform_control.disable_axes = disabled
            self._transform_control.disable_sliders = disabled
            self._transform_control.disable_rotations = disabled

    def _on_recompute_click(self, _) -> None:
        with self._lock:
            if self._computing or self._controller_active:
                return
            self._computing = True
            self._can_save = False
            self._validated_revision = -1
            self._validated_side = ""
            values = self._values.copy()
            position_mm = np.array([values["x"], values["y"], values["z"]])
            rpy_deg = np.array([values["roll"], values["pitch"], values["yaw"]])
            request = PoseTuningSnapshot(
                revision=self._revision,
                side=self._side,
                frame=self._frame,
                reference_pose=self._reference_pose.copy(),
                controller_active=self._controller_active,
                controller_revision=self._controller_revision,
                dirty=self._dirty,
                computing=True,
                position_mm=position_mm,
                rpy_deg=rpy_deg,
                position_m=position_mm / 1000.0,
                quaternion_wxyz=quaternion_wxyz_from_rpy_xyz(np.deg2rad(rpy_deg)),
            )
            self._recompute_request = request
        self._set_controls_disabled(True)
        self.set_status("Computing full IK…")

    def _on_save_click(self, _) -> None:
        with self._lock:
            is_current = (
                self._can_save
                and self._revision == self._validated_revision
                and self._side == self._validated_side
            )
            if self._computing or self._controller_active or not is_current:
                self._can_save = False
                should_restore = True
            else:
                should_restore = False
                self._computing = True
                values = self._values.copy()
                position_mm = np.array([values["x"], values["y"], values["z"]])
                rpy_deg = np.array([values["roll"], values["pitch"], values["yaw"]])
                self._save_request = PoseTuningSnapshot(
                    revision=self._revision,
                    side=self._side,
                    frame=self._frame,
                    reference_pose=self._reference_pose.copy(),
                    controller_active=self._controller_active,
                    controller_revision=self._controller_revision,
                    dirty=self._dirty,
                    computing=True,
                    position_mm=position_mm,
                    rpy_deg=rpy_deg,
                    position_m=position_mm / 1000.0,
                    quaternion_wxyz=quaternion_wxyz_from_rpy_xyz(
                        np.deg2rad(rpy_deg)
                    ),
                )
        if should_restore:
            self._save_button.disabled = True
            return
        self._set_controls_disabled(True)
        self.set_status("Saving validated episode…")

    def _load_side_values(self, side: str, frame: int) -> None:
        model = self._models[side]
        pose = model.forward_pose(model.baseline_joint_positions[frame])
        position_mm = pose[:3] * 1000.0
        rpy_deg = np.rad2deg(quaternion_wxyz_to_rpy_xyz(pose[3:]))
        values = {
            "x": float(position_mm[0]),
            "y": float(position_mm[1]),
            "z": float(position_mm[2]),
            "roll": float(rpy_deg[0]),
            "pitch": float(rpy_deg[1]),
            "yaw": float(rpy_deg[2]),
        }
        self._values = values
        self._reference_pose = pose.copy()

    def _on_side_update(self, _) -> None:
        side = str(self._side_widget.value).lower()
        if side not in self._models:
            return
        with self._lock:
            if self._computing or self._controller_active:
                side = self._side
                should_restore = True
            else:
                should_restore = False
            if should_restore:
                values = None
            else:
                self._syncing = True
                self._side = side
                self._load_side_values(side, self._frame)
                self._dirty = False
                self._can_save = False
                self._validated_revision = -1
                self._validated_side = ""
                self._revision += 1
                values = self._values.copy()
        if should_restore:
            self._side_widget.value = side.title()
            return
        for name, widget in self._widgets.items():
            widget.value = values[name]
        self._set_transform_control_pose(values)
        self._save_button.disabled = True
        with self._lock:
            self._syncing = False

    async def _on_transform_control_update(
        self, event: viser.TransformControlsEvent
    ) -> None:
        """Give an active gizmo ownership until its drag-release event."""
        if event.phase not in ("start", "update", "end"):
            return

        raw_position_mm = np.asarray(event.target.position, dtype=np.float64) * 1000.0
        position_mm = np.clip(
            raw_position_mm, self._POSITION_MIN_MM, self._POSITION_MAX_MM
        )
        was_clamped = not np.allclose(position_mm, raw_position_mm)
        rpy_deg = np.rad2deg(
            quaternion_wxyz_to_rpy_xyz(np.asarray(event.target.wxyz, dtype=np.float64))
        )
        values = {
            "x": float(position_mm[0]),
            "y": float(position_mm[1]),
            "z": float(position_mm[2]),
            "roll": float(rpy_deg[0]),
            "pitch": float(rpy_deg[1]),
            "yaw": float(rpy_deg[2]),
        }
        with self._lock:
            if self._computing:
                should_restore = True
                if event.phase == "end":
                    self._controller_active = False
                restore_values = self._values.copy()
            else:
                should_restore = False
                if event.phase == "start" or not self._controller_active:
                    self._controller_reference_pose = self._pose_from_values(
                        self._values
                    )
                    self._controller_drag_changed = False
                self._controller_active = event.phase != "end"
                pose_changed = any(
                    not np.isclose(values[name], self._values[name], atol=1e-6)
                    for name in values
                )
                self._syncing = True
                self._values = values
                if event.phase == "update" and pose_changed:
                    self._controller_drag_changed = True
                    self._controller_revision += 1
                if event.phase == "end":
                    drag_changed = self._controller_drag_changed or pose_changed
                    reference_pose = self._controller_reference_pose
                    self._controller_reference_pose = None
                    self._controller_drag_changed = False
                    if drag_changed:
                        self._reference_pose = (
                            reference_pose.copy()
                            if reference_pose is not None
                            else self._pose_from_values(values)
                        )
                        self._dirty = True
                        self._user_edit_pending = True
                        self._revision += 1
                        self._can_save = False
                        self._validated_revision = -1
                        self._validated_side = ""
                    else:
                        self._reference_pose = self._pose_from_values(values)
                if event.phase == "start":
                    self._side_widget.disabled = True
                    self._recompute_button.disabled = True
                    self._save_button.disabled = True
                    for widget in self._widgets.values():
                        widget.disabled = True
                elif event.phase == "end":
                    self._side_widget.disabled = False
                    self._recompute_button.disabled = False
                    self._save_button.disabled = not self._can_save
                    for widget in self._widgets.values():
                        widget.disabled = False
                restore_values = values.copy()

        for name, widget in self._widgets.items():
            widget.value = restore_values[name]
        if should_restore or was_clamped:
            self._set_transform_control_pose(restore_values)
        if should_restore:
            return

        with self._lock:
            self._syncing = False
            controller_active = self._controller_active
            dirty = self._dirty
        if controller_active:
            self.set_status("Controller owns pose; numeric values follow drag")
        elif dirty:
            self._save_button.disabled = True
            self.set_status("Recompute required")
        else:
            self.set_status("Ready")

    @staticmethod
    def _position_from_values(values: dict[str, float]) -> np.ndarray:
        return np.array([values["x"], values["y"], values["z"]]) / 1000.0

    @staticmethod
    def _pose_from_values(values: dict[str, float]) -> np.ndarray:
        return np.concatenate(
            (
                PoseTuningGUI._position_from_values(values),
                PoseTuningGUI._quaternion_from_values(values),
            )
        )

    @staticmethod
    def _quaternion_from_values(values: dict[str, float]) -> np.ndarray:
        return quaternion_wxyz_from_rpy_xyz(
            np.deg2rad([values["roll"], values["pitch"], values["yaw"]])
        )

    def _set_transform_control_pose(self, values: dict[str, float]) -> None:
        self._transform_control.position = tuple(self._position_from_values(values))
        self._transform_control.wxyz = tuple(self._quaternion_from_values(values))
