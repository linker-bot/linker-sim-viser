"""Sidebar playback controls: frame slider, play/pause, speed presets, loop.

v0 keeps everything in the Viser sidebar. When curation gets cramped, switch
to a companion-DOM timeline (Path 2 in the design doc).
"""

from __future__ import annotations

import time

import viser


class PlaybackGUI:
    """Owns playback state + its Viser widgets. Drive it from the main loop.

    Timing model: `tick()` advances `frame` by `elapsed_wall_time * speed / dt`,
    accumulating fractional progress across ticks so speed changes stay smooth.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        n_frames: int,
        dt: float,
        speed_presets: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
        default_speed: float = 1.0,
        default_loop: bool = True,
    ) -> None:
        self._n_frames = n_frames
        self._dt = dt
        self._frame_f = 0.0                     # float accumulator
        self._last_tick_wall: float | None = None

        with server.gui.add_folder("Playback"):
            self._slider = server.gui.add_slider(
                "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
            )
            self._play_btn = server.gui.add_button("Play")
            self._speed = server.gui.add_dropdown(
                "Speed",
                options=tuple(f"{s}x" for s in speed_presets),
                initial_value=f"{default_speed}x",
            )
            self._loop = server.gui.add_checkbox("Loop", initial_value=default_loop)
            self._time_label = server.gui.add_text(
                "Time", initial_value=self._format_time(0), disabled=True
            )

        self._playing = False
        self._play_btn.on_click(lambda _: self._toggle_play())
        self._slider.on_update(lambda _: self._on_scrub())

    @property
    def frame(self) -> int:
        return int(self._frame_f)

    def tick(self) -> None:
        """Call once per main-loop iteration."""
        now = time.monotonic()
        if not self._playing:
            self._last_tick_wall = now
            return
        if self._last_tick_wall is None:
            self._last_tick_wall = now
            return

        elapsed = now - self._last_tick_wall
        self._last_tick_wall = now
        speed = self._current_speed()
        self._frame_f += elapsed * speed / self._dt

        if self._frame_f >= self._n_frames:
            if self._loop.value:
                self._frame_f = self._frame_f % self._n_frames
            else:
                self._frame_f = float(self._n_frames - 1)
                self._set_playing(False)

        # Push to widgets. Guard against re-entering on_update on the slider.
        target = int(self._frame_f)
        if self._slider.value != target:
            self._slider.value = target
        self._time_label.value = self._format_time(target)

    def _current_speed(self) -> float:
        return float(self._speed.value.rstrip("x"))

    def _toggle_play(self) -> None:
        self._set_playing(not self._playing)

    def _set_playing(self, playing: bool) -> None:
        self._playing = playing
        self._play_btn.label = "Pause" if playing else "Play"
        self._last_tick_wall = None            # reset elapsed accumulator

    def _on_scrub(self) -> None:
        if not self._playing:
            self._frame_f = float(self._slider.value)
            self._time_label.value = self._format_time(self._slider.value)

    def _format_time(self, frame: int) -> str:
        t = frame * self._dt
        return f"{t:6.2f}s  ({frame}/{self._n_frames - 1})"
