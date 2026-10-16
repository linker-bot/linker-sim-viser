"""Frame-deadline pacing for `PlaybackGUI`.

Regression for issue #10 (playback stutter). The render loop used to sleep a
fixed 1/120s, so a frame was only *presented* on whichever loop iteration first
crossed its deadline. The loop period (~12ms: sleep + render cost) has no
relation to the frame period (33.3ms at 30fps), so frames landed 29-39ms apart
at 1x -- and at 4x the loop was slower than the frame rate, dropping ~130 of
411 frames in bursts. `app.run` now sleeps until the next frame is due, using
`seconds_until_next_frame()`.

These pin the deadline contract rather than wall-clock jitter, which would be
flaky under load. The jitter improvement itself was measured by hand, on both
episode_000000 (interval std 4.25ms -> 0.47ms at 1x, 130 dropped frames -> 0 at
4x) and episode_000009_O6 with the reporter's a7_lite_o6_dc config (4.23ms ->
0.50ms at 1x, 128 dropped frames -> 0 at 4x).
"""

import math

import pytest
import viser

from linker_sim_viser.timeline import PlaybackGUI

DT = 1 / 30
N_FRAMES = 411


@pytest.fixture
def gui():
    server = viser.ViserServer(port=0, verbose=False)
    try:
        yield PlaybackGUI(server, n_frames=N_FRAMES, dt=DT, default_loop=False)
    finally:
        server.stop()


def test_paused_has_no_deadline(gui):
    """Nothing is pending while paused, so the caller picks its own poll rate."""
    assert gui.seconds_until_next_frame() == math.inf


def test_on_a_frame_boundary_waits_one_full_frame(gui):
    gui._toggle_play()
    gui._frame_f = 7.0

    assert gui.seconds_until_next_frame() == pytest.approx(DT)


def test_mid_frame_waits_only_the_remainder(gui):
    """The fix hinges on this: partway into a frame, wait the rest of it, not
    a fixed interval that overshoots or undershoots the boundary."""
    gui._toggle_play()
    gui._frame_f = 7.25

    assert gui.seconds_until_next_frame() == pytest.approx(0.75 * DT)


@pytest.mark.parametrize("speed,factor", [("0.25x", 4.0), ("1.0x", 1.0),
                                          ("2.0x", 0.5), ("4.0x", 0.25)])
def test_deadline_scales_with_speed(gui, speed, factor):
    gui._speed.value = speed
    gui._toggle_play()
    gui._frame_f = 3.0

    assert gui.seconds_until_next_frame() == pytest.approx(DT * factor)


@pytest.mark.parametrize("frac", [0.0, 0.01, 0.5, 0.99])
def test_deadline_always_positive_and_within_one_frame(gui, frac):
    """Guarantees forward progress: a zero wait would busy-spin, and a wait
    longer than one frame period would sail past the frame it is pacing."""
    gui._speed.value = "2.0x"
    gui._toggle_play()
    gui._frame_f = 10 + frac

    wait = gui.seconds_until_next_frame()
    assert 0.0 < wait <= DT / 2 + 1e-12
