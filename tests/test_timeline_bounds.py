"""Frame-slider bounds for `PlaybackGUI`.

Regression for issue #11: typing a frame number past the end of the episode
used to raise `IndexError: index 7180 is out of bounds for axis 0 with size
719` out of the main loop and stop the server. Viser does not clamp typed
slider input to the widget's min/max, so `PlaybackGUI` has to.

These drive a real (loopback, OS-assigned port) ViserServer so the actual viser
slider setter/callback machinery is under test. Assigning `.value` from Python
takes the same no-clamp path and fires the same `on_update` callbacks as a
value arriving from the browser, so it stands in for a typed-in frame.
"""

import pytest
import viser

from linker_sim_viser.timeline import PlaybackGUI

N_FRAMES = 719          # the episode length from the report


@pytest.fixture
def gui():
    server = viser.ViserServer(port=0, verbose=False)
    try:
        yield PlaybackGUI(server, n_frames=N_FRAMES, dt=1 / 30)
    finally:
        server.stop()


def test_frame_past_end_clamps_to_last_frame(gui):
    """The reported case: 7180 typed into a 719-frame episode."""
    gui._slider.value = 7180

    assert gui.frame == N_FRAMES - 1        # a valid index -> no IndexError
    assert gui._slider.value == N_FRAMES - 1  # and the widget agrees


def test_frame_equal_to_count_clamps(gui):
    """Off-by-one: `n_frames` looks in-range but the last valid index is
    n_frames - 1. Verified against data/episode_000010_l20, where typing 1053
    into a 1053-frame episode raised
    "IndexError: index 1053 is out of bounds for axis 0 with size 1053".
    """
    gui._slider.value = N_FRAMES

    assert gui.frame == N_FRAMES - 1
    assert gui._slider.value == N_FRAMES - 1


def test_negative_frame_clamps_to_zero(gui):
    """The other end: a negative index would silently render from the tail."""
    gui._slider.value = -20

    assert gui.frame == 0
    assert gui._slider.value == 0


def test_in_range_scrub_is_untouched(gui):
    """Normal scrubbing still lands exactly where the user put it."""
    gui._slider.value = 300

    assert gui.frame == 300
    assert gui._slider.value == 300
