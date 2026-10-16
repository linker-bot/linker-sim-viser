"""Tail capping for `GrowingTrail`.

`trails.max_points` was declared in `configs/viewer.yaml` and parsed into
`TrailsConfig`, but nothing ever read it: `GrowingTrail` took no such parameter,
so the trail grew to the full episode length and the advertised 500-point cap
was silently unenforced. That made per-frame cost scale with episode length —
`update` rewrites the whole segment buffer every frame, and viser's client
rebuilds the line geometry from scratch on each points/colors assignment
(allocating a fresh LineSegmentsGeometry and disposing the old one), so per
trail a 3000-frame episode pushed ~88KB/frame where a 411-frame one pushed
~12KB. The trail now slides a fixed-size window, so the cost is a flat ~12KB
at any episode length.

The window is also why these assertions read the buffer shape: the shape *is*
the per-frame cost.
"""

import numpy as np
import pytest
import viser

from linker_sim_viser.trails import GrowingTrail


@pytest.fixture
def server():
    s = viser.ViserServer(port=0, verbose=False)
    try:
        yield s
    finally:
        s.stop()


def _ramp(n_frames: int) -> np.ndarray:
    """(T, 3) positions whose x coordinate equals the frame index."""
    xyz = np.zeros((n_frames, 3), dtype=np.float32)
    xyz[:, 0] = np.arange(n_frames)
    return xyz


def test_buffer_is_capped_independent_of_episode_length(server):
    """The point of the fix: a long episode must not cost more per frame."""
    short = GrowingTrail(server, positions=_ramp(50), name="/t/a", max_points=20)
    long_ = GrowingTrail(server, positions=_ramp(5000), name="/t/b", max_points=20)

    short.update(49)
    long_.update(4999)

    assert short._segments.points.shape == (19, 2, 3)
    assert long_._segments.points.shape == (19, 2, 3)


@pytest.mark.parametrize("frame", [0, 1, 25, 99])
def test_buffer_size_constant_across_frames(server, frame):
    """Cost must not grow as playback advances, either."""
    trail = GrowingTrail(server, positions=_ramp(100), name="/t", max_points=10)

    trail.update(frame)

    assert trail._segments.points.shape == (9, 2, 3)
    assert trail._segments.colors.shape == (9, 2, 3)


def test_window_slides_and_drops_oldest(server):
    """Past the window length the tail slides: the oldest points fall off rather
    than the buffer growing to hold them."""
    trail = GrowingTrail(server, positions=_ramp(100), name="/t", max_points=10)

    trail.update(50)

    xs = trail._segments.points[..., 0]
    # Window ends at the current frame and spans max_points-1 segments back.
    assert xs.max() == pytest.approx(50.0)
    assert xs.min() == pytest.approx(41.0)      # 50 - (10 - 1)


def test_uncapped_window_matches_full_growing_trail(server):
    """max_points >= T keeps the original behaviour, so short episodes are
    unchanged by the cap: every segment from frame 0 stays visible."""
    trail = GrowingTrail(server, positions=_ramp(10), name="/t", max_points=500)

    trail.update(5)

    pts = trail._segments.points
    assert pts.shape == (9, 2, 3)               # T-1, as before
    np.testing.assert_allclose(pts[:5, 0, 0], [0, 1, 2, 3, 4])
    np.testing.assert_allclose(pts[:5, 1, 0], [1, 2, 3, 4, 5])
    # Segments past the head collapse onto it (zero-length -> invisible).
    np.testing.assert_allclose(pts[5:, :, 0], 5.0)


def test_head_tracks_the_current_frame(server):
    trail = GrowingTrail(server, positions=_ramp(100), name="/t", max_points=10)

    trail.update(73)

    assert trail._head.position[0] == pytest.approx(73.0)


def test_fade_runs_dim_tail_to_bright_head(server):
    """The comet look survives the sliding window: the gradient is rebuilt over
    the visible portion, so the head is full colour wherever the window sits."""
    color = (100, 200, 255)
    trail = GrowingTrail(server, positions=_ramp(100), name="/t",
                        color=color, max_points=10)

    trail.update(50)

    cols = trail._segments.colors
    # Last visible vertex is the head, at full intensity.
    np.testing.assert_array_equal(cols[8, 1], color)
    # And the oldest visible vertex is dimmer.
    assert int(cols[0, 0].sum()) < int(cols[8, 1].sum())


def test_out_of_range_frame_is_clamped(server):
    """Matches the timeline's own clamping (issue #11) rather than IndexError-ing
    out of the render loop."""
    trail = GrowingTrail(server, positions=_ramp(100), name="/t", max_points=10)

    trail.update(10_000)

    assert trail._head.position[0] == pytest.approx(99.0)


@pytest.mark.parametrize("max_points", [0, 1, 2])
def test_window_never_degenerates_below_one_segment(server, max_points):
    """A nonsense cap must still leave a drawable buffer."""
    trail = GrowingTrail(server, positions=_ramp(100), name="/t",
                        max_points=max_points)

    trail.update(50)

    assert trail._segments.points.shape == (1, 2, 3)
