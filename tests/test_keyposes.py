"""Keypose detector: locks in the median-filter + threshold-crossing contract."""

import numpy as np

from linker_sim_viser.trails import keypose_events_from_hand


def _open_hand_sdk(T: int) -> np.ndarray:
    """SDK values near 200 (mostly open per linear-fit-v0)."""
    return np.full((T, 6), 200.0, dtype=np.float32)


def test_no_events_when_hand_holds_still():
    events = keypose_events_from_hand(_open_hand_sdk(30), side="left")
    assert events == []


def test_single_frame_dropout_is_rejected():
    hand = _open_hand_sdk(30)
    hand[15, :] = 0.0                                 # one-frame full-close spike
    assert keypose_events_from_hand(hand, side="left") == []


def test_two_frame_dropout_is_rejected():
    hand = _open_hand_sdk(30)
    hand[15:17, :] = 0.0                              # two-frame spike
    assert keypose_events_from_hand(hand, side="left") == []


def test_sustained_grasp_and_release_detected():
    hand = _open_hand_sdk(40)
    hand[10:25, :] = 10.0                             # 15-frame close
    events = keypose_events_from_hand(hand, side="left")
    assert len(events) == 2
    kinds = [e["kind"] for e in events]
    assert kinds == ["grasp_close", "release"]
    # Median-5 delays the crossing by ~2 frames from the transition edge.
    assert 8 <= events[0]["frame"] <= 14
    assert 22 <= events[1]["frame"] <= 28
    for e in events:
        assert e["side"] == "left"
