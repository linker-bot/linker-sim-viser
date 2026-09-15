"""SDK slot layout for the L25/L20/G20 hand (one hand, three names).

Regression for issue #9: the hand's SDK puts a 20-slot vector on the wire per
hand, four of them reserved (11-14), and the collection pipeline stores that
packet verbatim -- so qpos is 7+7+20+20 = 54 wide. `a7_lite_l25_dc.yaml` used
to declare contiguous 16-wide slices, `qpos[14:30]` and `qpos[30:46]`, which
fed reserved zeros to four PIP joints and straddled the hand boundary on the
right side (its block started 4 columns inside the left hand's data). Nothing
caught it: the width guard only rejected arrays *narrower* than declared.

Hand block widths now come from the hand's declared SDK layout via
`sdk_channel_width`, and the guard rejects a width mismatch either way.

Synthetic episodes, but the real robot config -- the config file is the thing
that was wrong, so it is under test rather than mocked.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from linker_sim_viser.config import DecoderSpec, StreamSpec, load_robot_config
from linker_sim_viser.episode import load_episode

REPO_ROOT = Path(__file__).resolve().parents[1]
L25_CONFIG = REPO_ROOT / "configs" / "robots" / "a7_lite_l25_dc.yaml"

# Needs the linker-robot-assets submodule for the L25 workstation + hand URDFs.
try:
    ROBOT = load_robot_config(L25_CONFIG)
except (FileNotFoundError, ImportError) as exc:      # pragma: no cover
    pytest.skip(f"linker-robot-assets unavailable: {exc}", allow_module_level=True)

ARM_COLS = 14
RAW_SLOTS = 20                       # SDK wire vector per hand
RESERVED = (11, 12, 13, 14)          # 预留 slots, no actuated joint
ACTIVE = tuple(i for i in range(RAW_SLOTS) if i not in RESERVED)


def _episode(tmp_path: Path, qpos: np.ndarray) -> Path:
    ep = tmp_path / "episode_synthetic"
    ep.mkdir(parents=True)
    np.savez(ep / "telemetry.npz", qpos=qpos.astype(np.float32))
    (ep / "metadata.json").write_text(json.dumps({}))
    return ep


def _raw_qpos(frames: int = 3, *, active: float, reserved: float) -> np.ndarray:
    """A 54-column recording: arms zeroed, hands as raw 20-slot SDK blocks."""
    q = np.zeros((frames, ARM_COLS + 2 * RAW_SLOTS), dtype=np.float32)
    for base in (ARM_COLS, ARM_COLS + RAW_SLOTS):
        for slot in range(RAW_SLOTS):
            q[:, base + slot] = reserved if slot in RESERVED else active
    return q


def test_hand_block_width_comes_from_the_sdk_layout():
    """Widths are derived, not hand-copied: 20 raw slots, not 16 active."""
    hands = [s for s in ROBOT.streams if s.decoder is not None]
    assert [s.columns() for s in hands] == [(14, 34), (34, 54)]
    assert all(s.decoder.slots == "raw" for s in hands)
    # The joint lists stay the actuated count -- decode_hand drops reserved.
    assert all(len(s.joints) == len(ACTIVE) for s in hands)


def test_reserved_slots_never_reach_a_joint(tmp_path):
    """The #9 failure, stated as a property.

    SDK 255 -> URDF lower limit, SDK 0 -> upper limit. Fill every active slot
    with 255 and every reserved slot with 0, then check each joint against the
    workstation URDF's own limits: a joint fed a reserved column would sit at
    its upper limit instead. Pre-fix, the four PIP joints did exactly that.
    """
    from linker_sim_viser.decoders import _urdf_joint_limits

    ep = _episode(tmp_path, _raw_qpos(active=255.0, reserved=0.0))
    episode = load_episode(ep, ROBOT)
    limits = _urdf_joint_limits(str(ROBOT.urdf_path))

    hand_joints = {j: q for j, q in episode.joint_positions.items() if "hand" in j}
    assert len(hand_joints) == 2 * len(ACTIVE)
    for joint, q in hand_joints.items():
        lower, upper = limits[joint]
        assert q[0] == pytest.approx(lower, abs=1e-5), (
            f"{joint} sits at {q[0]}, not its lower limit {lower} "
            f"(upper is {upper}) — a reserved slot reached it"
        )


def test_garbage_in_reserved_slots_is_ignored(tmp_path):
    """Reserved slots carry no contract, so their contents must not matter."""
    clean = _raw_qpos(active=200.0, reserved=0.0)
    dirty = clean.copy()
    for base in (ARM_COLS, ARM_COLS + RAW_SLOTS):
        for slot in RESERVED:
            dirty[:, base + slot] = 12345.0

    a = load_episode(_episode(tmp_path / "a", clean), ROBOT).joint_positions
    b = load_episode(_episode(tmp_path / "b", dirty), ROBOT).joint_positions

    assert a.keys() == b.keys()
    for joint in a:
        np.testing.assert_array_equal(a[joint], b[joint], err_msg=joint)


def test_extra_columns_stay_legal_but_are_noted(tmp_path, capsys):
    """Over-width is not a mismatch — a config may drive part of a recording on
    purpose (a7_lite_l6_umi_left reads 20 of 26). It earns a note, not an
    error, since unaccounted columns are the early smell of a #9-style
    layout mismatch."""
    ep = _episode(tmp_path, np.zeros((3, 60)))          # 60 > declared 54

    load_episode(ep, ROBOT)                             # must not raise

    note = capsys.readouterr().out
    assert "60 columns wide" in note and "reads 54" in note
    assert "6 column(s) unused" in note


def test_under_wide_qpos_still_rejected(tmp_path):
    """Issue #7's guard must survive: too narrow is still an error."""
    ep = _episode(tmp_path, np.zeros((3, 26)))          # an L6-shaped recording

    with pytest.raises(ValueError) as excinfo:
        load_episode(ep, ROBOT)

    msg = str(excinfo.value)
    assert "54" in msg and "26" in msg
    assert "does not match" in msg


def test_stripped_16_wide_recording_names_the_mode_to_switch(tmp_path):
    """The layout the config used to assume: 46 cols, reserved pre-stripped.

    We have no such recording, and guessing wrong is what #9 was. It errors,
    and names the one-word config change that would read it.
    """
    ep = _episode(tmp_path, np.zeros((3, 46)))

    with pytest.raises(ValueError) as excinfo:
        load_episode(ep, ROBOT)

    msg = str(excinfo.value)
    assert "54" in msg and "46" in msg
    assert "slots: active" in msg                       # points at the fix


def test_offset_without_a_decoder_is_an_error():
    """Only a decoded stream has a SDK layout to derive its width from."""
    stream = StreamSpec(key="qpos", joints=["a", "b"], offset=14)

    with pytest.raises(ValueError, match="without a decoder"):
        stream.columns()


def test_stream_needs_slice_or_offset():
    stream = StreamSpec(key="qpos", joints=["a"])

    with pytest.raises(ValueError, match="neither 'slice' nor 'offset'"):
        stream.columns()


def test_active_mode_still_derives_the_stripped_width():
    """`slots: active` remains available for a pre-stripped recording."""
    stream = StreamSpec(
        key="qpos",
        joints=["j"] * len(ACTIVE),
        offset=14,
        decoder=DecoderSpec(kind="linkerhand_l25", side="left", slots="active"),
    )

    assert stream.columns() == (14, 14 + len(ACTIVE))
