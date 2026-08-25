"""Convert a UMI-Dex mcap episode into a viewer-native episode (telemetry.npz + metadata.json).

Mirrors the linker-sim UMI pipeline
(``scripts/umi_bag_to_ee_poses.py`` + ``add_hand_to_npz.py`` + ``controllers/ik.py``)
but runs **entirely offline** with ``yourdfpy`` — no MuJoCo, no ROS, no rosbags:

    /vut/pose (wrist 6-DoF) + /hand/joint_states (0-100%)
        --[mcap-ros2-support]-->  read
        --> resample @hz  (position lerp + orientation Slerp / hand nearest)
        --> rebase frame-0 -> identity, anchor to arm_right:tool0 FK at default joints
        --> DLS IK  dq = Jᵀ(JJᵀ+λ²I)⁻¹dx  retargets the 7-DoF right arm
        --> decode_hand  percent -> radians (linear-fit-v0)
        --> qpos (T,26) [L-arm7, R-arm7, L-hand6, R-hand6]

The IK runs here (offline, warm-started, unlimited iterations) so the viewer stays a pure
joint replayer: it reads the emitted telemetry.npz through the existing loader + the
``a7_lite_l6_umi_right`` robot config, with no runtime changes.

The output npz also carries linker-sim-compatible ``arm_right`` (T,7 EE pose, wxyz),
``hand_right`` (T,6 radians), ``arm_right_qpos`` (T,7 IK joints), ``decoder_convention`` and
``channel_map`` keys for cross-checking / feeding linker-sim's own replay_ik.

Usage:
    .venv/bin/python scripts/umi_mcap_to_episode.py --episode data/episode_000004 --side right

Tuning knobs (default to identity, mirroring linker-sim) if the arm lands in an awkward pose:
    --dx/--dy/--dz, --anchor-roll/-pitch/-yaw, --remap-roll/-pitch/-yaw, --dyaw, --recenter
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import yaml
import yourdfpy
from scipy.spatial.transform import Rotation, Slerp

from linker_robot_assets import asset_root
from linker_robot_assets.decoders import decode_hand, CONVENTION

WORKSTATION = "a7_lite_l6_dc"
HAND_COMPONENT = "linkerhand_l6"
EE_LINK = {"left": "arm_left_L7_Link", "right": "arm_right_R7_Link"}


# ---------- SE(3) helpers (Hamilton wxyz <-> scipy xyzw) ----------

def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]])


def _pose_to_T(p: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(_wxyz_to_xyzw(q_wxyz)).as_matrix()
    T[:3, 3] = p
    return T


def _T_to_pose7(T: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate([T[:3, 3], _xyzw_to_wxyz(q_xyzw)])


# ---------- mcap reading ----------

def _read_topic(mcap_paths: list[Path], topic: str):
    """Yield decoded messages for one topic across bag segments, with log-time ns."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    ts: list[int] = []
    msgs: list = []
    for path in mcap_paths:
        with open(path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _schema, _channel, message, ros_msg in reader.iter_decoded_messages(topics=[topic]):
                ts.append(int(message.log_time))
                msgs.append(ros_msg)
    if not msgs:
        raise RuntimeError(f"no {topic} messages in {[str(p) for p in mcap_paths]}")
    return np.asarray(ts, dtype=np.int64), msgs


def _read_vut_pose(mcap_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    ts, msgs = _read_topic(mcap_paths, "/vut/pose")
    rows = []
    for m in msgs:
        p, q = m.pose.position, m.pose.orientation
        rows.append([p.x, p.y, p.z, q.w, q.x, q.y, q.z])       # store wxyz
    order = np.argsort(ts)
    return ts[order], np.asarray(rows, dtype=np.float64)[order]


def _read_hand_joint_states(mcap_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ts, msgs = _read_topic(mcap_paths, "/hand/joint_states")
    names = list(msgs[0].names)
    pct = np.asarray([list(m.positions) for m in msgs], dtype=np.float64)
    order = np.argsort(ts)
    return ts[order], pct[order], names


# ---------- resampling (ported from linker-sim) ----------

def _resample_pose(t_src_ns: np.ndarray, poses: np.ndarray, t_tgt_ns: np.ndarray) -> np.ndarray:
    """Linear-interp position + Slerp orientation onto t_tgt_ns. Returns (T, 7) wxyz."""
    t = (t_src_ns - t_src_ns[0]).astype(np.float64) / 1e9
    t_tgt = (t_tgt_ns - t_src_ns[0]).astype(np.float64) / 1e9
    pos = np.stack([np.interp(t_tgt, t, poses[:, i]) for i in range(3)], axis=1)
    rots = Rotation.from_quat(np.stack([_wxyz_to_xyzw(q) for q in poses[:, 3:7]]))
    slerp = Slerp(t, rots)
    quats = np.stack([_xyzw_to_wxyz(q) for q in slerp(np.clip(t_tgt, t.min(), t.max())).as_quat()])
    return np.concatenate([pos, quats], axis=1)


def _resample_nearest(t_src_ns: np.ndarray, vals: np.ndarray, t_tgt_ns: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(t_src_ns, t_tgt_ns, side="left"), 1, len(t_src_ns) - 1)
    left = idx - 1
    pick = np.where(np.abs(t_src_ns[idx] - t_tgt_ns) < np.abs(t_src_ns[left] - t_tgt_ns), idx, left)
    return vals[pick]


# ---------- workstation kinematics (fast analytic FK + geometric-Jacobian DLS IK) ----------

def _rodrigues(axis: np.ndarray, th: float) -> np.ndarray:
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


class ArmChain:
    """Fast analytic FK + geometric-Jacobian DLS IK for one arm.

    FK is a product of the base→tool0 joint chain extracted from the URDF (validated
    to machine precision against yourdfpy). The geometric Jacobian is exact:
    column i = [zᵢ × (p_ee − pᵢ); zᵢ] with zᵢ/pᵢ the world axis/origin of revolute
    joint i. IK is DLS `dq = Jᵀ(JJᵀ+λ²I)⁻¹dx`, ported from linker-sim controllers/ik.py.
    Everything is µs-scale, so a full-episode IK is seconds and an anchor search is feasible.
    """

    def __init__(self, urdf: yourdfpy.URDF, side: str, damping: float = 0.05):
        self.damping = damping
        ee_link = EE_LINK[side]
        jmap = {j.child: j for j in urdf.robot.joints}
        chain = []
        link = ee_link
        while link in jmap:
            chain.append(jmap[link])
            link = jmap[link].parent
        chain = chain[::-1]

        self._steps: list[tuple[np.ndarray, np.ndarray | None]] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        for j in chain:
            axis = None
            if j.type in ("revolute", "continuous"):
                axis = np.asarray(j.axis, dtype=float)
                axis = axis / np.linalg.norm(axis)
                self.lower.append(j.limit.lower if j.limit else -np.pi)
                self.upper.append(j.limit.upper if j.limit else np.pi)
            self._steps.append((np.asarray(j.origin, dtype=float), axis))
        self.lower = np.array(self.lower)
        self.upper = np.array(self.upper)
        self.n = len(self.lower)
        assert self.n == 7, f"expected 7 arm joints, got {self.n}"

    def fk(self, q: np.ndarray):
        """Return (T_ee 4x4, axes_world (7,3), origins_world (7,3))."""
        T = np.eye(4)
        ai = 0
        axes = np.empty((self.n, 3))
        origins = np.empty((self.n, 3))
        for origin, axis in self._steps:
            T = T @ origin
            if axis is not None:
                axes[ai] = T[:3, :3] @ axis
                origins[ai] = T[:3, 3]
                R = np.eye(4)
                R[:3, :3] = _rodrigues(axis, q[ai])
                T = T @ R
                ai += 1
        return T, axes, origins

    def tool0_default(self) -> np.ndarray:
        return self.fk(np.zeros(self.n))[0]

    def solve(self, target: np.ndarray, q_init: np.ndarray, max_iters: int,
              tol_pos: float = 1e-4, tol_rot: float = 1e-3,
              max_step: float = 0.3) -> tuple[np.ndarray, float, float]:
        """target: (7,) wxyz pose. Returns (q_arm, pos_err_norm, rot_err_norm)."""
        tgt_pos = target[:3]
        tgt_R = Rotation.from_quat(_wxyz_to_xyzw(target[3:7])).as_matrix()
        q = np.clip(q_init.copy(), self.lower, self.upper)
        pe = re = 0.0
        for _ in range(max_iters):
            T, axes, origins = self.fk(q)
            pos = T[:3, 3]
            pos_err = tgt_pos - pos
            orn_err = Rotation.from_matrix(tgt_R @ T[:3, :3].T).as_rotvec()
            pe, re = float(np.linalg.norm(pos_err)), float(np.linalg.norm(orn_err))
            if pe < tol_pos and re < tol_rot:
                break
            J = np.empty((6, self.n))
            J[:3, :] = np.cross(axes, pos - origins).T
            J[3:, :] = axes.T
            dx = np.concatenate([pos_err, orn_err])
            dq = J.T @ np.linalg.solve(J @ J.T + (self.damping ** 2) * np.eye(6), dx)
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, self.lower, self.upper)
        return q, pe, re

    def track(self, arm_pose: np.ndarray, warmup: int, iters: int):
        """IK over a full trajectory, warm-started. Returns (arm_q, pos_errs, rot_errs)."""
        n_frames = len(arm_pose)
        arm_q = np.zeros((n_frames, self.n))
        pe = np.zeros(n_frames)
        re = np.zeros(n_frames)
        q = np.zeros(self.n)
        for f in range(n_frames):
            q, pe[f], re[f] = self.solve(arm_pose[f], q, warmup if f == 0 else iters)
            arm_q[f] = q
        return arm_q, pe, re


# ---------- anchor: rebase + place trajectory, optional Nelder-Mead search ----------

def _apply_anchor(rebased: np.ndarray, T_ws_tool0: np.ndarray,
                  params: np.ndarray) -> np.ndarray:
    """params = [dx,dy,dz, roll,pitch,yaw]. Returns arm_pose (T,7) wxyz."""
    dx, dy, dz, roll, pitch, yaw = params
    T_extra = np.eye(4)
    T_extra[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    T_xyz = np.eye(4)
    T_xyz[:3, 3] = [dx, dy, dz]
    T_anchor = T_xyz @ T_ws_tool0 @ T_extra
    return np.stack([_T_to_pose7(T_anchor @ T) for T in rebased])


def _search_anchor(chain: ArmChain, rebased: np.ndarray, T_ws_tool0: np.ndarray,
                   init: np.ndarray, n_probe: int = 30, rot_weight: float = 0.05):
    """Nelder-Mead over [dx,dy,dz,roll,pitch,yaw], minimizing subsampled IK RMS."""
    from scipy.optimize import minimize

    probe_idx = np.linspace(0, len(rebased) - 1, min(n_probe, len(rebased))).astype(int)

    def cost(params: np.ndarray) -> float:
        arm_pose = _apply_anchor(rebased[probe_idx], T_ws_tool0, params)
        _, pe, re = chain.track(arm_pose, warmup=120, iters=40)
        return float(np.sqrt(np.mean(pe ** 2)) + rot_weight * np.sqrt(np.mean(re ** 2)))

    res = minimize(cost, init, method="Nelder-Mead",
                   options={"xatol": 2e-3, "fatol": 1e-3, "maxiter": 300})
    return res.x, res.fun


# ---------- hand percent -> SDK-channel-order reorder ----------

def _canon(name: str) -> tuple[str, str]:
    """(finger, motion) key, ignoring lh_/rh_ prefix and cmc/mcp segment."""
    toks = name.replace("lh_", "").replace("rh_", "").split("_")
    return toks[0], toks[-1]


def _reorder_to_sdk(pct: np.ndarray, msg_names: list[str], side: str) -> np.ndarray:
    """Reorder /hand/joint_states columns into the decoder.yaml SDK channel order.

    The bag may order thumb DoFs roll-first while the L6 decoder expects pitch-first;
    matching by (finger, motion) instead of position is robust to that.
    """
    spec = yaml.safe_load(
        (asset_root() / "components" / "hands" / HAND_COMPONENT / "decoder.yaml").read_text()
    )
    prefix = {"left": "l", "right": "r"}[side]
    channels = [c.replace("{S}", prefix) for c in spec["channels"]]
    msg_col = {_canon(n): i for i, n in enumerate(msg_names)}
    order = []
    for ch in channels:
        key = _canon(ch)
        if key not in msg_col:
            raise KeyError(f"decoder channel {ch!r} ({key}) not in bag names {msg_names}")
        order.append(msg_col[key])
    return pct[:, order]


# ---------- main ----------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episode", type=Path, required=True, help="episode dir containing *.mcap")
    p.add_argument("--side", choices=["left", "right"], default="right")
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--warmup-iters", type=int, default=200, help="IK iters on frame 0")
    p.add_argument("--iters", type=int, default=100, help="IK iters on subsequent frames")
    # Anchor placement. By default a Nelder-Mead search finds the SE(3) anchor that
    # minimizes IK tracking error (mirrors linker-sim's anchor_search). The manual
    # knobs seed the search, or are used directly with --no-search.
    p.add_argument("--no-search", dest="search", action="store_false",
                   help="skip the anchor search; use --dx/dy/dz/--anchor-* directly")
    p.set_defaults(search=True)
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--dz", type=float, default=0.0)
    p.add_argument("--anchor-roll", type=float, default=0.0)
    p.add_argument("--anchor-pitch", type=float, default=0.0)
    p.add_argument("--anchor-yaw", type=float, default=0.0)
    p.add_argument("--remap-roll", type=float, default=0.0)
    p.add_argument("--remap-pitch", type=float, default=0.0)
    p.add_argument("--remap-yaw", type=float, default=0.0)
    args = p.parse_args()

    mcap_paths = [Path(x) for x in sorted(glob.glob(str(args.episode / "*.mcap")))]
    if not mcap_paths:
        print(f"error: no *.mcap in {args.episode}")
        return 2
    print(f"[umi] reading {len(mcap_paths)} segment(s): {[p.name for p in mcap_paths]}", flush=True)

    t_pose_ns, poses = _read_vut_pose(mcap_paths)
    t_hand_ns, pct_raw, names = _read_hand_joint_states(mcap_paths)
    print(f"[umi] /vut/pose: {len(poses)} msgs ; /hand/joint_states: {len(pct_raw)} msgs "
          f"names={names}", flush=True)

    # Uniform grid over the pose∩hand time span.
    t0 = max(int(t_pose_ns[0]), int(t_hand_ns[0]))
    t1 = min(int(t_pose_ns[-1]), int(t_hand_ns[-1]))
    dt_ns = int(round(1e9 / args.hz))
    target_ns = np.arange(t0, t1, dt_ns, dtype=np.int64)
    if len(target_ns) < 2:
        print(f"error: sync window too short: {(t1 - t0)/1e9:.3f}s @ {args.hz}Hz")
        return 2
    n_frames = len(target_ns)
    poses_rs = _resample_pose(t_pose_ns, poses, target_ns)
    pct_rs = _resample_nearest(t_hand_ns, pct_raw, target_ns)
    print(f"[umi] resampled to {n_frames} frames @ {args.hz}Hz (dt={dt_ns/1e9:.4f}s)", flush=True)

    # Load workstation URDF (yourdfpy) for the tool0 anchor + build the fast IK chain.
    urdf_path = asset_root() / "workstations" / WORKSTATION / "workstation.urdf"
    urdf = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=False,
                              build_collision_scene_graph=False, load_collision_meshes=False)
    chain = ArmChain(urdf, args.side)

    # Rebase frame-0 -> identity, optional axis remap (pre-anchor).
    T0_inv = np.linalg.inv(_pose_to_T(poses_rs[0, :3], poses_rs[0, 3:7]))
    rebased = np.stack([T0_inv @ _pose_to_T(p[:3], p[3:7]) for p in poses_rs])
    R_remap = Rotation.from_euler("xyz", [args.remap_roll, args.remap_pitch, args.remap_yaw]).as_matrix()
    T_remap = np.eye(4); T_remap[:3, :3] = R_remap
    T_remap_inv = np.eye(4); T_remap_inv[:3, :3] = R_remap.T
    rebased = np.stack([T_remap @ d @ T_remap_inv for d in rebased])

    # Anchor placement. Seed = translation that centers the trajectory on tool0-default
    # (+ any manual offsets); then optionally refine with a Nelder-Mead search.
    T_ws_tool0 = chain.tool0_default()
    centroid = _apply_anchor(rebased, T_ws_tool0, np.zeros(6))[:, :3].mean(axis=0)
    seed_shift = T_ws_tool0[:3, 3] - centroid
    seed = np.array([seed_shift[0] + args.dx, seed_shift[1] + args.dy, seed_shift[2] + args.dz,
                     args.anchor_roll, args.anchor_pitch, args.anchor_yaw])
    if args.search:
        params, cost = _search_anchor(chain, rebased, T_ws_tool0, seed)
        print(f"[umi] anchor search: xyz={np.round(params[:3],3)} "
              f"rpy={np.round(np.degrees(params[3:]),1)}° cost≈{cost*1000:.1f}mm", flush=True)
    else:
        params = seed
    arm_pose = _apply_anchor(rebased, T_ws_tool0, params)
    print(f"[umi] EE target xyz range: "
          f"x[{arm_pose[:,0].min():.3f},{arm_pose[:,0].max():.3f}] "
          f"y[{arm_pose[:,1].min():.3f},{arm_pose[:,1].max():.3f}] "
          f"z[{arm_pose[:,2].min():.3f},{arm_pose[:,2].max():.3f}]", flush=True)

    # Full-resolution DLS IK, warm-started frame to frame.
    arm_q, pe, re = chain.track(arm_pose, args.warmup_iters, args.iters)
    print(f"[umi] IK residual: pos mean={pe.mean()*1000:.2f}mm max={pe.max()*1000:.2f}mm ; "
          f"rot mean={np.degrees(re.mean()):.2f}° max={np.degrees(re.max()):.2f}°", flush=True)

    # Hand: reorder bag columns into the decoder's SDK channel order (by name),
    # then decode percent -> radians. (linker-sim's legacy-wiring path is not
    # ported: no legacy client bags here, and the correct behaviour under a
    # name-based reorder can't be verified without one — add it with real data.)
    pct_sdk = _reorder_to_sdk(pct_rs, names, args.side)
    hand_rad = decode_hand(HAND_COMPONENT, args.side, pct_sdk).astype(np.float32)
    print(f"[umi] hand decoded ({CONVENTION}): shape={hand_rad.shape} "
          f"rad range " + ", ".join(f"{hand_rad[:,i].min():+.2f}..{hand_rad[:,i].max():+.2f}"
                                    for i in range(hand_rad.shape[1])), flush=True)

    # Assemble qpos (T,26) [L-arm7, R-arm7, L-hand6, R-hand6]; other side stays at defaults.
    qpos = np.zeros((n_frames, 26), dtype=np.float32)
    arm_cols = (0, 7) if args.side == "left" else (7, 14)
    hand_cols = (14, 20) if args.side == "left" else (20, 26)
    qpos[:, arm_cols[0]:arm_cols[1]] = arm_q.astype(np.float32)
    qpos[:, hand_cols[0]:hand_cols[1]] = hand_rad

    out_npz = args.episode / "telemetry.npz"
    np.savez(
        out_npz,
        qpos=qpos,
        **{f"arm_{args.side}": arm_pose.astype(np.float32)},
        **{f"arm_{args.side}_qpos": arm_q.astype(np.float32)},
        **{f"hand_{args.side}": hand_rad},
        decoder_convention=np.array(CONVENTION),
        channel_map=np.array("umi-dex"),
    )
    meta = {"fps": args.hz, "dt": dt_ns / 1e9, "frame_count": n_frames,
            "source": "umi_mcap", "side": args.side}
    (args.episode / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[umi] wrote {out_npz} and metadata.json "
          f"({n_frames} frames). Replay with:\n"
          f"      .venv/bin/python scripts/replay.py --robot a7_lite_l6_umi_{args.side} "
          f"--episode {args.episode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
