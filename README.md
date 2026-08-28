# linker-sim-viser

Language: [English](README.md) | [中文](README_zh.md)

Browser-based, physics-free replay viewer for LinkerBot recorded data. Built on [Viser](https://github.com/nerfstudio-project/viser) with `yourdfpy` for URDF loading (mimic joints resolve natively — LinkerHand tendon coupling works with no extra code).

Part of the `linker-sim` family. Sibling repos: `linker-sim-mujoco` (physics-aware replay; planned), `linker-sim-isaac` (forward sim + motion planning).

## Status

**v0 shipping.** Single-episode playback with comet-style EE trails and keypose stamps.

## What's in v0

- Load one recorded episode from `data/episode_XXXXXX/` (SDK format 1.4: `telemetry.npz` + `metadata.json`).
- Render the dual-arm + LinkerHand workstation URDF, driven by the episode's joint stream.
- Sidebar playback: frame slider, play/pause, speed presets (`0.25x` / `0.5x` / `1x` / `2x` / `4x`), loop, time readout.
- Hand decode via the installed `linker-robot-assets` package (`decode_hand`): raw SDK packets are reordered into URDF joint order and mapped to joint radians using limits read from the hand's URDF.
- **Comet trails** per configured EE frame: grows as playback advances, fades from bright head to dim tail, tracked by a coloured sphere.
- **Keypose stamps** at grasp/release events, detected from the SDK hand stream with a median-5 filter that rejects the 1–2 frame sensor dropouts.

## Install

**Prerequisite — the `linker-robot-assets` submodule.** This repo vendors the authoritative robot assets + hand decoders as a git submodule at [`packages/linker-robot-assets`](packages/linker-robot-assets) (the [`linker-sim-assets`](https://github.com/linker-bot/linker-sim-assets) repo, pinned to a release commit), installed as an editable path dependency; robot configs resolve URDFs through it (`pkg://` paths).

**Git LFS is required.** The submodule's meshes are LFS-tracked. Install `git-lfs` once per machine (`apt install git-lfs && git lfs install`), then clone with submodules and pull the meshes:

```bash
git clone --recurse-submodules <repo-url>
# already cloned without submodules / LFS?
git submodule update --init
git -C packages/linker-robot-assets lfs pull

uv venv
uv sync --extra dev
```

## Requirements

- **Recorded episodes** dropped in `data/episode_XXXXXX/`. `data/` is `.gitignore`'d (episodes are large binary bundles); obtain a recorded episode bundle and drop it in `data/`.

Robot URDFs and meshes load from the installed `linker-robot-assets` package (see [Install](#install)): `urdf_path` in `configs/robots/*.yaml` uses `pkg://` paths that resolve against the package's `asset_root()`. To point at a URDF that lives elsewhere, edit `urdf_path`.

## Run

```bash
.venv/bin/python scripts/replay.py --robot a7_lite_l6_dc --episode data/episode_000000
```

Open `http://localhost:8080`. `--robot NAME` resolves to `configs/robots/NAME.yaml`; `--episode PATH` is a directory containing `telemetry.npz` + `metadata.json`.

Flags:
- `--port INT` — override the viewer port (default from `configs/viewer.yaml`).
- `--viewer-config PATH` — use a different viewer config.

## Viewer config

`configs/viewer.yaml` holds playback + annotation defaults (override with `--viewer-config PATH`). The two visual annotations are **off by default** — enable them here as you wish:

```yaml
trails:
  enabled: true      # comet EE motion trail per configured ee_frame (default: false)
  max_points: 500    # downsample long trajectories to at most this many points
keyposes:
  enabled: true      # 3D "grasp_close@N" / "release@N" stamps at detected grasp/release events (default: false)
```

## Tests

```bash
.venv/bin/python -m pytest
```

Loader and decoder tests run against `data/episode_000000/`; they skip cleanly if that directory isn't present.

## UMI-Dex mcap replay (experimental)

UMI-Dex handheld bags (ROS 2 mcap: a 6-DOF wrist pose on `/vut/pose` + LinkerHand L6 percent on
`/hand/joint_states`) are replayed by first converting them to a viewer-native episode, mirroring
`linker-sim`'s UMI pipeline. The converter reads the bag, resamples to a fixed rate, anchors the
wrist pose to the arm's `tool0` via a Nelder-Mead search, solves DLS IK to retarget the 7-DOF right
arm, decodes the hand to radians, and writes `telemetry.npz` + `metadata.json` into the episode dir:

```bash
uv pip install -e ".[umi]"     # mcap + scipy (one-time)
.venv/bin/python scripts/umi_mcap_to_episode.py --episode data/episode_000004 --side right
.venv/bin/python scripts/replay.py --robot a7_lite_l6_umi_right --episode data/episode_000004
```

The converter prints the IK tracking residual (expect sub-mm to few-mm mean position error after the
anchor search). Only the right arm + right hand are driven; the left arm stays at its default pose.
If the arm lands awkwardly, `--no-search` plus manual `--dx/dy/dz`, `--anchor-roll/pitch/yaw`,
`--remap-roll/pitch/yaw` knobs are available (same semantics as linker-sim). This is offline
retargeting, not runtime forward simulation.

## Adding a new robot config

`configs/robots/*.yaml` maps npz columns to URDF joints and declares which links to trail:

```yaml
urdf_path: relative/or/absolute/path/to/workstation.urdf

streams:
  - key: qpos           # npz key
    slice: [0, 7]       # column range (optional)
    joints:             # URDF joint names in column order — authoritative
      - arm_left_L1_Joint
      - ...
  - key: qpos
    slice: [14, 20]
    joints: [...]
    decoder: {kind: linkerhand_l6, side: left, sdk_range: [0, 255]}

ee_frames:
  - {label: left_tcp,  link: hand_left_lh_hand_base_link,  color: [80, 140, 220]}
  - {label: right_tcp, link: hand_right_rh_hand_base_link, color: [230, 130, 60]}
```

## Non-goals (v0)

By design:

- **No physics.** Dynamics-aware replay is the domain of the planned `linker-sim-mujoco`.
- **No QC / anomaly detection.** Recorded data is treated as clean. Report upstream if you see systemic sensor issues.
- **No *runtime* forward simulation.** Playback never simulates forward. (Offline IK retargeting for replay — e.g. the UMI mcap converter — is allowed.)
- No multi-episode browser, no camera video sync, no annotations, no LeRobot parquet or ROS 2 `.mcap` ingest yet.

## Known limitations

Read these before filing bugs — they're deliberate v0 tradeoffs or unresolved upstream issues.

- **Hand decoding is approximate.** Decoding is delegated to `linker-robot-assets` (`decode_hand`), which rescales the raw SDK packet and linearly maps it into each joint's URDF `[lower, upper]`. The upstream package states outright that the true SDK angle convention hasn't been finalized yet. Symptom you'll see: fingertips of the thumb and index overlap slightly at rest. Every finger pose we render is off by an unknown amount. Treat all v0-rendered hand poses as approximate until the SDK ships a real convention.
- **Keypose detector is unverified on real grasps.** Synthetic tests prove it *rejects* single- and two-frame dropouts and *finds* sustained closures. The reported event frame lags the true crossing by up to `median_window // 2` frames (~67ms at 30fps). The 50% threshold is a starting guess. Expect to iterate on `median_window` and `threshold_percent` once episodes with real grasps land.
- **SDK sensor dropouts show through as visible jerks.** ~4% of frames per hand carry 1–2 frame spikes (index/thumb DoFs snapping fully closed then back). The keypose detector filters them; the pose stream doesn't (design policy: viewer stays faithful, upstream fixes the sensor path). If this becomes a blocker, we can add an opt-in render-side median filter — flag it and we'll reopen.
- **Precomputed EE poses in the npz are unusable.** `ee_poses_qpos_{left,right}` are in each arm's J2 frame, not workstation world (the collection pipeline used single-arm URDFs). We ignore them and FK from the workstation URDF instead. No cross-check is performed.
- **Camera + trail visuals are tuned for `a7_lite_l6_dc`.** Initial camera pose, `head_radius`, `tail_intensity`, and trail `line_width` are hardcoded for this workstation's scale. A taller/shorter robot will need re-tuning; no config path for that yet.
- **Hand decoding lives in `linker-robot-assets`, not vendored here.** The decoder (`linker_robot_assets.decoders.decode_hand`) and its per-hand `decoder.yaml` sidecar ship inside the installed package; this repo keeps only a small URDF joint-limit reader (`src/linker_sim_viser/decoders.py`) used for validation. There is no local copy to drift out of sync, but decode behavior is only as current as the pinned `linker-robot-assets` submodule commit.

## Roadmap

v1+ backlog: multi-episode browser, camera/video sync, side-by-side comparison, frame/span annotations, embedding-space episode map, and a companion-DOM timeline outside the Viser sidebar.

## License

Released under the [MIT License](LICENSE), © Linkerbot (Beijing) Technology Co., Ltd.
Third-party software and asset licenses are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
