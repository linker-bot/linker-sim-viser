# linker-sim-viser

Browser-based, physics-free replay viewer for LinkerBot recorded data. Built on [Viser](https://github.com/nerfstudio-project/viser) with `yourdfpy` for URDF loading (mimic joints resolve natively — LinkerHand tendon coupling works with no extra code).

Part of the three-way split from `linker-sim`. Sibling repos: `linker-sim-mujoco` (physics-aware replay, later), `linker-sim-isaac` (forward sim + motion planning, forked from colleague's work).

## Status

**v0 shipping.** Single-episode playback with comet-style EE trails and keypose stamps.

## What's in v0

- Load one recorded episode from `data/episode_XXXXXX/` (SDK format 1.4: `telemetry.npz` + `metadata.json`).
- Render the dual-arm + LinkerHand workstation URDF, driven by the episode's joint stream.
- Sidebar playback: frame slider, play/pause, speed presets (`0.25x` / `0.5x` / `1x` / `2x` / `4x`), loop, time readout.
- Two-step hand decode: raw SDK 0-255 → 0-100 → URDF radians via `linear-fit-v0`.
- **Comet trails** per configured EE frame: grows as playback advances, fades from bright head to dim tail, tracked by a coloured sphere.
- **Keypose stamps** at grasp/release events, detected from the SDK hand stream with a median-5 filter that rejects the 1–2 frame sensor dropouts.

## Install

```bash
uv venv
uv pip install -e ".[dev]"
```

**Git LFS is required to clone.** The ~40 vendored STL meshes under [`assets/`](assets/) are LFS-tracked (see [`.gitattributes`](.gitattributes)). Install `git-lfs` once per machine (`apt install git-lfs && git lfs install`), then `git clone` will fetch the meshes. If you've already cloned without LFS, run `git lfs install && git lfs pull`.

## Requirements

- **Recorded episodes** dropped in `data/episode_XXXXXX/`. `data/` is `.gitignore`'d (episodes are large binary bundles); ask the collection team for a set or copy `episode_000000/` from a shared drive.

The `a7_lite_l6_dc` workstation URDF and meshes for it are **vendored under `assets/`** for testing convenience (see the [Known limitations](#known-limitations) note on this). If you want to use a URDF that lives elsewhere, edit `urdf_path` in `configs/robots/*.yaml`.

## Run

```bash
.venv/bin/python scripts/replay.py --robot a7_lite_l6_dc --episode data/episode_000000
```

Open `http://localhost:8080`. `--robot NAME` resolves to `configs/robots/NAME.yaml`; `--episode PATH` is a directory containing `telemetry.npz` + `metadata.json`.

Flags:
- `--port INT` — override the viewer port (default from `configs/viewer.yaml`).
- `--viewer-config PATH` — use a different viewer config.

## Tests

```bash
.venv/bin/python -m pytest
```

Loader and decoder tests run against `data/episode_000000/`; they skip cleanly if that directory isn't present.

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

Per [docs/design.md](docs/design.md):

- **No physics.** For dynamics-aware replay use `linker-sim-mujoco`.
- **No QC / anomaly detection.** Recorded data is treated as clean. Report upstream if you see systemic sensor issues.
- **No forward simulation.** Motion planning + IK + cuMotion belong in `linker-sim-isaac`.
- No multi-episode browser, no camera video sync, no annotations, no LeRobot parquet or ROS 2 `.mcap` ingest yet.

## Known limitations

Read these before filing bugs — they're deliberate v0 tradeoffs or unresolved upstream issues.

- **Hand decoder is a placeholder.** `linear-fit-v0` is a linear map from SDK 0–100 into each joint's URDF `[lower, upper]`. The upstream `linker-robot-assets` module states outright that the true SDK angle convention hasn't been defined yet. Symptom you'll see: fingertips of the thumb and index overlap slightly at rest. Every finger pose we render is off by an unknown amount. When the SDK ships a real convention, `CONVENTION` gets bumped and all v0-rendered views should be treated as approximate.
- **Keypose detector is unverified on real grasps.** Synthetic tests prove it *rejects* single- and two-frame dropouts and *finds* sustained closures. The reported event frame lags the true crossing by up to `median_window // 2` frames (~67ms at 30fps). The 50% threshold is a starting guess. Expect to iterate on `median_window` and `threshold_percent` once episodes with real grasps land.
- **SDK sensor dropouts show through as visible jerks.** ~4% of frames per hand carry 1–2 frame spikes (index/thumb DoFs snapping fully closed then back). The keypose detector filters them; the pose stream doesn't (design policy: viewer stays faithful, upstream fixes the sensor path). If this becomes a blocker, we can add an opt-in render-side median filter — flag it and we'll reopen.
- **Precomputed EE poses in the npz are unusable.** `ee_poses_qpos_{left,right}` are in each arm's J2 frame, not workstation world (the collection pipeline used single-arm URDFs). We ignore them and FK from the workstation URDF instead. No cross-check is performed.
- **Camera + trail visuals are tuned for `a7_lite_l6_dc`.** Initial camera pose, `head_radius`, `tail_intensity`, and trail `line_width` are hardcoded for this workstation's scale. A taller/shorter robot will need re-tuning; no config path for that yet.
- **Vendored decoder can silently rot.** [`src/linker_sim_viser/hand_decoders/linkerhand_l6/decoder.yaml`](src/linker_sim_viser/hand_decoders/linkerhand_l6/decoder.yaml) is a manual copy of the sidecar from `linker-robot-assets`. If the collection team edits their copy, we won't pick it up. Delete this whole tree once the assets package ships.
- **Vendored asset bundle.** [`assets/`](assets/) contains a testing snapshot of the `a7_lite_l6_dc` workstation URDF + its ~40 STL meshes copied out of `linker-robot-assets`. Bundled so the colleague can test this repo without pulling the sibling `linker-sim/` checkout. Delete this tree (and edit `configs/robots/a7_lite_l6_dc.yaml`'s `urdf_path`) once `linker-robot-assets` ships as an installable package.

## Roadmap

See [docs/design.md](docs/design.md) for the v1+ backlog (multi-episode, video sync, side-by-side, annotations, embedding map, companion-DOM timeline).
