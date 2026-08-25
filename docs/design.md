# linker-sim-viser design

## Purpose

Browser-based, physics-free replay of LinkerBot recorded episodes. Not a simulator, not a teleop UI, not a training-data curation tool. Just: "show me what the robot did in the recording."

## Non-goals (explicit)

- **No physics.** Viser has none, and we don't add any. If we ever need dynamics-aware replay, that's `linker-sim-mujoco`.
- **No QC / anomaly detection.** Recorded data is assumed to have passed upstream QC and is treated as clean.
- **No live forward simulation.** Runtime motion planning + cuMotion belongs in `linker-sim-isaac`. The viewer never simulates forward at playback time.

### Scope amendment (2026-08): offline IK retargeting for replay

The viewer runtime stays a pure joint-replayer, but **offline preprocessing tooling** in this
repo may use kinematics (FK / Jacobian / IK) to turn a recorded end-effector trajectory into joint
angles that the viewer then plays back. See `scripts/umi_mcap_to_episode.py`: it ingests a UMI-Dex
mcap bag, anchors the wrist pose to the arm's `tool0` (Nelder-Mead search), solves DLS IK to
retarget the 7-DOF arm, decodes the hand, and writes a normal `telemetry.npz` + `metadata.json`.
This mirrors `linker-sim`'s UMI replay pipeline. It is *offline convert-time* work, distinct from
the still-out-of-scope *runtime* forward-sim/planning.

## Foundation

- **[Viser](https://github.com/nerfstudio-project/viser) ≥ 1.0.20** for the 3D scene, GUI widgets, and browser transport.
- **yourdfpy** (via `viser.extras.ViserUrdf`) for URDF loading. yourdfpy resolves URDF `<mimic>` natively → LinkerHand tendon coupling works without custom code.
- **Plain YAML** for config. No Hydra — the config surface here is narrow and Hydra's composition/sweep features don't buy anything for a viewer.

## v0 scope

1. Load one `.npz` episode.
2. Display dual-arm + dexterous-hand URDF in a Viser scene.
3. Sidebar playback controls: frame slider, play/pause, speed presets (0.25× / 0.5× / 1× / 2× / 4×), loop.
4. EE motion trails.
5. Keypose stamps derived from hand-percent transitions (grasp start, grasp close, release).

## Deferred to v1+

- Multi-episode browser (thumbnails, filters, tags)
- Camera video sync (GoPro wrist cams, third-person, PIP or side panel)
- Side-by-side episode comparison (aligned time cursors, delta plots)
- Frame / span annotations
- LeRobot parquet and ROS 2 `.mcap` ingest
- Companion-DOM timeline outside the Viser sidebar
- Embedding-space episode map

## Architectural fork (v0 → v1)

**Path 1 — Viser-sidebar-only (v0):** all controls live in the Viser GUI sidebar. Ships fast, easy to reason about. Ceiling: dataset curation and rich timelines get cramped in a vertical sidebar. See egoallo and mjviser for the shape.

**Path 2 — Viser + companion DOM (v1+):** Viser owns the 3D scene; a co-hosted web page hosts a bottom timeline, plots, and video panels. See LeRobot's dataset visualizer, Rerun, Foxglove. The `kimodo-viser` fork's existence is the signal that we'll want this eventually.

Start with Path 1, upgrade to Path 2 only when we hit the ceiling.

## Assets

Interim: reference URDFs and meshes in `../linker-sim/packages/linker-robot-assets/` via relative paths in the robot config. This is an interim approach until `linker-robot-assets` is published as an installable package. Once that lands, robot configs will resolve URDF paths through the installed package.

## Hand decoders

Live in this repo (`src/linker_sim_viser/decoders.py`), not in the assets package. They are only used by replay tools (this repo + `linker-sim-mujoco`), not by `linker-sim-isaac`, so keeping them out of the shared assets package keeps that package narrow.

## File layout

```
src/linker_sim_viser/
  __init__.py
  app.py         - main entry, orchestrates load + scene + GUI + playback loop
  episode.py     - Episode dataclass, .npz loader
  viewer.py     - SceneRobot: ViserUrdf wrapper
  timeline.py    - PlaybackGUI: sidebar playback widgets and state
  trails.py     - EE trails, keypose event detection
  decoders.py    - hand percent → joint angle
  config.py     - YAML load
configs/
  viewer.yaml            - viewer defaults
  robots/*.yaml          - per-robot: urdf_path, joint groups, npz layout, ee frames
scripts/
  replay.py     - CLI wrapper -> app.run()
tests/
  test_episode.py        - placeholder
```
