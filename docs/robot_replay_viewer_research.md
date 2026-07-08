# Browser-Based Robot Data Replay Viewer Research

Last updated: 2026-07-07

Context: this report targets a browser-based, physics-free robot data replay and inspection tool built on [Viser](https://github.com/nerfstudio-project/viser) and `yourdfpy`, for recorded real-robot teleoperation datasets: dual arms, dexterous LinkerHands, UMI-style wrist/hand poses, GoPro wrist cameras, `.npz`, ROS 2 `.mcap`, and LeRobot parquet.

The main finding is that most Viser projects are not dataset replay viewers. They are usually robot model viewers, IK tools, motion editors, retargeting demos, or simulation debug frontends. The strongest direct inspirations are:

- Viser-native replay/control examples: [egoallo](https://github.com/brentyi/egoallo), [mjviser](https://github.com/mujocolab/mjviser), parts of [PyRoki](https://pyroki-toolkit.github.io/), [robot_keyframe_kit](https://github.com/Stanford-TML/robot_keyframe_kit), and [kimodo-viser](https://github.com/nv-tlabs/kimodo-viser).
- Dataset/replay products outside Viser: [Rerun](https://rerun.io/docs/concepts/timelines), [Foxglove Studio](https://docs.foxglove.dev/docs/visualization/playback), [LeRobot visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset), [DROID visualizer](https://droid-dataset.github.io/visualizer/), [robomimic playback](https://raw.githubusercontent.com/ARISE-Initiative/robomimic/master/robomimic/scripts/playback_dataset.py), and [UMI](https://umi-gripper.github.io/).

## 1. Open-Source Robot Data Viewers Built On Viser

### Summary Table

| Project | Category | Timeline / Scrubbing UX | Multi-Episode Browsing | GUI Panels Exposed | Screenshots / GIFs | Relevance |
|---|---|---|---|---|---|---|
| [PyRoki](https://pyroki-toolkit.github.io/) / [GitHub](https://github.com/chungmin99/pyroki) | Kinematic optimization and retargeting toolkit, not a dataset replay viewer | Example scripts use simple Viser controls: `Timestep` slider plus `Playing` checkbox. Hand retargeting example adds `Retarget!` and per-weight tuning controls. | None found. Examples load one problem/sequence at a time. | Sliders for weights, target keypoint display, contact point display, object mesh, robot/hand visualization. | [Project page](https://pyroki-toolkit.github.io/) includes videos and demos. | Good for retargeting UI and live optimization controls, not enough for dataset review. |
| [egoallo](https://github.com/brentyi/egoallo) / [project page](https://egoallo.github.io/) | Human body/hand motion output visualizer; one of the best Viser replay references | Strong Viser-native playback folder: timestep slider, start/end multi-slider, next/prev buttons, playing checkbox, FPS slider, FPS preset buttons, `.viser` export, ego-video export. | File dropdown / refresh pattern for selecting outputs, but not a full dataset browser. | Attach camera to body/control frame, attach distance, show body/glasses/axes, wireframe, SMPL/HaMeR opacity, hand/wrist detections, point cloud size, Gaussian splat/point-cloud scene context. | [egoallo.github.io](https://egoallo.github.io/) and README media. | Directly useful for Viser replay mechanics and scene-context playback. |
| [hamer_helper](https://github.com/brentyi/hamer_helper) | HaMeR wrapper for hand estimation from RGB images | No replay UI found. | None. | Batch image inference and composite detection outputs. | README includes a Viser logo / image. | Relevant mainly as a hand-pose preprocessing dependency. |
| [Stanford-TML/robot_keyframe_kit](https://github.com/Stanford-TML/robot_keyframe_kit) | MuJoCo keyframe editor, not a replay viewer | Keyframe and timed-sequence editing. Saved files include keyframes, timed sequence durations, `time`, `qpos`, `qvel`, actions, and body/site trajectories. | Motion-file loading and save dirs, not dataset-level browsing. | Keyframe list, sequence list, mirror mode, physics test mode, IK targets, root pose/gizmo controls, save/load/export. | [docs/media](https://github.com/Stanford-TML/robot_keyframe_kit/tree/main/docs/media) and README demo GIF. | Excellent inspiration for keypose extraction, span editing, mirrored motion, and motion clips. |
| [nv-tlabs/kimodo-viser](https://github.com/nv-tlabs/kimodo-viser) | Viser fork for Kimodo motion synthesis | Fork adds a timeline UI and keyboard input support. This is a strong signal that stock Viser GUI widgets are not enough for rich motion replay. | None found. | Viser base GUI plus fork-specific timeline/keyboard support. | README. | Useful design signal: build or embed a dedicated timeline component instead of relying only on sidebar sliders. |
| [uynitsuj/IsaacLab-Viser](https://github.com/uynitsuj/IsaacLab-Viser) | Headless IsaacLab visualization through Viser | No polished replay/timeline UX found in README. | None documented. | Robot states, trajectories, sensor data, remote/headless simulation interaction. | [IsaacLabViser.gif](https://github.com/uynitsuj/IsaacLab-Viser/blob/main/media/IsaacLabViser.gif). | Useful for browser-based remote visualization, less relevant to replay/curation. |
| [mujocolab/mjviser](https://github.com/mujocolab/mjviser) | MuJoCo viewer on Viser | `motion_playback.py` uses a playback tab with frame slider, time label, play/pause button, speed buttons `0.25x` to `4x`, and loop checkbox. README also lists ghost overlay and timeline scrubber examples. | None found. | Simulation controls, joint/actuator sliders, contacts/forces, camera tracking, keyframes. | [examples](https://github.com/mujocolab/mjviser/tree/main/examples). | Good for compact playback controls, speed presets, looping, contacts, ghost overlays. |
| [zixingjiang/robot-viewer](https://github.com/zixingjiang/robot-viewer) | URDF/MJCF web robot model viewer | None. | Robot model library browsing, not trajectory browsing. | Joint-space and Cartesian controls, IK via Mink, collision/frames, 175+ robot descriptions. | [README gallery](https://github.com/zixingjiang/robot-viewer). | Good model-loading and FK/IK control reference. Not a replay viewer. |
| [legalaspro/robokin](https://github.com/legalaspro/robokin) | URDF FK/IK helper library with Viser and Rerun helpers | No Viser replay timeline found. Rerun logging can stream joint angles and EE trajectories. | None. | Viser gizmo control, URDF model loading, multiple IK backends, real-arm integrations via LeRobot/ROS 2. | README videos/table. | Relevant for URDF/IK plumbing and Rerun-style logging ideas. |
| [luckyrobots/luckylab](https://github.com/luckyrobots/luckylab) | Robot learning framework | Not Viser replay. Uses Rerun for live inspection of observations, actions, rewards, and camera feeds. | Dataset browse script supports repo ID and episode index. | Rerun web dashboards. | README. | Useful evidence that many robot-learning repos outsource replay UX to Rerun. |

### Takeaways For A Viser Replay Tool

- A simple Viser sidebar slider is fine for a demo but too weak for dataset curation.
- The best Viser-native controls to copy are from egoallo and mjviser: frame slider, range handles, speed presets, loop, next/previous frame, and export.
- The best editor concepts are from robot_keyframe_kit: keyframes, named sequences, durations, mirror mode, and motion export.
- Kimodo's fork is a useful warning: if timeline/keyboarding matter, consider a real timeline widget outside the default sidebar.

## 2. Timeline And Playback UX Patterns In Robotics Data Viewers

### Rerun

Reference: [Rerun timeline concepts](https://rerun.io/docs/concepts/timelines) and [Rerun timeline view docs](https://rerun.io/docs/reference/viewer/timeline).

Rerun is the strongest timeline reference. It supports multiple timelines, such as frame index, log time, sensor time, or custom sequence clocks. The timeline panel supports play/pause, stepping, scrubbing, loop selection, rate control, active timeline switching, time cursor, time ranges, and event streams.

Important patterns to steal:

- Treat frame index and timestamp as separate clocks.
- Allow multiple timelines: robot frame, camera frame, ROS log time, device time, wall time.
- Make missing data and discontinuities visible.
- Support a loop range, not just full-episode loop.
- Let plotted data, 3D state, and video follow the same time cursor.

### Foxglove Studio

References: [Foxglove playback docs](https://docs.foxglove.dev/docs/visualization/playback), [Foxglove panels](https://docs.foxglove.dev/docs/visualization/panels).

Foxglove is strongest for robotics log playback. Relevant conventions:

- Bottom playback bar.
- Buffered portions shown on the timeline.
- Adjustable start/end playback range.
- Loop playback.
- Speed controls.
- Keyboard shortcuts: Space to play/pause, arrows to step, Home/End to jump, shortcuts for speed changes.
- Lookback/latching behavior for sparse topics.
- Range loading for plots and 3D transforms.
- Modular panels for 3D, image, plot, raw messages, table, state transitions, and diagnostics.

Design adaptation: keep the playback mechanics, but avoid exposing raw ROS topic complexity as the primary UI unless the user explicitly enters a debug mode.

### LeRobot Dataset Visualizer

Reference: [LeRobot visualizer Space](https://huggingface.co/spaces/lerobot/visualize_dataset), [README source](https://huggingface.co/spaces/lerobot/visualize_dataset/raw/main/README.md).

The LeRobot visualizer is a close match for dataset inspection. It provides:

- Organization/dataset/episode navigation.
- Synchronized episode video and time-series charts.
- Overview metadata.
- Dataset statistics and episode-length histogram.
- Action insights: autocorrelation, state-action alignment, speed distribution, cross-episode variance heatmap.
- Filtering for low movement, jerky motion, and outlier length.
- Export of flagged episode IDs as a LeRobot CLI command.
- 3D URDF replay with end-effector trail.
- Annotation editing for LeRobot v3.1 language schema, including bbox/keypoint overlays.

This is the best direct template for a replay viewer that also does curation.

### DROID Visualizer

References: [DROID project](https://droid-dataset.github.io/), [DROID visualizer](https://droid-dataset.github.io/visualizer/).

DROID is strongest for dataset-scale exploration:

- Contact-sheet video browsing.
- Lazy-loaded video grid.
- Filters for scene type, task, object, and metadata.
- Per-filter counts.
- Random resampling.
- Language/task metadata overlaid with clips.
- Dataset-level analysis of camera viewpoints and first gripper close / interaction point.

For 100+ episodes, DROID's contact sheet is more useful than opening a single episode by ID.

### Robomimic Playback

Reference: [robomimic playback_dataset.py](https://raw.githubusercontent.com/ARISE-Initiative/robomimic/master/robomimic/scripts/playback_dataset.py).

Robomimic is CLI-first, but it contains useful practical conventions:

- Play back simulator states or actions.
- Render video outputs.
- Select cameras and concatenate camera views horizontally.
- Play a random subset or first N episodes.
- Use filter keys to select subsets.
- Warn about open-loop action divergence.

For your viewer, the useful piece is simple: side-by-side camera composites and subset playback matter.

### RoboCasa

Reference: [RoboCasa](https://github.com/robocasa/robocasa), [RoboCasa project](https://robocasa.ai/).

RoboCasa is more a simulation/data-generation ecosystem than a viewer UX reference. Useful concepts:

- Large-scale benchmark dataset organization.
- Human and synthetic demonstration split.
- Task-family navigation.
- Need for fast preview at scale.

### UMI Dataset / Viewers

References: [UMI project](https://umi-gripper.github.io/), [UMI GitHub](https://github.com/real-stanford/universal_manipulation_interface).

UMI does not appear to ship a polished browser replay viewer, but it is highly relevant for wrist-camera and trajectory synchronization:

- GoPro wrist camera observations.
- Camera-centric and relative trajectory actions.
- Latency matching.
- SLAM quality / demo validity checks.
- Kinematic feasibility checks for robot execution.

For a LinkerHand/GoPro viewer, UMI suggests making camera sync and validity checks first-class.

### What Makes A Good Scrub Timeline

A good robot replay timeline should include:

- A persistent bottom timeline, not a sidebar-only slider.
- Frame ticks and time ticks.
- Event markers: grasp, release, contact, annotation, failure, dropped frame, joint limit.
- Anomaly markers: NaN, out of range, discontinuity, sync drift, jerk spike.
- Range handles for focused replay and clip export.
- Play/pause, step forward/back, jump to next marker, jump to previous marker.
- Loop current range.
- Speed presets: `0.25x`, `0.5x`, `1x`, `2x`, `4x`.
- Keyboard shortcuts.
- Visual indication of missing data and stale/latching state.
- Multi-track lanes for robot joints, cameras, annotations, QC flags, and optionally left arm/right arm/hand.

## 3. Trajectory Visualization Ideas

### What People Have Actually Built

- LeRobot: 3D URDF frame replay with end-effector trails.
- mjviser: ghost overlays and contact/force visualization during playback.
- DROID: dataset-level viewpoint distributions and 3D interaction-point analysis, especially first gripper close.
- PyRoki: target keypoints, contact points, object meshes, and retargeted hand motion.
- egoallo: body/hand motion with point clouds or Gaussian splats as context; opacity and wireframe controls.
- robokin: Rerun logging of joint angles and EE trajectories.
- robot_keyframe_kit: keyframe trajectories, timed sequences, body/site trajectory export.

### Recommended For This Viewer

- Motion trails per end effector and per wrist camera.
- Ghost robot poses every N frames with fading opacity.
- Keypose stamps along paths: first motion, first contact, grasp close, lift, place, release.
- EE pose ribbons: position plus orientation frame stamps.
- Wrist-camera frustums in the 3D scene.
- Joint-space trace plots with limit bands.
- Action trace plots, separate from observed state.
- Left/right arm and left/right hand tracks.
- Delta-highlighting between commanded and observed pose where available.
- Side-by-side episode comparison with aligned time cursors.
- Workspace heatmap from selected episodes.
- Dataset-level interaction-point cloud, e.g. first gripper close, first object motion, highest contact confidence.

### Dual-Arm / Dexterous-Hand Specific Ideas

- Color-code left arm, right arm, left hand, right hand consistently across 3D, video overlays, plots, and timeline tracks.
- Show mimic/tendon-coupling residuals as a small hand-health plot.
- Add a "finger fan" miniature plot showing all hand joint positions in one compact view.
- Highlight joints near limits in the URDF itself.
- Show wrist-to-hand calibration frame and camera optical frame explicitly.

## 4. Data Quality Inspection Features That Surface Anomalies While Replaying

### Concrete Precedents

- LeRobot visualizer: low movement, jerky motion, outlier length, action autocorrelation, state-action alignment, speed distribution, cross-episode variance heatmap, and export of flagged episode IDs.
- robomimic: open-loop action playback divergence warnings and multi-camera playback/export.
- UMI: latency matching, SLAM usability, and validity checks under robot kinematic constraints.
- ARCap: AR feedback and haptic warnings for kinematic/collision constraints during collection; offline viewers can reuse the same warnings during replay.
- EgoAllo: contact labels and filtering of problematic motion sequences in preprocessing.
- Robot data curation research: mutual-information-based trajectory ranking for diversity and action predictability.

### High-Value QC Checks For This Tool

- NaN, Inf, and missing field detection.
- Joint name mismatch between data and URDF.
- Unit mismatch hints, e.g. degrees vs radians, mm vs meters.
- Joint limit violations.
- Velocity, acceleration, and jerk spikes.
- Control-frequency jitter.
- Dropped robot frames.
- Dropped or duplicated video frames.
- ROS message timestamp regressions.
- Camera/robot sync drift.
- Hand mimic/tendon-coupling residual violations.
- Gripper open/close discontinuities.
- Impossible wrist velocities.
- Long pauses or low-motion episodes.
- Outlier episode length.
- Calibration suspicion: camera frustum moving inconsistently with wrist pose.
- Left/right swap suspicion.
- Dual-arm collision proximity, even in physics-free replay.

### How To Surface QC In UX

- Put QC badges in the episode list.
- Put anomaly ticks directly on the timeline.
- Add "jump to next issue" and "jump to previous issue".
- Add a QC panel with severity, timestamp/frame, signal, measured value, and threshold.
- Highlight the relevant joint/link/camera in the 3D scene when the issue is selected.
- Allow suppressing or accepting a flag with reviewer notes.
- Export flags as sidecar JSON, CSV, and LeRobot-compatible filtered episode IDs.

## 5. Multi-Episode / Dataset-Level Navigation

### Existing Patterns

- DROID: contact-sheet video grid, lazy loading, filters by scene/task/object, counts per filter, random resampling.
- LeRobot: sidebar navigation, pagination, overview metadata, dataset stats, filtering, flagged episode export.
- robomimic: CLI subset selection by filters and random/N episode options.
- Foxglove: modular layouts and panels, but not dataset-curation-oriented by default.

### Recommended UI For 100+ Episodes

- Left sidebar with dataset selector and episode table.
- Columns: episode ID, task/language, duration, frames, FPS, cameras present, robot config, success/failure, QC score, tags, modified status.
- Thumbnail/contact-sheet mode with short looping preview clips.
- Filters: task, object, scene, operator, date, success, QC issue, duration, camera availability, hand/arm side, file source.
- Search over language and metadata.
- Sort by anomaly count, duration, motion amount, or recency.
- Episode detail panel with metadata and file provenance.
- Tagging: keep, reject, needs review, calibration issue, camera issue, bad hand pose, good demo.
- Batch actions: export selected clips, flag selected, compare selected, open random sample.
- Dataset summary: duration histogram, FPS histogram, camera availability matrix, QC issue histogram.
- Embedding map as an advanced view, not the primary navigation.

### Side-By-Side Episode Comparison

Useful comparison modes:

- Two episodes with synchronized normalized progress.
- Same episode with raw vs retargeted joints.
- Same episode with command vs observed state.
- Different retiming or filtering results.
- Successful vs failed demonstration for the same task.

Show:

- Split 3D view or overlaid ghost robots.
- Split video or synchronized camera panels.
- Delta plots by joint group.
- Difference highlights on the timeline.

## 6. Camera / Video Sync

### Existing Patterns

- LeRobot: synchronized video and time-series charts.
- DROID: video contact sheets for browsing.
- robomimic: horizontal multi-camera video concatenation.
- UMI: GoPro wrist camera data, camera-centric action representation, latency matching.
- Rerun: separate timelines and video/image entities, useful as a model for sensor time vs frame index.
- egoallo: attach camera to body/control frame, adjust distance, and visualize motion with scene context.

### Recommended Layout

- Main center: 3D scene.
- Right panel: selected camera video, with tabs or grid for wrist left, wrist right, third-person, overhead.
- Bottom: timeline and compact plots.
- Optional floating PIP only for the currently selected camera.
- Filmstrip mode for rapid scrubbing.

### Sync Features To Include

- Display current robot frame, camera frame, robot timestamp, camera timestamp, and delta.
- Show nearest-frame matching mode: exact, nearest, interpolated, stale, missing.
- Camera frustums in 3D, colored by camera stream.
- Video overlay of frame number and timestamp.
- Timeline track per camera, with dropped-frame markers.
- Drift plot over the episode.
- Manual offset adjustment for debugging.
- Optional "lock video to robot time" vs "lock robot to video time".

### Viser-Specific Pattern

Viser can own the 3D scene, robot model, trails, camera frustums, and GUI widgets. For video, a custom HTML panel or adjacent web UI is likely better than trying to make video a 3D texture. Use Viser for 3D state and a dedicated DOM component for rich media/timeline behavior.

## 7. Annotation And Labeling Tools Built Into Replay Viewers

### Existing References

- LeRobot visualizer supports annotation editing for language atoms, bbox/keypoint overlays, parquet rewrite, and optional Hub push.
- Foxglove supports events around the playback timeline and event search.
- robot_keyframe_kit provides a useful mental model for named keyframes and timed motion sequences.

### Recommended Annotation Features

- Frame tags: contact, grasp start, grasp end, lift, place, release, failure.
- Span annotations: subtask, object interaction, hesitation, recovery, invalid segment.
- Region selection on timeline to create clips.
- Keypose extraction from selected frames.
- Reviewer labels: keep, reject, needs relabel, calibration issue.
- Per-camera bbox/keypoint labels where useful.
- Per-event notes and confidence.
- Snap-to-frame and snap-to-nearest-marker behavior.
- Export sidecar JSON and optional LeRobot parquet update.
- Maintain annotation provenance: user, timestamp, schema version, source file hash.

### Clip Creation

Clip creation should be first-class:

- Select range on timeline.
- Preview range loop.
- Export clip with robot states, actions, camera frames, annotations, and metadata.
- Preserve source episode ID and frame range.
- Optionally export a lightweight MP4 contact sheet for review.

## 8. Novel / Creative Uses Of Viser Or Similar Tools

### Ideas Seen In The Wild

- Rerun: multiple timelines and discontinuity handling.
- Kimodo-viser: forked Viser to add timeline UI and keyboard input.
- PyRoki: live retargeting with weight tuning and contact/keypoint visualization.
- robot_keyframe_kit: mirror mode, keyframe sequence editing, and export of body/site trajectories.
- mjviser: ghost overlay and contact replay.
- egoallo: replay with point clouds / Gaussian splats and body-attached camera views.
- UMI: latency-aware wrist-camera demonstrations and kinematic validity checks.
- ARCap: real-time kinematic/collision feedback during collection.

### Novel Features Worth Considering

- Interactive retiming: stretch/compress a selected segment and preview changed timing.
- "What-if" pose interpolation between selected keyframes.
- Mirror/reflect demonstrations for symmetric data augmentation.
- In-browser retargeting preview from UMI wrist pose to Linker dual-arm/hand configuration.
- Command vs observation comparison with latency shift slider.
- Dataset "motion fingerprint" for finding near-duplicates.
- Automatic subtask proposal from gripper events, velocity changes, and contact heuristics.
- Embedding-space episode map using features from trajectories, language, and QC metrics.
- Calibration debug mode that shows wrist camera frustum residuals.
- "Replay recipe" export: selected episode, time range, camera layout, plots, and active QC filters.

## 9. Design Mistakes To Avoid

### Viser-Specific

- Do not rely only on sidebar sliders for timeline control.
- Do not put all controls into one long Viser GUI column.
- Do not make users scrub frame-by-frame without keyboard shortcuts.
- Do not make 3D state updates depend on expensive full-scene rebuilds.
- Do not make video playback subordinate to 3D rendering; use a dedicated video UI.

### Rerun / Foxglove Lessons

- Rerun is powerful, but generic entity/timeline models can feel abstract to users who just want to inspect an episode.
- Foxglove is excellent for ROS debugging, but topic-centric layouts can overwhelm dataset-curation workflows.
- Lookback/latching behavior is useful, but stale data must be visibly marked.
- Multiple timestamps are necessary, but they must be explained in the UI.

### MeshCat / Isaac Sim Lessons

- MeshCat-style viewers often lack serious timeline, media, and dataset browsing UX.
- Isaac Sim is too heavy for quick replay inspection and can blur the line between simulator and replay viewer.
- Avoid requiring GPU-heavy simulation stacks for a physics-free viewer.

### Dataset Tool Mistakes

- Do not preload the whole dataset.
- Do not hide corrupted episodes until a crash happens.
- Do not force users to know episode IDs.
- Do not separate video, 3D, plots, and QC into disconnected tools.
- Do not make annotations hard to export.
- Do not mutate source data without clear sidecar/provenance behavior.
- Do not assume every dataset has identical camera/joint schema.

## 10. Papers, Blog Posts, And Talks From 2024-2026

This area has more dataset/model papers than viewer-specific papers. The useful materials tend to mention visualization, curation, and replay as part of larger data pipelines.

| Source | Date | Relevance |
|---|---:|---|
| [Viser technical report](https://arxiv.org/abs/2507.22885) | 2025 | Confirms Viser's role as imperative web-based 3D visualization with GUI primitives. Useful baseline for what Viser provides and what you must build yourself. |
| [PyRoki](https://pyroki-toolkit.github.io/) | 2025 | Kinematic optimization and retargeting, with Viser-based examples that show live controls, contacts, and trajectory playback. |
| [EgoAllo](https://egoallo.github.io/) | 2025 | Strong Viser-based output visualization for body/hand motion, point clouds/splats, and playback controls. |
| [DROID](https://droid-dataset.github.io/) | 2024-2025 | Dataset-scale robot manipulation data browser, contact-sheet exploration, metadata filters, and interaction-point analysis. |
| [UMI](https://umi-gripper.github.io/) | 2024 | GoPro wrist camera demonstrations, latency matching, camera-centric actions, SLAM/validity concerns. |
| [ARCap](https://stanford-tml.github.io/ARCap/) | 2024 | Data quality feedback during collection; useful offline warning types for replay viewers. |
| [Robot Data Curation with Mutual Information Estimators](https://arxiv.org/abs/2502.08623) | 2025 | Automated trajectory ranking and curation for robot policy learning. |
| [Re-Mix](https://arxiv.org/abs/2408.14037) | 2024 | Data mixture optimization for robot imitation learning; motivates dataset-level metadata and filtering. |
| [Octo](https://arxiv.org/abs/2405.12213) | 2024 | Open X-Embodiment policy; reinforces need to inspect heterogeneous multi-robot datasets. |
| [pi0](https://arxiv.org/abs/2410.24164) | 2024 | Vision-language-action model trained over diverse robot data; motivates scalable data inspection. |
| [pi0.7](https://arxiv.org/abs/2604.15483) | 2026 | Future-looking multi-environment robot data and context-conditioned behavior. |
| [RoboCasa](https://robocasa.ai/) / RoboCasa365 | 2024-2026 | Large-scale simulation and demonstration data organization; useful for dataset navigation and benchmark-scale browsing. |
| [NVIDIA GR00T N1](https://arxiv.org/abs/2503.14734) | 2025 | Large heterogeneous robot data pipeline; motivates real/synthetic/human data provenance tracking. |

## Top 10 Features To Steal

Ranked by practical build effort x impact for a Viser + yourdfpy replay viewer.

| Rank | Feature | Effort | Impact | Why |
|---:|---|---|---|---|
| 1 | Dedicated bottom timeline with scrub, range handles, loop, speed, step, keyboard shortcuts, and anomaly ticks | Medium | Very high | This is the core UX gap in most Viser demos. |
| 2 | Synchronized 3D + multi-camera video + joint/action plots | Medium | Very high | The user needs one shared time cursor across robot, video, and signals. |
| 3 | Episode browser with thumbnails, filters, tags, QC badges, and metadata | Medium | Very high | Essential once there are more than a few episodes. |
| 4 | QC precompute pass: NaNs, limits, mimic residuals, jitter, dropped frames, sync drift, jerk | Medium | Very high | Turns replay from viewing into inspection. |
| 5 | EE trails, wrist-camera frustums, ghost poses, and keypose stamps | Small-Medium | High | High visual payoff and directly helps manipulation debugging. |
| 6 | Export flagged episodes and clips as JSON/CSV/LeRobot-compatible selections | Small-Medium | High | Makes the viewer actionable for training-data cleanup. |
| 7 | Camera sync debugger with timestamp deltas, nearest-frame status, drift plot, and manual offset | Medium | High | Especially important for GoPro wrist-camera teleop data. |
| 8 | Side-by-side episode comparison with aligned cursors and delta plots | Medium-Large | High | Useful for success/failure comparison and retargeting validation. |
| 9 | Annotation panel for frame tags, spans, grasp/contact labels, and failure reasons | Medium | Medium-High | Needed for curation, keypose extraction, and downstream training. |
| 10 | Embedding/UMAP dataset map for large collections | Large | Medium-High | Powerful at scale, but less important than timeline, QC, and episode browsing. |

## Recommended First Build Slice

If building incrementally, start with:

1. Load one episode and display URDF state in Viser.
2. Add a real bottom timeline with play/pause, speed, loop, frame step, and range selection.
3. Add synchronized video panel and joint/action plots.
4. Add QC precompute and timeline markers.
5. Add episode table with thumbnails and QC badges.
6. Add annotations and clip export.

This order gets to a useful replay and inspection tool before adding advanced comparison, retargeting, or embedding maps.
