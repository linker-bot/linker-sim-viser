# linker-sim-viser

基于浏览器的无物理回放查看器，用于 LinkerBot 录制数据。构建在 [Viser](https://github.com/nerfstudio-project/viser) 之上，URDF 加载使用 `yourdfpy`（原生支持 mimic 关节 —— LinkerHand 的腱耦合无需额外代码即可正确渲染）。

本仓库是 `linker-sim` 三分拆的一部分。姊妹仓库：`linker-sim-mujoco`（含物理的回放，稍后建立），`linker-sim-isaac`（正向仿真 + 运动规划，从同事的仓库 fork）。

## 状态

**v0 已发布。** 支持单条 episode 回放、彗星轨迹、关键姿态标记。

## v0 功能

- 从 `data/episode_XXXXXX/` 加载一条录制的 episode（SDK 格式 1.4：`telemetry.npz` + `metadata.json`）。
- 渲染双臂 + LinkerHand 工作站 URDF，由 episode 的关节流驱动。
- 侧栏回放控件：帧滑块、播放/暂停、速度预设（`0.25x` / `0.5x` / `1x` / `2x` / `4x`）、循环、时间读数。
- 两段式手部解码：SDK 原始 0–255 → 0–100 → 通过 `linear-fit-v0` 映射到 URDF 弧度。
- **彗星轨迹**：每个配置的末端执行器一条，随回放推进而生长，从明亮的头端渐变到暗淡的尾端，由一个彩色小球标记当前位置。
- **关键姿态标记**：从 SDK 手部流中检测抓取/释放事件，使用 5 点中值滤波去除 1–2 帧的传感器丢帧再触发。

## 安装

```bash
uv venv
uv pip install -e ".[dev]"
```

**克隆时需要 Git LFS。** [`assets/`](assets/) 下约 40 个内置的 STL 网格由 LFS 管理（见 [`.gitattributes`](.gitattributes)）。每台机器安装一次 `git-lfs`（`apt install git-lfs && git lfs install`），然后 `git clone` 会自动拉取网格。如果你已经在没启用 LFS 的情况下 clone 了，运行 `git lfs install && git lfs pull` 补拉即可。

## 前置条件

- **录制数据**：放到 `data/episode_XXXXXX/`。`data/` 目录已被 `.gitignore` 忽略（episode 是大二进制包），请找采集组要一份，或从共享盘复制一份 `episode_000000/` 过来。

`a7_lite_l6_dc` 工作站的 URDF 及其网格已 **内置在 `assets/` 下**，方便测试（详见 [已知限制](#已知限制)）。若想指向别处的 URDF，编辑 `configs/robots/*.yaml` 中的 `urdf_path` 即可。

## 运行

```bash
.venv/bin/python scripts/replay.py --robot a7_lite_l6_dc --episode data/episode_000000
```

浏览器打开 `http://localhost:8080`。`--robot NAME` 会解析为 `configs/robots/NAME.yaml`；`--episode PATH` 是一个包含 `telemetry.npz` + `metadata.json` 的目录。

可选参数：
- `--port INT` —— 覆盖查看器端口（默认从 `configs/viewer.yaml` 读取）。
- `--viewer-config PATH` —— 使用另一份查看器配置。

## 测试

```bash
.venv/bin/python -m pytest
```

加载器和解码器的测试针对 `data/episode_000000/` 运行；若该目录不存在，测试会自动跳过。

## 添加新的机器人配置

`configs/robots/*.yaml` 把 npz 的列映射到 URDF 关节，并声明要绘制轨迹的连杆：

```yaml
urdf_path: 相对或绝对路径/指向/workstation.urdf

streams:
  - key: qpos           # npz 键名
    slice: [0, 7]       # 列范围（可选）
    joints:             # 按列顺序排列的 URDF 关节名 —— 以此为准
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

## 非目标 (v0)

见 [docs/design.md](docs/design.md)：

- **不做物理仿真。** 需要动力学感知的回放请用 `linker-sim-mujoco`。
- **不做 QC / 异常检测。** 录制数据默认视为干净。若发现系统性传感器问题，请反馈到上游修复。
- **不做正向仿真。** 运动规划 + IK + cuMotion 属于 `linker-sim-isaac`。
- v0 暂不支持多 episode 浏览、相机视频同步、标注、LeRobot parquet 或 ROS 2 `.mcap` 摄入。

## 已知限制

提交 bug 前请先读这些 —— 它们是 v0 有意接受的取舍或尚未解决的上游问题。

- **手部解码器是占位实现。** `linear-fit-v0` 是把 SDK 0–100 线性映射到每个关节 URDF `[lower, upper]` 的暂时方案。上游 `linker-robot-assets` 模块明确说明 SDK 真正的角度约定还没定。可见现象：静止时拇指和食指指尖会有轻微重叠。渲染出来的手指姿态都存在未知量级的偏差。等 SDK 出台正式约定后，`CONVENTION` 会 bump，届时所有 v0 渲染的画面都应视为近似。
- **关键姿态检测器未在真实抓取数据上验证。** 合成测试证明它能 *排除* 1–2 帧丢帧、能 *识别* 持续闭合。但报告的事件帧会比真实跨越点滞后最多 `median_window // 2` 帧（30fps 约 67ms）。50% 阈值只是起点。等有真实抓取的 episode 之后，`median_window` 和 `threshold_percent` 会需要调参。
- **SDK 传感器丢帧会导致回放画面有可见的抖动。** 每只手约 4% 的帧带 1–2 帧的尖峰（食指/拇指自由度瞬间闭合再回弹）。关键姿态检测器会过滤这些，但姿态流本身不做过滤（设计策略：查看器保持忠实，传感器路径由上游修复）。如果这成为阻塞问题，可以加一个可选的渲染端中值滤波 —— 反馈一下我们再重开这个讨论。
- **npz 中预计算的末端执行器姿态不能直接使用。** `ee_poses_qpos_{left,right}` 是相对每条臂的 J2 坐标系的（采集流水线用的是单臂 URDF），并非工作站世界系。我们忽略这些字段，改用工作站 URDF 做 FK。也没有做对照校验。
- **相机和轨迹视觉效果针对 `a7_lite_l6_dc` 调优。** 初始相机位姿、`head_radius`、`tail_intensity`、轨迹 `line_width` 都为这个工作站的尺度硬编码。换更高或更矮的机器人需要重新调，目前还没有对应的配置项。
- **内置解码器 sidecar 可能悄悄过时。** [`src/linker_sim_viser/hand_decoders/linkerhand_l6/decoder.yaml`](src/linker_sim_viser/hand_decoders/linkerhand_l6/decoder.yaml) 是 `linker-robot-assets` 那份 sidecar 的手动拷贝。如果采集组改了他们那一份，这边不会自动同步。等 assets 包正式发布后，整棵目录删掉即可。
- **内置资产包。** [`assets/`](assets/) 目录是 `a7_lite_l6_dc` 工作站 URDF 及其约 40 个 STL 网格的一个测试快照，从 `linker-robot-assets` 拷贝而来。之所以打包进来，是为了让同事在没有 sibling `linker-sim/` checkout 的情况下也能测试这个仓库。等 `linker-robot-assets` 正式作为可安装包发布后，删除这棵目录并修改 `configs/robots/a7_lite_l6_dc.yaml` 的 `urdf_path` 即可。

## 路线图

v1+ 待办详见 [docs/design.md](docs/design.md)（多 episode 浏览、视频同步、并排对比、标注、embedding map、独立于 Viser 侧栏的伴生 DOM 时间轴等等）。
