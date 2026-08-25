# linker-sim-viser

语言：[中文](README_zh.md) | [English](README.md)

基于浏览器的无物理回放查看器，用于 LinkerBot 录制数据。构建在 [Viser](https://github.com/nerfstudio-project/viser) 之上，URDF 加载使用 `yourdfpy`（原生支持 mimic 关节 —— LinkerHand 的腱耦合无需额外代码即可正确渲染）。

本仓库是 `linker-sim` 家族的一部分。姊妹仓库：`linker-sim-mujoco`（含物理的回放，规划中），`linker-sim-isaac`（正向仿真 + 运动规划）。

## 状态

**v0 已发布。** 支持单条 episode 回放、彗星轨迹、关键姿态标记。

## v0 功能

- 从 `data/episode_XXXXXX/` 加载一条录制的 episode（SDK 格式 1.4：`telemetry.npz` + `metadata.json`）。
- 渲染双臂 + LinkerHand 工作站 URDF，由 episode 的关节流驱动。
- 侧栏回放控件：帧滑块、播放/暂停、速度预设（`0.25x` / `0.5x` / `1x` / `2x` / `4x`）、循环、时间读数。
- 手部解码由已安装的 `linker-robot-assets` 包（`decode_hand`）完成：将原始 SDK 数据包按 URDF 关节顺序重排，并使用从手部 URDF 读取的限位映射为关节弧度。
- **彗星轨迹**：每个配置的末端执行器一条，随回放推进而生长，从明亮的头端渐变到暗淡的尾端，由一个彩色小球标记当前位置。
- **关键姿态标记**：从 SDK 手部流中检测抓取/释放事件，使用 5 点中值滤波去除 1–2 帧的传感器丢帧再触发。

## 安装

**前置条件 —— 姊妹仓库 `linker-sim` checkout。** 本仓库无法独立安装。`pyproject.toml` 通过指向 `../linker-sim/packages/linker-robot-assets` 的可编辑路径依赖引入 `linker-robot-assets`，机器人配置也通过该包（`pkg://` 路径）解析 URDF。请先把 `linker-sim` 作为同级目录 clone 下来，使相对本仓库的 `../linker-sim/packages/linker-robot-assets` 存在 —— 安装和运行都依赖它。

```bash
uv venv
uv pip install -e ".[dev]"
```

**克隆时需要 Git LFS。** [`assets/`](assets/) 下 15 个内置的 STL 网格由 LFS 管理（见 [`.gitattributes`](.gitattributes)）。每台机器安装一次 `git-lfs`（`apt install git-lfs && git lfs install`），然后 `git clone` 会自动拉取网格。如果你已经在没启用 LFS 的情况下 clone 了，运行 `git lfs install && git lfs pull` 补拉即可。

## 前置条件

- **录制数据**：放到 `data/episode_XXXXXX/`。`data/` 目录已被 `.gitignore` 忽略（episode 是大二进制包）；请自行获取一份录制的 episode 包并放到 `data/` 下。

机器人的 URDF 与网格从已安装的 `linker-robot-assets` 包加载（见 [安装](#安装)）：`configs/robots/*.yaml` 中的 `urdf_path` 使用 `pkg://` 路径，相对该包的 `asset_root()` 解析。若想指向别处的 URDF，编辑 `urdf_path` 即可。

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

## UMI-Dex mcap 回放（实验性）

UMI-Dex 手持设备的 bag（ROS 2 mcap：`/vut/pose` 上的 6-DOF 手腕位姿 + `/hand/joint_states` 上的
LinkerHand L6 百分比）通过先转换为查看器原生 episode 来回放，流程对齐 `linker-sim` 的 UMI 管线：
转换器读取 bag、重采样到固定帧率、用 Nelder-Mead 搜索把手腕位姿锚定到机械臂的 `tool0`、解 DLS IK
把 7-DOF 右臂重定向到该轨迹、把手部解码为弧度，然后把 `telemetry.npz` + `metadata.json` 写进 episode 目录：

```bash
uv pip install -e ".[umi]"     # mcap + scipy（一次性）
.venv/bin/python scripts/umi_mcap_to_episode.py --episode data/episode_000004 --side right
.venv/bin/python scripts/replay.py --robot a7_lite_l6_umi_right --episode data/episode_000004
```

转换器会打印 IK 跟踪残差（锚点搜索后位置平均误差通常在亚毫米到几毫米级）。只驱动右臂 + 右手，左臂保持默认位姿。
如果机械臂落点别扭，可用 `--no-search` 配合手动 `--dx/dy/dz`、`--anchor-roll/pitch/yaw`、
`--remap-roll/pitch/yaw`（语义与 linker-sim 一致）。这是离线重定向 —— 见 [docs/design.md](docs/design.md) 的范围修订说明。

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

- **不做物理仿真。** 需要动力学感知的回放属于规划中的 `linker-sim-mujoco`。
- **不做 QC / 异常检测。** 录制数据默认视为干净。若发现系统性传感器问题，请反馈到上游修复。
- **不做*运行时*正向仿真。** 回放时从不做前向仿真。（用于回放的*离线* IK 重定向 —— 例如 UMI mcap 转换器 —— 是允许的；见 [docs/design.md](docs/design.md) 的范围修订。）
- v0 暂不支持多 episode 浏览、相机视频同步、标注、LeRobot parquet 或 ROS 2 `.mcap` 摄入。

## 已知限制

提交 bug 前请先读这些 —— 它们是 v0 有意接受的取舍或尚未解决的上游问题。

- **手部解码是近似的。** 解码委托给 `linker-robot-assets`（`decode_hand`）：它先对原始 SDK 数据包重新缩放，再线性映射到每个关节 URDF 的 `[lower, upper]`。上游包明确说明 SDK 真正的角度约定尚未最终确定。可见现象：静止时拇指和食指指尖会有轻微重叠。渲染出来的手指姿态都存在未知量级的偏差。在 SDK 出台正式约定之前，所有 v0 渲染的手部姿态都应视为近似。
- **关键姿态检测器未在真实抓取数据上验证。** 合成测试证明它能 *排除* 1–2 帧丢帧、能 *识别* 持续闭合。但报告的事件帧会比真实跨越点滞后最多 `median_window // 2` 帧（30fps 约 67ms）。50% 阈值只是起点。等有真实抓取的 episode 之后，`median_window` 和 `threshold_percent` 会需要调参。
- **SDK 传感器丢帧会导致回放画面有可见的抖动。** 每只手约 4% 的帧带 1–2 帧的尖峰（食指/拇指自由度瞬间闭合再回弹）。关键姿态检测器会过滤这些，但姿态流本身不做过滤（设计策略：查看器保持忠实，传感器路径由上游修复）。如果这成为阻塞问题，可以加一个可选的渲染端中值滤波 —— 反馈一下我们再重开这个讨论。
- **npz 中预计算的末端执行器姿态不能直接使用。** `ee_poses_qpos_{left,right}` 是相对每条臂的 J2 坐标系的（采集流水线用的是单臂 URDF），并非工作站世界系。我们忽略这些字段，改用工作站 URDF 做 FK。也没有做对照校验。
- **相机和轨迹视觉效果针对 `a7_lite_l6_dc` 调优。** 初始相机位姿、`head_radius`、`tail_intensity`、轨迹 `line_width` 都为这个工作站的尺度硬编码。换更高或更矮的机器人需要重新调，目前还没有对应的配置项。
- **手部解码位于 `linker-robot-assets`，未在本仓库内置拷贝。** 解码器（`linker_robot_assets.decoders.decode_hand`）及其每只手的 `decoder.yaml` sidecar 都在已安装的包里；本仓库只保留一个用于校验的小型 URDF 关节限位读取器（`src/linker_sim_viser/decoders.py`）。这里没有本地拷贝，不会出现不同步的漂移，但解码行为只与姊妹 checkout 中 `linker-robot-assets` 的版本一致。
- **部分内置网格快照。** [`assets/`](assets/) 目录保存了 15 个由 LFS 管理的 STL 网格（a7_lite 的左右臂变体加一个躯干基座），从 `linker-robot-assets` 拷贝而来。权威的 URDF 与全分辨率网格来自已安装的 `linker-robot-assets` 包（见 [安装](#安装)）；机器人配置通过 `pkg://` 相对该包解析 `urdf_path`。这个快照并非自包含 —— 安装和运行仍需要姊妹仓库 `linker-sim` 的 checkout。

## 路线图

v1+ 待办详见 [docs/design.md](docs/design.md)（多 episode 浏览、视频同步、并排对比、标注、embedding map、独立于 Viser 侧栏的伴生 DOM 时间轴等等）。

## 许可

本项目基于 [MIT License](LICENSE) 发布，© Linkerbot (Beijing) Technology Co., Ltd.。
第三方软件与素材许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
