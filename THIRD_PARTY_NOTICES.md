# Third-Party Notices

This project depends on third-party software and assets with separate
licenses. End users and redistributors are responsible for ensuring full
compliance with all applicable third-party licenses when shipping source,
binaries, containers, or integrated products.

## Viser

- Project: Viser
- Upstream: [https://github.com/nerfstudio-project/viser](https://github.com/nerfstudio-project/viser)
- License: Apache License 2.0.
- Notes: Browser-based 3D visualization/transport layer for the replay viewer
  (installed via `viser[urdf]`). Viser's PyPI metadata is inconsistent about
  its license label; the shipped LICENSE text and upstream repository are
  Apache-2.0.

## yourdfpy

- Project: yourdfpy
- Upstream: [https://github.com/clemense/yourdfpy](https://github.com/clemense/yourdfpy)
- License: MIT.
- Notes: URDF loader, pulled in via `viser[urdf]` and used directly by the
  mcap→episode pipeline.

## trimesh

- Project: trimesh
- Upstream: [https://github.com/mikedh/trimesh](https://github.com/mikedh/trimesh)
- License: MIT.
- Notes: Mesh loading via `viser[urdf]`.

## NumPy

- Project: NumPy
- Upstream: [https://github.com/numpy/numpy](https://github.com/numpy/numpy)
- License: BSD 3-Clause.

## PyYAML

- Project: PyYAML
- Upstream: [https://github.com/yaml/pyyaml](https://github.com/yaml/pyyaml)
- License: MIT.

## MCAP (mcap / mcap-ros2-support)

- Project: MCAP Python libraries
- Upstream: [https://github.com/foxglove/mcap](https://github.com/foxglove/mcap)
- License: MIT.
- Notes: Optional dependencies (installed via the `umi` extra) for reading
  UMI-Dex mcap episode logs.

## SciPy

- Project: SciPy
- Upstream: [https://github.com/scipy/scipy](https://github.com/scipy/scipy)
- License: BSD 3-Clause.
- Notes: Optional dependency (`umi` extra); used for rotation/Slerp math in the
  offline IK retargeting.

## websockets

- Project: websockets
- Upstream: [https://github.com/python-websockets/websockets](https://github.com/python-websockets/websockets)
- License: BSD 3-Clause.
- Notes: Viser transport dependency.

## UMI-Dex (Linkerbot)

- Project: UMI-Dex (Linkerbot)
- Upstream: [https://github.com/Linkerbot/UMI-Dex](https://github.com/Linkerbot/UMI-Dex)
- License: Apache 2.0.
- Notes: The mcap→episode pipeline mirrors the linker-sim UMI pipeline. No
  third-party UMI-Dex source is copied into this repository — the pipeline is a
  Linkerbot-authored reimplementation — but the attribution is kept parallel to
  the sibling repositories.

## linker-robot-assets (Linkerbot)

- Project: linker-robot-assets (Linkerbot)
- Notes: Provides the workstation URDFs, hand decoders, and full robot mesh
  set consumed by this viewer via `pkg://` resolution. It is a Linkerbot
  package obtained from the co-located `linker-sim` checkout (see the README
  Install section). The robot meshes it provides are Linkerbot hardware CAD.

## Robot meshes

The 3D meshes shipped under `assets/` (and those rendered via
`linker-robot-assets`) are Linkerbot hardware CAD released as open-source by
the manufacturer:

- A7 Lite arm meshes (A7 family) — Linkerbot.
- A7 Lite torso meshes — Linkerbot.
- LinkerHand L6 / O6 / L25 / L20lite hand meshes (rendered via
  `linker-robot-assets`) — Linkerbot.

Each mesh is included in good faith based on the upstream open-source release.
Redistributors should retain the upstream attribution.

## Responsibility

End users and redistributors are responsible for ensuring full compliance with
all applicable third-party licenses when shipping source, binaries,
containers, or integrated products.
