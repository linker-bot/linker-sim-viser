"""CLI entry: python scripts/replay.py --robot NAME --episode DIR"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from linker_sim_viser import app

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOTS_DIR = REPO_ROOT / "configs" / "robots"


def main() -> int:
    p = argparse.ArgumentParser(description="Replay one recorded episode in Viser.")
    p.add_argument(
        "--robot",
        required=True,
        help=f"robot config name (looked up under {ROBOTS_DIR}/<NAME>.yaml)",
    )
    p.add_argument(
        "--episode",
        required=True,
        help="path to an episode directory (must contain telemetry.npz + metadata.json)",
    )
    p.add_argument(
        "--viewer-config",
        default=None,
        help="override viewer.yaml (default: configs/viewer.yaml)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="override viewer port from viewer.yaml",
    )
    args = p.parse_args()

    robot_config = ROBOTS_DIR / f"{args.robot}.yaml"
    if not robot_config.is_file():
        available = sorted(f.stem for f in ROBOTS_DIR.glob("*.yaml"))
        print(f"error: robot config not found: {robot_config}", file=sys.stderr)
        print(f"       available: {', '.join(available)}", file=sys.stderr)
        return 2

    episode_dir = Path(args.episode)
    if not (episode_dir / "telemetry.npz").is_file():
        print(
            f"error: {episode_dir}/telemetry.npz missing "
            f"(episode must be a directory, not the .npz itself)",
            file=sys.stderr,
        )
        return 2

    app.run(
        robot_config_path=robot_config,
        episode_path=episode_dir,
        viewer_config_path=args.viewer_config,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
