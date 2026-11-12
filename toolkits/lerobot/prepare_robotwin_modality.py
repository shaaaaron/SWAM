#!/usr/bin/env python3
"""Inject RoboTwin modality metadata into LeRobot task folders.

Example:
    python toolkits/lerobot/prepare_robotwin_modality.py \
        --data-root /path/to/robotwin-clean-and-aug-lerobot
"""

import argparse
import json
import os
from pathlib import Path
from typing import Union

MODALITY_JSON = {
    "state": {
        "eef_position": {
            "start": 0,
            "end": 16,
            "original_key": "observation.state",
        }
    },
    "action": {
        "eef_position": {
            "start": 0,
            "end": 16,
            "original_key": "action",
        }
    },
    "video": {
        "cam_high": {"original_key": "observation.images.cam_high"},
        "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
        "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
    },
    "annotation": {
        "language.action_text": {"original_key": "task_index"},
    },
}

EMBODIMENT_JSON = {"robot_type": "robotwin", "embodiment_tag": "robotwin"}

DEFAULT_SUBSETS = ["lerobot_robotwin_eef_aug_500", "lerobot_robotwin_eef_clean_50"]


def prepare_robotwin_modality(
    data_root: Union[Path, str],
    subsets: list[str],
    *,
    dry_run: bool = False,
) -> int:
    """Write ``modality.json`` and ``embodiment.json`` for each RoboTwin task."""
    data_root = Path(data_root)
    total_written = 0

    for subset_name in subsets:
        subset_dir = data_root / subset_name
        if not subset_dir.is_dir():
            print(f"[SKIP] Subset directory not found: {subset_dir}")
            continue

        for task_name in sorted(os.listdir(subset_dir)):
            task_dir = subset_dir / task_name
            info_path = task_dir / "meta" / "info.json"
            if not info_path.is_file():
                continue

            modality_path = task_dir / "meta" / "modality.json"
            embodiment_path = task_dir / "meta" / "embodiment.json"
            if dry_run:
                status = "EXISTS" if modality_path.exists() else "WRITE"
                print(f"[{status}] {modality_path}")
                continue

            modality_path.parent.mkdir(parents=True, exist_ok=True)
            with open(modality_path, "w", encoding="utf-8") as f:
                json.dump(MODALITY_JSON, f, indent=4)
            with open(embodiment_path, "w", encoding="utf-8") as f:
                json.dump(EMBODIMENT_JSON, f, indent=4)
            total_written += 1

    return total_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare modality metadata for RoboTwin LeRobot datasets"
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing RoboTwin LeRobot subsets",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=DEFAULT_SUBSETS,
        help="Subset directory names to process",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="Print what would be done without writing files",
    )
    args = parser.parse_args()

    total_written = prepare_robotwin_modality(
        args.data_root,
        args.subsets,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\nDry run complete. No files were written.")
    else:
        print(f"\nDone. Wrote RoboTwin metadata to {total_written} task folders.")


if __name__ == "__main__":
    main()
