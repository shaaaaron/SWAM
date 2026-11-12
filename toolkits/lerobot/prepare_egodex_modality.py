#!/usr/bin/env python3
"""校验 EgoDex LeRobot task roots 并写入 DreamZero modality metadata。

Example:
    python toolkits/lerobot/prepare_egodex_modality.py \
        --data-root /path/to/egodex \
        --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


STATE_NAMES = [
    "left_hand.x",
    "left_hand.y",
    "left_hand.z",
    "right_hand.x",
    "right_hand.y",
    "right_hand.z",
]
ACTION_NAMES = [
    "delta_left_hand.x",
    "delta_left_hand.y",
    "delta_left_hand.z",
    "delta_right_hand.x",
    "delta_right_hand.y",
    "delta_right_hand.z",
]
STATE_CONFIDENCE_KEY = "observation.state_confidence"

MODALITY_JSON = {
    "state": {
        "left_hand_position": {
            "start": 0,
            "end": 3,
            "original_key": "observation.state",
            "absolute": True,
        },
        "right_hand_position": {
            "start": 3,
            "end": 6,
            "original_key": "observation.state",
            "absolute": True,
        },
    },
    "action": {
        "left_hand_delta": {
            "start": 0,
            "end": 3,
            "original_key": "action",
            "absolute": False,
        },
        "right_hand_delta": {
            "start": 3,
            "end": 6,
            "original_key": "action",
            "absolute": False,
        },
    },
    "video": {
        "egocentric": {
            "original_key": "observation.images.egocentric",
        }
    },
    "annotation": {
        "task": {
            "original_key": "task_index",
        }
    },
}


@dataclass(frozen=True)
class PrepareSummary:
    found: int
    missing: int
    matching: int
    mismatched: int
    written: int


def _load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象并校验顶层类型。"""
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, got {type(value).__name__}"
        )
    return value


def _flatten_string_names(value: Any) -> list[str]:
    """递归展平字符串或字符串序列。"""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        names: list[str] = []
        for item in value:
            names.extend(_flatten_string_names(item))
        return names
    return []


def _validate_vector_feature(
    features: dict[str, Any],
    *,
    key: str,
    expected_names: list[str],
) -> list[str]:
    """校验指定 6D 特征的 dtype、shape 和 names。"""
    errors: list[str] = []
    feature = features.get(key)
    if not isinstance(feature, dict):
        return [f"missing feature {key!r}"]
    if feature.get("dtype") != "float32":
        errors.append(f"{key}.dtype must be 'float32', got {feature.get('dtype')!r}")
    if feature.get("shape") != [6]:
        errors.append(f"{key}.shape must be [6], got {feature.get('shape')!r}")
    names = _flatten_string_names(feature.get("names"))
    if names != expected_names:
        errors.append(f"{key}.names must be {expected_names!r}, got {names!r}")
    return errors


def validate_egodex_info(info: dict[str, Any]) -> list[str]:
    """输入 info.json 内容，返回不符合 EgoDex 6D 固定切片的错误。"""
    errors: list[str] = []
    if info.get("robot_type") != "human_hands":
        errors.append(
            f"robot_type must be 'human_hands', got {info.get('robot_type')!r}"
        )

    features = info.get("features")
    if not isinstance(features, dict):
        return [*errors, "features must be a JSON object"]

    errors.extend(
        _validate_vector_feature(
            features,
            key="observation.state",
            expected_names=STATE_NAMES,
        )
    )
    errors.extend(
        _validate_vector_feature(
            features,
            key="action",
            expected_names=ACTION_NAMES,
        )
    )

    confidence = features.get(STATE_CONFIDENCE_KEY)
    if not isinstance(confidence, dict):
        errors.append(f"missing feature {STATE_CONFIDENCE_KEY!r}")
    else:
        if confidence.get("dtype") != "float32":
            errors.append(
                f"{STATE_CONFIDENCE_KEY}.dtype must be 'float32', "
                f"got {confidence.get('dtype')!r}"
            )
        if confidence.get("shape") != [2]:
            errors.append(
                f"{STATE_CONFIDENCE_KEY}.shape must be [2], "
                f"got {confidence.get('shape')!r}"
            )

    video = features.get("observation.images.egocentric")
    if not isinstance(video, dict):
        errors.append("missing feature 'observation.images.egocentric'")
    elif video.get("dtype") != "video":
        errors.append(
            "observation.images.egocentric.dtype must be 'video', "
            f"got {video.get('dtype')!r}"
        )
    if "task_index" not in features:
        errors.append("missing feature 'task_index'")
    return errors


def discover_egodex_roots(data_root: Path | str) -> list[Path]:
    """输入单 task 或集合目录，返回按路径排序的 EgoDex task roots。"""
    root = Path(data_root).expanduser()
    if (root / "meta" / "info.json").is_file():
        return [root]
    roots = sorted(path.parent.parent for path in root.rglob("meta/info.json"))
    if not roots:
        raise FileNotFoundError(
            f"No EgoDex LeRobot roots found under {root}; expected meta/info.json files"
        )
    return roots


def prepare_egodex_modality(
    data_root: Path | str,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> PrepareSummary:
    """校验全部 root 后写入缺失或明确要求覆盖的 modality.json。"""
    roots = discover_egodex_roots(data_root)

    schema_errors: list[str] = []
    for task_root in roots:
        info_path = task_root / "meta" / "info.json"
        try:
            info = _load_json(info_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            schema_errors.append(f"{info_path}: {exc}")
            continue
        errors = validate_egodex_info(info)
        schema_errors.extend(f"{info_path}: {error}" for error in errors)
    if schema_errors:
        details = "\n".join(f"- {error}" for error in schema_errors)
        raise ValueError(f"EgoDex schema validation failed:\n{details}")

    missing: list[Path] = []
    matching: list[Path] = []
    mismatched: list[Path] = []
    for task_root in roots:
        modality_path = task_root / "meta" / "modality.json"
        if not modality_path.is_file():
            missing.append(modality_path)
            continue
        try:
            current = _load_json(modality_path)
        except (OSError, json.JSONDecodeError, ValueError):
            mismatched.append(modality_path)
            continue
        (matching if current == MODALITY_JSON else mismatched).append(modality_path)

    if mismatched and not overwrite:
        details = "\n".join(f"- {path}" for path in mismatched)
        raise ValueError(
            "Existing modality.json files differ from the EgoDex template. "
            "No files were written. Re-run with --overwrite only after review:\n"
            f"{details}"
        )

    write_paths = [*missing, *mismatched]
    for path in matching:
        print(f"[EXISTS] {path}")
    for path in missing:
        print(f"[WRITE] {path}")
    for path in mismatched:
        print(f"[OVERWRITE] {path}")

    written = 0
    if not dry_run:
        payload = json.dumps(MODALITY_JSON, indent=4) + "\n"
        for path in write_paths:
            path.write_text(payload, encoding="utf-8")
            written += 1

    return PrepareSummary(
        found=len(roots),
        missing=len(missing),
        matching=len(matching),
        mismatched=len(mismatched),
        written=written,
    )


def main() -> None:
    """解析命令行参数并校验、生成 EgoDex modality sidecar。"""
    parser = argparse.ArgumentParser(
        description="Validate EgoDex LeRobot roots and prepare modality.json sidecars"
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        type=Path,
        required=True,
        help="A single EgoDex task root or a collection containing task roots",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="Validate and print planned writes without changing files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing modality.json files that differ from the template",
    )
    args = parser.parse_args()

    summary = prepare_egodex_modality(
        args.data_root,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    mode = "Dry run" if args.dry_run else "Done"
    print(
        f"\n{mode}: found={summary.found} missing={summary.missing} "
        f"matching={summary.matching} mismatched={summary.mismatched} "
        f"written={summary.written}"
    )


if __name__ == "__main__":
    main()
