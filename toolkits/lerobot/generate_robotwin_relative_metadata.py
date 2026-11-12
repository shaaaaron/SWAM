#!/usr/bin/env python3
"""Generate one RoboTwin relative DreamZero metadata file from a task collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.data.datasets.dreamzero.relative_actions import (
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_POSE_REFERENCE_PER_ARM,
    resolve_relative_action_pose_reference,
    ROBOTWIN_EEF_ACTION_KEY,
    ROBOTWIN_EEF_STATE_KEY,
)
from toolkits.lerobot import generate_dreamzero_metadata as gm


def _task_roots(collection_root: Path) -> list[Path]:
    roots = sorted(
        p for p in collection_root.glob("*/*") if (p / "meta/info.json").is_file()
    )
    if not roots:
        raise ValueError(f"No task LeRobot roots found under {collection_root}")
    return roots


def _assert_same_modality(
    reference: dict[str, Any], current: dict[str, Any], root: Path
) -> None:
    if current != reference:
        raise ValueError(f"meta/modality.json differs from the first task root: {root}")


def _read_episode_arrays(
    task_root: Path,
    *,
    state_key: str,
    action_key: str,
    action_horizon: int,
    max_chunk_size: int,
    anchor_stride: int,
    relative_action_mode: str,
    relative_action_pose_reference: str,
    max_episodes: int | None,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    dict[str, Any],
    dict[str, Any],
]:
    meta_dir = task_root / "meta"
    info = gm._load_json(meta_dir / "info.json")
    modality_cfg = gm._load_modality_json(meta_dir)
    episodes = gm._load_episodes(meta_dir)
    if max_episodes is not None:
        episodes = episodes[: int(max_episodes)]

    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    relative_state_chunks: list[np.ndarray] = []
    relative_action_chunks: list[np.ndarray] = []

    for episode_index in episodes:
        parquet_path = gm._episode_path(task_root, info, int(episode_index))
        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"Episode {episode_index} is listed in {meta_dir / 'episodes.jsonl'} "
                f"but parquet is missing: {parquet_path}"
            )
        schema_names = list(pq.read_schema(parquet_path).names)
        resolved_state_key = gm._resolve_column(
            schema_names, state_key, aliases=["observation.state", "state"]
        )
        resolved_action_key = gm._resolve_column(
            schema_names, action_key, aliases=["action", "actions"]
        )
        table = pq.read_table(
            parquet_path, columns=[resolved_state_key, resolved_action_key]
        )
        state_episode = gm._column_to_2d_array(table, resolved_state_key)
        action_episode = gm._column_to_2d_array(table, resolved_action_key)
        state_chunks.append(state_episode)
        action_chunks.append(action_episode)
        (
            relative_state_episode,
            relative_action_episode,
        ) = gm._robotwin_eef_relative_samples(
            state_episode,
            action_episode,
            modality_cfg,
            relative_action_mode=relative_action_mode,
            action_horizon=action_horizon,
            max_chunk_size=max_chunk_size,
            anchor_stride=anchor_stride,
            episode_index=int(episode_index),
            relative_action_pose_reference=relative_action_pose_reference,
        )
        if relative_state_episode.size > 0:
            relative_state_chunks.append(relative_state_episode)
        if relative_action_episode.size > 0:
            relative_action_chunks.append(relative_action_episode)

    return (
        state_chunks,
        action_chunks,
        relative_state_chunks,
        relative_action_chunks,
        info,
        modality_cfg,
    )


def _build_robotwin_metadata(
    *,
    info: dict[str, Any],
    modality_cfg: dict[str, Any],
    state: np.ndarray,
    action: np.ndarray,
    relative_state_eef: np.ndarray,
    relative_action_eef: np.ndarray,
    relative_action_mode: str,
    relative_action_pose_reference: str,
    state_key: str,
    action_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_stats_full = gm._compute_stats(state)
    action_stats_full = gm._compute_stats(action)
    relative_state_eef_stats = gm._compute_stats(relative_state_eef)
    relative_action_eef_stats = gm._compute_stats(relative_action_eef)

    state_stats = gm._build_split_statistics(state_stats_full, modality_cfg, kind="state")
    action_stats = gm._build_split_statistics(
        action_stats_full, modality_cfg, kind="action"
    )
    state_stats["eef_position"] = relative_state_eef_stats
    action_stats["eef_position"] = relative_action_eef_stats

    state_modalities = gm._build_split_modalities(
        modality_cfg, kind="state", full_dim=int(state.shape[-1])
    )
    action_modalities = gm._build_split_modalities(
        modality_cfg, kind="action", full_dim=int(action.shape[-1])
    )
    video_modalities = gm._video_modalities_from_modality(info, modality_cfg)

    lerobot_stats = {
        state_key: state_stats_full,
        action_key: action_stats_full,
    }
    for subkey, spec in (modality_cfg.get("state") or {}).items():
        start, end = int(spec["start"]), int(spec["end"])
        lerobot_stats[f"state.{subkey}"] = gm._slice_stats(state_stats_full, start, end)
    for subkey, spec in (modality_cfg.get("action") or {}).items():
        start, end = int(spec["start"]), int(spec["end"])
        lerobot_stats[f"action.{subkey}"] = gm._slice_stats(
            action_stats_full, start, end
        )
    lerobot_stats[ROBOTWIN_EEF_STATE_KEY] = relative_state_eef_stats
    lerobot_stats[ROBOTWIN_EEF_ACTION_KEY] = relative_action_eef_stats

    metadata = {
        "robotwin": {
            "statistics": {
                "state": state_stats,
                "action": action_stats,
            },
            "modalities": {
                "video": video_modalities,
                "state": state_modalities,
                "action": action_modalities,
            },
            "embodiment_tag": "robotwin",
            "relative_action_mode": relative_action_mode,
            "relative_action_pose_reference": relative_action_pose_reference,
        }
    }
    return metadata, lerobot_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--output-stats", type=Path, required=True)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--action-horizon", type=int, required=True)
    parser.add_argument("--max-chunk-size", type=int, required=True)
    parser.add_argument(
        "--anchor-stride",
        type=int,
        default=20,
        help="Frame stride between s0 anchors used for metadata stats sampling.",
    )
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--relative-action-mode",
        choices=(
            RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
            RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
        ),
        default=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
    )
    parser.add_argument(
        "--relative-action-pose-reference",
        choices=("per_arm", "left_hand"),
        default=RELATIVE_ACTION_POSE_REFERENCE_PER_ARM,
    )
    args = parser.parse_args()

    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive.")
    if args.max_chunk_size <= 0:
        raise ValueError("--max-chunk-size must be positive.")
    if args.anchor_stride <= 0:
        raise ValueError("--anchor-stride must be positive.")

    roots = _task_roots(args.dataset_root)
    if args.max_datasets is not None:
        roots = roots[: int(args.max_datasets)]

    print(
        "[RoboTwin relative metadata] "
        f"dataset_root={args.dataset_root} num_task_roots={len(roots)} "
        f"mode={args.relative_action_mode!r} "
        f"action_horizon={args.action_horizon} max_chunk_size={args.max_chunk_size} "
        f"anchor_stride={args.anchor_stride} "
        f"pose_reference={args.relative_action_pose_reference!r}",
        flush=True,
    )

    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    relative_state_chunks: list[np.ndarray] = []
    relative_action_chunks: list[np.ndarray] = []
    reference_info: dict[str, Any] | None = None
    reference_modality: dict[str, Any] | None = None

    for idx, root in enumerate(roots, start=1):
        (
            task_state,
            task_action,
            task_relative_state,
            task_relative_action,
            info,
            modality_cfg,
        ) = _read_episode_arrays(
            root,
            state_key=args.state_key,
            action_key=args.action_key,
            action_horizon=args.action_horizon,
            max_chunk_size=args.max_chunk_size,
            anchor_stride=args.anchor_stride,
            relative_action_mode=args.relative_action_mode,
            max_episodes=args.max_episodes,
            relative_action_pose_reference=resolve_relative_action_pose_reference(
                args.relative_action_pose_reference,
                relative_action_mode=args.relative_action_mode,
            ),
        )
        if reference_info is None:
            reference_info = info
            reference_modality = modality_cfg
        else:
            assert reference_modality is not None
            _assert_same_modality(reference_modality, modality_cfg, root)
        state_chunks.extend(task_state)
        action_chunks.extend(task_action)
        relative_state_chunks.extend(task_relative_state)
        relative_action_chunks.extend(task_relative_action)
        print(
            f"[RoboTwin relative metadata] loaded {idx}/{len(roots)} "
            f"root={root} episodes={len(task_state)} "
            f"rel_state_chunks={len(task_relative_state)} "
            f"rel_action_chunks={len(task_relative_action)}",
            flush=True,
        )

    if reference_info is None or reference_modality is None:
        raise ValueError("No task roots were loaded.")
    if not relative_state_chunks or not relative_action_chunks:
        raise ValueError("No relative state/action samples were generated.")

    state = np.concatenate(state_chunks, axis=0)
    action = np.concatenate(action_chunks, axis=0)
    relative_state_eef = np.concatenate(relative_state_chunks, axis=0)
    relative_action_eef = np.concatenate(relative_action_chunks, axis=0)
    print(
        "[RoboTwin relative metadata] concatenated "
        f"state={state.shape} action={action.shape} "
        f"relative_state_eef={relative_state_eef.shape} "
        f"relative_action_eef={relative_action_eef.shape}",
        flush=True,
    )

    metadata, stats = _build_robotwin_metadata(
        info=reference_info,
        modality_cfg=reference_modality,
        state=state,
        action=action,
        relative_state_eef=relative_state_eef,
        relative_action_eef=relative_action_eef,
        relative_action_mode=args.relative_action_mode,
        relative_action_pose_reference=args.relative_action_pose_reference,
        state_key=args.state_key,
        action_key=args.action_key,
    )

    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_stats.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        f.write("\n")
    with open(args.output_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.output_metadata}", flush=True)
    print(f"Wrote {args.output_stats}", flush=True)


if __name__ == "__main__":
    main()
