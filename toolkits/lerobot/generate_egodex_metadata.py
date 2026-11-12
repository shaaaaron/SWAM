#!/usr/bin/env python3
"""从 EgoDex LeRobot task 集合生成 absolute state/action metadata。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from rlinf.data.datasets.dreamzero.egodex_confidence import (
    filter_egodex_episode_indices,
)

STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_NAMES = ("left_hand_position", "right_hand_position")
ACTION_NAMES = ("left_hand_position", "right_hand_position")


def discover_task_roots(data_root: Path) -> list[Path]:
    """输入 EgoDex 集合目录，返回按路径排序的 LeRobot task roots。"""
    roots = sorted(path.parent.parent for path in data_root.rglob("meta/info.json"))
    if not roots:
        raise FileNotFoundError(f"No EgoDex task roots found under {data_root}")
    return roots


def episode_paths(task_root: Path) -> tuple[list[Path], int]:
    """按 sidecar 筛选单个 task，并返回 parquet 路径和总帧数。"""
    meta_dir = task_root / "meta"
    records: dict[int, dict] = {}
    with open(meta_dir / "episodes.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            episode_index = int(record["episode_index"])
            if episode_index in records:
                raise ValueError(
                    f"Duplicate episode_index={episode_index} in "
                    f"{meta_dir / 'episodes.jsonl'}"
                )
            records[episode_index] = record
    actual: dict[int, Path] = {}
    for path in (task_root / "data").rglob("episode_*.parquet"):
        episode_index = int(path.stem.removeprefix("episode_"))
        if episode_index in actual:
            raise ValueError(
                f"Duplicate parquet episode_index={episode_index} under {task_root}"
            )
        actual[episode_index] = path
    if set(actual) != set(records):
        raise ValueError(
            f"Episode set mismatch under {task_root}: "
            f"declared={len(records)} actual={len(actual)}"
        )
    kept = filter_egodex_episode_indices(meta_dir, records)
    total_frames = 0
    for index in kept:
        record = records[index]
        for key in ("episode_length", "length", "num_frames", "num_steps"):
            if record.get(key) is not None:
                total_frames += int(record[key])
                break
        else:
            total_frames += pq.read_metadata(actual[index]).num_rows
    print(
        f"[egodex metadata] confidence filter root={task_root} "
        f"kept={len(kept)} skipped={len(records) - len(kept)}",
        flush=True,
    )
    return [actual[index] for index in sorted(kept)], total_frames


def _statistics(
    minimum: np.ndarray,
    maximum: np.ndarray,
    total: np.ndarray,
    total_square: np.ndarray,
    count: np.ndarray,
    samples: np.ndarray,
) -> dict[str, list[float]]:
    """输入流式累计量和分位数样本，输出 DreamZero 六项统计量。"""
    if np.any(count <= 0):
        raise ValueError(
            f"Every EgoDex dimension needs valid values, got count={count}."
        )
    mean = total / count
    variance = np.maximum(total_square / count - mean * mean, 0.0)
    q01: list[float] = []
    q99: list[float] = []
    for dim in range(samples.shape[1]):
        valid_samples = samples[np.isfinite(samples[:, dim]), dim]
        if len(valid_samples) == 0:
            raise ValueError(f"No quantile samples for EgoDex dimension {dim}.")
        q01.append(float(np.quantile(valid_samples, 0.01)))
        q99.append(float(np.quantile(valid_samples, 0.99)))
    return {
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "q01": q01,
        "q99": q99,
    }


def _update_statistics(
    values: np.ndarray,
    *,
    minimum: np.ndarray,
    maximum: np.ndarray,
    total: np.ndarray,
    total_square: np.ndarray,
    count: np.ndarray,
) -> None:
    """使用 clean episode 的全部维度更新流式统计量。"""
    if values.ndim != 2:
        raise ValueError(f"EgoDex values must have shape [T,D], got {values.shape}.")
    if len(values) == 0:
        return
    minimum[:] = np.minimum(minimum, values.min(axis=0))
    maximum[:] = np.maximum(maximum, values.max(axis=0))
    total += values.sum(axis=0)
    total_square += np.square(values).sum(axis=0)
    count += len(values)


def compute_statistics(
    parquet_paths: list[Path],
    *,
    total_frames: int,
    quantile_sample_limit: int,
    progress_every: int = 10000,
) -> tuple[dict[str, list[float]], dict[str, list[float]], int]:
    """扫描 parquet，输出 6D absolute state/action 统计和分位数样本数。"""
    if quantile_sample_limit <= 0:
        raise ValueError("quantile_sample_limit must be positive")
    sample_stride = max(1, math.ceil(total_frames / quantile_sample_limit))
    sample_capacity = math.ceil(total_frames / sample_stride)
    state_min = np.full(6, np.inf, dtype=np.float64)
    state_max = np.full(6, -np.inf, dtype=np.float64)
    state_sum = np.zeros(6, dtype=np.float64)
    state_square_sum = np.zeros(6, dtype=np.float64)
    action_min = np.full(6, np.inf, dtype=np.float64)
    action_max = np.full(6, -np.inf, dtype=np.float64)
    action_sum = np.zeros(6, dtype=np.float64)
    action_square_sum = np.zeros(6, dtype=np.float64)
    state_count = np.zeros(6, dtype=np.int64)
    action_count = np.zeros(6, dtype=np.int64)
    state_samples = np.full((sample_capacity, 6), np.nan, dtype=np.float32)
    action_samples = np.full((sample_capacity, 6), np.nan, dtype=np.float32)
    sample_count = 0
    frames_read = 0

    for file_index, path in enumerate(parquet_paths, start=1):
        table = pq.read_table(path, columns=[STATE_KEY, ACTION_KEY])
        state = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float64)
        delta = np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != 6 or delta.shape != state.shape:
            raise ValueError(
                f"Expected state/delta [T,6] in {path}, got {state.shape}/{delta.shape}"
            )
        if not np.isfinite(state).all() or not np.isfinite(delta).all():
            raise ValueError(f"Non-finite state/action value in {path}")

        action = state[:-1] + delta[:-1]
        if not np.allclose(action, state[1:], rtol=1e-5, atol=1e-6):
            max_error = float(np.max(np.abs(action - state[1:])))
            raise ValueError(
                "EgoDex delta does not reconstruct the next state in "
                f"{path}; max_abs_error={max_error}."
            )

        _update_statistics(
            state,
            minimum=state_min,
            maximum=state_max,
            total=state_sum,
            total_square=state_square_sum,
            count=state_count,
        )
        _update_statistics(
            action,
            minimum=action_min,
            maximum=action_max,
            total=action_sum,
            total_square=action_square_sum,
            count=action_count,
        )

        start = (-frames_read) % sample_stride
        sampled_rows = np.arange(start, len(state), sample_stride, dtype=np.int64)
        end = sample_count + len(sampled_rows)
        state_samples[sample_count:end] = state[sampled_rows]
        sampled_action_exists = sampled_rows < len(action)
        if np.any(sampled_action_exists):
            action_rows = sampled_rows[sampled_action_exists]
            action_slots = np.arange(sample_count, end)[sampled_action_exists]
            action_samples[action_slots] = action[action_rows]
        sample_count = end
        frames_read += len(state)

        if progress_every > 0 and file_index % progress_every == 0:
            print(
                f"[egodex metadata] files={file_index}/{len(parquet_paths)} "
                f"frames={frames_read}",
                flush=True,
            )

    if frames_read != total_frames:
        raise ValueError(f"Expected {total_frames} frames, read {frames_read}")
    state_samples = state_samples[:sample_count]
    action_samples = action_samples[:sample_count]
    state_stats = _statistics(
        state_min,
        state_max,
        state_sum,
        state_square_sum,
        state_count,
        state_samples,
    )
    action_stats = _statistics(
        action_min,
        action_max,
        action_sum,
        action_square_sum,
        action_count,
        action_samples,
    )
    return state_stats, action_stats, sample_count


def build_metadata(
    state_stats: dict[str, list[float]],
    action_stats: dict[str, list[float]],
) -> dict:
    """输入完整 6D 统计，输出左右手 3D absolute DreamZero metadata。"""

    def split(stats: dict[str, list[float]], start: int, end: int) -> dict:
        """按维度切分完整 6D 统计量。"""
        return {name: values[start:end] for name, values in stats.items()}

    return {
        "egodex": {
            "statistics": {
                "state": {
                    STATE_NAMES[0]: split(state_stats, 0, 3),
                    STATE_NAMES[1]: split(state_stats, 3, 6),
                },
                "action": {
                    ACTION_NAMES[0]: split(action_stats, 0, 3),
                    ACTION_NAMES[1]: split(action_stats, 3, 6),
                },
            },
            "modalities": {
                "video": {
                    "egocentric": {
                        "resolution": [1920, 1080],
                        "channels": 3,
                        "fps": 30.0,
                    }
                },
                "state": {
                    name: {
                        "absolute": True,
                        "rotation_type": None,
                        "shape": [3],
                        "continuous": True,
                    }
                    for name in STATE_NAMES
                },
                "action": {
                    name: {
                        "absolute": True,
                        "rotation_type": None,
                        "shape": [3],
                        "continuous": True,
                    }
                    for name in ACTION_NAMES
                },
            },
            "embodiment_tag": "egodex",
        }
    }


def main() -> None:
    """解析命令行参数并生成 EgoDex metadata 文件。"""
    parser = argparse.ArgumentParser(
        description="Generate absolute state/action DreamZero metadata for EgoDex"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--quantile-sample-limit", type=int, default=2_000_000)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    roots = discover_task_roots(args.data_root)
    paths: list[Path] = []
    total_frames = 0
    for root in roots:
        root_paths, root_frames = episode_paths(root)
        paths.extend(root_paths)
        total_frames += root_frames
    if args.max_files is not None:
        paths = paths[: args.max_files]
        total_frames = sum(pq.read_metadata(path).num_rows for path in paths)

    state_stats, action_stats, sample_count = compute_statistics(
        paths,
        total_frames=total_frames,
        quantile_sample_limit=args.quantile_sample_limit,
    )
    metadata = build_metadata(state_stats, action_stats)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(
        json.dumps(metadata, indent=4) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {args.output_metadata}: roots={len(roots)} files={len(paths)} "
        f"frames={total_frames} quantile_samples={sample_count}"
    )


if __name__ == "__main__":
    main()
