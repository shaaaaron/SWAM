#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate DreamZero ``metadata.json`` from a local LeRobot v2 dataset.

The output format matches ``model_path/experiment_cfg/metadata.json`` used by
DreamZero training and inference.

Examples:

  # Libero (simple state/actions columns)
  python toolkits/lerobot/generate_dreamzero_metadata.py \
    --preset libero_sim \
    --dataset-root /path/libero \
    --output-metadata /path/to/output/metadata.json

  # DROID / OXE (split state/action via meta/modality.json)
  python toolkits/lerobot/generate_dreamzero_metadata.py \
    --preset oxe_droid \
    --dataset-root /path/droid \
    --output-metadata /path/to/output/metadata.json \
    --merge

  # Both presets in one metadata.json
  python toolkits/lerobot/generate_dreamzero_metadata.py \
    --preset libero_sim oxe_droid \
    --dataset-root /path/libero /path/droid \
    --output-metadata /path/to/output/metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlinf.data.datasets.dreamzero.noe_robotics_contract import (  # noqa: E402
    validate_noe_robotics_modality,
)
from rlinf.data.datasets.dreamzero.relative_actions import (  # noqa: E402
    RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_POSE_REFERENCE_PER_ARM,
    ROBOTWIN_EEF_ACTION_KEY,
    ROBOTWIN_EEF_DIM,
    ROBOTWIN_EEF_STATE_KEY,
    ACTION_QUATERNION_MODE_PRESERVE,
    ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS,
    ensure_robotwin_eef_dim,
    resolve_action_quaternion_mode,
    resolve_relative_action_pose_reference,
    robotwin_eef_abs_to_rel_wxyz,
    temporal_continuous_robotwin_eef_action_wxyz,
)

_NOE_UMI_16D_TAG = "noe_umi_16d"
_NOE_ROBOTICS_16D_TAG = "noe_robotics_16d"
_ABSOLUTE_PACKED_EEF_16D_TAGS = {
    _NOE_UMI_16D_TAG,
    _NOE_ROBOTICS_16D_TAG,
}
_PARENT_DATASET_EMBODIMENT_TAGS = {"robotwin", _NOE_UMI_16D_TAG}
_NOE_UMI_QUALIFICATION_FIELDS = (
    "slam_validity_all",
    "width_detection",
    "width_calibration",
)
_NOE_UMI_EEF_MODALITY_CFG = {
    "state": {"eef_position": {"start": 0, "end": ROBOTWIN_EEF_DIM}},
    "action": {"eef_position": {"start": 0, "end": ROBOTWIN_EEF_DIM}},
}

# Preset registry: (embodiment_tag, state_key, action_key, video_keys, use_modality_json)
PRESETS: dict[str, dict[str, Any]] = {
    "libero_sim": {
        "embodiment_tag": "libero_sim",
        "state_key": "state",
        "action_key": "actions",
        "video_keys": ["image", "wrist_image"],
        "use_modality_json": False,
        "default_dataset_root": None,
    },
    "oxe_droid": {
        "embodiment_tag": "oxe_droid",
        "state_key": "observation.state",
        "action_key": "action",
        "video_keys": None,
        "use_modality_json": True,
        "default_dataset_root": None,
    },
    "real_panda_single_arm": {
        "embodiment_tag": "real_panda_single_arm",
        "state_key": "state",
        "action_key": "actions",
        "video_keys": [
            "observation.images.image",
            "observation.images.wrist_image",
        ],
        "use_modality_json": False,
        "default_dataset_root": None,
    },
    "noe_umi_16d": {
        "embodiment_tag": "noe_umi_16d",
        "state_key": "observation.state",
        "action_key": "action",
        "video_keys": [
            "observation.images.image",
            "observation.images.wrist_image_1",
            "observation.images.wrist_image_2",
        ],
        "use_modality_json": False,
        "default_dataset_root": None,
    },
    "noe_robotics_16d": {
        "embodiment_tag": "noe_robotics_16d",
        "state_key": "observation.state",
        "action_key": "action",
        "video_keys": [
            "observation.images.image",
            "observation.images.wrist_image_1",
            "observation.images.wrist_image_2",
        ],
        "use_modality_json": False,
        "streaming_stats": True,
        "default_dataset_root": None,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_episodes(
    meta_dir: Path,
    *,
    filter_noe_umi_qualified_episodes: bool = False,
    episode_exclusion_manifest: Path | None = None,
) -> list[int]:
    episodes_path = meta_dir / "episodes.jsonl"
    episodes: list[int] = []
    known_episodes: set[int] = set()
    filtered_count = 0
    invalid_count = 0
    invalid_examples: list[dict[str, Any]] = []
    with open(episodes_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            episode_index = int(record["episode_index"])
            known_episodes.add(episode_index)
            if filter_noe_umi_qualified_episodes:
                missing = [
                    field
                    for field in _NOE_UMI_QUALIFICATION_FIELDS
                    if field not in record
                ]
                invalid_types = {
                    field: type(record[field]).__name__
                    for field in _NOE_UMI_QUALIFICATION_FIELDS
                    if field in record and type(record[field]) is not bool
                }
                if missing or invalid_types:
                    invalid_count += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append(
                            {
                                "episode_index": episode_index,
                                "missing": missing,
                                "invalid_types": invalid_types,
                            }
                        )
                    continue
                if not all(
                    record[field] is True for field in _NOE_UMI_QUALIFICATION_FIELDS
                ):
                    filtered_count += 1
                    continue
            episodes.append(episode_index)
    if filter_noe_umi_qualified_episodes:
        print(
            "[noe_umi_16d] qualification filter "
            f"kept={len(episodes)} filtered={filtered_count} "
            f"invalid={invalid_count} path={episodes_path}",
            flush=True,
        )
        if invalid_examples:
            print(
                f"[noe_umi_16d] qualification invalid examples={invalid_examples}",
                flush=True,
            )
    if episode_exclusion_manifest is not None:
        manifest_path = meta_dir.parent / episode_exclusion_manifest
        excluded: set[int] = set()
        with manifest_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                episode_index = json.loads(line)["episode_index"]
                if type(episode_index) is not int or episode_index < 0:
                    raise ValueError(
                        "Exclusion episode_index must be a non-negative integer."
                    )
                excluded.add(episode_index)
        unknown = excluded - known_episodes
        if unknown:
            raise ValueError(
                f"Unknown excluded episode indices in {manifest_path}: "
                f"{sorted(unknown)[:5]}."
            )
        before = len(episodes)
        episodes = [index for index in episodes if index not in excluded]
        print(
            f"[episode exclusions] source={len(known_episodes)} "
            f"manifest_entries={len(excluded)} filtered={before - len(episodes)} "
            f"kept={len(episodes)} path={manifest_path}",
            flush=True,
        )
    if not episodes:
        raise ValueError(f"No episodes found in {episodes_path}")
    return episodes


def _resolve_metadata_dataset_roots(
    dataset_root: Path,
    *,
    embodiment_tag: str,
) -> list[Path]:
    """Resolve one LeRobot root or an allowed homogeneous parent."""
    if (dataset_root / "meta" / "info.json").is_file():
        return [dataset_root]
    if embodiment_tag not in _PARENT_DATASET_EMBODIMENT_TAGS:
        raise FileNotFoundError(
            f"Dataset root is missing meta/info.json: {dataset_root}"
        )
    roots = sorted(
        child
        for child in dataset_root.iterdir()
        if child.is_dir() and (child / "meta" / "info.json").is_file()
    )
    if not roots:
        raise FileNotFoundError(
            f"Dataset root is missing meta/info.json and has no LeRobot child "
            f"datasets: {dataset_root}"
        )
    print(
        f"[{embodiment_tag}] expanded parent {dataset_root} to "
        f"{len(roots)} datasets",
        flush=True,
    )
    return roots


def _episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size") or 1000)
    data_tmpl = info.get("data_path")
    if not data_tmpl:
        raise KeyError("meta/info.json missing data_path")
    rel = data_tmpl.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    return root / rel


def _resolve_column(schema_names: list[str], preferred: str, aliases: list[str]) -> str:
    for name in [preferred, *aliases]:
        if name in schema_names:
            return name
    raise KeyError(
        f"Could not find column {preferred!r}; tried {[preferred, *aliases]} "
        f"in schema {schema_names}"
    )


def _column_to_2d_array(table: Any, column_name: str) -> np.ndarray:
    values = table.column(column_name).to_pylist()
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(
            f"{column_name} must be a 1D/2D vector column, got {arr.shape}"
        )
    return arr


def _compute_stats(
    arr: np.ndarray, *, quantile_sample_limit: int | None = None
) -> dict[str, list[float]]:
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D array, got {arr.shape}")
    arr64 = arr.astype(np.float64, copy=False)
    quantile_arr = arr64
    if quantile_sample_limit is not None:
        limit = int(quantile_sample_limit)
        if limit <= 0:
            raise ValueError("quantile_sample_limit must be positive")
        sample_stride = max(1, (len(arr64) + limit - 1) // limit)
        quantile_arr = arr64[::sample_stride]
    return {
        "min": np.min(arr64, axis=0).tolist(),
        "max": np.max(arr64, axis=0).tolist(),
        "mean": np.mean(arr64, axis=0).tolist(),
        "std": np.std(arr64, axis=0).tolist(),
        "q01": np.quantile(quantile_arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(quantile_arr, 0.99, axis=0).tolist(),
    }


class _StreamingStatsAccumulator:
    """Streaming moments plus a bounded deterministic quantile reservoir."""

    def __init__(self, *, quantile_sample_limit: int, seed: int = 42) -> None:
        limit = int(quantile_sample_limit)
        if limit <= 0:
            raise ValueError("quantile_sample_limit must be positive")
        self._limit = limit
        self._rng = np.random.default_rng(int(seed))
        self._count = 0
        self._dim: int | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._reservoir: np.ndarray | None = None
        self._priorities: np.ndarray | None = None

    @property
    def count(self) -> int:
        return self._count

    @property
    def feature_dim(self) -> int:
        if self._dim is None:
            raise ValueError("Streaming stats accumulator has no observations")
        return self._dim

    @property
    def quantile_sample_rows(self) -> int:
        return 0 if self._reservoir is None else len(self._reservoir)

    def update(self, chunk: np.ndarray) -> None:
        values = np.asarray(chunk)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(f"Expected non-empty 2D chunk, got {values.shape}")
        values = values.astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("Streaming stats input contains non-finite values")
        rows, dim = values.shape
        if self._dim is None:
            self._dim = int(dim)
            self._count = int(rows)
            self._min = values.min(axis=0)
            self._max = values.max(axis=0)
            self._mean = values.mean(axis=0)
            centered = values - self._mean
            self._m2 = np.sum(centered * centered, axis=0)
        else:
            if dim != self._dim:
                raise ValueError(
                    "Streaming stats feature dimension changed: "
                    f"expected={self._dim}, got={dim}"
                )
            assert self._min is not None and self._max is not None
            assert self._mean is not None and self._m2 is not None
            chunk_mean = values.mean(axis=0)
            centered = values - chunk_mean
            chunk_m2 = np.sum(centered * centered, axis=0)
            total = self._count + int(rows)
            delta = chunk_mean - self._mean
            self._mean = self._mean + delta * (rows / total)
            self._m2 = self._m2 + chunk_m2 + delta * delta * (self._count * rows / total)
            self._min = np.minimum(self._min, values.min(axis=0))
            self._max = np.maximum(self._max, values.max(axis=0))
            self._count = total

        priorities = self._rng.random(rows)
        if self._reservoir is None:
            reservoir, kept = values, priorities
        else:
            reservoir = np.concatenate([self._reservoir, values], axis=0)
            kept = np.concatenate([self._priorities, priorities], axis=0)
        if len(reservoir) > self._limit:
            indices = np.argpartition(kept, len(kept) - self._limit)[-self._limit:]
            reservoir, kept = reservoir[indices], kept[indices]
        self._reservoir = reservoir.copy()
        self._priorities = kept.copy()

    def finalize(self) -> dict[str, list[float]]:
        if self._count == 0 or self._reservoir is None:
            raise ValueError("Streaming stats accumulator has no observations")
        assert self._min is not None and self._max is not None
        assert self._mean is not None and self._m2 is not None
        variance = np.maximum(self._m2 / self._count, 0.0)
        result = {
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "mean": self._mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "q01": np.quantile(self._reservoir, 0.01, axis=0).tolist(),
            "q99": np.quantile(self._reservoir, 0.99, axis=0).tolist(),
        }
        if any(not np.isfinite(np.asarray(value)).all() for value in result.values()):
            raise ValueError("Streaming stats result contains non-finite values")
        return result

def _slice_stats(
    full_stats: dict[str, list[float]], start: int, end: int
) -> dict[str, list[float]]:
    return {key: values[start:end] for key, values in full_stats.items()}


def _feature_resolution(feature: dict[str, Any]) -> list[int]:
    shape = feature.get("shape") or []
    names = feature.get("names") or []
    if len(shape) < 2:
        raise ValueError(f"Cannot infer resolution from feature shape {shape}")
    # LeRobot 的 shape 是数组维度 [H, W, C]；DreamZero metadata 的 resolution 约定是 [W, H]。
    if "height" in names and "width" in names:
        height = int(shape[names.index("height")])
        width = int(shape[names.index("width")])
        return [width, height]
    return [int(shape[1]), int(shape[0])]


def _video_metadata_from_feature(
    info: dict[str, Any],
    feature_key: str,
    *,
    short_name: str | None = None,
    resolution: list[int] | None = None,
) -> dict[str, Any]:
    feature = (info.get("features") or {}).get(feature_key, {})
    video_info = feature.get("video_info") or feature.get("info") or {}
    fps = float(video_info.get("video.fps", info.get("fps", 10)))
    names = feature.get("names") or []
    shape = feature.get("shape") or []
    if len(shape) != len(names):
        raise ValueError(
            f"Video feature {feature_key!r} has mismatched shape and names: "
            f"shape={shape}, names={names}"
        )
    if "channel" in names:
        channel_name = "channel"
    elif "channels" in names:
        channel_name = "channels"
    else:
        raise ValueError(
            f"Video feature {feature_key!r} must declare 'channel' or 'channels': "
            f"shape={shape}, names={names}"
        )
    channels = int(shape[names.index(channel_name)])
    return {
        "resolution": resolution if resolution is not None else _feature_resolution(feature),
        "channels": channels,
        "fps": fps,
    }


def _shape_dim(feature: dict[str, Any], dim_name: str) -> int | None:
    names = feature.get("names")
    shape = feature.get("shape")
    if not names or not shape or dim_name not in names:
        return None
    return int(shape[names.index(dim_name)])


def _load_modality_json(meta_dir: Path) -> dict[str, Any]:
    modality_path = meta_dir / "modality.json"
    if not modality_path.exists():
        raise FileNotFoundError(
            f"modality.json not found at {modality_path}; required for split state/action"
        )
    return _load_json(modality_path)


def _build_split_modalities(
    modality_cfg: dict[str, Any],
    *,
    kind: str,
    full_dim: int,
) -> dict[str, dict[str, Any]]:
    entries = modality_cfg.get(kind) or {}
    result: dict[str, dict[str, Any]] = {}
    for subkey, spec in entries.items():
        start = int(spec["start"])
        end = int(spec["end"])
        result[subkey] = {
            "absolute": bool(spec.get("absolute", True)),
            "rotation_type": None,
            "shape": [end - start],
            "continuous": True,
        }
    covered = sum(v["shape"][0] for v in result.values())
    if covered != full_dim:
        raise ValueError(
            f"{kind} modality slices cover {covered} dims but column has {full_dim}"
        )
    return result


def _build_split_statistics(
    full_stats: dict[str, list[float]],
    modality_cfg: dict[str, Any],
    *,
    kind: str,
) -> dict[str, dict[str, list[float]]]:
    entries = modality_cfg.get(kind) or {}
    return {
        subkey: _slice_stats(full_stats, int(spec["start"]), int(spec["end"]))
        for subkey, spec in entries.items()
    }


def _short_relative_action_keys(keys: list[str] | None) -> set[str]:
    """把 relative action key 统一成短 key；例如 action.eef_position -> eef_position。"""
    return {str(key).split(".", 1)[-1] for key in (keys or [])}


def _modality_slice(
    modality_cfg: dict[str, Any],
    *,
    kind: str,
    subkey: str,
) -> slice:
    """从 meta/modality.json 里读取某个 state/action 子字段对应的切片范围。"""
    entries = modality_cfg.get(kind) or {}
    if subkey not in entries:
        raise KeyError(f"meta/modality.json missing {kind}.{subkey}")
    spec = entries[subkey]
    return slice(int(spec["start"]), int(spec["end"]))


def _iter_robotwin_eef_relative_samples(
    state: np.ndarray,
    action: np.ndarray,
    modality_cfg: dict[str, Any],
    *,
    relative_action_mode: str,
    relative_action_pose_reference: str = RELATIVE_ACTION_POSE_REFERENCE_PER_ARM,
    action_horizon: int,
    max_chunk_size: int,
    anchor_stride: int = 1,
    episode_index: int,
    batch_size: int = 256,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """按训练 sample 的 s0 分批生成统计样本，限制长 episode 的中间内存。"""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    state_sl = _modality_slice(modality_cfg, kind="state", subkey="eef_position")
    action_sl = _modality_slice(modality_cfg, kind="action", subkey="eef_position")
    state_eef = state[:, state_sl]
    action_eef = action[:, action_sl]
    ensure_robotwin_eef_dim("state.eef_position", state_eef)
    ensure_robotwin_eef_dim("action.eef_position", action_eef)
    horizon = int(action_horizon)
    if horizon <= 0:
        raise ValueError(
            "--action-horizon must be positive for RoboTwin relative stats."
        )
    chunks = int(max_chunk_size)
    if chunks <= 0:
        raise ValueError(
            "--max-chunk-size must be positive for RoboTwin relative stats."
        )
    anchor_step = int(anchor_stride)
    if anchor_step <= 0:
        raise ValueError(
            "--anchor-stride must be positive for RoboTwin relative stats."
        )
    # Metadata stats mirror the default DreamZero macro layout: each state anchor is
    # one action horizon apart. The dataset sampler may expose more knobs, but stats
    # generation intentionally keeps this fixed.
    stride = horizon
    if len(state_eef) != len(action_eef):
        raise ValueError(
            "RoboTwin EEF relative stats require state/action episode lengths to "
            f"match; episode_index={episode_index}, state_len={len(state_eef)}, "
            f"action_len={len(action_eef)}."
        )
    max_len = min(len(state_eef), len(action_eef))
    usable = max_len - horizon + 1
    if usable <= 0:
        warnings.warn(
            "RoboTwin EEF s0 relative stats require each episode to contain at "
            "least one complete action horizon; "
            f"episode_index={episode_index}, "
            f"episode_len={len(action_eef)}, action_horizon={horizon}, "
            f"max_chunk_size={chunks}. "
            "Skipping this episode for relative stats.",
            stacklevel=2,
        )
        return

    state_chunks = []
    action_chunks = []
    for anchor_idx in range(0, usable, anchor_step):
        ref = state_eef[anchor_idx : anchor_idx + 1]
        state_indices = []
        action_indices = []
        for chunk_idx in range(chunks):
            chunk_anchor = anchor_idx + chunk_idx * stride
            if chunk_anchor + horizon > max_len:
                break
            state_indices.append(chunk_anchor)
            action_indices.extend(range(chunk_anchor, chunk_anchor + horizon))
        if not state_indices or not action_indices:
            continue
        state_indices = np.asarray(state_indices, dtype=np.int64)
        action_indices = np.asarray(action_indices, dtype=np.int64)
        absolute_state = state_eef[state_indices]
        if relative_action_mode == RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ:
            state_anchor = absolute_state
            chunk_actions = action_eef[action_indices].reshape(
                len(state_indices), horizon, ROBOTWIN_EEF_DIM
            )
            relative_action = robotwin_eef_abs_to_rel_wxyz(
                chunk_actions,
                absolute_state[:, None, :],
                relative_action_pose_reference,
            ).reshape(-1, ROBOTWIN_EEF_DIM)
        else:
            state_anchor = ref
            relative_action = robotwin_eef_abs_to_rel_wxyz(
                action_eef[action_indices],
                ref,
                relative_action_pose_reference,
            )
        state_chunks.append(
            robotwin_eef_abs_to_rel_wxyz(
                absolute_state,
                state_anchor,
                relative_action_pose_reference,
            )
        )
        action_chunks.append(relative_action)
        if len(state_chunks) >= batch_size:
            yield np.concatenate(state_chunks, axis=0), np.concatenate(action_chunks, axis=0)
            state_chunks.clear()
            action_chunks.clear()
    if state_chunks:
        yield np.concatenate(state_chunks, axis=0), np.concatenate(action_chunks, axis=0)


def _robotwin_eef_relative_samples(
    state: np.ndarray,
    action: np.ndarray,
    modality_cfg: dict[str, Any],
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """保留原有物化接口；流式统计直接消费同一采样迭代器。"""
    batches = list(_iter_robotwin_eef_relative_samples(state, action, modality_cfg, **kwargs))
    if not batches:
        empty = np.empty((0, ROBOTWIN_EEF_DIM), dtype=np.float32)
        return empty, empty
    states, actions = zip(*batches)
    return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)


def _prepare_relative_stats(
    *,
    embodiment_tag: str,
    use_modality_json: bool,
    modality_cfg: dict[str, Any],
    relative_action_mode: str | None,
    relative_action_pose_reference: str | None,
    relative_action_keys: list[str] | None,
    action_horizon: int | None,
    max_chunk_size: int | None,
) -> tuple[bool, int | None, int | None]:
    """校验 relative stats 配置；返回是否需要收集 relative stats 以及采样超参。"""
    mode = "" if relative_action_mode is None else str(relative_action_mode)
    pose_reference = resolve_relative_action_pose_reference(
        relative_action_pose_reference,
        relative_action_mode=mode if mode not in ("", "None") else None,
    )
    keys = _short_relative_action_keys(relative_action_keys)
    if mode in ("", "None"):
        if keys or action_horizon is not None or max_chunk_size is not None:
            raise ValueError(
                "Relative stats require --relative-action-mode "
                "robotwin_eef_se3_s0_wxyz."
            )
        return False, None, None
    mode_to_embodiment_tag = {
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ: "robotwin",
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ: "robotwin",
        RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ: _NOE_UMI_16D_TAG,
        RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ: _NOE_ROBOTICS_16D_TAG,
    }
    if mode not in mode_to_embodiment_tag:
        raise ValueError(
            f"Unsupported relative_action_mode {mode!r}; "
            f"supported modes: {tuple(mode_to_embodiment_tag)}."
        )
    expected_embodiment_tag = mode_to_embodiment_tag[mode]
    if embodiment_tag != expected_embodiment_tag:
        raise ValueError(
            f"relative_action_mode={mode!r} requires "
            f"--embodiment-tag {expected_embodiment_tag}, got {embodiment_tag!r}."
        )
    if (
        pose_reference != "per_arm"
        and expected_embodiment_tag != "robotwin"
    ):
        raise ValueError(
            "--relative-action-pose-reference left_hand currently only "
            "supports RoboTwin relative stats."
        )
    if mode in (
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
    ) and not use_modality_json:
        raise ValueError(
            "relative_action_mode='robotwin_eef_se3_s0_wxyz' requires --use-modality-json."
        )
    if expected_embodiment_tag in _ABSOLUTE_PACKED_EEF_16D_TAGS and use_modality_json:
        raise ValueError(
            f"relative_action_mode={mode!r} uses the fixed "
            "first-16D EEF layout and does not accept --use-modality-json."
        )
    if not keys:
        raise ValueError(
            f"relative_action_mode={mode!r} requires explicit "
            "--relative-action-keys eef_position."
        )
    if keys != {"eef_position"}:
        raise ValueError(
            f"relative_action_mode={mode!r} only supports "
            "--relative-action-keys eef_position."
        )
    if mode in (
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
    ):
        _modality_slice(modality_cfg, kind="state", subkey="eef_position")
        _modality_slice(modality_cfg, kind="action", subkey="eef_position")
    if action_horizon is None:
        raise ValueError(
            f"--action-horizon is required for relative_action_mode={mode!r}."
        )
    if max_chunk_size is None:
        raise ValueError(
            f"--max-chunk-size is required for relative_action_mode={mode!r}."
        )
    return True, int(action_horizon), int(max_chunk_size)


def _video_modalities_from_modality(
    info: dict[str, Any],
    modality_cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    video_cfg = modality_cfg.get("video") or {}
    result: dict[str, dict[str, Any]] = {}
    for short_name, spec in video_cfg.items():
        original_key = spec.get("original_key", short_name)
        result[short_name] = _video_metadata_from_feature(info, original_key)
    return result


def _infer_video_keys(info: dict[str, Any]) -> list[str]:
    """Pick LeRobot feature keys for video modalities from meta/info.json."""
    features = info.get("features") or {}
    video_keys = [key for key, feat in features.items() if feat.get("dtype") == "video"]
    if video_keys:
        return sorted(video_keys)
    for key in ("image", "wrist_image"):
        if key in features and features[key].get("dtype") == "image":
            video_keys.append(key)
    if video_keys:
        return video_keys
    raise ValueError(
        "Could not infer video_keys from meta/info.json; pass --video-keys explicitly"
    )


def _video_modalities_simple(
    info: dict[str, Any],
    video_keys: list[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for video_key in video_keys:
        short_name = video_key.removeprefix("observation.images.")
        result[short_name] = _video_metadata_from_feature(info, video_key)
    return result


def _noe_umi_16d_video_modalities(
    info: dict[str, Any], video_keys: list[str], video_resolution: list[int] | None = None
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for video_key in video_keys:
        short_name = video_key.removeprefix("observation.images.")
        metadata = _video_metadata_from_feature(info, video_key, resolution=video_resolution)
        result[short_name] = metadata
    return result


def _validate_noe_umi_episode_relative_options(
    *,
    pose_reference: str | None,
    quaternion_mode: str,
    anchor_stride: int,
    relative_action_keys: list[str] | None,
) -> None:
    if pose_reference not in (None, "per_arm"):
        raise ValueError("UMI episode-relative only supports per_arm")
    if quaternion_mode != ACTION_QUATERNION_MODE_PRESERVE:
        raise ValueError("UMI episode-relative uses existing preserve quaternion semantics")
    if anchor_stride != 1:
        raise ValueError("Episode reference is fixed; anchor_stride does not apply")
    if _short_relative_action_keys(relative_action_keys) != {"eef_position"}:
        raise ValueError("Episode-relative requires relative_action_keys=['eef_position']")


def build_metadata(
    dataset_root: Path,
    *,
    embodiment_tag: str,
    state_key: str,
    action_key: str,
    video_keys: list[str] | None,
    use_modality_json: bool,
    max_episodes: int | None,
    quantile_sample_limit: int | None = None,
    streaming_stats: bool = False,
    stats_seed: int = 42,
    episode_stride: int = 1,
    skip_missing_episodes: bool = False,
    episode_exclusion_manifest: Path | None = None,
    relative_action_mode: str | None = None,
    video_resolution: list[int] | None = None,
    relative_action_pose_reference: str | None = None,
    relative_action_keys: list[str] | None = None,
    action_horizon: int | None = None,
    max_chunk_size: int | None = None,
    anchor_stride: int = 1,
    filter_noe_umi_qualified_episodes: bool = False,
    action_quaternion_mode: str = ACTION_QUATERNION_MODE_PRESERVE,
    apply_noe_umi_frame_quality: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if relative_action_mode == RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ:
        if embodiment_tag != _NOE_UMI_16D_TAG or use_modality_json:
            raise ValueError("UMI episode-relative metadata requires the fixed noe_umi_16d contract")
        _validate_noe_umi_episode_relative_options(
            pose_reference=relative_action_pose_reference,
            quaternion_mode=action_quaternion_mode,
            anchor_stride=anchor_stride,
            relative_action_keys=relative_action_keys,
        )
        from toolkits.lerobot.noe_umi_episode_metadata import build_noe_umi_episode_metadata

        return build_noe_umi_episode_metadata(
            _resolve_metadata_dataset_roots(dataset_root, embodiment_tag=embodiment_tag),
            episode_stride=episode_stride, max_episodes=max_episodes,
            quantile_sample_limit=100000 if quantile_sample_limit is None else quantile_sample_limit, stats_seed=stats_seed,
            apply_frame_quality=apply_noe_umi_frame_quality, state_key=state_key, action_key=action_key,
            video_keys=video_keys, video_resolution=video_resolution,
            episode_exclusion_manifest=episode_exclusion_manifest,
            filter_qualified_episodes=filter_noe_umi_qualified_episodes,
            skip_missing_episodes=skip_missing_episodes,
        )
    if apply_noe_umi_frame_quality:
        raise ValueError("--apply-noe-umi-frame-quality is supported by UMI episode-relative metadata")
    if episode_stride <= 0:
        raise ValueError("episode_stride must be positive")
    if video_resolution is not None:
        if embodiment_tag not in _ABSOLUTE_PACKED_EEF_16D_TAGS or use_modality_json:
            raise ValueError("video_resolution requires a packed NOE 16D preset")
        if len(video_resolution) != 2 or any(type(value) is not int or value <= 0 for value in video_resolution):
            raise ValueError("video_resolution must contain positive integer width and height")
    if streaming_stats and (quantile_sample_limit is None or int(quantile_sample_limit) <= 0):
        raise ValueError("quantile_sample_limit must be positive when streaming_stats=True")
    if filter_noe_umi_qualified_episodes and embodiment_tag != _NOE_UMI_16D_TAG:
        raise ValueError(
            "--filter-noe-umi-qualified-episodes requires "
            f"embodiment_tag={_NOE_UMI_16D_TAG!r}."
        )
    action_quaternion_mode = resolve_action_quaternion_mode(action_quaternion_mode)
    if (
        action_quaternion_mode == ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS
        and relative_action_mode not in (None, "", "None")
    ):
        raise ValueError(
            "action_quaternion_mode='temporal_continuous' currently only "
            "supports absolute action metadata; generate relative metadata separately."
        )
    if action_quaternion_mode == ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS:
        if action_horizon is None or int(action_horizon) <= 0:
            raise ValueError(
                "action_horizon is required for action_quaternion_mode="
                "'temporal_continuous'."
            )
    dataset_roots = _resolve_metadata_dataset_roots(
        dataset_root,
        embodiment_tag=embodiment_tag,
    )
    relative_action_pose_reference = resolve_relative_action_pose_reference(
        relative_action_pose_reference,
        relative_action_mode=(
            relative_action_mode
            if relative_action_mode not in (None, "", "None")
            else None
        ),
    )
    first_meta_dir = dataset_roots[0] / "meta"
    info = _load_json(first_meta_dir / "info.json")
    modality_cfg = _load_modality_json(first_meta_dir) if use_modality_json else {}
    (
        collect_relative_stats,
        relative_action_horizon,
        relative_max_chunk_size,
    ) = _prepare_relative_stats(
        embodiment_tag=embodiment_tag,
        use_modality_json=use_modality_json,
        modality_cfg=modality_cfg,
        relative_action_mode=relative_action_mode,
        relative_action_pose_reference=relative_action_pose_reference,
        relative_action_keys=relative_action_keys,
        action_horizon=(
            action_horizon
            if relative_action_mode not in (None, "", "None")
            else None
        ),
        max_chunk_size=(
            max_chunk_size
            if relative_action_mode not in (None, "", "None")
            else None
        ),
    )
    print(
        "[DreamZero metadata] relative_stats "
        f"enabled={collect_relative_stats}, "
        f"mode={relative_action_mode!r}, "
        f"pose_reference={relative_action_pose_reference!r}, "
        f"keys={relative_action_keys}, "
        f"action_horizon={relative_action_horizon}, "
        f"max_chunk_size={relative_max_chunk_size}",
        flush=True,
    )
    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    state_accumulator = (
        _StreamingStatsAccumulator(quantile_sample_limit=int(quantile_sample_limit), seed=stats_seed)
        if streaming_stats else None
    )
    action_accumulator = (
        _StreamingStatsAccumulator(quantile_sample_limit=int(quantile_sample_limit), seed=stats_seed)
        if streaming_stats else None
    )
    relative_state_eef_chunks: list[np.ndarray] = []
    relative_action_eef_chunks: list[np.ndarray] = []
    relative_state_accumulator = (
        _StreamingStatsAccumulator(quantile_sample_limit=int(quantile_sample_limit), seed=stats_seed)
        if streaming_stats and collect_relative_stats else None
    )
    relative_action_accumulator = (
        _StreamingStatsAccumulator(quantile_sample_limit=int(quantile_sample_limit), seed=stats_seed)
        if streaming_stats and collect_relative_stats else None
    )
    missing_episodes: list[tuple[int, Path]] = []

    relative_modality_cfg = modality_cfg
    if collect_relative_stats and embodiment_tag in _ABSOLUTE_PACKED_EEF_16D_TAGS:
        # UMI 和 Robotics 共享双臂 SE(3) 转换，源列都只取前 16D。
        relative_modality_cfg = _NOE_UMI_EEF_MODALITY_CFG

    for current_root in dataset_roots:
        meta_dir = current_root / "meta"
        current_info = _load_json(meta_dir / "info.json")
        if collect_relative_stats and embodiment_tag == _NOE_ROBOTICS_16D_TAG:
            validate_noe_robotics_modality(_load_modality_json(meta_dir))
        if current_root != dataset_roots[0]:
            reference_features = info.get("features") or {}
            current_features = current_info.get("features") or {}
            for feature_key in [state_key, action_key, *(video_keys or [])]:
                if (reference_features.get(feature_key) or {}).get("shape") != (
                    current_features.get(feature_key) or {}
                ).get("shape"):
                    raise ValueError(
                        f"Feature shape differs across dataset roots for {feature_key!r}: "
                        f"root={current_root}"
                    )
            if use_modality_json and _load_modality_json(meta_dir) != modality_cfg:
                raise ValueError(
                    f"meta/modality.json differs across dataset roots: {current_root}"
                )
        episodes = _load_episodes(
            meta_dir,
            filter_noe_umi_qualified_episodes=(filter_noe_umi_qualified_episodes),
            episode_exclusion_manifest=episode_exclusion_manifest,
        )
        source_episode_count = len(episodes)
        episodes = episodes[::episode_stride]
        if max_episodes is not None:
            episodes = episodes[: int(max_episodes)]
        print(
            f"[{embodiment_tag}] loading root={current_root} "
            f"episodes={len(episodes)}/{source_episode_count} "
            f"episode_stride={episode_stride}",
            flush=True,
        )
        for loaded_index, episode_index in enumerate(episodes, start=1):
            parquet_path = _episode_path(current_root, current_info, episode_index)
            if not parquet_path.is_file():
                if not skip_missing_episodes:
                    raise FileNotFoundError(
                        f"Episode {episode_index} is listed in "
                        f"{meta_dir / 'episodes.jsonl'} but parquet is missing: "
                        f"{parquet_path}"
                    )
                missing_episodes.append((episode_index, parquet_path))
                continue
            schema_names = list(pq.read_schema(parquet_path).names)
            resolved_state_key = _resolve_column(
                schema_names,
                state_key,
                aliases=["observation.state", "state"],
            )
            resolved_action_key = _resolve_column(
                schema_names,
                action_key,
                aliases=["action", "actions"],
            )
            table = pq.read_table(
                parquet_path,
                columns=[resolved_state_key, resolved_action_key],
            )
            state_episode = _column_to_2d_array(table, resolved_state_key)
            action_episode = _column_to_2d_array(table, resolved_action_key)
            if action_quaternion_mode == ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS:
                if action_episode.shape[-1] != ROBOTWIN_EEF_DIM:
                    raise ValueError(
                        "action_quaternion_mode='temporal_continuous' requires "
                        f"16D EEF actions, got shape={action_episode.shape}."
                    )
                if state_episode.shape[-1] < ROBOTWIN_EEF_DIM:
                    raise ValueError(
                        "action_quaternion_mode='temporal_continuous' requires "
                        f"at least 16D state anchors, got shape={state_episode.shape}."
                    )
                action_episode = temporal_continuous_robotwin_eef_action_wxyz(
                    action_episode,
                    state_episode[..., :ROBOTWIN_EEF_DIM],
                    action_horizon=int(action_horizon),
                )
            if streaming_stats:
                assert state_accumulator is not None and action_accumulator is not None
                state_accumulator.update(state_episode)
                action_accumulator.update(action_episode)
            else:
                state_chunks.append(state_episode)
                action_chunks.append(action_episode)
            if collect_relative_stats:
                assert relative_action_horizon is not None
                assert relative_max_chunk_size is not None
                for relative_state_episode, relative_action_episode in _iter_robotwin_eef_relative_samples(
                    state_episode,
                    action_episode,
                    relative_modality_cfg,
                    relative_action_mode=str(relative_action_mode),
                    relative_action_pose_reference=relative_action_pose_reference,
                    action_horizon=relative_action_horizon,
                    max_chunk_size=relative_max_chunk_size,
                    anchor_stride=anchor_stride,
                    episode_index=int(episode_index),
                ):
                    if streaming_stats:
                        assert relative_state_accumulator is not None
                        assert relative_action_accumulator is not None
                        relative_state_accumulator.update(relative_state_episode)
                        relative_action_accumulator.update(relative_action_episode)
                    else:
                        relative_state_eef_chunks.append(relative_state_episode)
                        relative_action_eef_chunks.append(relative_action_episode)
            if loaded_index % 500 == 0 or loaded_index == len(episodes):
                print(
                    f"[{embodiment_tag}] loaded root={current_root} "
                    f"episodes={loaded_index}/{len(episodes)}",
                    flush=True,
                )

    if missing_episodes:
        print(
            f"[{embodiment_tag}] skipped {len(missing_episodes)} missing episode parquet files:"
        )
        for episode_index, parquet_path in missing_episodes:
            print(
                f"[{embodiment_tag}]   episode_index={episode_index} path={parquet_path}"
            )
    if streaming_stats:
        assert state_accumulator is not None and action_accumulator is not None
        if state_accumulator.count == 0 or action_accumulator.count == 0:
            raise ValueError(f"No usable parquet episodes found under {dataset_root}")
        state_dim = state_accumulator.feature_dim
        action_dim = action_accumulator.feature_dim
        state = None
        action = None
        state_stats_full = state_accumulator.finalize()
        action_stats_full = action_accumulator.finalize()
        print(
            f"[streaming stats] state_rows={state_accumulator.count} "
            f"action_rows={action_accumulator.count}",
            flush=True,
        )
    else:
        if not state_chunks or not action_chunks:
            raise ValueError(f"No usable parquet episodes found under {dataset_root}")
        state = np.concatenate(state_chunks, axis=0)
        action = np.concatenate(action_chunks, axis=0)
        state_dim = int(state.shape[-1])
        action_dim = int(action.shape[-1])
        state_stats_full = _compute_stats(state, quantile_sample_limit=quantile_sample_limit)
        action_stats_full = _compute_stats(action, quantile_sample_limit=quantile_sample_limit)
    relative_state_eef_stats = None
    relative_action_eef_stats = None
    if collect_relative_stats and streaming_stats:
        assert relative_state_accumulator is not None
        assert relative_action_accumulator is not None
        relative_state_eef_stats = relative_state_accumulator.finalize()
        relative_action_eef_stats = relative_action_accumulator.finalize()
        print(
            f"[streaming relative stats] state_rows={relative_state_accumulator.count} "
            f"action_rows={relative_action_accumulator.count}",
            flush=True,
        )
    elif collect_relative_stats:
        if not relative_state_eef_chunks or not relative_action_eef_chunks:
            raise ValueError(
                "No RoboTwin EEF s0 relative state/action samples were generated; "
                "check --action-horizon, --max-chunk-size, and episode lengths."
            )
        relative_state_eef_stats = _compute_stats(
            np.concatenate(relative_state_eef_chunks, axis=0),
            quantile_sample_limit=quantile_sample_limit,
        )
        relative_action_eef_stats = _compute_stats(
            np.concatenate(relative_action_eef_chunks, axis=0),
            quantile_sample_limit=quantile_sample_limit,
        )

    if use_modality_json:
        state_stats = _build_split_statistics(
            state_stats_full, modality_cfg, kind="state"
        )
        action_stats = _build_split_statistics(
            action_stats_full, modality_cfg, kind="action"
        )
        state_modalities = _build_split_modalities(
            modality_cfg, kind="state", full_dim=state_dim
        )
        action_modalities = _build_split_modalities(
            modality_cfg, kind="action", full_dim=action_dim
        )
        video_modalities = _video_modalities_from_modality(info, modality_cfg)
        lerobot_stats = {
            state_key: state_stats_full,
            action_key: action_stats_full,
        }
        for subkey, spec in (modality_cfg.get("state") or {}).items():
            start, end = int(spec["start"]), int(spec["end"])
            lerobot_stats[f"state.{subkey}"] = _slice_stats(
                state_stats_full, start, end
            )
        for subkey, spec in (modality_cfg.get("action") or {}).items():
            start, end = int(spec["start"]), int(spec["end"])
            lerobot_stats[f"action.{subkey}"] = _slice_stats(
                action_stats_full, start, end
            )
        if relative_state_eef_stats is not None:
            state_stats["eef_position"] = relative_state_eef_stats
            lerobot_stats[ROBOTWIN_EEF_STATE_KEY] = relative_state_eef_stats
        if relative_action_eef_stats is not None:
            action_stats["eef_position"] = relative_action_eef_stats
            lerobot_stats[ROBOTWIN_EEF_ACTION_KEY] = relative_action_eef_stats
    else:
        if collect_relative_stats and embodiment_tag not in _ABSOLUTE_PACKED_EEF_16D_TAGS:
            raise ValueError(
                "RoboTwin EEF relative stats are only supported with --use-modality-json."
            )
        if not video_keys:
            video_keys = _infer_video_keys(info)
        if embodiment_tag in _ABSOLUTE_PACKED_EEF_16D_TAGS:
            if state_dim < 16 or action_dim < 16:
                raise ValueError(
                    "noe_umi_16d expects at least 16D state/action, "
                    f"got state={state_dim} action={action_dim}"
                )
            state_stats_16d = (
                relative_state_eef_stats
                if relative_state_eef_stats is not None
                else _slice_stats(state_stats_full, 0, ROBOTWIN_EEF_DIM)
            )
            action_stats_16d = (
                relative_action_eef_stats
                if relative_action_eef_stats is not None
                else _slice_stats(action_stats_full, 0, ROBOTWIN_EEF_DIM)
            )
            state_stats = {"eef_position": state_stats_16d}
            action_stats = {"eef_position": action_stats_16d}
            state_modalities = {
                "eef_position": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [16],
                    "continuous": True,
                }
            }
            action_modalities = {
                "eef_position": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [16],
                    "continuous": True,
                }
            }
            video_modalities = _noe_umi_16d_video_modalities(info, video_keys, video_resolution)
            lerobot_stats = {
                state_key: state_stats_full,
                action_key: action_stats_full,
                "state.eef_position": state_stats_16d,
                "action.eef_position": action_stats_16d,
            }
        else:
            state_stats = {"state": state_stats_full}
            action_stats = {"actions": action_stats_full}
            state_modalities = {
                "state": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [state_dim],
                    "continuous": True,
                }
            }
            action_modalities = {
                "actions": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [action_dim],
                    "continuous": True,
                }
            }
            video_modalities = _video_modalities_simple(info, video_keys)
            lerobot_stats = {
                "state": state_stats_full,
                "actions": action_stats_full,
                "state.state": state_stats_full,
                "action.actions": action_stats_full,
            }

    metadata_entry = {
        "statistics": {
            "state": state_stats,
            "action": action_stats,
        },
        "modalities": {
            "video": video_modalities,
            "state": state_modalities,
            "action": action_modalities,
        },
        "embodiment_tag": embodiment_tag,
        "action_quaternion_mode": action_quaternion_mode,
    }
    if collect_relative_stats:
        metadata_entry["relative_action_mode"] = str(relative_action_mode)
        metadata_entry["relative_action_pose_reference"] = (
            relative_action_pose_reference
        )
        metadata_entry["relative_action_stats"] = {
            "action_horizon": relative_action_horizon,
            "max_chunk_size": relative_max_chunk_size,
            "anchor_stride": anchor_stride,
            "episode_stride": episode_stride,
            "max_episodes": max_episodes,
            "streaming_stats": streaming_stats,
            "quantile_sample_limit": quantile_sample_limit,
            "stats_seed": stats_seed,
        }
    metadata = {
        embodiment_tag: {
            **metadata_entry,
        }
    }
    return metadata, lerobot_stats


def _resolve_preset(name: str) -> dict[str, Any]:
    if name not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}; known presets: {known}")
    return PRESETS[name].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DreamZero metadata.json from LeRobot datasets."
    )
    parser.add_argument(
        "--preset",
        nargs="+",
        default=None,
        help=f"Built-in presets: {', '.join(sorted(PRESETS))}",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        nargs="+",
        default=None,
        help="One root per preset (or single root for manual mode).",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("/mnt/project_rlinf/guozhen/test_wam/test_dataset/metadata.json"),
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        nargs="*",
        default=None,
        help="Optional stats.json paths (one per dataset). Defaults to DATASET_ROOT/meta/stats.json.",
    )
    parser.add_argument(
        "--merge", action="store_true", help="Merge into existing output file."
    )
    parser.add_argument("--embodiment-tag", default=None)
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--action-key", default=None)
    parser.add_argument("--video-keys", nargs="+", default=None)
    parser.add_argument(
        "--video-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
        help=("NOE 16D output video metadata resolution override for variable-size sources; "
              "requires per-view resize in the training transform."),
    )
    parser.add_argument(
        "--use-modality-json",
        action="store_true",
        help="Split state/action using meta/modality.json.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--episode-exclusion-manifest",
        type=Path,
        default=None,
        help=(
            "JSONL episode exclusions; relative paths resolve from each dataset root. "
            "Applied before episode-stride, max-episodes and parquet reads."
        ),
    )
    parser.add_argument(
        "--quantile-sample-limit",
        type=int,
        default=None,
        help=(
            "Optional upper bound for deterministic uniform q01/q99 samples. "
            "Min/max/mean/std still use all loaded rows."
        ),
    )
    parser.add_argument(
        "--streaming-stats",
        action="store_true",
        help="Compute exact moments without materializing all state/action rows.",
    )
    parser.add_argument("--stats-seed", type=int, default=42)
    parser.add_argument(
        "--episode-stride",
        type=int,
        default=1,
        help=("Keep every Nth selected episode in each dataset root (default: 1)."),
    )
    parser.add_argument(
        "--relative-action-mode",
        default=None,
        help=(
            "双臂 EEF 相对统计模式：robotwin_eef_se3_s0_wxyz、"
            "robotwin_eef_se3_chunk_wxyz、noe_umi_eef_se3_s0_wxyz 或 "
            "noe_robotics_eef_se3_s0_wxyz；省略则生成绝对统计。"
        ),
    )
    parser.add_argument(
        "--relative-action-keys",
        nargs="*",
        default=None,
        help="Action subkeys for relative stats, e.g. eef_position.",
    )
    parser.add_argument(
        "--relative-action-pose-reference",
        choices=("per_arm", "left_hand"),
        default="per_arm",
        help=(
            "Pose anchor for dual-arm SE(3) relative stats. left_hand makes "
            "both arm poses relative to the left-hand state anchor."
        ),
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Action horizon used to generate relative stats.",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=None,
        help="Max macro chunks per training sample used to generate s0 relative stats.",
    )
    parser.add_argument(
        "--anchor-stride",
        type=int,
        default=1,
        help="Frame stride between s0 anchors used for relative stats sampling.",
    )
    parser.add_argument(
        "--action-quaternion-mode",
        choices=(
            ACTION_QUATERNION_MODE_PRESERVE,
            ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS,
        ),
        default=ACTION_QUATERNION_MODE_PRESERVE,
        help="Absolute action 四元数符号处理方式；默认 preserve。",
    )
    parser.add_argument(
        "--skip-missing-episodes",
        action="store_true",
        help="Skip episodes listed in meta/episodes.jsonl whose parquet files are missing.",
    )
    parser.add_argument(
        "--filter-noe-umi-qualified-episodes",
        action="store_true",
        help=(
            "Keep only UMI episodes whose slam_validity_all, width_detection, "
            "and width_calibration fields are all true."
        ),
    )
    parser.add_argument(
        "--apply-noe-umi-frame-quality", action="store_true",
        help="Apply existing UMI frame quality and episode reference gate to episode-relative stats.",
    )
    parser.add_argument(
        "--dataset-roots-file", type=Path,
        help="Explicit UMI episode-relative dataset roots as a JSON list of directory paths.",
    )
    args = parser.parse_args()

    if args.relative_action_mode == RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ:
        if args.preset not in (None, ["noe_umi_16d"]):
            raise SystemExit("Episode-relative requires --preset noe_umi_16d")
        _validate_noe_umi_episode_relative_options(
            pose_reference=args.relative_action_pose_reference,
            quaternion_mode=args.action_quaternion_mode,
            anchor_stride=args.anchor_stride,
            relative_action_keys=args.relative_action_keys,
        )
        if not args.output_stats or len(args.output_stats) != 1:
            raise SystemExit("Episode-relative requires one explicit --output-stats path")
        from toolkits.lerobot.noe_umi_episode_metadata import build_noe_umi_episode_metadata

        if args.dataset_roots_file:
            manifest = _load_json(args.dataset_roots_file)
            specs = manifest["roots"] if isinstance(manifest, dict) else manifest
            roots = [Path(root) for root in specs]
        elif args.dataset_root:
            roots = [p for root in args.dataset_root for p in _resolve_metadata_dataset_roots(root, embodiment_tag=_NOE_UMI_16D_TAG)]
        else:
            raise SystemExit("Provide --dataset-roots-file or --dataset-root")
        metadata, stats = build_noe_umi_episode_metadata(
            roots, episode_stride=args.episode_stride, max_episodes=args.max_episodes,
            quantile_sample_limit=100000 if args.quantile_sample_limit is None else args.quantile_sample_limit, stats_seed=args.stats_seed,
            apply_frame_quality=args.apply_noe_umi_frame_quality,
            state_key=args.state_key or "observation.state", action_key=args.action_key or "action",
            video_keys=args.video_keys, video_resolution=args.video_resolution,
            episode_exclusion_manifest=args.episode_exclusion_manifest,
            filter_qualified_episodes=args.filter_noe_umi_qualified_episodes,
            skip_missing_episodes=args.skip_missing_episodes,
            audit_dir=args.output_stats[0].parent,
        )
        for path, obj in [(args.output_metadata, metadata), (args.output_stats[0], stats)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote UMI episode-relative metadata: {args.output_metadata}", flush=True)
        return
    if args.dataset_roots_file or args.apply_noe_umi_frame_quality:
        raise SystemExit("New UMI roots/quality options require the UMI episode-relative mode")

    if args.preset:
        if args.dataset_root and len(args.dataset_root) not in (1, len(args.preset)):
            raise SystemExit(
                "Provide either one --dataset-root for all presets, "
                "or one root per preset."
            )
        combined: dict[str, Any] = {}
        if args.merge and args.output_metadata.exists():
            combined = _load_json(args.output_metadata)

        for idx, preset_name in enumerate(args.preset):
            preset = _resolve_preset(preset_name)
            if args.dataset_root:
                root = (
                    args.dataset_root[idx]
                    if len(args.dataset_root) == len(args.preset)
                    else args.dataset_root[0]
                )
            else:
                root = Path(preset["default_dataset_root"])

            metadata, lerobot_stats = build_metadata(
                root,
                embodiment_tag=preset["embodiment_tag"],
                state_key=preset["state_key"],
                action_key=preset["action_key"],
                video_keys=preset.get("video_keys"),
                use_modality_json=bool(preset["use_modality_json"]),
                max_episodes=args.max_episodes,
                quantile_sample_limit=args.quantile_sample_limit,
                streaming_stats=bool(preset.get("streaming_stats", False) or args.streaming_stats),
                stats_seed=args.stats_seed,
                episode_stride=args.episode_stride,
                skip_missing_episodes=args.skip_missing_episodes,
                episode_exclusion_manifest=args.episode_exclusion_manifest,
                video_resolution=args.video_resolution,
                relative_action_mode=args.relative_action_mode,
                relative_action_pose_reference=(
                    args.relative_action_pose_reference
                ),
                relative_action_keys=args.relative_action_keys,
                action_horizon=args.action_horizon,
                max_chunk_size=args.max_chunk_size,
                anchor_stride=args.anchor_stride,
                filter_noe_umi_qualified_episodes=(
                    args.filter_noe_umi_qualified_episodes
                ),
                action_quaternion_mode=args.action_quaternion_mode,
            )
            combined.update(metadata)

            stats_path = None
            if args.output_stats:
                stats_path = (
                    args.output_stats[idx] if idx < len(args.output_stats) else None
                )
            stats_path = stats_path or (root / "meta" / "stats.json")
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(lerobot_stats, f, indent=2)
                f.write("\n")
            tag = preset["embodiment_tag"]
            print(f"[{tag}] dataset_root={root}")
            print(f"[{tag}] wrote stats {stats_path}")

        args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_metadata, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=4)
            f.write("\n")
        print(f"Wrote merged metadata: {args.output_metadata}")
        print(f"embodiment keys: {list(combined.keys())}")
        return

    if not args.dataset_root:
        raise SystemExit("--dataset-root is required without --preset")
    if args.embodiment_tag is None:
        raise SystemExit("--embodiment-tag is required without --preset")

    root = args.dataset_root[0]
    metadata, lerobot_stats = build_metadata(
        root,
        embodiment_tag=args.embodiment_tag,
        state_key=args.state_key or "state",
        action_key=args.action_key or "actions",
        video_keys=args.video_keys,
        use_modality_json=args.use_modality_json,
        max_episodes=args.max_episodes,
        quantile_sample_limit=args.quantile_sample_limit,
        streaming_stats=args.streaming_stats,
        stats_seed=args.stats_seed,
        episode_stride=args.episode_stride,
        skip_missing_episodes=args.skip_missing_episodes,
        episode_exclusion_manifest=args.episode_exclusion_manifest,
        video_resolution=args.video_resolution,
        relative_action_mode=args.relative_action_mode,
        relative_action_pose_reference=args.relative_action_pose_reference,
        relative_action_keys=args.relative_action_keys,
        action_horizon=args.action_horizon,
        max_chunk_size=args.max_chunk_size,
        anchor_stride=args.anchor_stride,
        filter_noe_umi_qualified_episodes=(args.filter_noe_umi_qualified_episodes),
        action_quaternion_mode=args.action_quaternion_mode,
    )

    output = args.output_metadata
    if args.merge and output.exists():
        existing = _load_json(output)
        existing.update(metadata)
        metadata = existing

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        f.write("\n")

    stats_path = (args.output_stats[0] if args.output_stats else None) or (
        root / "meta" / "stats.json"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(lerobot_stats, f, indent=2)
        f.write("\n")

    tag = args.embodiment_tag
    print(f"Wrote {output}")
    print(f"Wrote {stats_path}")
    print(f"embodiment_tag={tag}")


if __name__ == "__main__":
    main()
