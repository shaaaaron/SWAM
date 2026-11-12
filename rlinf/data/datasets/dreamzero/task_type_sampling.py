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

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

TASK_TYPE_VALUES = (
    "Basic Manipulation",
    "Cleaning",
    "Electromechanical Operations",
    "Fabric Handling",
    "Food Processing",
    "Gardening",
    "Handcrafting",
    "Organization",
    "Other/Unclear",
)
ACTION_TYPE_VALUES = (
    "assemble",
    "attach",
    "clean",
    "close",
    "connect/charge",
    "cook/prepare",
    "cut",
    "fold",
    "move",
    "needs_review",
    "open",
    "organize",
    "others",
    "package/serve",
    "pick up",
    "place",
    "pour",
    "press",
    "pull",
    "push",
    "remove/take out",
    "replace/refill",
    "rotate",
    "stack",
    "stir/mix",
    "thread/wear",
    "unfold",
    "write/mark",
)
_LABEL_VALUES = {   # 保留顺序
    "task_type": TASK_TYPE_VALUES,
    "action_type": ACTION_TYPE_VALUES,
}
_LABEL_VALUE_SETS = {   # 不保留顺序
    label_space: frozenset(values) for label_space, values in _LABEL_VALUES.items()
}
_DEFAULT_METADATA_FILES = {
    "task_type": "meta/task_type/task_type_v2.jsonl",
    "action_type": "meta/task_type/action_type.jsonl",
}


@dataclass(frozen=True)
class TaskTypeSamplingConfig:
    enabled: bool = False
    label_space: str = "task_type"
    metadata_file: str = _DEFAULT_METADATA_FILES["task_type"]
    weights: dict[str, float] = field(default_factory=dict)

    @property
    def label_values(self) -> tuple[str, ...]:
        return _LABEL_VALUES[self.label_space]

    @property
    def label_value_set(self) -> frozenset[str]:
        return _LABEL_VALUE_SETS[self.label_space]

    @classmethod
    def from_value(cls, value: Any) -> "TaskTypeSamplingConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, Mapping):  # 需要输入为字典
            raise TypeError(
                f"data.task_type_sampling must be a mapping, got {type(value)!r}."
            )

        enabled = bool(value.get("enabled", False))
        label_space = str(value.get("label_space", "task_type")).strip().lower()
        if label_space not in _LABEL_VALUES:
            raise ValueError(
                "Unknown label space for type sampling: "
                f"{label_space!r}. Legal values: {sorted(_LABEL_VALUES)}."
            )
        metadata_file = str(
            value.get("metadata_file", _DEFAULT_METADATA_FILES[label_space])
        ).strip()
        if not enabled:
            return cls(
                enabled=False,
                label_space=label_space,
                metadata_file=metadata_file,
            )
        if not metadata_file:
            raise ValueError(
                "Enabled task type sampling requires a non-empty metadata_file."
            )

        raw_weights = value.get("weights")
        if raw_weights is None:
            if label_space == "action_type":
                raise ValueError(
                    "Action type sampling requires explicit weights."
                )
            weights = dict.fromkeys(_LABEL_VALUES[label_space], 1.0)
        else:
            if not isinstance(raw_weights, Mapping):
                raise TypeError(
                    "data.task_type_sampling.weights must be a mapping, "
                    f"got {type(raw_weights)!r}."
                )
            weights = {
                str(label): float(weight)
                for label, weight in raw_weights.items()
            }

        label_value_set = _LABEL_VALUE_SETS[label_space]
        unknown = sorted(set(weights) - label_value_set)
        if unknown:
            raise ValueError(
                f"Unknown {label_space} sampling weight keys: "
                f"{unknown}. Legal values: {list(_LABEL_VALUES[label_space])}."
            )
        if label_space == "action_type":    # action type必须全部提供，task type允许静默
            missing = sorted(label_value_set - set(weights))
            if missing:
                raise ValueError(
                    "Action type sampling weights must list all action types; "
                    f"missing: {missing}."
                )
        invalid = {
            label: weight
            for label, weight in weights.items()
            if not math.isfinite(weight) or weight < 0.0
        }
        if invalid:
            raise ValueError(
                "Task type sampling weights must be finite and non-negative: "
                f"{invalid}."
            )
        if not weights:
            raise ValueError("Enabled task type sampling requires weights.")
        if not any(weight > 0.0 for weight in weights.values()):
            raise ValueError(
                "Enabled task type sampling requires at least one positive weight."
            )
        return cls(
            enabled=True,
            label_space=label_space,
            metadata_file=metadata_file,
            weights=weights,
        )


@dataclass(frozen=True)
class TaskTypeRanges:
    """
    表示某一个 task type 对应的所有 episode frame 区间

    episode A: global frame 0 ~ 99
    episode B: global frame 100 ~ 109

    starts = [0, 100]
    lengths = [100, 10]
    cumulative_frames = [100, 110]
    """
    starts: np.ndarray
    lengths: np.ndarray
    cumulative_frames: np.ndarray

    @property
    def episode_count(self) -> int:
        """
        包含多少episode
        """
        return int(self.starts.size)

    @property
    def frame_count(self) -> int:
        """
        包含多少frame
        """
        return int(self.cumulative_frames[-1])

    def sample_from_offsets(self, offsets: np.ndarray) -> np.ndarray:
        """
        将task type内部的offset 转换成 dataset 的全局 sample_idx
        """
        offsets = np.asarray(offsets, dtype=np.int64)
        range_ids = np.searchsorted(
            self.cumulative_frames,
            offsets,
            side="right",
        )
        previous = np.zeros_like(offsets)
        has_previous = range_ids > 0
        previous[has_previous] = self.cumulative_frames[range_ids[has_previous] - 1]
        return self.starts[range_ids] + offsets - previous


@dataclass(frozen=True)
class TaskTypeFrameIndex:
    ranges: dict[str, TaskTypeRanges]
    episode_count: int
    frame_count: int
    sidecar_rows: int
    extra_rows: int
    roots: tuple[str, ...]

    """
    TaskTypeFrameIndex
    └── ranges
        ├── "Cleaning"       → TaskTypeRanges
        ├── "Handcrafting"   → TaskTypeRanges
        ├── "Organization"   → TaskTypeRanges
        └── ...
    """
    @property
    def episode_counts(self) -> dict[str, int]:
        return {
            task_type: task_ranges.episode_count
            for task_type, task_ranges in self.ranges.items()
        }

    @property
    def frame_counts(self) -> dict[str, int]:
        return {
            task_type: task_ranges.frame_count
            for task_type, task_ranges in self.ranges.items()
        }


@dataclass(frozen=True)
class TaskTypeSamplingPlan:
    index: TaskTypeFrameIndex
    task_types: tuple[str, ...]
    probabilities: torch.Tensor
    effective_weights: dict[str, float]
    unavailable_types: tuple[str, ...]
    disabled_types: tuple[str, ...]
    """
    TaskTypeSamplingConfig
    用户希望怎么抽

    TaskTypeFrameIndex
        数据实际上有哪些类型和 frame

    TaskTypeSamplingPlan
        根据前两者生成的最终抽样方案
    """
    @classmethod
    def from_config(
        cls,
        index: TaskTypeFrameIndex,
        config: TaskTypeSamplingConfig,
    ) -> "TaskTypeSamplingPlan":
        if not config.enabled:
            raise ValueError("Cannot build a task type sampling plan when disabled.")

        missing_weights = sorted(set(index.ranges) - set(config.weights))   # 检查是否都设置了weight
        if missing_weights:
            raise ValueError(
                f"{config.label_space} sampling weights are missing labels used by "
                "the dataset: "
                f"{missing_weights}."
            )
        task_types = tuple(  # 只保留实际存在且权重大于零的 task type
            label
            for label in config.label_values
            if label in index.ranges and config.weights[label] > 0.0
        )
        if not task_types:
            raise ValueError(
                "Task type frame index contains no enabled available types."
            )
        raw_weights = torch.as_tensor(
            [config.weights[label] for label in task_types],
            dtype=torch.double,
        )
        probabilities = raw_weights / raw_weights.sum() # 进行归一化
        effective_weights = {   # 转化为方便显示的字典
            label: float(probability)
            for label, probability in zip(
                task_types,
                probabilities.tolist(),
            )
        }
        unavailable_types = tuple(  # 配置了正权重，但数据中不存在
            label
            for label, weight in config.weights.items()
            if weight > 0.0 and label not in index.ranges
        )
        disabled_types = tuple(  # 零权重表示显式禁用
            label for label, weight in config.weights.items() if weight == 0.0
        )
        return cls(
            index=index,
            task_types=task_types,
            probabilities=probabilities,
            effective_weights=effective_weights,
            unavailable_types=unavailable_types,
            disabled_types=disabled_types,
        )

    def sample_many(
        self,
        count: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        先为每个样本抽 task type，再在对应类型内部均匀抽 frame offset，转换成 dataset 全局 sample_idx
        """
        count = int(count)
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}.")
        if count == 0:
            return torch.empty((0,), dtype=torch.long)

        type_ids = torch.multinomial(
            self.probabilities,
            count,
            replacement=True,
            generator=generator,
        )
        sample_ids = torch.empty(count, dtype=torch.long)
        for type_id, label in enumerate(self.task_types):
            mask = type_ids == type_id
            type_count = int(mask.sum().item())
            if not type_count:
                continue
            task_ranges = self.index.ranges[label]
            offsets = torch.randint(
                task_ranges.frame_count,
                (type_count,),
                generator=generator,
                dtype=torch.long,
            )   # 抽取任务内部index
            selected = task_ranges.sample_from_offsets(offsets.numpy()) # 转化为数据集整体index
            sample_ids[mask] = torch.from_numpy(selected.astype(np.int64, copy=False))
        return sample_ids


def _build_ranges(
    starts_by_type: Mapping[str, Sequence[int]],
    lengths_by_type: Mapping[str, Sequence[int]],
    label_values: Sequence[str] | None = None,
) -> dict[str, TaskTypeRanges]:
    """按 task type 构建 episode frame 区间索引。

    starts_by_type = {
        "Cleaning": [0, 120],
        "Organization": [100],
    }

    lengths_by_type = {
        "Cleaning": [100, 50],
        "Organization": [20],
    }

    输出：
    {
        "Cleaning": TaskTypeRanges(
            starts=np.array([0, 120]),
            lengths=np.array([100, 50]),
            cumulative_frames=np.array([100, 150]),
        ),
        "Organization": TaskTypeRanges(
            starts=np.array([100]),
            lengths=np.array([20]),
            cumulative_frames=np.array([20]),
        ),
    }
    """
    result: dict[str, TaskTypeRanges] = {}
    labels = label_values if label_values is not None else starts_by_type
    for label in labels:
        starts = starts_by_type.get(label)
        if not starts:
            continue
        starts_array = np.asarray(starts, dtype=np.int64)
        lengths_array = np.asarray(lengths_by_type[label], dtype=np.int64)
        if starts_array.size != lengths_array.size:
            raise ValueError(
                f"Type range size mismatch for {label!r}: "
                f"{starts_array.size} starts vs {lengths_array.size} lengths."
            )
        if np.any(lengths_array <= 0):
            raise ValueError(
                f"Type {label!r} contains non-positive episode lengths."
            )
        result[label] = TaskTypeRanges(
            starts=starts_array,
            lengths=lengths_array,
            cumulative_frames=np.cumsum(lengths_array, dtype=np.int64),
        )
    return result


def build_task_type_frame_index(
    data_root: str | Path,
    episodes: Sequence[int],
    episode_starts: Sequence[int],
    config: TaskTypeSamplingConfig,
) -> TaskTypeFrameIndex:
    # 为单个数据集 root 建立 task-type frame 索引
    if not config.enabled:
        raise ValueError("Cannot build a task type frame index when disabled.")

    root = Path(data_root).resolve()
    metadata_path = Path(config.metadata_file)
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Task type metadata file not found: {metadata_path}")

    task_types_by_episode: dict[int, str] = {}
    with open(metadata_path, encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {metadata_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object in {metadata_path}:{line_number}."
                )
            episode_index = row.get("episode_index")
            task_type = row.get("task_type")
            if type(episode_index) is not int or episode_index < 0:
                raise ValueError(
                    f"Invalid episode_index in {metadata_path}:{line_number}: "
                    f"{episode_index!r}."
                )
            if not isinstance(task_type, str) or task_type not in config.label_value_set:
                raise ValueError(
                    f"Invalid {config.label_space} in {metadata_path}:{line_number}: "
                    f"{task_type!r}."
                )
            if episode_index in task_types_by_episode:
                raise ValueError(
                    f"Duplicate episode_index={episode_index} in {metadata_path}."
                )
            task_types_by_episode[episode_index] = task_type

    # episodes = [2, 5]
    # episode_starts = [0, 100, 160], 包含终止边界

    # task_types_by_episode = {
    #     2: "Cleaning",
    #     5: "Organization",
    #     9: "Handcrafting",
    # }
    episode_ids = [int(episode_index) for episode_index in episodes]
    starts_array = np.asarray(episode_starts, dtype=np.int64)
    if starts_array.size != len(episode_ids) + 1:
        raise ValueError(
            f"Episode starts length mismatch under {root}: "
            f"{starts_array.size} vs {len(episode_ids) + 1}."
        )
    episode_lengths = np.diff(starts_array)
    if np.any(episode_lengths < 0):
        raise ValueError(f"Negative episode length found under {root}.")

    usable_episode_ids = {
        episode_index
        for episode_index, episode_length in zip(episode_ids, episode_lengths)
        if episode_length > 0
    }
    missing = sorted(usable_episode_ids - set(task_types_by_episode))
    if missing:
        raise ValueError(
            f"Task type metadata {metadata_path} is missing {len(missing)} usable "
            f"episodes; examples={missing[:10]}."
        )

    starts_by_type: dict[str, list[int]] = {}
    lengths_by_type: dict[str, list[int]] = {}
    for position, episode_index in enumerate(episode_ids):  # （筛选后的）新坐标 和 （筛选前、数据集的）原坐标
        episode_length = int(episode_lengths[position])
        if episode_length == 0:
            continue
        task_type = task_types_by_episode[episode_index]
        starts_by_type.setdefault(task_type, []).append(int(starts_array[position]))
        lengths_by_type.setdefault(task_type, []).append(episode_length)

    ranges = _build_ranges(
        starts_by_type,
        lengths_by_type,
        config.label_values,
    )
    return TaskTypeFrameIndex(
        ranges=ranges,
        episode_count=int(np.count_nonzero(episode_lengths)),
        frame_count=int(episode_lengths.sum()),
        sidecar_rows=len(task_types_by_episode),
        extra_rows=len(set(task_types_by_episode) - usable_episode_ids),
        roots=(str(root),),
    )


def merge_task_type_frame_indices(
    indices: Sequence[TaskTypeFrameIndex],
    offsets: Sequence[int],
) -> TaskTypeFrameIndex:
    """
    多个数据根各自的 TaskTypeFrameIndex 合并成一个索引
    """
    if len(indices) != len(offsets):
        raise ValueError(
            f"Task type index/offset length mismatch: {len(indices)} vs {len(offsets)}."
        )
    if not indices:
        raise ValueError("At least one task type frame index is required.")

    starts_by_type: dict[str, list[int]] = {}
    lengths_by_type: dict[str, list[int]] = {}
    for index, offset in zip(indices, offsets):
        offset = int(offset)
        if offset < 0:
            raise ValueError(f"Task type index offset must be non-negative: {offset}.")
        for task_type, task_ranges in index.ranges.items():
            starts_by_type.setdefault(task_type, []).extend(
                (task_ranges.starts + offset).tolist()
            )
            lengths_by_type.setdefault(task_type, []).extend(
                task_ranges.lengths.tolist()
            )

    roots = tuple(root for index in indices for root in index.roots)
    return TaskTypeFrameIndex(
        ranges=_build_ranges(starts_by_type, lengths_by_type),
        episode_count=sum(index.episode_count for index in indices),
        frame_count=sum(index.frame_count for index in indices),
        sidecar_rows=sum(index.sidecar_rows for index in indices),
        extra_rows=sum(index.extra_rows for index in indices),
        roots=roots,
    )
