# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Post-sampling NOE subtask lookup for DreamZero datasets."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

NoeSubtaskMode = Literal["off", "prefer"]
NoeSubtaskMissingMetadataPolicy = Literal["error", "fallback"]
SubtaskResolutionStatus = Literal[
    "matched",  # 唯一共同文本
    "ambiguous_random",  # 多个共同文本，随机选择
    "no_match",  # 存在未命中的 chunk
    "mixed",  # 各 chunk 命中但无共同文本
]


def _config_get(config: Any, key: str, default: Any) -> Any:
    # 兼容普通字典与 OmegaConf DictConfig 的统一取值接口。
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if getter is None:
        raise TypeError(
            f"data.noe_subtask must be a mapping-like object, got {type(config)!r}."
        )
    return getter(key, default)


def _normalize_text(value: Any) -> str:
    # 合并多余空白，确保重复标注映射到同一个文本 ID。
    return " ".join(str(value or "").split())


@dataclass(frozen=True)
class NoeSubtaskConfig:
    """Configuration for optional post-sampling NOE subtask lookup."""

    mode: NoeSubtaskMode = "off"
    metadata_file: str = "meta/subtask_segments.parquet"
    missing_metadata_policy: NoeSubtaskMissingMetadataPolicy = "error"
    instruction_key: str = "subtask_en"
    required_quality: int = 1
    ambiguous_policy: str = "deterministic_random"
    seed: int = 42

    @classmethod
    def from_value(cls, value: Any) -> NoeSubtaskConfig:
        """Parse and validate a mapping-like config value."""
        # 将 YAML 或字典配置收敛为经过校验的不可变配置。
        if value is None:
            return cls()

        mode = str(_config_get(value, "mode", "off")).strip().lower()
        if mode not in {"off", "prefer"}:
            raise ValueError(
                f"data.noe_subtask.mode must be 'off' or 'prefer', got {mode!r}."
            )

        metadata_file = str(
            _config_get(value, "metadata_file", cls.metadata_file)
        ).strip()
        if not metadata_file:
            raise ValueError("data.noe_subtask.metadata_file must not be empty.")

        missing_metadata_policy = str(
            _config_get(
                value,
                "missing_metadata_policy",
                cls.missing_metadata_policy,
            )
        ).strip().lower()
        if missing_metadata_policy not in {"error", "fallback"}:
            raise ValueError(
                "data.noe_subtask.missing_metadata_policy must be "
                f"'error' or 'fallback', got {missing_metadata_policy!r}."
            )

        instruction_key = str(
            _config_get(value, "instruction_key", cls.instruction_key)
        ).strip()
        if instruction_key not in {"subtask_en", "subtask_cn"}:
            raise ValueError(
                "data.noe_subtask.instruction_key must be 'subtask_en' or "
                f"'subtask_cn', got {instruction_key!r}."
            )

        required_quality = int(
            _config_get(value, "required_quality", cls.required_quality)
        )
        if required_quality not in {0, 1}:
            raise ValueError(
                "data.noe_subtask.required_quality must be 0 or 1, "
                f"got {required_quality!r}."
            )

        ambiguous_policy = str(
            _config_get(value, "ambiguous_policy", cls.ambiguous_policy)
        ).strip()
        if ambiguous_policy != "deterministic_random":
            raise ValueError(
                "data.noe_subtask.ambiguous_policy currently supports only "
                f"'deterministic_random', got {ambiguous_policy!r}."
            )

        return cls(
            mode=mode,
            metadata_file=metadata_file,
            missing_metadata_policy=missing_metadata_policy,
            instruction_key=instruction_key,
            required_quality=required_quality,
            ambiguous_policy=ambiguous_policy,
            seed=int(_config_get(value, "seed", cls.seed)),
        )


@dataclass(frozen=True)
class SubtaskResolution:
    """Result of resolving one sampled multi-anchor action sequence."""

    text: str | None
    status: SubtaskResolutionStatus
    candidate_count: int = 0


class NoeSubtaskIndex:
    """基于单个数据集的 subtask parquet 构建内存区间索引。"""

    def __init__(
        self,
        *,
        source_path: Path,
        dataset_id: str,
        config: NoeSubtaskConfig,
        episode_bounds: dict[int, tuple[int, int]],
        starts: np.ndarray,
        ends: np.ndarray,
        text_ids: np.ndarray,
        texts: tuple[str, ...],
    ) -> None:
        # 使用紧凑数组保存区间，仅保留 episode 到数组切片的映射。
        self.source_path = source_path
        self.dataset_id = dataset_id
        self.config = config
        self._episode_bounds = episode_bounds
        self._starts = starts
        self._ends = ends
        self._text_ids = text_ids
        self._texts = texts
        self.num_segments = int(starts.size)

    @property
    def num_episodes(self) -> int:
        """返回至少包含一个已索引 segment 的 episode 数量。"""
        return len(self._episode_bounds)

    @property
    def num_texts(self) -> int:
        """返回规范化并去重后的指令文本数量。"""
        return len(self._texts)

    @classmethod
    def load(
        cls,
        dataset_root: str | Path,
        config: NoeSubtaskConfig,
    ) -> NoeSubtaskIndex:
        """加载并校验单个数据集的 subtask parquet。"""
        import pyarrow as pa
        import pyarrow.parquet as pq

        root = Path(dataset_root).resolve()
        configured_path = Path(config.metadata_file)
        source_path = (
            configured_path if configured_path.is_absolute() else root / configured_path
        ).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"NOE subtask mode='prefer' 需要元数据 parquet：{source_path}"
            )

        schema = pq.read_schema(source_path)
        expected_types = {
            "episode_index": pa.int32(),
            "start_frame": pa.int32(),
            "end_frame": pa.int32(),
            config.instruction_key: pa.string(),
            "quality": pa.int8(),
        }
        missing = [name for name in expected_types if name not in schema.names]
        if missing:
            raise ValueError(f"NOE subtask parquet {source_path} 缺少字段：{missing}。")
        invalid_types = {
            name: str(schema.field(name).type)
            for name, expected in expected_types.items()
            if not schema.field(name).type.equals(expected)
        }
        if invalid_types:
            raise ValueError(
                f"NOE subtask parquet {source_path} 的字段类型无效：{invalid_types}。"
            )

        table = pq.read_table(source_path, columns=list(expected_types))

        # 非法 quality 只告警并忽略，避免单条坏标注阻断整个数据集加载。
        quality_column = table["quality"].combine_chunks()
        quality_is_null = np.asarray(
            quality_column.is_null().to_numpy(zero_copy_only=False),
            dtype=bool,
        )
        quality_values = quality_column.fill_null(-1).to_numpy()
        invalid_quality = quality_is_null | (
            (quality_values != 0) & (quality_values != 1)
        )
        invalid_rows = np.flatnonzero(invalid_quality)
        if invalid_rows.size:
            row_index = int(invalid_rows[0])
            invalid_value = (
                None if quality_is_null[row_index] else int(quality_values[row_index])
            )
            warnings.warn(
                (
                    f"忽略 {source_path} 中 {invalid_rows.size} 行 quality 非法的数据；"
                    f"quality 应为 0 或 1。首个非法行 {row_index} 的 "
                    f"quality={invalid_value!r}。"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        selected_mask = quality_values == config.required_quality
        table = table.filter(pa.array(selected_mask))

        # episode_indices[i], starts[i], ends[i], raw_texts[i] 共同描述第 i 条 segment
        episode_indices = (
            table["episode_index"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.int32, copy=True)
        )
        starts = (
            table["start_frame"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.int32, copy=True)
        )
        ends = (
            table["end_frame"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.int32, copy=True)
        )
        raw_texts = table[config.instruction_key].combine_chunks().to_pylist()

        # 检查segment合法性
        invalid_episode = np.flatnonzero(episode_indices < 0)
        if invalid_episode.size:
            row_index = int(invalid_episode[0])
            raise ValueError(
                f"{source_path} 筛选后的第 {row_index} 行包含非法 "
                f"episode_index={int(episode_indices[row_index])}。"
            )
        invalid_interval = np.flatnonzero((starts < 0) | (starts >= ends))
        if invalid_interval.size:
            row_index = int(invalid_interval[0])
            raise ValueError(
                f"{source_path} 筛选后的第 {row_index} 行包含非法区间 "
                f"[{int(starts[row_index])}, {int(ends[row_index])})。"
            )

        # 将文本转化为id，通过texts查找
        text_to_id: dict[str, int] = {}
        texts: list[str] = []
        text_ids = np.empty(len(raw_texts), dtype=np.int32)
        for row_index, raw_text in enumerate(raw_texts):
            text = _normalize_text(raw_text)
            if not text:
                raise ValueError(
                    f"{source_path} 筛选后的第 {row_index} 行包含空的 "
                    f"{config.instruction_key!r}。"
                )
            text_id = text_to_id.get(text)
            if text_id is None:
                text_id = len(texts)
                text_to_id[text] = text_id
                texts.append(text)
            text_ids[row_index] = text_id

        # 同一个 episode 的所有 segment 会连续存放，并且内部按开始帧排列
        order = np.lexsort((text_ids, ends, starts, episode_indices))
        episode_indices = episode_indices[order]
        starts = starts[order]
        ends = ends[order]
        text_ids = text_ids[order]

        # 得到 episode_bounds = {10: (0, 19), 11: (19, 34), 12: (34, 51)}，用于快速查找
        episode_bounds: dict[int, tuple[int, int]] = {}
        if episode_indices.size:
            unique_episodes, first_positions = np.unique(
                episode_indices,
                return_index=True,
            )
            last_positions = np.append(first_positions[1:], episode_indices.size)
            episode_bounds = {
                int(episode_index): (int(first), int(last))
                for episode_index, first, last in zip(
                    unique_episodes,
                    first_positions,
                    last_positions,
                    strict=True,
                )
            }

        return cls(
            source_path=source_path,
            dataset_id=root.name,
            config=config,
            episode_bounds=episode_bounds,
            starts=starts,
            ends=ends,
            text_ids=text_ids,
            texts=tuple(texts),
        )

    def _covering_text_ids(
        self,
        episode_bounds: tuple[int, int],
        chunk_start: int,
        chunk_end: int,
    ) -> set[int]:
        # 返回满足条件的文本id
        first, last = episode_bounds
        starts = self._starts[first:last]
        right = int(np.searchsorted(starts, int(chunk_start), side="right"))
        if right == 0:
            return set()
        candidate_ends = self._ends[first : first + right]
        candidate_text_ids = self._text_ids[first : first + right]
        return set(
            map(
                int,
                candidate_text_ids[candidate_ends >= int(chunk_end)].tolist(),
            )
        )

    def _choose_text(
        self,
        text_ids: set[int],
        *,
        episode_index: int,
        frame_index: int,
    ) -> str:
        # 从共同候选文本中选择最终指令，
        candidates = sorted(self._texts[text_id] for text_id in text_ids)
        if len(candidates) == 1:
            return candidates[0]
        payload = (
            f"{self.dataset_id}\0{int(episode_index)}\0{int(frame_index)}\0"
            f"{self.config.seed}"
        ).encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        selected_index = int.from_bytes(digest, byteorder="big") % len(candidates)
        return candidates[selected_index]

    def resolve(
        self,
        episode_index: int,
        frame_index: int,
        action_chunks: list[tuple[int, int]],
    ) -> SubtaskResolution:
        """
        resolve() 解析所有已采样 action chunk 共有的一条指令
        ├─ 对每个 chunk 调用 _covering_text_ids()
        ├─ 对候选文本 ID 求交集
        └─ 调用 _choose_text() 选择最终文本
        """
        episode_bounds = self._episode_bounds.get(int(episode_index))
        if episode_bounds is None or not action_chunks:
            return SubtaskResolution(text=None, status="no_match")

        common: set[int] | None = None
        for chunk_start, chunk_end in action_chunks:
            chunk_start = int(chunk_start)
            chunk_end = int(chunk_end)
            if chunk_start < 0 or chunk_start >= chunk_end:
                raise ValueError(
                    f"采样得到的 action chunk 区间非法：[{chunk_start}, {chunk_end})。"
                )
            candidates = self._covering_text_ids(
                episode_bounds,
                chunk_start,
                chunk_end,
            )
            if not candidates:
                return SubtaskResolution(text=None, status="no_match")
            common = candidates if common is None else common & candidates
            if not common:
                return SubtaskResolution(text=None, status="mixed")

        assert common is not None
        text = self._choose_text(
            common,
            episode_index=episode_index,
            frame_index=frame_index,
        )
        status: SubtaskResolutionStatus = (
            "matched" if len(common) == 1 else "ambiguous_random"
        )
        return SubtaskResolution(
            text=text,
            status=status,
            candidate_count=len(common),
        )
