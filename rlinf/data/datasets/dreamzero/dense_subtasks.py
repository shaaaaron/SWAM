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

"""Shared dense-subtask lookup for DreamZero UMI and Ego datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


class DenseSubtaskIndex:
    """Compact per-episode index over valid manipulation annotations."""

    def __init__(
        self,
        *,
        source_path: Path,
        caption_granularity: str,
        num_rows: int,
        total_segments: int,
        episode_bounds: dict[int, tuple[int, int]],
        starts: np.ndarray,
        ends: np.ndarray,
        captions: tuple[str, ...],
        episode_captions: dict[int, str],
    ) -> None:
        self.source_path = source_path
        self.caption_granularity = caption_granularity
        self.num_rows = int(num_rows)
        self.total_segments = int(total_segments)
        self._episode_bounds = episode_bounds
        self._starts = starts
        self._ends = ends
        self._captions = captions
        self._episode_captions = episode_captions

    @property
    def num_episodes(self) -> int:
        return len(self._episode_bounds)

    @property
    def num_segments(self) -> int:
        return int(self._starts.size)

    @property
    def num_filtered_segments(self) -> int:
        return self.total_segments - self.num_segments

    @classmethod
    def load(
        cls,
        dataset_root: str | Path,
        metadata_file: str,
        *,
        caption_field: str = "action_detail_en",
        caption_granularity: str = "subtask",
        caption_extra: str = "none",
    ) -> DenseSubtaskIndex:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        if caption_field not in (
            "action_detail_en",
            "action_sent_en",
            "action_short_en",
        ):
            raise ValueError(
                "subtask_caption_field must be 'action_detail_en', "
                "'action_sent_en', or 'action_short_en', "
                f"got {caption_field!r}."
            )

        caption_granularity = str(caption_granularity).strip().lower()
        if caption_granularity not in {"subtask", "episode"}:
            raise ValueError(
                "caption_granularity must be 'subtask' or 'episode', "
                f"got {caption_granularity!r}."
            )

        if caption_extra not in ("none", "scene_light", "state"):
            raise ValueError(
                "subtask_caption_extra must be 'none', 'scene_light', or 'state', "
                f"got {caption_extra!r}."
            )
        extra_fields = {
            "none": (),
            "scene_light": ("scene_en", "lighting_en"),
            "state": ("state_detail_en",),
        }[caption_extra]

        root = Path(dataset_root).resolve()
        configured_path = Path(metadata_file)
        source_path = (
            configured_path if configured_path.is_absolute() else root / configured_path
        ).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"unified_dense subtask mode requires parquet: {source_path}"
            )

        schema = pq.read_schema(source_path)
        if caption_granularity == "episode":
            # 缺失字段
            missing = [
                name for name in ("episode_index", "caption") if name not in schema.names
            ]
            if missing:
                raise ValueError(
                    f"Episode caption parquet {source_path} is missing fields: {missing}."
                )

            # 检查字段类型
            caption_type = schema.field("caption").type
            if not pa.types.is_struct(caption_type):
                raise ValueError(
                    f"Episode caption parquet {source_path} has non-struct caption: "
                    f"{caption_type}."
                )
            caption_fields = {field.name: field.type for field in caption_type}
            for field in (caption_field, *extra_fields):
                if field not in caption_fields:
                    raise ValueError(
                        f"Episode caption parquet {source_path} is missing caption field "
                        f"{field!r}."
                    )
                if not (
                    pa.types.is_string(caption_fields[field])
                    or pa.types.is_large_string(caption_fields[field])
                ):
                    raise ValueError(
                        f"Episode caption parquet {source_path} has invalid type for "
                        f"caption.{field}: {caption_fields[field]}."
                    )

            # 建立从 episode index 到 caption 的映射
            table = pq.read_table(source_path, columns=["episode_index", "caption"])
            episode_rows = table["episode_index"].combine_chunks()
            if episode_rows.null_count:
                raise ValueError(
                    f"Episode caption parquet {source_path} contains null episode_index values."
                )
            episode_indices = episode_rows.to_numpy(zero_copy_only=False).astype(
                np.int64, copy=True
            )
            if np.any(episode_indices < 0):
                raise ValueError(
                    f"Episode caption parquet {source_path} contains negative episode_index values."
                )
            if np.unique(episode_indices).size != episode_indices.size:
                raise ValueError(
                    f"Episode caption parquet {source_path} contains duplicate episode_index rows."
                )
            captions = table["caption"].combine_chunks()
            caption_values = captions.field(caption_field).to_pylist()
            extra_captions = [captions.field(field).to_pylist() for field in extra_fields]
            episode_captions = {}
            for index, (episode_index, value) in enumerate(
                zip(episode_indices.tolist(), caption_values, strict=True)
            ):
                caption = _normalize_text(value)
                # Empty action captions stay empty, even when extras are present.
                if caption and extra_fields:
                    caption = _normalize_text(
                        " ".join(
                            [caption]
                            + [_normalize_text(values[index]) for values in extra_captions]
                        )
                    )
                episode_captions[int(episode_index)] = caption
            return cls(
                source_path=source_path,
                caption_granularity=caption_granularity,
                num_rows=table.num_rows,
                total_segments=0,
                episode_bounds={int(index): (0, 0) for index in episode_indices},
                starts=np.empty(0, dtype=np.int64),
                ends=np.empty(0, dtype=np.int64),
                captions=(),
                episode_captions=episode_captions,
            )

        missing = [
            name
            for name in ("episode_index", "subtasks")
            if name not in schema.names
        ]
        if missing:
            raise ValueError(
                f"Dense subtask parquet {source_path} is missing fields: {missing}."
            )

        # 检查字段类型
        episode_type = schema.field("episode_index").type
        subtasks_type = schema.field("subtasks").type
        if not pa.types.is_integer(episode_type):
            raise ValueError(
                f"Dense subtask parquet {source_path} has non-integer "
                f"episode_index: {episode_type}."
            )
        if not (
            pa.types.is_list(subtasks_type) or pa.types.is_large_list(subtasks_type)
        ):
            raise ValueError(
                f"Dense subtask parquet {source_path} has non-list subtasks: "
                f"{subtasks_type}."
            )
        segment_type = subtasks_type.value_type
        if not pa.types.is_struct(segment_type):
            raise ValueError(
                f"Dense subtask parquet {source_path} has non-struct subtask values: "
                f"{segment_type}."
            )

        required_segment_types = {
            "start_frame": pa.types.is_integer,
            "end_frame": pa.types.is_integer,
            caption_field: lambda value: pa.types.is_string(value)
            or pa.types.is_large_string(value),
            "segment_type": lambda value: pa.types.is_string(value)
            or pa.types.is_large_string(value),
            "visual_clear": pa.types.is_boolean,
        }
        required_segment_types.update(
            {
                field: lambda value: pa.types.is_string(value)
                or pa.types.is_large_string(value)
                for field in extra_fields
            }
        )
        segment_fields = {field.name: field.type for field in segment_type}
        missing_segments = [
            name for name in required_segment_types if name not in segment_fields
        ]
        if missing_segments:
            raise ValueError(
                f"Dense subtask parquet {source_path} is missing subtask fields: "
                f"{missing_segments}."
            )
        invalid_segments = {
            name: str(segment_fields[name])
            for name, predicate in required_segment_types.items()
            if not predicate(segment_fields[name])
        }
        if invalid_segments:
            raise ValueError(
                f"Dense subtask parquet {source_path} has invalid subtask field "
                f"types: {invalid_segments}."
            )

        table = pq.read_table(
            source_path,
            columns=["episode_index", "subtasks"],
        )

        # 检查 episode_index 的有效性
        episode_rows = table["episode_index"].combine_chunks()
        if episode_rows.null_count:
            raise ValueError(
                f"Dense subtask parquet {source_path} contains null episode_index values."
            )
        row_episode_indices = episode_rows.to_numpy(zero_copy_only=False).astype(
            np.int64, copy=True
        )
        if np.any(row_episode_indices < 0):
            raise ValueError(
                f"Dense subtask parquet {source_path} contains negative episode_index values."
            )
        if np.unique(row_episode_indices).size != row_episode_indices.size:
            raise ValueError(
                f"Dense subtask parquet {source_path} contains duplicate episode_index rows."
            )
        # 展平subtasks 并过滤有效的操作段
        subtask_lists = table["subtasks"].combine_chunks()
        parent_rows = pc.list_parent_indices(subtask_lists)
        segments = pc.list_flatten(subtask_lists)
        total_segments = len(segments)

        visual_clear = pc.fill_null(segments.field("visual_clear"), False)
        is_manipulation = pc.fill_null(
            pc.equal(segments.field("segment_type"), "manipulation"), False
        )
        captions = pc.utf8_trim_whitespace(
            pc.fill_null(segments.field(caption_field), "")
        )
        has_caption = pc.greater(pc.utf8_length(captions), 0)
        selected = pc.and_(pc.and_(visual_clear, is_manipulation), has_caption)

        selected_episode_indices = (
            pc.filter(pc.take(episode_rows, parent_rows), selected)
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=True)
        )
        selected_segments = pc.filter(segments, selected)
        selected_captions = pc.filter(captions, selected)
        starts = (
            selected_segments.field("start_frame")
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=True)
        )
        ends = (
            selected_segments.field("end_frame")
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=True)
        )

        invalid_intervals = np.flatnonzero((starts < 0) | (starts >= ends))
        if invalid_intervals.size:
            index = int(invalid_intervals[0])
            raise ValueError(
                f"Dense subtask parquet {source_path} contains invalid selected "
                f"interval [{int(starts[index])}, {int(ends[index])}) at row {index}."
            )

        order = np.lexsort((ends, starts, selected_episode_indices))
        selected_episode_indices = selected_episode_indices[order]
        starts = starts[order]
        ends = ends[order]
        raw_captions = selected_captions.to_pylist()
        extra_captions = [
            selected_segments.field(field).to_pylist() for field in extra_fields
        ]
        normalized_captions = tuple(
            _normalize_text(
                " ".join(
                    [raw_captions[int(index)]]
                    + [_normalize_text(values[int(index)]) for values in extra_captions]
                )
            )
            for index in order
        )

        episode_bounds: dict[int, tuple[int, int]] = {}
        if selected_episode_indices.size:
            unique_episodes, first_positions = np.unique(
                selected_episode_indices, return_index=True
            )
            last_positions = np.append(
                first_positions[1:], selected_episode_indices.size
            )
            episode_bounds = {
                int(episode_index): (int(first), int(last))
                for episode_index, first, last in zip(
                    unique_episodes, first_positions, last_positions, strict=True
                )
            }

        return cls(
            source_path=source_path,
            caption_granularity=caption_granularity,
            num_rows=table.num_rows,
            total_segments=total_segments,
            episode_bounds=episode_bounds,
            starts=starts,
            ends=ends,
            captions=normalized_captions,
            episode_captions={},
        )

    def resolve(self, episode_index: int, video_indices: np.ndarray) -> str | None:
        """Return the valid subtask covering the most sampled video frames."""
        if self.caption_granularity == "episode":
            return self._episode_captions.get(int(episode_index), "")
        episode_bounds = self._episode_bounds.get(int(episode_index))
        indices = np.unique(np.asarray(video_indices, dtype=np.int64).reshape(-1))
        if episode_bounds is None or indices.size == 0:
            return None

        first, last = episode_bounds
        starts = self._starts[first:last]
        ends = self._ends[first:last]
        overlap_counts = np.searchsorted(indices, ends, side="left") - np.searchsorted(
            indices, starts, side="left"
        )
        best_offset = int(np.argmax(overlap_counts))
        if int(overlap_counts[best_offset]) <= 0:
            return None
        return self._captions[first + best_offset]
