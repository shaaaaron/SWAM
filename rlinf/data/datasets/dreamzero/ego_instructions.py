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

"""Optional Ego subtask, dense-caption, and scene instruction lookup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EGO_CAPTION_SEGMENT_MODES = {"then", "max"}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def append_scene_instruction(instruction: str, scene_caption: str) -> str:
    """Append scene context without changing the selected base instruction."""
    instruction = _normalize_text(instruction)
    scene_caption = _normalize_text(scene_caption)
    if not scene_caption:
        return instruction
    if not instruction:
        return f"Scene: {scene_caption}"
    separator = "" if instruction.endswith((".", "!", "?")) else "."
    return f"{instruction}{separator} Scene: {scene_caption}"


def compose_dense_instruction(action_caption: str, object_caption: str) -> str:
    """Combine the two dense-caption fields as natural consecutive sentences."""
    sentences = []
    for value in (action_caption, object_caption):
        text = _normalize_text(value)
        if not text:
            continue
        if not text.endswith((".", "!", "?")):
            text = f"{text}."
        sentences.append(text)
    return " ".join(sentences)


@dataclass(frozen=True)
class EgoSubtaskSegment:
    start_frame: int
    end_frame: int
    caption: str
    tag: str


@dataclass(frozen=True)
class EgoDenseCaptionSegment:
    start_frame: int
    end_frame: int
    action_caption: str
    object_caption: str
    tag: str


class EgoInstructionIndex:
    """In-memory index for the optional Ego annotation sidecars."""

    def __init__(
        self,
        *,
        subtasks_by_episode: dict[int, tuple[EgoSubtaskSegment, ...]],
        dense_captions_by_episode: dict[
            int, tuple[EgoDenseCaptionSegment, ...]
        ],
        scenarios_by_episode: dict[int, dict[str, Any]],
        use_subtask_instruction: bool,
        use_dense_caption_instruction: bool,
        dense_caption_include_object: bool,
        ego_caption_segment_mode: str,
        use_scene_instruction: bool,
    ) -> None:
        self._subtasks_by_episode = subtasks_by_episode
        self._dense_captions_by_episode = dense_captions_by_episode
        self._scenarios_by_episode = scenarios_by_episode
        self.use_subtask_instruction = bool(use_subtask_instruction)
        self.use_dense_caption_instruction = bool(use_dense_caption_instruction)
        self.dense_caption_include_object = bool(dense_caption_include_object)
        self.ego_caption_segment_mode = ego_caption_segment_mode
        self.use_scene_instruction = bool(use_scene_instruction)

    @classmethod
    def load(
        cls,
        dataset_root: str | Path,
        *,
        use_subtask_instruction: bool,
        use_scene_instruction: bool,
        use_dense_caption_instruction: bool = False,
        dense_caption_include_object: bool = True,
        ego_caption_segment_mode: str | None = None,
    ) -> "EgoInstructionIndex":
        if use_subtask_instruction and use_dense_caption_instruction:
            raise ValueError(
                "use_subtask_instruction and use_dense_caption_instruction "
                "are mutually exclusive."
            )
        if ego_caption_segment_mode is None:
            ego_caption_segment_mode = (
                "max" if use_dense_caption_instruction else "then"
            )
        else:
            ego_caption_segment_mode = str(ego_caption_segment_mode).strip().lower()
        if ego_caption_segment_mode not in _EGO_CAPTION_SEGMENT_MODES:
            raise ValueError(
                "ego_caption_segment_mode must be 'then' or 'max', "
                f"got {ego_caption_segment_mode!r}."
            )
        root = Path(dataset_root).resolve()
        meta_dir = root / "meta"
        subtasks: dict[int, list[EgoSubtaskSegment]] = {}
        dense_captions: dict[int, list[EgoDenseCaptionSegment]] = {}
        scenarios: dict[int, dict[str, Any]] = {}

        if use_subtask_instruction:
            path = meta_dir / "subtask" / "subtasks_v1.jsonl"
            if not path.is_file():
                raise FileNotFoundError(
                    "use_subtask_instruction=True requires Ego subtask sidecar: "
                    f"{path}"
                )
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    episode_index = int(record["episode_index"])
                    rows = record["subtasks"]
                    episode_rows = subtasks.setdefault(episode_index, [])
                    for row in rows:
                        start_frame = int(row["start_frame"])
                        end_frame = int(row["end_frame"])
                        caption = _normalize_text(row["caption"])
                        episode_rows.append(
                            EgoSubtaskSegment(
                                start_frame=start_frame,
                                end_frame=end_frame,
                                caption=caption,
                                tag=_normalize_text(row.get("tag")).lower(),
                            )
                        )
            subtasks_by_episode = {
                episode_index: tuple(
                    sorted(
                        rows,
                        key=lambda segment: (
                            segment.start_frame,
                            segment.end_frame,
                        ),
                    )
                )
                for episode_index, rows in subtasks.items()
            }
        else:
            subtasks_by_episode = {}

        if use_dense_caption_instruction:
            path = meta_dir / "dense_caption" / "dense_captions_v1.jsonl"
            if not path.is_file():
                raise FileNotFoundError(
                    "use_dense_caption_instruction=True requires Ego dense "
                    f"caption sidecar: {path}"
                )
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    episode_index = int(record["episode_index"])
                    rows = record["dense_captions"]
                    episode_rows = dense_captions.setdefault(episode_index, [])
                    for row in rows:
                        episode_rows.append(
                            EgoDenseCaptionSegment(
                                start_frame=int(row["start_frame"]),
                                end_frame=int(row["end_frame"]),
                                action_caption=_normalize_text(
                                    row["action_caption"]
                                ),
                                object_caption=_normalize_text(
                                    row["object_caption"]
                                ),
                                tag=_normalize_text(row.get("tag")).lower(),
                            )
                        )
            dense_captions_by_episode = {
                episode_index: tuple(
                    sorted(
                        rows,
                        key=lambda segment: (
                            segment.start_frame,
                            segment.end_frame,
                        ),
                    )
                )
                for episode_index, rows in dense_captions.items()
            }
        else:
            dense_captions_by_episode = {}

        if use_scene_instruction:
            path = meta_dir / "episodes_extra.jsonl"
            if not path.is_file():
                raise FileNotFoundError(
                    "use_scene_instruction=True requires Ego episode sidecar: "
                    f"{path}"
                )
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    episode_index = int(record["episode_index"])
                    metadata = record.get("metadata", record)
                    tags = metadata.get("scenarios", [])
                    if isinstance(tags, str):
                        tags = [tags]
                    scenarios[episode_index] = {
                        "scenario_tags": [
                            _normalize_text(tag)
                            for tag in tags
                            if _normalize_text(tag)
                        ],
                        "scenario_caption": _normalize_text(
                            metadata.get("scenario_caption")
                        ),
                        "task_type": _normalize_text(metadata.get("task_type")),
                        "episode_caption": _normalize_text(metadata.get("caption")),
                    }

        return cls(
            subtasks_by_episode=subtasks_by_episode,
            dense_captions_by_episode=dense_captions_by_episode,
            scenarios_by_episode=scenarios,
            use_subtask_instruction=use_subtask_instruction,
            use_dense_caption_instruction=use_dense_caption_instruction,
            dense_caption_include_object=dense_caption_include_object,
            ego_caption_segment_mode=ego_caption_segment_mode,
            use_scene_instruction=use_scene_instruction,
        )

    def get_ego_block_prompt(
        self,
        episode_index: int,
        start_frame: int,
        end_frame: int,
        *,
        episode_instruction: str,
    ) -> str:
        """Resolve valid captions intersecting a sampled action block."""
        fallback = _normalize_text(episode_instruction)
        if self.use_dense_caption_instruction:
            if self.ego_caption_segment_mode == "then":
                captions: list[str] = []
                previous: str | None = None
                for segment in self._dense_captions_by_episode.get(
                    int(episode_index), ()
                ):
                    if segment.tag != "valid":
                        continue
                    overlap = min(segment.end_frame, int(end_frame)) - max(
                        segment.start_frame, int(start_frame)
                    )
                    if overlap <= 0:
                        continue
                    caption = compose_dense_instruction(
                        segment.action_caption,
                        (
                            segment.object_caption
                            if self.dense_caption_include_object
                            else ""
                        ),
                    )
                    if not caption or caption == previous:
                        continue
                    captions.append(caption)
                    previous = caption
                return " then ".join(captions) if captions else fallback

            best_segment: EgoDenseCaptionSegment | None = None
            best_overlap = 0
            for segment in self._dense_captions_by_episode.get(
                int(episode_index), ()
            ):
                if segment.tag != "valid":
                    continue
                overlap = min(segment.end_frame, int(end_frame)) - max(
                    segment.start_frame, int(start_frame)
                )
                if overlap > best_overlap:
                    best_segment = segment
                    best_overlap = overlap
            if best_segment is None:
                return fallback
            return compose_dense_instruction(
                best_segment.action_caption,
                (
                    best_segment.object_caption
                    if self.dense_caption_include_object
                    else ""
                ),
            ) or fallback
        if not self.use_subtask_instruction:
            return fallback

        if self.ego_caption_segment_mode == "max":
            best_segment: EgoSubtaskSegment | None = None
            best_overlap = 0
            for segment in self._subtasks_by_episode.get(int(episode_index), ()):
                if segment.tag != "valid":
                    continue
                overlap = min(segment.end_frame, int(end_frame)) - max(
                    segment.start_frame, int(start_frame)
                )
                if overlap > best_overlap:
                    best_segment = segment
                    best_overlap = overlap
            return best_segment.caption if best_segment is not None else fallback

        captions: list[str] = []
        previous: str | None = None
        for segment in self._subtasks_by_episode.get(int(episode_index), ()):
            if segment.tag != "valid":
                continue
            if (
                segment.start_frame >= int(end_frame)
                or segment.end_frame <= int(start_frame)
            ):
                continue
            if segment.caption == previous:
                continue
            captions.append(segment.caption)
            previous = segment.caption
        return " then ".join(captions) if captions else fallback

    def get_ego_scenario(self, episode_index: int) -> dict[str, Any]:
        """Return structured scene metadata for one episode."""
        return dict(
            self._scenarios_by_episode.get(
                int(episode_index),
                {
                    "scenario_tags": [],
                    "scenario_caption": "",
                    "task_type": "",
                    "episode_caption": "",
                },
            )
        )
