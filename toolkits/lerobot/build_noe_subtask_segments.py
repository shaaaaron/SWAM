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

"""Build a normalized NOE subtask index for one LeRobot dataset.

The source layout is::

    <dataset>/meta/episodes.jsonl
    <dataset>/subtask/<episode_uuid>.json

The generated index is::

    <dataset>/meta/subtask_segments.parquet

Raw subtask ranges use one-based boundaries. The output stores zero-based,
half-open intervals ``[start_frame, end_frame)``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SUBTASK_SCHEMA = pa.schema(
    [
        pa.field("episode_index", pa.int32(), nullable=False),
        pa.field("episode_uuid", pa.string(), nullable=False),
        pa.field("start_frame", pa.int32(), nullable=False),
        pa.field("end_frame", pa.int32(), nullable=False),
        pa.field("subtask_en", pa.string(), nullable=False),
        pa.field("subtask_cn", pa.string(), nullable=False),
        pa.field("quality", pa.int8(), nullable=False),
    ]
)

_RANGE_RE = re.compile(r"^(\d+)_(\d+)$")
_SKIPPED_STATUSES = {"failed", "skipped"}


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    episode_uuid: str
    length: int
    status: str | None


@dataclass
class BuildStats:
    source_subtask_files: int = 0
    source_segments: int = 0
    emitted_segments: int = 0
    orphan_files: int = 0
    orphan_segments: int = 0
    skipped_episode_files: int = 0
    skipped_episode_segments: int = 0
    zero_length_segments: int = 0
    out_of_bounds_segments: int = 0
    quality_0_segments: int = 0
    quality_1_segments: int = 0


def _load_episodes(dataset_root: Path) -> dict[str, EpisodeInfo]:
    # 读取 episode 主 metadata，并建立 UUID 到索引、长度和状态的映射。
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing episodes metadata: {episodes_path}")

    episodes: dict[str, EpisodeInfo] = {}
    with open(episodes_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                episode_uuid = str(record["episode_uuid"]).strip()
                episode_index = int(record["episode_index"])
                length = int(record.get("length") or 0)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid episode row at {episodes_path}:{line_number}: "
                    f"{record!r}"
                ) from exc
            if not episode_uuid:
                raise ValueError(
                    f"Empty episode_uuid at {episodes_path}:{line_number}"
                )
            if episode_uuid in episodes:
                raise ValueError(
                    f"Duplicate episode_uuid {episode_uuid!r} in {episodes_path}"
                )
            raw_status = record.get("status")
            status = None if raw_status is None else str(raw_status).strip().lower()
            episodes[episode_uuid] = EpisodeInfo(
                episode_index=episode_index,
                episode_uuid=episode_uuid,
                length=length,
                status=status,
            )
    if not episodes:
        raise ValueError(f"No episodes found in {episodes_path}")
    return episodes


def _load_subtask(path: Path) -> dict[str, Any]:
    # 读取单个 UUID subtask JSON，并校验根结构和文件名一致性。
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    if not isinstance(record, dict):
        raise ValueError(f"Subtask root must be an object: {path}")
    clips = record.get("clip_descriptions")
    if not isinstance(clips, dict):
        raise ValueError(f"clip_descriptions must be an object: {path}")
    episode_uuid = str(record.get("episode_uuid") or "").strip()
    if not episode_uuid:
        raise ValueError(f"Missing episode_uuid in {path}")
    if path.stem != episode_uuid:
        raise ValueError(
            f"Subtask filename/episode_uuid mismatch: {path.stem!r} != "
            f"{episode_uuid!r} in {path}"
        )
    return record


def _parse_segment(
    *,
    range_key: str,
    clip: Any,
    episode: EpisodeInfo,
) -> tuple[dict[str, Any] | None, str | None]:
    # 将原始 1-based boundary 转成 0-based 半开区间并校验双语标注。
    match = _RANGE_RE.fullmatch(str(range_key))
    if match is None:
        raise ValueError(
            f"Invalid subtask range {range_key!r} for {episode.episode_uuid}"
        )
    if not isinstance(clip, dict):
        raise ValueError(
            f"Subtask clip must be an object for {episode.episode_uuid}:{range_key}"
        )

    raw_start, raw_end = (int(value) for value in match.groups())
    start_frame = raw_start - 1
    end_frame = raw_end - 1
    if start_frame >= end_frame:
        return None, "zero_length"
    if start_frame < 0 or end_frame > episode.length:
        return None, "out_of_bounds"

    quality = clip.get("quality")
    if type(quality) is not int or quality not in (0, 1):
        raise ValueError(
            f"quality must be integer 0/1 for "
            f"{episode.episode_uuid}:{range_key}, got {quality!r}"
        )
    subtask_en = clip.get("subtask_en")
    subtask_cn = clip.get("subtask_cn")
    if not isinstance(subtask_en, str) or not subtask_en.strip():
        raise ValueError(
            f"Missing subtask_en for {episode.episode_uuid}:{range_key}"
        )
    if not isinstance(subtask_cn, str) or not subtask_cn.strip():
        raise ValueError(
            f"Missing subtask_cn for {episode.episode_uuid}:{range_key}"
        )

    return (
        {
            "episode_index": episode.episode_index,
            "episode_uuid": episode.episode_uuid,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "subtask_en": subtask_en.strip(),
            "subtask_cn": subtask_cn.strip(),
            "quality": quality,
        },
        None,
    )


def collect_subtask_rows(
    dataset_root: Path,
) -> tuple[list[dict[str, Any]], BuildStats]:
    # 关联 episode 与 subtask 文件，过滤无效区间并生成稳定排序的行。
    dataset_root = dataset_root.resolve()
    subtask_dir = dataset_root / "subtask"
    if not subtask_dir.is_dir():
        raise FileNotFoundError(f"Missing subtask directory: {subtask_dir}")

    episodes = _load_episodes(dataset_root)
    stats = BuildStats()
    rows: list[dict[str, Any]] = []

    for path in sorted(subtask_dir.glob("*.json")):
        stats.source_subtask_files += 1
        record = _load_subtask(path)
        clips = record["clip_descriptions"]
        stats.source_segments += len(clips)

        episode_uuid = str(record["episode_uuid"])
        episode = episodes.get(episode_uuid)
        if episode is None:
            stats.orphan_files += 1
            stats.orphan_segments += len(clips)
            continue
        if episode.length <= 0 or episode.status in _SKIPPED_STATUSES:
            stats.skipped_episode_files += 1
            stats.skipped_episode_segments += len(clips)
            continue

        for range_key, clip in clips.items():
            row, rejection = _parse_segment(
                range_key=str(range_key),
                clip=clip,
                episode=episode,
            )
            if rejection == "zero_length":
                stats.zero_length_segments += 1
                continue
            if rejection == "out_of_bounds":
                stats.out_of_bounds_segments += 1
                continue
            assert row is not None
            rows.append(row)
            stats.emitted_segments += 1
            if row["quality"] == 1:
                stats.quality_1_segments += 1
            else:
                stats.quality_0_segments += 1

    rows.sort(
        key=lambda row: (
            row["episode_index"],
            row["start_frame"],
            row["end_frame"],
            row["quality"],
            row["subtask_en"],
            row["subtask_cn"],
        )
    )
    return rows, stats


def write_subtask_parquet(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    # 使用临时文件写入并校验 parquet，最后通过原子替换发布结果。
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(rows, schema=SUBTASK_SCHEMA)
    temp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    try:
        pq.write_table(
            table,
            temp_path,
            compression="zstd",
            use_dictionary=["episode_uuid", "subtask_en", "subtask_cn"],
            write_statistics=True,
        )
        check = pq.read_table(temp_path)
        if check.schema != SUBTASK_SCHEMA:
            raise ValueError(
                f"Written schema mismatch for {temp_path}: {check.schema}"
            )
        if check.num_rows != len(rows):
            raise ValueError(
                f"Written row count mismatch for {temp_path}: "
                f"{check.num_rows} != {len(rows)}"
            )
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def build_subtask_index(
    dataset_root: Path,
    *,
    output_path: Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    # 执行单个数据集的收集、可选写入和统计汇总流程。
    dataset_root = dataset_root.resolve()
    output_path = (
        output_path.resolve()
        if output_path is not None
        else dataset_root / "meta" / "subtask_segments.parquet"
    )
    rows, stats = collect_subtask_rows(dataset_root)
    if not dry_run:
        write_subtask_parquet(rows, output_path, overwrite=overwrite)

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "output_path": str(output_path),
        "dry_run": dry_run,
        **asdict(stats),
    }
    if not dry_run:
        summary["output_size_bytes"] = output_path.stat().st_size
    return summary


def parse_args() -> argparse.Namespace:
    # 定义单数据集生成工具的命令行参数。
    parser = argparse.ArgumentParser(
        description="Build meta/subtask_segments.parquet for one NOE dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="LeRobot dataset root containing meta/ and subtask/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path (default: <dataset>/meta/subtask_segments.parquet).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print counts without writing parquet.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file atomically.",
    )
    return parser.parse_args()


def main() -> None:
    # 解析参数、构建索引并将统计结果打印为 JSON。
    args = parse_args()
    summary = build_subtask_index(
        args.dataset_root,
        output_path=args.output,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
