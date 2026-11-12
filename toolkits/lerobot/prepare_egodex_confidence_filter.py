#!/usr/bin/env python3
"""扫描 EgoDex confidence 并生成 Dataset 初始化使用的 episode sidecar。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from rlinf.data.datasets.dreamzero.egodex_confidence import (
    EGODEX_CONFIDENCE_SIDECAR_NAME,
    EGODEX_CONFIDENCE_SOURCE_KEY,
    EGODEX_CONFIDENCE_STATUS_ERROR,
    EGODEX_CONFIDENCE_STATUS_VALID,
    EGODEX_CONFIDENCE_STATUSES,
    classify_egodex_confidence,
)


def discover_task_roots(data_root: Path) -> list[Path]:
    """输入 EgoDex 集合目录，返回按路径排序的 LeRobot task roots。"""
    roots = sorted(path.parent.parent for path in data_root.rglob("meta/info.json"))
    if not roots:
        raise FileNotFoundError(f"No EgoDex task roots found under {data_root}")
    return roots


def load_episode_records(task_root: Path) -> list[dict[str, Any]]:
    """读取单个 task 的 episodes.jsonl，并校验 episode index 唯一。"""
    episodes_path = task_root / "meta" / "episodes.jsonl"
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    with open(episodes_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            episode_index = record.get("episode_index")
            if type(episode_index) is not int or episode_index < 0:
                raise ValueError(
                    f"Invalid episode_index at {episodes_path}:{line_number}: "
                    f"{episode_index!r}"
                )
            if episode_index in seen:
                raise ValueError(
                    f"Duplicate episode_index={episode_index} in {episodes_path}"
                )
            seen.add(episode_index)
            records.append(record)
    if not records:
        raise ValueError(f"No episode records found in {episodes_path}")
    return records


def discover_episode_paths(task_root: Path) -> dict[int, Path]:
    """发现单个 task 的 parquet，并校验文件名中的 episode index 唯一。"""
    paths: dict[int, Path] = {}
    for path in (task_root / "data").rglob("episode_*.parquet"):
        episode_index = int(path.stem.removeprefix("episode_"))
        if episode_index in paths:
            raise ValueError(
                f"Duplicate parquet episode_index={episode_index} under {task_root}: "
                f"{paths[episode_index]} and {path}"
            )
        paths[episode_index] = path
    return paths


def _declared_episode_length(record: dict[str, Any]) -> int | None:
    """从 episode metadata 中读取声明长度，缺失时返回 None。"""
    for key in ("episode_length", "length", "num_frames", "num_steps"):
        if record.get(key) is not None:
            length = int(record[key])
            if length <= 0:
                raise ValueError(
                    f"Episode {record['episode_index']} has invalid {key}={length}"
                )
            return length
    return None


def scan_episode(item: tuple[str, int, Path | None, int | None]) -> dict[str, Any]:
    """读取一个 episode 的 confidence，并返回统一 sidecar 记录和帧数。"""
    task_key, episode_index, parquet_path, declared_length = item
    try:
        if parquet_path is None:
            raise FileNotFoundError(
                f"Parquet is missing for episode_index={episode_index}"
            )
        table = pq.read_table(parquet_path, columns=[EGODEX_CONFIDENCE_SOURCE_KEY])
        expected_length = table.num_rows if declared_length is None else declared_length
        status = classify_egodex_confidence(
            table[EGODEX_CONFIDENCE_SOURCE_KEY],
            expected_length=expected_length,
            source=str(parquet_path),
        )
        return {
            "task_key": task_key,
            "episode_index": episode_index,
            "confidence_status": status,
            "error_reason": "",
            "frames": int(table.num_rows),
        }
    except Exception as exc:
        return {
            "task_key": task_key,
            "episode_index": episode_index,
            "confidence_status": EGODEX_CONFIDENCE_STATUS_ERROR,
            "error_reason": f"{type(exc).__name__}: {exc}",
            "frames": int(declared_length or 0),
        }


def scan_dataset(
    data_root: Path,
    *,
    workers: int,
    batch_size: int = 4096,
    progress_every: int = 10_000,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """并行扫描全部 task，返回逐 task sidecar 记录和全局审计统计。"""
    task_roots = discover_task_roots(data_root)
    work_items: list[tuple[str, int, Path | None, int | None]] = []
    declared_by_task: dict[str, set[int]] = {}
    extra_parquets: list[str] = []
    for task_root in task_roots:
        task_key = task_root.relative_to(data_root).as_posix()
        records = load_episode_records(task_root)
        paths = discover_episode_paths(task_root)
        declared = {int(record["episode_index"]) for record in records}
        declared_by_task[task_key] = declared
        extra_parquets.extend(
            str(paths[index]) for index in sorted(set(paths) - declared)
        )
        for record in records:
            episode_index = int(record["episode_index"])
            work_items.append(
                (
                    task_key,
                    episode_index,
                    paths.get(episode_index),
                    _declared_episode_length(record),
                )
            )

    records_by_task = {task_key: [] for task_key in declared_by_task}
    status_counts = dict.fromkeys(EGODEX_CONFIDENCE_STATUSES, 0)
    status_frames = dict.fromkeys(EGODEX_CONFIDENCE_STATUSES, 0)
    started = time.monotonic()
    processed = 0
    next_progress = progress_every
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        for start in range(0, len(work_items), batch_size):
            batch = work_items[start : start + batch_size]
            for result in executor.map(scan_episode, batch):
                task_key = result.pop("task_key")
                frames = int(result.pop("frames"))
                records_by_task[task_key].append(result)
                status = result["confidence_status"]
                status_counts[status] += 1
                status_frames[status] += frames
                processed += 1
                if progress_every > 0 and processed >= next_progress:
                    elapsed = time.monotonic() - started
                    print(
                        f"[egodex confidence] episodes={processed}/{len(work_items)} "
                        f"rate={processed / elapsed:.1f}/s",
                        flush=True,
                    )
                    next_progress += progress_every

    task_summaries: dict[str, dict[str, Any]] = {}
    for task_key, records in records_by_task.items():
        records.sort(key=lambda record: int(record["episode_index"]))
        counts = dict.fromkeys(EGODEX_CONFIDENCE_STATUSES, 0)
        for record in records:
            counts[record["confidence_status"]] += 1
        task_summaries[task_key] = {
            "episodes": len(records),
            "kept": counts[EGODEX_CONFIDENCE_STATUS_VALID],
            "kept_ratio": counts[EGODEX_CONFIDENCE_STATUS_VALID] / len(records),
            "status_counts": counts,
        }

    elapsed = time.monotonic() - started
    summary = {
        "data_root": str(data_root),
        "task_roots": len(task_roots),
        "episodes": len(work_items),
        "frames": sum(status_frames.values()),
        "kept": status_counts[EGODEX_CONFIDENCE_STATUS_VALID],
        "kept_frames": status_frames[EGODEX_CONFIDENCE_STATUS_VALID],
        "kept_episode_ratio": (
            status_counts[EGODEX_CONFIDENCE_STATUS_VALID] / len(work_items)
        ),
        "kept_frame_ratio": (
            status_frames[EGODEX_CONFIDENCE_STATUS_VALID] / sum(status_frames.values())
        ),
        "status_counts": status_counts,
        "status_frames": status_frames,
        "extra_parquets": extra_parquets,
        "empty_tasks": sorted(
            task_key
            for task_key, task_summary in task_summaries.items()
            if task_summary["kept"] == 0
        ),
        "tasks": task_summaries,
        "workers": max(1, int(workers)),
        "elapsed_seconds": elapsed,
    }
    return records_by_task, summary


def _sidecar_path(
    data_root: Path,
    task_key: str,
    output_root: Path | None,
) -> Path:
    """根据是否使用暂存根目录，返回单个 task 的 sidecar 输出路径。"""
    root = data_root if output_root is None else output_root
    return root / task_key / "meta" / EGODEX_CONFIDENCE_SIDECAR_NAME


def _check_free_space(path: Path, min_free_gb: float) -> None:
    """写入前检查目标文件系统剩余空间，不足时立即停止。"""
    probe = path
    while not probe.exists():
        probe = probe.parent
    free_bytes = shutil.disk_usage(probe).free
    required_bytes = int(float(min_free_gb) * (1024**3))
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient free space under {probe}: "
            f"free={free_bytes / (1024**3):.2f} GiB, "
            f"required={min_free_gb:.2f} GiB"
        )


def write_outputs(
    data_root: Path,
    records_by_task: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
    *,
    output_root: Path | None,
    summary_json: Path,
    overwrite: bool,
    min_free_gb: float,
) -> None:
    """原子写入所有 sidecar 和审计 summary，避免留下半成品文件。"""
    targets = [
        _sidecar_path(data_root, task_key, output_root) for task_key in records_by_task
    ]
    targets.append(summary_json)
    _check_free_space(targets[0], min_free_gb)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing outputs; "
            f"first={existing[0]}"
        )

    for task_key, records in records_by_task.items():
        target = _sidecar_path(data_root, task_key, output_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        with open(temporary, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, target)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_json.with_name(f".{summary_json.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_json)


def main() -> None:
    """解析命令行参数，扫描 confidence 并写入 sidecar 与审计文件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else None
    records_by_task, summary = scan_dataset(
        data_root,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    write_outputs(
        data_root,
        records_by_task,
        summary,
        output_root=output_root,
        summary_json=args.summary_json,
        overwrite=args.overwrite,
        min_free_gb=args.min_free_gb,
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "tasks"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
