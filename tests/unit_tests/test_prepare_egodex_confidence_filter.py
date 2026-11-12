# Copyright 2026 The RLinf Authors.

import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from rlinf.data.datasets.dreamzero.egodex_confidence import (  # noqa: E402
    EGODEX_CONFIDENCE_SIDECAR_NAME,
    EGODEX_CONFIDENCE_STATUS_CONTAINS_ZERO,
    EGODEX_CONFIDENCE_STATUS_MISSING_ALL_ONE,
    EGODEX_CONFIDENCE_STATUS_VALID,
    classify_egodex_confidence,
    filter_egodex_episode_indices,
)
from toolkits.lerobot.prepare_egodex_confidence_filter import (  # noqa: E402
    scan_dataset,
)


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _write_confidence_parquet(path, confidence):
    confidence = np.asarray(confidence, dtype=np.float32)
    pq.write_table(
        pa.Table.from_pydict(
            {
                "observation.state_confidence": pa.array(
                    confidence.tolist(), type=pa.list_(pa.float32())
                )
            }
        ),
        path,
    )


def test_classify_egodex_confidence_returns_mutually_exclusive_statuses():
    assert (
        classify_egodex_confidence(
            [[0.9, 0.8], [0.7, 0.6]], expected_length=2, source="test"
        )
        == EGODEX_CONFIDENCE_STATUS_VALID
    )
    assert (
        classify_egodex_confidence(
            [[1.0, 1.0], [1.0, 1.0]], expected_length=2, source="test"
        )
        == EGODEX_CONFIDENCE_STATUS_MISSING_ALL_ONE
    )
    assert (
        classify_egodex_confidence(
            [[0.9, 0.0], [0.7, 0.6]], expected_length=2, source="test"
        )
        == EGODEX_CONFIDENCE_STATUS_CONTAINS_ZERO
    )


@pytest.mark.parametrize(
    "confidence",
    [
        [[0.9], [0.8]],
        [[0.9, np.nan], [0.8, 0.7]],
        [[0.9, 1.1], [0.8, 0.7]],
    ],
)
def test_classify_egodex_confidence_rejects_invalid_values(confidence):
    with pytest.raises(ValueError):
        classify_egodex_confidence(confidence, expected_length=2, source="invalid-test")


def test_filter_egodex_episode_indices_keeps_only_valid_declared_records(tmp_path):
    _write_jsonl(
        tmp_path / EGODEX_CONFIDENCE_SIDECAR_NAME,
        [
            {"episode_index": 0, "confidence_status": "valid", "error_reason": ""},
            {
                "episode_index": 1,
                "confidence_status": "missing_all_one",
                "error_reason": "",
            },
            {
                "episode_index": 2,
                "confidence_status": "contains_zero",
                "error_reason": "",
            },
            {
                "episode_index": 3,
                "confidence_status": "error",
                "error_reason": "broken parquet",
            },
            {"episode_index": 5, "confidence_status": 7, "error_reason": ""},
        ],
    )

    kept = filter_egodex_episode_indices(tmp_path, range(6))

    assert kept == {0}


def test_scan_dataset_reports_episode_and_frame_retention(tmp_path):
    task_root = tmp_path / "part1" / "task_a"
    meta_dir = task_root / "meta"
    data_dir = task_root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text("{}", encoding="utf-8")
    _write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {"episode_index": 0, "length": 2},
            {"episode_index": 1, "length": 3},
        ],
    )
    _write_confidence_parquet(
        data_dir / "episode_000000.parquet", [[0.9, 0.8], [0.7, 0.6]]
    )
    _write_confidence_parquet(
        data_dir / "episode_000001.parquet",
        [[0.9, 0.8], [0.7, 0.0], [0.6, 0.5]],
    )

    records_by_task, summary = scan_dataset(tmp_path, workers=2, progress_every=0)

    assert [
        record["confidence_status"] for record in records_by_task["part1/task_a"]
    ] == [
        "valid",
        "contains_zero",
    ]
    assert summary["episodes"] == 2
    assert summary["frames"] == 5
    assert summary["kept"] == 1
    assert summary["kept_frames"] == 2
    assert summary["empty_tasks"] == []
