import json

import pytest

pytest.importorskip("pyarrow")

import pyarrow.parquet as pq  # noqa: E402

from toolkits.lerobot.build_noe_subtask_segments import (  # noqa: E402
    SUBTASK_SCHEMA,
    build_subtask_index,
    collect_subtask_rows,
)


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_fixture(dataset_root):
    meta_dir = dataset_root / "meta"
    subtask_dir = dataset_root / "subtask"
    meta_dir.mkdir(parents=True)
    subtask_dir.mkdir()

    episodes = [
        {
            "episode_index": 42,
            "episode_uuid": "episode-good",
            "length": 10,
        },
        {
            "episode_index": 43,
            "episode_uuid": "episode-failed",
            "length": 10,
            "status": "failed",
        },
        {
            "episode_index": 44,
            "episode_uuid": "episode-empty",
            "length": 0,
        },
    ]
    (meta_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes),
        encoding="utf-8",
    )

    _write_json(
        subtask_dir / "episode-good.json",
        {
            "episode_uuid": "episode-good",
            "clip_descriptions": {
                "1_5": {
                    "subtask_en": "Pick up the object",
                    "subtask_cn": "拿起物体",
                    "quality": 1,
                },
                "5_11": {
                    "subtask_en": "Put down the object",
                    "subtask_cn": "放下物体",
                    "quality": 0,
                },
                "3_3": {
                    "subtask_en": "Zero length",
                    "subtask_cn": "零长度",
                    "quality": 1,
                },
                "1_20": {
                    "subtask_en": "Out of bounds",
                    "subtask_cn": "越界",
                    "quality": 1,
                },
            },
        },
    )
    _write_json(
        subtask_dir / "episode-failed.json",
        {
            "episode_uuid": "episode-failed",
            "clip_descriptions": {
                "1_5": {
                    "subtask_en": "Failed episode",
                    "subtask_cn": "失败片段",
                    "quality": 1,
                }
            },
        },
    )
    _write_json(
        subtask_dir / "episode-empty.json",
        {
            "episode_uuid": "episode-empty",
            "clip_descriptions": {
                "1_5": {
                    "subtask_en": "Empty episode",
                    "subtask_cn": "空片段",
                    "quality": 1,
                }
            },
        },
    )
    _write_json(
        subtask_dir / "episode-orphan.json",
        {
            "episode_uuid": "episode-orphan",
            "clip_descriptions": {
                "1_5": {
                    "subtask_en": "Orphan episode",
                    "subtask_cn": "孤立片段",
                    "quality": 1,
                }
            },
        },
    )


def test_collect_subtask_rows_normalizes_and_filters(tmp_path):
    _write_fixture(tmp_path)

    rows, stats = collect_subtask_rows(tmp_path)

    assert rows == [
        {
            "episode_index": 42,
            "episode_uuid": "episode-good",
            "start_frame": 0,
            "end_frame": 4,
            "subtask_en": "Pick up the object",
            "subtask_cn": "拿起物体",
            "quality": 1,
        },
        {
            "episode_index": 42,
            "episode_uuid": "episode-good",
            "start_frame": 4,
            "end_frame": 10,
            "subtask_en": "Put down the object",
            "subtask_cn": "放下物体",
            "quality": 0,
        },
    ]
    assert stats.source_subtask_files == 4
    assert stats.source_segments == 7
    assert stats.emitted_segments == 2
    assert stats.orphan_files == 1
    assert stats.orphan_segments == 1
    assert stats.skipped_episode_files == 2
    assert stats.skipped_episode_segments == 2
    assert stats.zero_length_segments == 1
    assert stats.out_of_bounds_segments == 1
    assert stats.quality_0_segments == 1
    assert stats.quality_1_segments == 1


def test_build_subtask_index_writes_expected_parquet(tmp_path):
    _write_fixture(tmp_path)
    output_path = tmp_path / "meta" / "subtask_segments.parquet"

    summary = build_subtask_index(tmp_path)

    assert summary["emitted_segments"] == 2
    assert summary["output_path"] == str(output_path)
    table = pq.read_table(output_path)
    assert table.schema == SUBTASK_SCHEMA
    assert table.to_pylist()[0]["subtask_cn"] == "拿起物体"

    with pytest.raises(FileExistsError):
        build_subtask_index(tmp_path)


def test_build_subtask_index_dry_run_does_not_write(tmp_path):
    _write_fixture(tmp_path)
    output_path = tmp_path / "meta" / "subtask_segments.parquet"

    summary = build_subtask_index(tmp_path, dry_run=True)

    assert summary["dry_run"] is True
    assert summary["emitted_segments"] == 2
    assert not output_path.exists()
