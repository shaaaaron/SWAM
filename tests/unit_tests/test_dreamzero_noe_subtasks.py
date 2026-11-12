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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rlinf.data.datasets.dreamzero.noe_subtasks import (
    NoeSubtaskConfig,
    NoeSubtaskIndex,
)


def _write_segments(path, rows, *, start_type=pa.int32()):
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("episode_index", pa.int32()),
            pa.field("start_frame", start_type),
            pa.field("end_frame", pa.int32()),
            pa.field("subtask_en", pa.string()),
            pa.field("subtask_cn", pa.string()),
            pa.field("quality", pa.int8()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _row(episode, start, end, text, *, quality=1):
    normalized_text = " ".join(text.split())
    return {
        "episode_index": episode,
        "start_frame": start,
        "end_frame": end,
        "subtask_en": text,
        "subtask_cn": f"cn:{normalized_text}",
        "quality": quality,
    }


def test_noe_subtask_config_supports_only_off_and_prefer():
    assert NoeSubtaskConfig.from_value(None).mode == "off"
    assert NoeSubtaskConfig.from_value({"mode": "prefer"}).mode == "prefer"
    assert (
        NoeSubtaskConfig.from_value({"mode": "prefer"}).missing_metadata_policy
        == "error"
    )
    assert (
        NoeSubtaskConfig.from_value(
            {"mode": "prefer", "missing_metadata_policy": "fallback"}
        ).missing_metadata_policy
        == "fallback"
    )

    with pytest.raises(ValueError, match="must be 'off' or 'prefer'"):
        NoeSubtaskConfig.from_value({"mode": "only"})

    with pytest.raises(ValueError, match="missing_metadata_policy"):
        NoeSubtaskConfig.from_value(
            {"mode": "prefer", "missing_metadata_policy": "ignore"}
        )


def test_noe_subtask_index_resolves_common_text_after_sampling(tmp_path):
    path = tmp_path / "meta" / "subtask_segments.parquet"
    _write_segments(
        path,
        [
            _row(0, 0, 96, "  shared   task  "),
            _row(0, 0, 144, "shared task"),
            _row(0, 0, 144, "ignored low quality", quality=0),
            _row(1, 0, 48, "first task"),
            _row(1, 48, 96, "second task"),
            _row(2, 0, 144, "candidate a"),
            _row(2, 0, 144, "candidate b"),
            _row(3, 10, 58, "exact boundary"),
        ],
    )
    config = NoeSubtaskConfig.from_value(
        {
            "mode": "prefer",
            "metadata_file": "meta/subtask_segments.parquet",
            "seed": 7,
        }
    )
    index = NoeSubtaskIndex.load(tmp_path, config)

    shared = index.resolve(0, 0, [(0, 48), (48, 96)])
    assert shared.text == "shared task"
    assert shared.status == "matched"
    assert shared.candidate_count == 1

    mixed = index.resolve(1, 0, [(0, 48), (48, 96)])
    assert mixed.text is None
    assert mixed.status == "mixed"

    ambiguous_1 = index.resolve(2, 12, [(12, 60), (60, 108)])
    ambiguous_2 = index.resolve(2, 12, [(12, 60), (60, 108)])
    assert ambiguous_1 == ambiguous_2
    assert ambiguous_1.text in {"candidate a", "candidate b"}
    assert ambiguous_1.status == "ambiguous_random"
    assert ambiguous_1.candidate_count == 2

    exact = index.resolve(3, 10, [(10, 58)])
    assert exact.text == "exact boundary"
    assert index.resolve(3, 11, [(11, 59)]).status == "no_match"
    assert index.resolve(999, 0, [(0, 48)]).status == "no_match"

    chinese_index = NoeSubtaskIndex.load(
        tmp_path,
        NoeSubtaskConfig.from_value(
            {
                "mode": "prefer",
                "instruction_key": "subtask_cn",
            }
        ),
    )
    chinese = chinese_index.resolve(0, 0, [(0, 48), (48, 96)])
    assert chinese.text == "cn:shared task"


def test_noe_subtask_index_validates_file_and_schema(tmp_path):
    config = NoeSubtaskConfig.from_value({"mode": "prefer"})
    with pytest.raises(FileNotFoundError, match="需要元数据 parquet"):
        NoeSubtaskIndex.load(tmp_path, config)

    path = tmp_path / "meta" / "subtask_segments.parquet"
    _write_segments(path, [_row(0, 0, 48, "task")], start_type=pa.int64())
    with pytest.raises(ValueError, match="字段类型无效"):
        NoeSubtaskIndex.load(tmp_path, config)


def test_noe_subtask_index_warns_and_skips_invalid_quality(tmp_path):
    path = tmp_path / "meta" / "subtask_segments.parquet"
    config = NoeSubtaskConfig.from_value({"mode": "prefer"})

    null_quality_row = _row(0, 0, 48, "task")
    null_quality_row["quality"] = None
    _write_segments(
        path,
        [
            _row(0, 0, 48, "valid task"),
            _row(1, 0, 48, "invalid task", quality=2),
            null_quality_row,
        ],
    )
    with pytest.warns(RuntimeWarning, match=r"2 行 quality 非法"):
        index = NoeSubtaskIndex.load(tmp_path, config)

    assert index.num_segments == 1
    assert index.resolve(0, 0, [(0, 48)]).text == "valid task"
    assert index.resolve(1, 0, [(0, 48)]).text is None
