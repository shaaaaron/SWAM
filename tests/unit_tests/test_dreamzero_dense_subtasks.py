# Copyright 2026 The RLinf Authors.

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rlinf.data.datasets.dreamzero import dreamzero as dreamzero_module
from rlinf.data.datasets.dreamzero.dense_subtasks import DenseSubtaskIndex


def _subtask_type(*, include_visual_clear=True, caption_fields=("action_detail_en",)):
    fields = [
        pa.field("segment_index", pa.int32()),
        pa.field("start_s", pa.float64()),
        pa.field("end_s", pa.float64()),
        pa.field("start_frame", pa.int32()),
        pa.field("end_frame", pa.int32()),
        *[pa.field(name, pa.string()) for name in caption_fields],
        pa.field("segment_type", pa.string()),
    ]
    if include_visual_clear:
        fields.append(pa.field("visual_clear", pa.bool_()))
    return pa.struct(fields)


def _segment(
    index,
    start,
    end,
    caption,
    *,
    segment_type="manipulation",
    visual_clear=True,
):
    return {
        "segment_index": index,
        "start_s": start / 30,
        "end_s": end / 30,
        "start_frame": start,
        "end_frame": end,
        "action_detail_en": caption,
        "segment_type": segment_type,
        "visual_clear": visual_clear,
    }


def _write_dense_subtasks(
    path, rows, *, include_visual_clear=True, caption_fields=("action_detail_en",)
):
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("episode_index", pa.int64()),
            pa.field("relative_video_path", pa.string()),
            pa.field(
                "subtasks",
                pa.list_(
                    _subtask_type(
                        include_visual_clear=include_visual_clear,
                        caption_fields=caption_fields,
                    )
                ),
            ),
        ]
    )
    if not include_visual_clear:
        rows = [
            {
                **row,
                "subtasks": [
                    {
                        key: value
                        for key, value in segment.items()
                        if key != "visual_clear"
                    }
                    for segment in row["subtasks"]
                ],
            }
            for row in rows
        ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _episode(index, subtasks):
    return {
        "episode_index": index,
        "relative_video_path": f"videos/episode_{index:06d}.mp4",
        "task_name": "test task",
        "subtasks": subtasks,
    }


def test_dense_subtask_index_filters_and_selects_maximum_overlap(tmp_path):
    path = tmp_path / "meta/subtask/subtasks_dense_seed.parquet"
    _write_dense_subtasks(
        path,
        [
            _episode(
                0,
                [
                    _segment(0, 0, 4, "  early   action  "),
                    _segment(1, 4, 10, "later action"),
                    _segment(2, 0, 10, "ignore no action", segment_type="no_action"),
                    _segment(3, 0, 10, "ignore unclear", visual_clear=False),
                    _segment(4, 0, 10, "   "),
                ],
            ),
            _episode(1, []),
        ],
    )

    index = DenseSubtaskIndex.load(tmp_path, "meta/subtask/subtasks_dense_seed.parquet")

    assert index.num_rows == 2
    assert index.total_segments == 5
    assert index.num_segments == 2
    assert index.num_filtered_segments == 3
    assert index.num_episodes == 1
    assert index.resolve(0, np.asarray([1, 2, 4, 5, 6])) == "later action"
    assert index.resolve(0, np.asarray([1, 2, 4, 5])) == "early action"
    assert index.resolve(0, np.asarray([1, 1, 4, 4])) == "early action"
    assert index.resolve(0, np.asarray([20, 21])) is None
    assert index.resolve(1, np.asarray([0, 1])) is None
    assert index.resolve(999, np.asarray([0, 1])) is None


def test_dense_subtask_index_validates_file_and_nested_schema(tmp_path):
    relative_path = "meta/subtask/subtasks_dense_seed.parquet"
    with pytest.raises(FileNotFoundError, match="requires parquet"):
        DenseSubtaskIndex.load(tmp_path, relative_path)

    path = tmp_path / relative_path
    _write_dense_subtasks(
        path,
        [_episode(0, [_segment(0, 0, 10, "action")])],
        include_visual_clear=False,
    )
    with pytest.raises(ValueError, match="missing subtask fields"):
        DenseSubtaskIndex.load(tmp_path, relative_path)


@pytest.mark.parametrize(
    "caption_field", ["action_detail_en", "action_sent_en", "action_short_en"]
)
def test_dense_subtask_index_selects_configured_file_and_caption(tmp_path, caption_field):
    dense_path = "meta/subtask/subtasks_dense_seed.parquet"
    all_path = "meta/subtask/subtasks_all_seed.parquet"
    _write_dense_subtasks(
        tmp_path / dense_path,
        [_episode(0, [_segment(0, 0, 10, "old dense caption")])],
    )
    captions = {
        "action_detail_en": "detailed action",
        "action_sent_en": "action sentence",
        "action_short_en": "short action",
    }
    _write_dense_subtasks(
        tmp_path / all_path,
        [_episode(0, [{**_segment(0, 0, 10, ""), **captions}])],
        caption_fields=tuple(captions),
    )

    index = DenseSubtaskIndex.load(tmp_path, all_path, caption_field=caption_field)

    assert index.resolve(0, np.asarray([0, 1])) == captions[caption_field]
    assert (
        DenseSubtaskIndex.load(tmp_path, dense_path).resolve(0, np.asarray([0, 1]))
        == "old dense caption"
    )


@pytest.mark.parametrize("caption_field", ["action_sent_en", "action_short_en"])
def test_dense_subtask_index_requires_only_the_selected_caption(tmp_path, caption_field):
    relative_path = "meta/subtask/subtasks_all_seed.parquet"
    _write_dense_subtasks(
        tmp_path / relative_path,
        [_episode(0, [{**_segment(0, 0, 10, ""), caption_field: "  selected  text "}])],
        caption_fields=(caption_field,),
    )

    index = DenseSubtaskIndex.load(tmp_path, relative_path, caption_field=caption_field)

    assert index.resolve(0, np.asarray([0, 1])) == "selected text"


@pytest.mark.parametrize("caption_field", ["action_sent_en", "action_short_en"])
def test_dense_subtask_index_rejects_missing_selected_caption(tmp_path, caption_field):
    relative_path = "meta/subtask/subtasks_dense_seed.parquet"
    _write_dense_subtasks(
        tmp_path / relative_path, [_episode(0, [_segment(0, 0, 10, "detail")])]
    )

    with pytest.raises(ValueError, match=f"missing subtask fields.*{caption_field}"):
        DenseSubtaskIndex.load(tmp_path, relative_path, caption_field=caption_field)


@pytest.mark.parametrize("caption_field", ["action_sent_en", "action_short_en"])
@pytest.mark.parametrize("empty_caption", [None, "", " \t\n "])
def test_dense_subtask_index_filters_empty_selected_caption(
    tmp_path, caption_field, empty_caption
):
    relative_path = "meta/subtask/subtasks_all_seed.parquet"
    segments = [
        {**_segment(0, 0, 10, "do not use detail"), caption_field: empty_caption},
        {**_segment(1, 8, 10, "other detail"), caption_field: "valid selected text"},
        {**_segment(2, 0, 10, "", visual_clear=False), caption_field: "unclear"},
        {**_segment(3, 0, 10, "", segment_type="no_action"), caption_field: "idle"},
    ]
    _write_dense_subtasks(
        tmp_path / relative_path,
        [_episode(0, segments)],
        caption_fields=("action_detail_en", caption_field),
    )

    index = DenseSubtaskIndex.load(tmp_path, relative_path, caption_field=caption_field)

    assert index.num_segments == 1
    assert index.resolve(0, np.asarray([0, 1])) is None
    assert index.resolve(0, np.asarray([0, 1, 8])) == "valid selected text"


@pytest.mark.parametrize("caption_field", ["action_detail", "scene_en", "", None])
def test_dense_subtask_index_rejects_unsupported_caption_field(tmp_path, caption_field):
    with pytest.raises(ValueError, match="subtask_caption_field must be"):
        DenseSubtaskIndex.load(tmp_path, "unused.parquet", caption_field=caption_field)


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [
                _episode(0, [_segment(0, 0, 10, "action")]),
                _episode(0, []),
            ],
            "duplicate episode_index",
        ),
        (
            [_episode(0, [_segment(0, 10, 10, "action")])],
            "invalid selected interval",
        ),
    ],
)
def test_dense_subtask_index_rejects_ambiguous_identity_and_intervals(
    tmp_path, rows, message
):
    path = tmp_path / "meta/subtask/subtasks_dense_seed.parquet"
    _write_dense_subtasks(path, rows)

    with pytest.raises(ValueError, match=message):
        DenseSubtaskIndex.load(tmp_path, "meta/subtask/subtasks_dense_seed.parquet")


@pytest.mark.parametrize(
    "mode, suffix",
    [("none", ""), ("scene_light", " A kitchen. Soft light."), ("state", " The cup is upright.")],
)
def test_caption_extra_preserves_filtering_sorting_and_overlap(tmp_path, mode, suffix):
    path = "meta/subtask/subtasks_all_seed_20260910.parquet"
    extra = {
        "scene_en": " A kitchen. ",
        "lighting_en": "Soft light.",
        "state_detail_en": "The cup is upright.",
    }
    _write_dense_subtasks(
        tmp_path / path,
        [
            _episode(
                7,
                [
                    {**_segment(1, 4, 10, "late action"), **extra},
                    {**_segment(0, 0, 4, " early   action "), "scene_en": "Other scene."},
                    {**_segment(2, 0, 20, " "), **extra},
                    {**_segment(3, 0, 20, "unclear", visual_clear=False), **extra},
                    {**_segment(4, 0, 20, "idle", segment_type="no_action"), **extra},
                ],
            )
        ],
        caption_fields=("action_detail_en", *extra),
    )
    index = DenseSubtaskIndex.load(tmp_path, path, caption_extra=mode)
    assert index.num_segments == 2
    assert index.num_filtered_segments == 3
    assert index.resolve(7, np.asarray([1, 4, 5, 6])) == "late action" + suffix
    early_suffix = " Other scene." if mode == "scene_light" else ""
    assert index.resolve(7, np.asarray([1, 2, 4, 5])) == "early action" + early_suffix
    assert index.resolve(7, np.asarray([11, 12])) is None
    assert index.resolve(8, np.asarray([1])) is None


@pytest.mark.parametrize("empty", [None, "", " \t\n "])
def test_empty_extra_keeps_original_caption(tmp_path, empty):
    path = "meta/subtask/all.parquet"
    _write_dense_subtasks(
        tmp_path / path,
        [_episode(0, [{**_segment(0, 0, 10, " original   action "), "state_detail_en": empty}])],
        caption_fields=("action_detail_en", "state_detail_en"),
    )
    baseline = DenseSubtaskIndex.load(tmp_path, path)
    appended = DenseSubtaskIndex.load(tmp_path, path, caption_extra="state")
    frames = np.asarray([0, 1])
    assert appended.resolve(0, frames) == baseline.resolve(0, frames) == "original action"


@pytest.mark.parametrize("mode", ["scene_light", "state"])
def test_caption_extra_requires_selected_fields(tmp_path, mode):
    path = "meta/subtask/dense.parquet"
    _write_dense_subtasks(tmp_path / path, [_episode(0, [_segment(0, 0, 10, "action")])])
    assert DenseSubtaskIndex.load(tmp_path, path).num_segments == 1
    with pytest.raises(ValueError, match="missing subtask fields"):
        DenseSubtaskIndex.load(tmp_path, path, caption_extra=mode)


@pytest.mark.parametrize("mode", ["all", "scene", "", None])
def test_caption_extra_rejects_unknown_mode(tmp_path, mode):
    with pytest.raises(ValueError, match="subtask_caption_extra must be"):
        DenseSubtaskIndex.load(tmp_path, "unused.parquet", caption_extra=mode)


def _write_episode_captions(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = [
            pa.field(name, pa.string())
            for name in ("action_detail_en", "scene_en", "lighting_en", "state_detail_en")
        ]
    schema = pa.schema([
        pa.field("episode_index", pa.int64()),
        pa.field("caption", pa.struct(fields)),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


@pytest.mark.parametrize("caption_field", [
    "action_detail_en", "action_sent_en", "action_short_en",
])
@pytest.mark.parametrize("field_type", [pa.string(), pa.large_string()])
@pytest.mark.parametrize("mode, first_suffix, second_suffix", [
    ("scene_light", " A kitchen. Soft light.", " A table. Bright light."),
    ("state", " The cup is raised.", " The cup is on the table."),
])
def test_episode_caption_extra_uses_same_episode_for_every_window(
    tmp_path, caption_field, field_type, mode, first_suffix, second_suffix
):
    path = "meta/subtask/episode_all_qwen.parquet"
    _write_episode_captions(
        tmp_path / path,
        [
            {"episode_index": 7, "caption": {
                caption_field: " Pick  up the cup. ",
                "scene_en": " A kitchen. ", "lighting_en": "Soft\nlight.",
                "state_detail_en": " The cup  is raised. ",
            }},
            {"episode_index": 2, "caption": {
                caption_field: "Place the cup.",
                "scene_en": "A table.", "lighting_en": "Bright light.",
                "state_detail_en": "The cup is on the table.",
            }},
        ],
        fields=[pa.field(name, field_type) for name in (
            caption_field, "scene_en", "lighting_en", "state_detail_en",
        )],
    )
    index = DenseSubtaskIndex.load(
        tmp_path, path, caption_field=caption_field,
        caption_granularity="episode", caption_extra=mode,
    )
    baseline = DenseSubtaskIndex.load(
        tmp_path, path, caption_field=caption_field, caption_granularity="episode",
    )
    for frames in (np.asarray([0, 1]), np.asarray([100, 200]), np.asarray([])):
        assert index.resolve(7, frames) == "Pick up the cup." + first_suffix
        assert index.resolve(2, frames) == "Place the cup." + second_suffix
        assert index.resolve(8, frames) == ""
        assert baseline.resolve(7, frames) == "Pick up the cup."
    assert index.num_episodes == 2 and index.num_segments == 0


def test_unified_dense_supports_robotwin_episode_captions():
    assert "robotwin" in dreamzero_module._DENSE_SUBTASK_SUPPORTED_TAGS


@pytest.mark.parametrize("mode, extras, suffix", [
    ("scene_light", {"scene_en": None, "lighting_en": "Bright light."}, " Bright light."),
    ("scene_light", {"scene_en": "A table.", "lighting_en": None}, " A table."),
    ("scene_light", {"scene_en": " \t ", "lighting_en": "\n"}, ""),
    ("scene_light", {"scene_en": None, "lighting_en": None}, ""),
    ("state", {"state_detail_en": None}, ""),
    ("state", {"state_detail_en": ""}, ""),
    ("state", {"state_detail_en": " \t\n "}, ""),
])
def test_episode_caption_extra_skips_empty_extra_values(tmp_path, mode, extras, suffix):
    path = "meta/subtask/episode_all_qwen.parquet"
    _write_episode_captions(tmp_path / path, [{
        "episode_index": 0, "caption": {
            "action_detail_en": " original   action ",
            **extras,
        },
    }])
    index = DenseSubtaskIndex.load(
        tmp_path, path, caption_granularity="episode", caption_extra=mode,
    )
    assert index.resolve(0, np.asarray([0])) == "original action" + suffix


@pytest.mark.parametrize("action", [None, "", " \t\n "])
@pytest.mark.parametrize("mode", ["scene_light", "state"])
def test_episode_caption_extra_preserves_empty_action_caption(tmp_path, action, mode):
    path = "meta/subtask/episode_all_qwen.parquet"
    _write_episode_captions(tmp_path / path, [
        {"episode_index": 0, "caption": {
            "action_detail_en": action,
            "scene_en": "A table.", "lighting_en": "Bright light.",
            "state_detail_en": "The cup is raised.",
        }},
        {"episode_index": 1, "caption": None},
    ])
    index = DenseSubtaskIndex.load(
        tmp_path, path, caption_granularity="episode", caption_extra=mode,
    )
    assert index.resolve(0, np.asarray([0])) == ""
    assert index.resolve(1, np.asarray([0])) == ""


@pytest.mark.parametrize("mode, field", [
    ("scene_light", "scene_en"), ("scene_light", "lighting_en"),
    ("state", "state_detail_en"),
])
@pytest.mark.parametrize("invalid_type", [None, pa.int32()])
def test_episode_caption_extra_validates_fields(tmp_path, mode, field, invalid_type):
    path = "meta/subtask/episode_all_qwen.parquet"
    fields = [pa.field("action_detail_en", pa.string())]
    for name in ("scene_en", "lighting_en", "state_detail_en"):
        if name != field:
            fields.append(pa.field(name, pa.string()))
        elif invalid_type is not None:
            fields.append(pa.field(name, invalid_type))
    _write_episode_captions(tmp_path / path, [{
        "episode_index": 0, "caption": {"action_detail_en": "Action."},
    }], fields=fields)
    # Default none does not require or validate unused extra fields.
    baseline = DenseSubtaskIndex.load(tmp_path, path, caption_granularity="episode")
    assert baseline.resolve(0, np.asarray([0])) == "Action."
    message = "missing caption field" if invalid_type is None else "invalid type"
    with pytest.raises(ValueError, match=message + ".*" + field):
        DenseSubtaskIndex.load(
            tmp_path, path, caption_granularity="episode", caption_extra=mode,
        )
