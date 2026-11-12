# Copyright 2026 The RLinf Authors.

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("groot")

from rlinf.data.datasets.dreamzero import dreamzero as dreamzero_module  # noqa: E402
from rlinf.data.datasets.dreamzero.data_transforms.dream_transform import (  # noqa: E402
    DreamTransform,
    DreamTransformBase,
)
from rlinf.data.datasets.dreamzero.dreamzero import (  # noqa: E402
    DreamZeroLeRobotDataset,
    _resolve_dreamzero_data_paths,
)
from rlinf.data.datasets.dreamzero.relative_actions import (  # noqa: E402
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
)

VIDEO_KEYS = [
    "video.image",
    "video.wrist_image_1",
    "video.wrist_image_2",
]


class _CaptureLogger:
    def __init__(self):
        self.info_calls = []
        self.warning_calls = []

    def info(self, *args):
        self.info_calls.append(args)

    def warning(self, *args):
        self.warning_calls.append(args)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_noe_umi_tree(root):
    meta_dir = root / "meta"
    data_dir = root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    video_path = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    video_features = {
        "observation.images.image": {
            "dtype": "video",
            "shape": [240, 320, 3],
        },
        "observation.images.wrist_image_1": {
            "dtype": "video",
            "shape": [320, 240, 3],
        },
        "observation.images.wrist_image_2": {
            "dtype": "video",
            "shape": [320, 240, 3],
        },
    }
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_episodes": 4,
        "total_frames": 26,
        "total_videos": 12,
        "chunks_size": 1000,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": video_path,
        "features": {
            **video_features,
            "observation.state": {"dtype": "float32", "shape": [23]},
            "action": {"dtype": "float32", "shape": [23]},
            "task_index": {"dtype": "int64"},
        },
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f)

    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": "test"}])
    _write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {
                "episode_index": 0,
                "length": 3,
                "task_annotation": "qualified",
                "slam_validity_all": True,
                "width_detection": True,
                "width_calibration": True,
            },
            {
                "episode_index": 1,
                "length": 5,
                "task_annotation": "slam invalid",
                "slam_validity_all": False,
                "width_detection": True,
                "width_calibration": True,
            },
            {
                "episode_index": 2,
                "length": 7,
                "task_annotation": "missing field",
                "slam_validity_all": True,
                "width_calibration": True,
            },
            {
                "episode_index": 3,
                "length": 11,
                "task_annotation": "invalid type",
                "slam_validity_all": True,
                "width_detection": "true",
                "width_calibration": True,
            },
        ],
    )

    for episode_index in range(4):
        (data_dir / f"episode_{episode_index:06d}.parquet").write_bytes(b"")
        for video_key in video_features:
            path = root / video_path.format(
                episode_chunk=0,
                video_key=video_key,
                episode_index=episode_index,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")


def _write_subtask_segments(root):
    schema = pa.schema(
        [
            pa.field("episode_index", pa.int32()),
            pa.field("start_frame", pa.int32()),
            pa.field("end_frame", pa.int32()),
            pa.field("subtask_en", pa.string()),
            pa.field("subtask_cn", pa.string()),
            pa.field("quality", pa.int8()),
        ]
    )
    table = pa.Table.from_pylist(
        [
            {
                "episode_index": 0,
                "start_frame": 0,
                "end_frame": 3,
                "subtask_en": "fine-grained instruction",
                "subtask_cn": "细粒度指令",
                "quality": 1,
            }
        ],
        schema=schema,
    )
    pq.write_table(table, root / "meta" / "subtask_segments.parquet")


def _write_dense_subtasks(
    root, *, caption_field="action_detail_en", caption="dense manipulation", extra=None
):
    extra = extra or {}
    subtask_type = pa.struct(
        [
            pa.field("segment_index", pa.int32()),
            pa.field("start_s", pa.float64()),
            pa.field("end_s", pa.float64()),
            pa.field("start_frame", pa.int32()),
            pa.field("end_frame", pa.int32()),
            pa.field(caption_field, pa.string()),
            *[pa.field(field, pa.string()) for field in extra],
            pa.field("segment_type", pa.string()),
            pa.field("visual_clear", pa.bool_()),
        ]
    )
    rows = [
        {
            "episode_index": 0,
            "relative_video_path": "videos/episode_000000.mp4",
            "task_name": "dense episode fallback",
            "subtasks": [
                {
                    "segment_index": 0,
                    "start_s": 0.0,
                    "end_s": 0.1,
                    "start_frame": 0,
                    "end_frame": 3,
                    caption_field: caption,
                    **extra,
                    "segment_type": "manipulation",
                    "visual_clear": True,
                }
            ],
        },
        {
            "episode_index": 1,
            "relative_video_path": "videos/episode_000001.mp4",
            "task_name": "dense episode fallback",
            "subtasks": [
                {
                    "segment_index": 0,
                    "start_s": 0.0,
                    "end_s": 1 / 6,
                    "start_frame": 0,
                    "end_frame": 5,
                    caption_field: "no action",
                    "segment_type": "no_action",
                    "visual_clear": True,
                }
            ],
        },
    ]
    schema = pa.schema(
        [
            pa.field("episode_index", pa.int64()),
            pa.field("relative_video_path", pa.string()),
            pa.field("task_name", pa.string()),
            pa.field("subtasks", pa.list_(subtask_type)),
        ]
    )
    path = root / "meta" / "subtask" / "subtasks_dense_seed.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _build_dataset(
    root,
    *,
    filter_qualified,
    quality_mode="episode",
    lazy_load=True,
    relative_action=False,
    relative_action_mode=None,
    action_horizon=48,
    noe_subtask=None,
    subtask_mode="legacy",
    subtask_file="meta/subtask/subtasks_dense_seed.parquet",
    subtask_caption_field="action_detail_en",
    subtask_caption_extra="none",
    caption_granularity="subtask",
    episode_instruction_key="task_annotation",
    episode_discovery_mode="scan",
    pq_cache_max_episodes=128,
):
    return DreamZeroLeRobotDataset(
        data_path=str(root),
        video_keys=VIDEO_KEYS,
        state_keys=["state.eef_position"],
        action_keys=["action.eef_position"],
        language_keys=["annotation.language.action_text"],
        data_transform=lambda raw: raw,
        lazy_load=lazy_load,
        num_frames=33,
        state_horizon=1,
        action_horizon=action_horizon,
        max_chunk_size=4,
        relative_action=relative_action,
        relative_action_keys=["eef_position"] if relative_action else None,
        relative_action_mode=relative_action_mode,
        sampling_mode="multi_anchor",
        multi_anchor_macro_stride=48,
        multi_anchor_video_offsets=[0, 6, 12, 18, 24, 30, 36, 42],
        embodiment_tag="noe_umi_16d",
        episode_instruction_key=episode_instruction_key,
        filter_noe_umi_qualified_episodes=filter_qualified,
        noe_umi_quality_mode=quality_mode,
        noe_subtask=noe_subtask,
        subtask_mode=subtask_mode,
        subtask_file=subtask_file,
        subtask_caption_field=subtask_caption_field,
        subtask_caption_extra=subtask_caption_extra,
        caption_granularity=caption_granularity,
        episode_discovery_mode=episode_discovery_mode,
        pq_cache_max_episodes=pq_cache_max_episodes,
    )


def test_noe_umi_qualification_filter_is_disabled_by_default(tmp_path):
    _write_noe_umi_tree(tmp_path)

    dataset = _build_dataset(tmp_path, filter_qualified=False)

    assert dataset._episodes == [0, 1, 2, 3]
    assert dataset._episode_lengths == [3, 5, 7, 11]
    assert len(dataset) == 26


def test_noe_umi_metadata_discovery_skips_recursive_file_scan(
    tmp_path, monkeypatch
):
    _write_noe_umi_tree(tmp_path)

    def fail_rglob(*_args, **_kwargs):
        pytest.fail("metadata discovery must not recursively scan the data tree")

    monkeypatch.setattr(Path, "rglob", fail_rglob)
    dataset = _build_dataset(
        tmp_path,
        filter_qualified=False,
        episode_discovery_mode="metadata",
    )

    assert dataset._episodes == [0, 1, 2, 3]
    assert dataset._episode_lengths == [3, 5, 7, 11]


def test_noe_umi_requires_episode_index(tmp_path):
    _write_noe_umi_tree(tmp_path)
    episodes_path = tmp_path / "meta" / "episodes.jsonl"
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    rows[0].pop("episode_index")
    _write_jsonl(episodes_path, rows)

    with pytest.raises(KeyError, match="episode_index"):
        _build_dataset(tmp_path, filter_qualified=False)


def test_episode_instruction_empty_values_warn_once_when_metadata_loads(
    tmp_path, monkeypatch
):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    _write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {"episode_index": 0, "task_annotation": "pick up the cup"},
            {"episode_index": 1, "task_annotation": ""},
            {"episode_index": 2, "task_annotation": "   "},
            {"episode_index": 3},
        ],
    )
    capture_logger = _CaptureLogger()
    monkeypatch.setattr(dreamzero_module, "logger", capture_logger)
    dataset = DreamZeroLeRobotDataset.__new__(DreamZeroLeRobotDataset)
    dataset.episode_instruction_key = "task_annotation"
    dataset.embodiment_tag = "noe_umi_16d"
    dataset.data_path = str(tmp_path)
    dataset._meta_dir = meta_dir

    texts = dataset._load_episode_instruction_texts()

    assert texts[0] == "pick up the cup"
    assert texts[1] == ""
    assert texts[2] == "   "
    assert texts[3] is None
    assert len(capture_logger.warning_calls) == 1
    warning_call = capture_logger.warning_calls[0]
    assert warning_call[3:7] == (3, 4, 75.0, [1, 2, 3])


def test_episode_instruction_resolution_returns_empty_for_unknown_indices(monkeypatch):
    capture_logger = _CaptureLogger()
    monkeypatch.setattr(dreamzero_module, "logger", capture_logger)
    dataset = DreamZeroLeRobotDataset.__new__(DreamZeroLeRobotDataset)
    dataset.episode_instruction_key = "task_annotation"
    dataset.data_path = "/dataset"
    dataset._episode_instruction_texts = {1: "", 2: "  valid instruction  "}

    assert dataset._resolve_episode_instruction_text({"episode_index": 1}) == ""
    assert (
        dataset._resolve_episode_instruction_text({"episode_index": 2})
        == "valid instruction"
    )
    assert dataset._resolve_episode_instruction_text({"episode_index": 3}) == ""
    assert dataset._resolve_episode_instruction_text({"episode_index": 4}) == ""
    assert capture_logger.warning_calls == []


@pytest.mark.parametrize(
    "cache_size, expected_reads, expected_entries",
    [(0, 2, 0), (1, 1, 1)],
)
def test_episode_parquet_cache_supports_disabled_mode(
    tmp_path, monkeypatch, cache_size, expected_reads, expected_entries
):
    _write_noe_umi_tree(tmp_path)
    parquet_path = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    pq.write_table(pa.table({"task_index": [0, 0, 0]}), parquet_path)
    dataset = _build_dataset(
        tmp_path,
        filter_qualified=False,
        pq_cache_max_episodes=cache_size,
    )

    read_calls = []
    real_read_table = pq.read_table

    def tracked_read_table(*args, **kwargs):
        read_calls.append(args[0])
        return real_read_table(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", tracked_read_table)
    first = dataset._get_episode_table(0)
    second = dataset._get_episode_table(0)

    assert first.column_names == ["task_index"]
    assert second.column_names == ["task_index"]
    assert len(read_calls) == expected_reads
    assert len(dataset._pq_cache) == expected_entries


def test_episode_parquet_cache_rejects_negative_size(tmp_path):
    _write_noe_umi_tree(tmp_path)

    with pytest.raises(ValueError, match="must be non-negative"):
        _build_dataset(
            tmp_path,
            filter_qualified=False,
            pq_cache_max_episodes=-1,
        )


def test_episode_language_labels_reuse_supplied_table(tmp_path, monkeypatch):
    _write_noe_umi_tree(tmp_path)
    dataset = _build_dataset(
        tmp_path,
        filter_qualified=False,
        pq_cache_max_episodes=0,
    )
    dataset.episode_instruction_key = None
    table = pa.table({"task_index": [0, 1, 1]})

    monkeypatch.setattr(
        dataset,
        "_get_episode_table",
        lambda _episode_index: pytest.fail("supplied table should be reused"),
    )

    labels = dataset._read_episode_language_labels(0, episode_table=table)

    assert labels.tolist() == [0, 1, 1]


def test_noe_umi_frame_quality_requires_enable_switch(tmp_path, monkeypatch):
    _write_noe_umi_tree(tmp_path)
    parquet_path = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    pq.write_table(pa.table({"task_index": [0]}), parquet_path)
    capture_logger = _CaptureLogger()
    monkeypatch.setattr(dreamzero_module, "logger", capture_logger)

    dataset = _build_dataset(
        tmp_path,
        filter_qualified=False,
        quality_mode="frame",
    )
    table = dataset._get_episode_table(0)

    assert dataset._uses_noe_umi_frame_quality() is False
    assert table.column_names == ["task_index"]
    assert dataset._noe_umi_quality_fallback_warned is False
    assert capture_logger.warning_calls == []
    assert dataset._noe_umi_quality_masks(
        0,
        table,
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
    ) is None


def test_noe_umi_qualification_filter_keeps_only_all_true(tmp_path, monkeypatch):
    _write_noe_umi_tree(tmp_path)
    capture_logger = _CaptureLogger()
    monkeypatch.setattr(dreamzero_module, "logger", capture_logger)

    dataset = _build_dataset(tmp_path, filter_qualified=True)

    assert dataset._episodes == [0]
    assert dataset._episode_lengths == [3]
    assert len(dataset) == 3
    assert len(capture_logger.warning_calls) == 1
    assert capture_logger.warning_calls[0][1] == 2
    assert any(
        call[0].startswith("NOE/UMI qualification filter:")
        for call in capture_logger.info_calls
    )


def test_noe_umi_qualification_filter_rejects_non_lazy_loading(tmp_path):
    _write_noe_umi_tree(tmp_path)

    with pytest.raises(ValueError, match="requires the lazy local MP4 dataset path"):
        _build_dataset(tmp_path, filter_qualified=True, lazy_load=False)


def test_noe_umi_frame_quality_uses_complete_parquet_fields(tmp_path):
    _write_noe_umi_tree(tmp_path)
    dataset = _build_dataset(
        tmp_path,
        filter_qualified=True,
        quality_mode="frame",
        action_horizon=3,
    )
    table = pa.table(
        {
            "slam_diagnostics_valid_mask.right": [True, True, True, False, True, True, True],
            "slam_diagnostics_valid_mask.left": [True, True, True, False, True, True, True],
            "slam_diagnostics_valid_mask.ego": [True, True, True, False, True, True, True],
            "width_valid_mask.right": [True, True, False, True, True, True, True],
            "width_valid_mask.left": [True, True, False, True, True, True, True],
        }
    )

    state_valid, action_valid = dataset._noe_umi_quality_masks(
        0,
        table,
        np.asarray([0, 3], dtype=np.int64),
        np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
    )

    assert state_valid.tolist() == [True, False]
    # action[t] is checked against the quality row at t+1.
    assert action_valid.tolist() == [True, False, False, True, True, True]
    assert dataset._noe_umi_quality_fallback_warned is False


def test_noe_umi_frame_quality_falls_back_once_to_episode_fields(tmp_path, monkeypatch):
    _write_noe_umi_tree(tmp_path)
    capture_logger = _CaptureLogger()
    monkeypatch.setattr(dreamzero_module, "logger", capture_logger)
    dataset = _build_dataset(
        tmp_path,
        filter_qualified=True,
        quality_mode="frame",
        action_horizon=2,
    )
    table = pa.table({"unrelated": [1, 2, 3]})
    indices = np.asarray([0, 1], dtype=np.int64)

    state_valid, action_valid = dataset._noe_umi_quality_masks(
        0, table, indices[:1], indices
    )
    dataset._noe_umi_quality_masks(0, table, indices[:1], indices)

    assert state_valid.tolist() == [True]
    assert action_valid.tolist() == [True, True]
    assert dataset._noe_umi_quality_fallback_warned is True
    assert len(capture_logger.warning_calls) == 1


def test_noe_umi_frame_quality_masks_bad_steps_and_bad_anchor_chunks(monkeypatch):
    monkeypatch.setattr(
        DreamTransformBase,
        "apply_single",
        lambda self, data: {
            "state_mask": np.ones((2, 4), dtype=bool),
            "action_mask": np.ones((4, 4), dtype=bool),
        },
    )
    transform = DreamTransform.model_construct(training=True, action_horizon=2)

    transformed = transform.apply_single(
        {
            "__noe_umi_state_quality": np.asarray([True, False]),
            "__noe_umi_action_quality": np.asarray([True, False, True, True]),
        }
    )

    assert transformed["state_mask"].all(axis=1).tolist() == [True, False]
    assert transformed["action_mask"].all(axis=1).tolist() == [
        True,
        False,
        False,
        False,
    ]


def test_noe_umi_relative_mode_is_accepted_and_robotwin_mode_is_rejected(tmp_path):
    _write_noe_umi_tree(tmp_path)

    dataset = _build_dataset(
        tmp_path,
        filter_qualified=False,
        relative_action=True,
        relative_action_mode=RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ,
    )

    assert dataset.relative_action is True
    assert (
        dataset.relative_action_mode
        == RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ
    )

    with pytest.raises(ValueError, match="requires embodiment_tag='robotwin'"):
        _build_dataset(
            tmp_path,
            filter_qualified=False,
            relative_action=True,
            relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
        )


def test_noe_umi_prefers_common_subtask_after_action_sampling(tmp_path):
    _write_noe_umi_tree(tmp_path)
    _write_subtask_segments(tmp_path)
    dataset = _build_dataset(
        tmp_path,
        filter_qualified=False,
        action_horizon=2,
        noe_subtask={"mode": "prefer"},
    )

    action_indices = np.asarray([0, 1, 1, 2], dtype=np.int64)
    resolution = dataset._resolve_noe_subtask(
        episode_index=0,
        frame_index=0,
        action_indices=action_indices,
    )
    assert resolution is not None
    assert resolution.text == "fine-grained instruction"

    sample = {
        "episode_index": 0,
        "frame_index": 0,
        "observation.state": np.zeros((1, 23), dtype=np.float32),
        "action": np.zeros((4, 23), dtype=np.float32),
        **{
            key: np.zeros((1, 2, 2, 3), dtype=np.uint8)
            for key in VIDEO_KEYS
        },
    }
    dataset._attach_noe_subtask_instruction(
        sample,
        episode_index=0,
        frame_index=0,
        action_indices=action_indices,
    )
    output = dataset._build_modality_dict(sample)
    assert output["annotation.language.action_text"] == "fine-grained instruction"
    assert dataset._noe_subtask_resolution_counts["total"] == 1
    assert dataset._noe_subtask_resolution_counts["matched"] == 1


def test_noe_umi_prefers_episode_instruction_when_no_common_subtask(tmp_path):
    _write_noe_umi_tree(tmp_path)
    _write_subtask_segments(tmp_path)
    dataset = _build_dataset(
        tmp_path,
        filter_qualified=False,
        action_horizon=2,
        noe_subtask={"mode": "prefer"},
    )

    sample = {
        "episode_index": 1,
        "frame_index": 0,
        "observation.state": np.zeros((1, 23), dtype=np.float32),
        "action": np.zeros((2, 23), dtype=np.float32),
        **{
            key: np.zeros((1, 2, 2, 3), dtype=np.uint8)
            for key in VIDEO_KEYS
        },
    }
    dataset._attach_noe_subtask_instruction(
        sample,
        episode_index=1,
        frame_index=0,
        action_indices=np.asarray([0, 1], dtype=np.int64),
    )
    output = dataset._build_modality_dict(sample)
    assert output["annotation.language.action_text"] == "slam invalid"
    assert dataset._noe_subtask_resolution_counts["total"] == 1
    assert dataset._noe_subtask_resolution_counts["no_match"] == 1


@pytest.mark.parametrize(
    "caption_field, caption, expected",
    [
        (None, "dense manipulation", "dense manipulation"),
        ("action_detail_en", "detail text", "detail text"),
        ("action_sent_en", "sentence text", "sentence text"),
        ("action_short_en", "short text", "short text"),
        ("action_sent_en", " \t ", "qualified"),
        ("action_short_en", None, "qualified"),
    ],
)
def test_unified_dense_subtask_uses_manipulation_and_falls_back_to_task_annotation(
    tmp_path, monkeypatch, caption_field, caption, expected
):
    _write_noe_umi_tree(tmp_path)
    _write_dense_subtasks(
        tmp_path, caption_field=caption_field or "action_detail_en", caption=caption
    )
    capture_logger = _CaptureLogger()
    monkeypatch.setattr(dreamzero_module, "logger", capture_logger)
    options = {} if caption_field is None else {"subtask_caption_field": caption_field}
    dataset = _build_dataset(
        tmp_path,
        filter_qualified=False,
        action_horizon=2,
        subtask_mode="unified_dense",
        **options,
    )

    assert dataset._episode_instruction_texts[1] == "slam invalid"
    assert dataset.subtask_caption_field == (caption_field or "action_detail_en")
    assert any(
        "caption_field=%s" in call[0] and call[3] == dataset.subtask_caption_field
        for call in capture_logger.info_calls
    )

    def sample(episode_index):
        return {
            "episode_index": episode_index,
            "frame_index": 0,
            "observation.state": np.zeros((1, 23), dtype=np.float32),
            "action": np.zeros((2, 23), dtype=np.float32),
            **{
                key: np.zeros((1, 2, 2, 3), dtype=np.uint8)
                for key in VIDEO_KEYS
            },
        }

    matched = sample(0)
    dataset._attach_dense_subtask_instruction(
        matched,
        episode_index=0,
        video_indices=np.asarray([0, 1], dtype=np.int64),
    )
    assert (
        dataset._build_modality_dict(matched)["annotation.language.action_text"]
        == expected
    )

    fallback = sample(1)
    dataset._attach_dense_subtask_instruction(
        fallback,
        episode_index=1,
        video_indices=np.asarray([0, 1], dtype=np.int64),
    )
    assert (
        dataset._build_modality_dict(fallback)["annotation.language.action_text"]
        == "slam invalid"
    )


def test_legacy_mode_does_not_load_dense_caption_field(tmp_path):
    _write_noe_umi_tree(tmp_path)

    dataset = _build_dataset(
        tmp_path, filter_qualified=False, subtask_caption_field="action_short_en"
    )

    assert dataset.subtask_mode == "legacy"
    assert dataset._dense_subtask_index is None


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("none", "dense manipulation"),
        ("scene_light", "dense manipulation A table. Bright light."),
        ("state", "dense manipulation The box is closed."),
    ],
)
def test_unified_dense_caption_extra_reaches_instruction(tmp_path, mode, expected):
    _write_noe_umi_tree(tmp_path)
    _write_dense_subtasks(
        tmp_path,
        extra={
            "scene_en": "A table.",
            "lighting_en": "Bright light.",
            "state_detail_en": "The box is closed.",
        },
    )
    dataset = _build_dataset(
        tmp_path,
        filter_qualified=False,
        action_horizon=2,
        subtask_mode="unified_dense",
        subtask_caption_extra=mode,
    )
    for episode, text in [(0, expected), (1, "slam invalid")]:
        sample = {
            "episode_index": episode,
            "frame_index": 0,
            "observation.state": np.zeros((1, 23), dtype=np.float32),
            "action": np.zeros((2, 23), dtype=np.float32),
            **{key: np.zeros((1, 2, 2, 3), dtype=np.uint8) for key in VIDEO_KEYS},
        }
        dataset._attach_dense_subtask_instruction(
            sample, episode_index=episode, video_indices=np.asarray([0, 1])
        )
        assert dataset._build_modality_dict(sample)["annotation.language.action_text"] == text


@pytest.mark.parametrize("extra", ["scene_light", "state"])
def test_caption_extra_requires_unified_dense_mode(tmp_path, extra):
    with pytest.raises(ValueError, match="requires subtask_mode='unified_dense'"):
        _build_dataset(tmp_path, filter_qualified=False, subtask_caption_extra=extra)


@pytest.mark.parametrize("extra, expected", [
    ("none", "Pick up the cup."),
    ("scene_light", "Pick up the cup. A table. Bright light."),
    ("state", "Pick up the cup. The cup is raised."),
])
def test_episode_caption_extra_reaches_instruction(tmp_path, extra, expected):
    _write_noe_umi_tree(tmp_path)
    path = tmp_path / "meta/subtask/episode_all_qwen.parquet"
    path.parent.mkdir(parents=True)
    fields = [pa.field(name, pa.string()) for name in (
        "action_detail_en", "scene_en", "lighting_en", "state_detail_en",
    )]
    schema = pa.schema([
        pa.field("episode_index", pa.int64()),
        pa.field("caption", pa.struct(fields)),
    ])
    pq.write_table(pa.Table.from_pylist([
        {"episode_index": 0, "caption": {
            "action_detail_en": "Pick up the cup.",
            "scene_en": "A table.", "lighting_en": "Bright light.",
            "state_detail_en": "The cup is raised.",
        }},
        {"episode_index": 1, "caption": {
            "action_detail_en": "", "scene_en": "Ignored scene.",
            "state_detail_en": "Ignored state.",
        }},
    ], schema=schema), path)
    dataset = _build_dataset(
        tmp_path, filter_qualified=False, action_horizon=2,
        subtask_mode="unified_dense", subtask_file=str(path),
        caption_granularity="episode", subtask_caption_extra=extra,
    )
    for episode, text in [(0, expected), (1, ""), (2, "")]:
        for frames in (np.asarray([0, 1]), np.asarray([100, 101])):
            sample = {
                "episode_index": episode, "frame_index": 0,
                "observation.state": np.zeros((1, 23), dtype=np.float32),
                "action": np.zeros((2, 23), dtype=np.float32),
                **{key: np.zeros((1, 2, 2, 3), dtype=np.uint8) for key in VIDEO_KEYS},
            }
            dataset._attach_dense_subtask_instruction(
                sample, episode_index=episode, video_indices=frames,
            )
            assert dataset._build_modality_dict(sample)["annotation.language.action_text"] == text


def test_unified_dense_subtask_rejects_legacy_instruction_sources(tmp_path):
    _write_noe_umi_tree(tmp_path)

    with pytest.raises(ValueError, match="cannot be combined with legacy"):
        _build_dataset(
            tmp_path,
            filter_qualified=False,
            subtask_mode="unified_dense",
            noe_subtask={"mode": "prefer"},
        )


def test_unified_dense_umi_requires_episode_instruction_for_fallback(tmp_path):
    _write_noe_umi_tree(tmp_path)
    _write_dense_subtasks(tmp_path)

    with pytest.raises(ValueError, match="requires data.episode_instruction_key"):
        _build_dataset(
            tmp_path,
            filter_qualified=False,
            subtask_mode="unified_dense",
            episode_instruction_key=None,
        )


@pytest.mark.parametrize(
    "caption_field", ["action_detail_en", "action_sent_en", "action_short_en"]
)
def test_unified_dense_miss_keeps_ego_episode_task_fallback(tmp_path, caption_field):
    _write_dense_subtasks(tmp_path, caption_field=caption_field, caption=None)
    dataset = DreamZeroLeRobotDataset.__new__(DreamZeroLeRobotDataset)
    dataset._dense_subtask_index = dreamzero_module.DenseSubtaskIndex.load(
        tmp_path, "meta/subtask/subtasks_dense_seed.parquet", caption_field=caption_field
    )
    dataset._source_video_key = {}
    dataset._state_components = {}
    dataset._action_components = {}
    dataset.return_episode_first_images = False
    dataset.embodiment_tag = "ego4d"
    dataset.action_quaternion_mode = dreamzero_module.ACTION_QUATERNION_MODE_PRESERVE
    dataset.relative_action = False
    dataset.relative_action_mode = None
    dataset._ego_instruction_index = None
    dataset.episode_instruction_key = None
    dataset.language_keys = ["annotation.language.action_text"]
    dataset._language_sources = {}
    dataset._tasks = {}

    sample = {"episode_index": 0, "task": "ego episode task"}
    dataset._attach_dense_subtask_instruction(
        sample, episode_index=0, video_indices=np.asarray([0, 1])
    )
    output = dataset._build_modality_dict(sample)

    assert output["annotation.language.action_text"] == "ego episode task"


def test_noe_subtask_resolution_stats_warn_after_warmup(tmp_path, monkeypatch):
    _write_noe_umi_tree(tmp_path)
    _write_subtask_segments(tmp_path)
    capture_logger = _CaptureLogger()
    monkeypatch.setattr(dreamzero_module, "logger", capture_logger)
    dataset = _build_dataset(
        tmp_path,
        filter_qualified=False,
        action_horizon=2,
        noe_subtask={"mode": "prefer"},
    )
    counts = dataset._noe_subtask_resolution_counts
    counts["total"] = 999
    counts["no_match"] = 999

    sample = {}
    dataset._attach_noe_subtask_instruction(
        sample,
        episode_index=1,
        frame_index=0,
        action_indices=np.asarray([0, 1], dtype=np.int64),
    )

    assert sample == {}
    assert counts["total"] == 1000
    assert counts["no_match"] == 1000
    stats_call = next(
        call
        for call in capture_logger.info_calls
        if call[0].startswith("NOE subtask resolution stats:")
    )
    assert stats_call[3] == 1000
    assert stats_call[6] == 1000
    assert stats_call[8] == 0.0
    assert stats_call[9] == 100.0
    warning_call = next(
        call
        for call in capture_logger.warning_calls
        if call[0].startswith("NOE subtask prefer mode produced zero matches")
    )
    assert warning_call[1] == 1000


def test_noe_umi_collection_root_discovers_direct_symlink_datasets(tmp_path):
    collection_root = tmp_path / "collection"
    embedded_root = collection_root / "embedded"
    external_root = tmp_path / "external"
    (embedded_root / "meta").mkdir(parents=True)
    (external_root / "meta").mkdir(parents=True)
    (embedded_root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (external_root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    linked_root = collection_root / "linked"
    linked_root.symlink_to(external_root, target_is_directory=True)

    roots = _resolve_dreamzero_data_paths(str(collection_root), "noe_umi_16d")

    assert roots == [str(embedded_root), str(linked_root)]


def test_egoverse_lerobot_root_remains_single_dataset(tmp_path):
    dataset_root = tmp_path / "egoverse_480x640"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}", encoding="utf-8")

    roots = _resolve_dreamzero_data_paths(str(dataset_root), "egoverse")

    assert roots == [str(dataset_root)]


def test_egoverse_profile_root_discovers_shards(tmp_path):
    profile_root = tmp_path / "egostand"
    shard_roots = [
        profile_root / "shard-000000",
        profile_root / "shard-000001",
    ]
    for shard_root in shard_roots:
        (shard_root / "meta").mkdir(parents=True)
        (shard_root / "meta" / "info.json").write_text("{}", encoding="utf-8")

    roots = _resolve_dreamzero_data_paths(str(profile_root), "egoverse")

    assert roots == [str(root) for root in shard_roots]
