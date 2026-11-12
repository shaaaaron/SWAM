# Copyright 2026 The RLinf Authors.

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from groot.vla.data.transform.base import ComposedModalityTransform
from groot.vla.model.dreamzero.transform import dreamzero_cotrain
from omegaconf import OmegaConf

from rlinf.data.datasets.dreamzero import dreamzero as dreamzero_module
from rlinf.data.datasets.dreamzero.data_transforms import (
    DEFAULT_EMBODIMENT_TAG_MAPPING,
    build_dreamzero_composed_transform,
    collect_dreamzero_dataset_keys,
    load_dreamzero_dataset_metadata,
)
from rlinf.data.datasets.dreamzero.data_transforms.ego_video import (
    EgoverseDataTransform,
)
from rlinf.data.datasets.dreamzero.data_transforms.egopro import EgoProDataTransform
from rlinf.data.datasets.dreamzero.data_transforms.noe_robotics_16d import (
    NoeRobotics16DDataTransform,
)
from rlinf.data.datasets.dreamzero.data_transforms.robotwin import RobotwinDataTransform
from rlinf.data.datasets.dreamzero.dreamzero import (
    DreamZeroCollator,
    DreamZeroLeRobotDataset,
    _resolve_dreamzero_data_paths,
)
from rlinf.data.datasets.dreamzero.mixed import build_mixed_dreamzero_sft_dataloader

_REPO = Path(__file__).resolve().parents[2]
_CAMERAS = ("head_left", "left_wrist", "right_wrist")
_LANGUAGE_KEY = "annotation.language.action_text"
_CAPTION = "Place the cup on the shelf."


class RecordingTokenizer:
    def __init__(self, **_kwargs):
        self.calls = []

    def __call__(self, texts, **_kwargs):
        self.calls.append(texts)
        shape = (len(texts), 4)
        return np.zeros(shape, dtype=np.int64), np.ones(shape, dtype=np.int64)


@pytest.fixture(autouse=True)
def stub_tokenizer(monkeypatch):
    monkeypatch.setattr(dreamzero_cotrain, "HuggingfaceTokenizer", RecordingTokenizer)


def _model_cfg(tag="egopro"):
    return OmegaConf.create(
        {
            "embodiment_tag": tag,
            "metadata_json_path": str(_REPO / f".assets/{tag}_480x640_metadata.json"),
            "has_real_action": False,
            "image_resolution_height": 2 if tag == "egopro" else 4,
            "image_resolution_width": 4 if tag == "egopro" else 8,
            "tokenizer_path": "unused",
            "max_seq_len": 16,
            "state_horizon": 1,
            "action_horizon": 48,
            "max_state_dim": 16,
            "max_action_dim": 16,
            "color_jitter": {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0},
            "action_head_cfg": {
                "config": {
                    "num_frames": 33,
                    "diffusion_model_cfg": {"max_chunk_size": 4},
                    "video_condition_on_action_state": False,
                    "use_past_action_context": False,
                }
            },
        }
    )


def _synthetic_metadata(tag):
    metadata = load_dreamzero_dataset_metadata(_model_cfg(tag))
    for video in metadata.modalities.video.values():
        video.resolution = (4, 2)
    return metadata


def _transform(tag="egopro"):
    cfg = _model_cfg(tag)
    transform = build_dreamzero_composed_transform(cfg, "unused")
    transform.set_metadata(_synthetic_metadata(tag))
    transform.train()
    return transform


def _raw_pro(num_frames, height=2, width=4):
    pattern = (
        np.arange(num_frames * height * width * 3).reshape(num_frames, height, width, 3)
        % 40
    )
    return {
        **{
            f"video.{camera}": (pattern + 50 * index).astype(np.uint8)
            for index, camera in enumerate(_CAMERAS)
        },
        _LANGUAGE_KEY: np.asarray([_CAPTION]),
    }


def _robotwin_video(raw, cfg, resolution=(4, 2)):
    reference = RobotwinDataTransform.get_transform(
        tokenizer_path="unused",
        cfg=cfg,
        embodiment_tag_mapping=RobotwinDataTransform.DEFAULT_TAG_MAPPING,
    )
    prefix = ComposedModalityTransform(transforms=reference.transforms[:5])
    assert [type(stage).__name__ for stage in prefix.transforms] == [
        "VideoToTensor",
        "VideoCrop",
        "VideoResize",
        "VideoColorJitter",
        "VideoToNumpy",
    ]
    robot_keys = RobotwinDataTransform.get_modality_config()["video"].modality_keys
    metadata = _synthetic_metadata("egopro")
    metadata.modalities.video = {
        key.removeprefix("video."): metadata.modalities.video[camera]
        for key, camera in zip(robot_keys, _CAMERAS)
    }
    for video in metadata.modalities.video.values():
        video.resolution = resolution
    prefix.set_metadata(metadata)
    prefix.train()
    result = prefix(
        {
            key: raw[f"video.{camera}"].copy()
            for key, camera in zip(robot_keys, _CAMERAS)
        }
    )
    return {f"video.{camera}": result[key] for key, camera in zip(robot_keys, _CAMERAS)}


@pytest.mark.parametrize("num_frames", [9, 17, 25, 33, 34])
def test_pro_transform_preserves_view_order_orientation_and_video_only_registers(
    num_frames,
):
    raw = _raw_pro(num_frames)
    torch.manual_seed(7)
    expected = _robotwin_video(raw, _model_cfg())

    torch.manual_seed(7)
    sample = _transform()(raw)

    assert sample["images"].shape == (num_frames, 4, 8, 3)
    np.testing.assert_array_equal(
        sample["images"][:, :2], np.repeat(expected["video.head_left"], 2, axis=2)
    )
    np.testing.assert_array_equal(
        sample["images"][:, 2:, :4], expected["video.left_wrist"]
    )
    np.testing.assert_array_equal(
        sample["images"][:, 2:, 4:], expected["video.right_wrist"]
    )
    chunks = (num_frames - 1) // 8
    assert sample["state"].shape == (chunks, 16)
    assert sample["action"].shape == (chunks * 48, 16)
    for key in ("state", "state_mask", "action", "action_mask", "has_real_action"):
        assert not sample[key].any()
    assert sample["embodiment_id"] == 40
    assert sample["text"] == _CAPTION
    assert "lapa_action" not in sample


@pytest.mark.parametrize("seed", [0, 7, 29])
@pytest.mark.parametrize("color_jitter", [False, True])
def test_pro_video_preprocessing_matches_robotwin(seed, color_jitter):
    cfg = _model_cfg()
    if color_jitter:
        del cfg.color_jitter
    raw = _raw_pro(9, height=20, width=40)
    torch.manual_seed(seed)
    expected = _robotwin_video(raw, cfg, resolution=(40, 20))
    images = np.stack([expected[f"video.{camera}"] for camera in _CAMERAS])
    canvas = RobotwinDataTransform.concat_multiview_video(
        images.transpose(0, 1, 4, 2, 3)
    )[0].transpose(0, 2, 3, 1)
    transform = build_dreamzero_composed_transform(cfg, "unused")
    metadata = _synthetic_metadata("egopro")
    for video in metadata.modalities.video.values():
        video.resolution = (40, 20)
    transform.set_metadata(metadata)
    transform.train()
    torch.manual_seed(seed)

    sample = transform(raw)

    np.testing.assert_array_equal(sample["images"], canvas)


@pytest.mark.parametrize("views", [1, 2, 4])
def test_pro_canvas_rejects_incomplete_or_extra_views(views):
    with pytest.raises(ValueError, match="exactly 3 views"):
        EgoProDataTransform.concat_multiview_video(
            np.zeros((views, 9, 3, 2, 4), dtype=np.uint8)
        )


@pytest.mark.parametrize("has_real_action", [None, True])
def test_pro_requires_explicit_video_only_training(has_real_action):
    cfg = _model_cfg()
    if has_real_action is None:
        del cfg.has_real_action
    else:
        cfg.has_real_action = has_real_action
    with pytest.raises(ValueError, match="has_real_action: false"):
        build_dreamzero_composed_transform(cfg, "unused")


@pytest.mark.parametrize("height,width", [(0, 160), (-2, 160), (128, 0), (128, -2)])
def test_pro_rejects_invalid_per_view_size(height, width):
    cfg = _model_cfg()
    cfg.image_resolution_height = height
    cfg.image_resolution_width = width
    with pytest.raises(ValueError, match="positive per-view image resolution"):
        build_dreamzero_composed_transform(cfg, "unused")


@pytest.mark.parametrize(
    "resolution,expected_shape",
    [(None, (9, 256, 320, 3)), ((128, 160), (9, 256, 320, 3)), ((3, 5), (9, 6, 10, 3))],
)
def test_pro_resolution_is_per_view(resolution, expected_shape):
    cfg = _model_cfg()
    if resolution is None:
        del cfg.image_resolution_height
        del cfg.image_resolution_width
    else:
        cfg.image_resolution_height, cfg.image_resolution_width = resolution
    transform = build_dreamzero_composed_transform(cfg, "unused")
    transform.set_metadata(_synthetic_metadata("egopro"))
    transform.train()

    sample = transform(_raw_pro(9))

    assert sample["images"].shape == expected_shape


def test_pro_registration_and_mixed_collation_keep_single_head_identity():
    mappings = DEFAULT_EMBODIMENT_TAG_MAPPING
    assert mappings["egopro"] == {"egopro": 40}
    assert all(
        40 not in mapping.values()
        for tag, mapping in mappings.items()
        if tag != "egopro"
    )
    pro = _transform()(_raw_pro(17))
    head = _transform("egoverse")(
        {
            "video.cam_high": np.full((9, 2, 4, 3), 42, dtype=np.uint8),
            _LANGUAGE_KEY: np.asarray([_CAPTION]),
        }
    )
    tokenizer = RecordingTokenizer()

    batch = DreamZeroCollator.collate_batch(
        [head, pro], tokenizer, {"egoverse": 36, "egopro": 40}
    )

    assert batch["images"].shape == (2, 17, 4, 8, 3)
    assert batch["embodiment_id"].tolist() == [36, 40]
    assert not batch["has_real_action"].any()
    assert not batch["action_mask"].any()
    assert not batch["state_mask"].any()
    assert np.all(batch["images"][0].numpy() == 42)
    np.testing.assert_array_equal(batch["images"][1].numpy(), pro["images"])
    expected_prompts = [
        EgoverseDataTransform.format_training_prompt(_CAPTION.lower()),
        EgoProDataTransform.format_training_prompt(_CAPTION.lower()),
    ]
    assert expected_prompts in tokenizer.calls
    assert "head-mounted" in expected_prompts[1]
    assert "robot" not in expected_prompts[1]


def test_collator_keeps_prompt_identity_when_embodiment_ids_overlap():
    pro = _transform()(_raw_pro(17))
    pro["embodiment_id"] = 39
    robotics_prompt = dict(pro)
    robotics_prompt["prompt_style"] = "robotics"
    tokenizer = RecordingTokenizer()

    batch = DreamZeroCollator.collate_batch(
        [pro, robotics_prompt],
        tokenizer,
        {"noe_robotics_16d": 39, "egopro": 39},
    )

    expected_prompts = [
        EgoProDataTransform.format_training_prompt(_CAPTION.lower()),
        NoeRobotics16DDataTransform.format_training_prompt(_CAPTION.lower()),
    ]
    assert expected_prompts in tokenizer.calls
    assert "head-mounted" in expected_prompts[0]
    assert "robot" not in expected_prompts[0]
    assert "robot" in expected_prompts[1]
    assert "prompt_style" not in batch
    assert batch["embodiment_id"].tolist() == [39, 39]
    assert not batch["has_real_action"].any()
    assert not batch["action_mask"].any()
    assert not batch["state_mask"].any()


def _write_tree(root, cameras=_CAMERAS):
    meta = root / "meta"
    data = root / "data/chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": 256,
        "total_videos": len(cameras),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            **{
                f"observation.images.{camera}": {"dtype": "video", "shape": [3, 2, 4]}
                for camera in cameras
            },
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 256}) + "\n"
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "coarse task"}) + "\n"
    )
    pq.write_table(pa.table({"task_index": [0] * 256}), data / "episode_000000.parquet")
    (meta / "subtask").mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 0,
                    "subtasks": [
                        {
                            "start_frame": 0,
                            "end_frame": 256,
                            "segment_type": "manipulation",
                            "visual_clear": True,
                            "action_detail_en": _CAPTION,
                            "tag": "valid",
                        }
                    ],
                }
            ]
        ),
        meta / "subtask/subtasks_v1.parquet",
    )
    for camera in cameras:
        video_dir = root / f"videos/chunk-000/observation.images.{camera}"
        video_dir.mkdir(parents=True)
        (video_dir / "episode_000000.mp4").touch()


def _dataset(root):
    transform = _transform()
    video, state, action, language = collect_dreamzero_dataset_keys(transform, "egopro")
    return DreamZeroLeRobotDataset(
        data_path=str(root),
        video_keys=video,
        state_keys=state,
        action_keys=action,
        language_keys=language,
        data_transform=transform,
        embodiment_tag="egopro",
        lazy_load=True,
        num_frames=33,
        state_horizon=1,
        action_horizon=48,
        max_chunk_size=4,
        multi_anchor_macro_stride=48,
        multi_anchor_video_offsets=[0, 6, 12, 18, 24, 30, 36, 42],
        pq_cache_max_episodes=0,
        subtask_mode="unified_dense",
        subtask_file="meta/subtask/subtasks_v1.parquet",
    )


@pytest.fixture
def decoded_cameras(monkeypatch):
    from lerobot.datasets import video_utils

    calls = []

    def decode(path, timestamps, **_kwargs):
        camera = path.parent.name.removeprefix("observation.images.")
        calls.append((camera, timestamps))
        return np.full(
            (len(timestamps), 2, 4, 3),
            40 * (_CAMERAS.index(camera) + 1),
            dtype=np.uint8,
        )

    monkeypatch.setattr(dreamzero_module, "probe_video_container_fps", lambda _: 30.0)
    monkeypatch.setattr(video_utils, "decode_video_frames", decode)
    return calls


def test_pro_loader_uses_shared_timestamps_and_parquet_caption(
    tmp_path, decoded_cameras
):
    _write_tree(tmp_path)
    sample = _dataset(tmp_path)[0]

    assert [camera for camera, _ in decoded_cameras] == list(_CAMERAS)
    assert decoded_cameras[0][1] == decoded_cameras[1][1] == decoded_cameras[2][1]
    assert sample["text"] == _CAPTION
    assert np.all(sample["images"][:, :2] == 40)
    assert np.all(sample["images"][:, 2:, :4] == 80)
    assert np.all(sample["images"][:, 2:, 4:] == 120)


@pytest.mark.parametrize("missing", _CAMERAS)
def test_pro_loader_rejects_missing_camera_at_construction(tmp_path, missing):
    _write_tree(tmp_path, cameras=[camera for camera in _CAMERAS if camera != missing])
    with pytest.raises(KeyError, match=missing):
        _dataset(tmp_path)


def test_pro_resolver_expands_profiles_without_duplicating_shards(tmp_path):
    roots = [tmp_path / "egopro/shard-000001", tmp_path / "egopro-body/shard-000001"]
    for root in roots:
        _write_tree(root)
    resolved = _resolve_dreamzero_data_paths(
        [str(tmp_path / "egopro"), str(tmp_path / "egopro-body"), str(roots[0])],
        "egopro",
    )
    assert resolved == sorted(str(root) for root in roots)


def _mixed_cfg(tmp_path):
    entries = []
    for profile, tag in (("egostand", "egoverse"), ("egopro", "egopro")):
        root = tmp_path / profile / "shard-000000"
        _write_tree(root, cameras=_CAMERAS if tag == "egopro" else ("head_left",))
        metadata_path = tmp_path / f"{tag}_metadata.json"
        metadata_path.write_text(
            json.dumps({tag: _synthetic_metadata(tag).model_dump(mode="json")})
        )
        model_cfg = _model_cfg(tag)
        entries.append(
            {
                "name": profile,
                "weight": 1.0,
                "embodiment_tag": tag,
                "metadata_json_path": str(metadata_path),
                "has_real_action": False,
                "image_resolution_height": model_cfg.image_resolution_height,
                "image_resolution_width": model_cfg.image_resolution_width,
                "data_paths": [str(root.parent)],
                "video_key_mapping": {}
                if tag == "egopro"
                else {"video.cam_high": "observation.images.head_left"},
                "subtask_mode": "unified_dense",
                "subtask_file": "meta/subtask/subtasks_v1.parquet",
            }
        )
    return OmegaConf.create(
        {
            "actor": {"model": _model_cfg("egoverse"), "micro_batch_size": 2},
            "data": {
                "train_datasets": entries,
                "num_workers": 0,
                "parquet_cache_size": 0,
                "multi_anchor_macro_stride": 48,
                "multi_anchor_video_offsets": [0, 6, 12, 18, 24, 30, 36, 42],
                "mixture": {"epoch_size": 16, "seed": 7},
            },
        }
    )


def test_real_mixed_builder_accepts_stand_and_pro(tmp_path, decoded_cameras):
    loader, info = build_mixed_dreamzero_sft_dataloader(_mixed_cfg(tmp_path), 1, 0)
    assert info["mixture_names"] == ["egostand", "egopro"]
    assert info["mixture_lengths"] == [256, 256]
    samples = [dataset[0] for dataset in loader.dataset.datasets]
    batch = loader.collate_fn(samples)
    assert batch["embodiment_id"].tolist() == [36, 40]
    assert batch["images"].shape == (2, 33, 4, 8, 3)
    seen = set()
    for batch in loader:
        seen.update(batch["embodiment_id"].tolist())
        assert not batch["has_real_action"].any()
        assert not batch["action_mask"].any()
    assert seen == {36, 40}


@pytest.mark.parametrize(
    "key", ["video_condition_on_action_state", "use_past_action_context"]
)
def test_pro_mixed_builder_enforces_video_only_model_settings(tmp_path, key):
    cfg = _mixed_cfg(tmp_path)
    cfg.data.train_datasets = [cfg.data.train_datasets[1]]
    cfg.actor.model.action_head_cfg.config[key] = True
    with pytest.raises(ValueError, match=key):
        build_mixed_dreamzero_sft_dataloader(cfg, 1, 0)
