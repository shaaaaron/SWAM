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

import importlib
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from rlinf.data.datasets.dreamzero.task_type_sampling import (
    TaskTypeFrameIndex,
    TaskTypeRanges,
    TaskTypeSamplingConfig,
    TaskTypeSamplingPlan,
)
class _ListDataset(Dataset):
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _import_mixed_with_stubs(monkeypatch):
    stub_dreamzero = types.ModuleType("rlinf.data.datasets.dreamzero.dreamzero")

    class DummyCollator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, features):
            return {"items": list(features), "num_items": len(features)}

    class DummyLeRobotDataset(Dataset):
        def __init__(self, *args, **kwargs):
            self._items = [{"dummy": True}]

        def __len__(self):
            return len(self._items)

        def __getitem__(self, index):
            return self._items[index]

    stub_dreamzero.DreamZeroCollator = DummyCollator
    stub_dreamzero.DreamZeroLeRobotDataset = DummyLeRobotDataset
    stub_dreamzero._build_dreamzero_temporal_pad_lengths = (
        lambda *, num_frames, max_chunk_size, state_horizon, action_horizon: {
            "images": int(num_frames),
            "state": int(max_chunk_size) * int(state_horizon),
            "action": int(max_chunk_size) * int(action_horizon),
        }
    )
    stub_dreamzero._resolve_dreamzero_data_paths = lambda data_paths, embodiment_tag: (
        [data_paths] if isinstance(data_paths, str) else list(data_paths)
    )
    stub_dreamzero._prepare_noe_subtask_configs = (
        lambda data_paths, noe_subtask, **kwargs: [noe_subtask] * len(data_paths)
    )
    stub_dreamzero._suppress_ffmpeg_log = lambda worker_id=0: None

    stub_transforms = types.ModuleType("rlinf.data.datasets.dreamzero.data_transforms")

    def embodiment_tag_mapping_for_embodiment(tag, mapping=None):
        defaults = {
            "ego4d": {"ego4d": 35},
            "egoverse": {"egoverse": 36},
            "egodex": {"egodex": 37},
            "openvid": {"openvid": 38},
            "robotwin": {"robotwin": 33},
            "noe_umi_16d": {"noe_umi_16d": 34},
            "noe_robotics_16d": {"noe_robotics_16d": 39},
            "oxe_droid": {"oxe_droid": 17},
        }
        if mapping is not None:
            return dict(mapping)
        return dict(defaults[tag])

    stub_transforms.embodiment_tag_mapping_for_embodiment = (
        embodiment_tag_mapping_for_embodiment
    )

    monkeypatch.setitem(
        sys.modules,
        "rlinf.data.datasets.dreamzero.dreamzero",
        stub_dreamzero,
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.data.datasets.dreamzero.data_transforms",
        stub_transforms,
    )
    sys.modules.pop("rlinf.data.datasets.dreamzero.mixed", None)
    return importlib.import_module("rlinf.data.datasets.dreamzero.mixed")


def _equal_task_type_plan():
    index = TaskTypeFrameIndex(
        ranges={
            "Basic Manipulation": TaskTypeRanges(
                starts=np.asarray([0], dtype=np.int64),
                lengths=np.asarray([4], dtype=np.int64),
                cumulative_frames=np.asarray([4], dtype=np.int64),
            ),
            "Cleaning": TaskTypeRanges(
                starts=np.asarray([4], dtype=np.int64),
                lengths=np.asarray([4], dtype=np.int64),
                cumulative_frames=np.asarray([4], dtype=np.int64),
            ),
        },
        episode_count=2,
        frame_count=8,
        sidecar_rows=2,
        extra_rows=0,
        roots=("/tmp/example",),
    )
    config = TaskTypeSamplingConfig.from_value(
        {
            "enabled": True,
            "weights": {
                "Basic Manipulation": 1.0,
                "Cleaning": 1.0,
            },
        }
    )
    return TaskTypeSamplingPlan.from_config(index, config)


@pytest.fixture
def mixed_module(monkeypatch):
    return _import_mixed_with_stubs(monkeypatch)


def test_mixture_dataset_supports_tuple_and_flat_indices(mixed_module):
    dataset = mixed_module.DreamZeroMixtureDataset(
        [
            _ListDataset(["rw0", "rw1"]),
            _ListDataset(["noe0", "noe1", "noe2"]),
        ],
        ["robotwin", "noe_umi"],
    )

    assert len(dataset) == 5
    assert dataset[(0, 1)] == "rw1"
    assert dataset[(1, 2)] == "noe2"
    assert dataset[0] == "rw0"
    assert dataset[3] == "noe1"
    with pytest.raises(IndexError):
        dataset[7]


def test_weighted_sampler_respects_epoch_and_ratio(mixed_module):
    sampler = mixed_module.DistributedWeightedMixtureSampler(
        [8, 8],
        [3.0, 1.0],
        epoch_size=4000,
        num_replicas=1,
        rank=0,
        seed=7,
        drop_last=True,
    )

    pairs_epoch0 = list(iter(sampler))
    sampler.set_epoch(1)
    pairs_epoch1 = list(iter(sampler))

    assert len(pairs_epoch0) == 4000
    assert len(pairs_epoch1) == 4000
    assert pairs_epoch0 != pairs_epoch1

    counts = [0, 0]
    for dataset_idx, _sample_idx in pairs_epoch0:
        counts[dataset_idx] += 1
    ratio = counts[0] / sum(counts)
    assert 0.70 <= ratio <= 0.80


def test_nine_way_equal_weight_sampler_matches_target_ratio(mixed_module):
    sampler = mixed_module.DistributedWeightedMixtureSampler(
        [
            56_858_070,
            26_906_700,
            5_102_104,
            19_920_362,
            9_922_606,
            4_806_875,
            8_372_255,
            5_175_924,
            7_553_160,
        ],
        [1.0] * 9,
        epoch_size=180_000,
        num_replicas=1,
        rank=0,
        seed=42,
        drop_last=True,
    )

    counts = Counter(dataset_idx for dataset_idx, _sample_idx in sampler)

    assert set(counts) == set(range(9))
    for count in counts.values():
        assert abs(count / 180_000 - 1 / 9) <= 0.01


def test_weighted_sampler_shards_evenly_across_ranks(mixed_module):
    rank0 = mixed_module.DistributedWeightedMixtureSampler(
        [5, 7],
        [1.0, 1.0],
        epoch_size=120,
        num_replicas=2,
        rank=0,
        seed=19,
        drop_last=True,
    )
    rank1 = mixed_module.DistributedWeightedMixtureSampler(
        [5, 7],
        [1.0, 1.0],
        epoch_size=120,
        num_replicas=2,
        rank=1,
        seed=19,
        drop_last=True,
    )

    pairs0 = list(iter(rank0))
    pairs1 = list(iter(rank1))

    assert len(pairs0) == 60
    assert len(pairs1) == 60
    assert pairs0 != pairs1


def test_weighted_sampler_uses_task_type_plan(mixed_module):
    sampler = mixed_module.DistributedWeightedMixtureSampler(
        [8],
        [1.0],
        epoch_size=4000,
        num_replicas=1,
        rank=0,
        seed=23,
        drop_last=True,
        task_type_plans=[_equal_task_type_plan()],
    )

    sample_ids = torch.as_tensor([sample_idx for _, sample_idx in sampler])
    basic_ratio = float((sample_ids < 4).double().mean())

    assert 0.46 <= basic_ratio <= 0.54


def test_task_type_sampler_keeps_global_sequence_before_rank_slice(mixed_module):
    kwargs = {
        "dataset_lengths": [8],
        "weights": [1.0],
        "epoch_size": 120,
        "seed": 31,
        "drop_last": True,
        "task_type_plans": [_equal_task_type_plan()],
    }
    global_pairs = list(
        mixed_module.DistributedWeightedMixtureSampler(
            **kwargs,
            num_replicas=1,
            rank=0,
        )
    )
    rank0_pairs = list(
        mixed_module.DistributedWeightedMixtureSampler(
            **kwargs,
            num_replicas=2,
            rank=0,
        )
    )
    rank1_pairs = list(
        mixed_module.DistributedWeightedMixtureSampler(
            **kwargs,
            num_replicas=2,
            rank=1,
        )
    )
    reconstructed = [None] * len(global_pairs)
    reconstructed[0::2] = rank0_pairs
    reconstructed[1::2] = rank1_pairs

    assert reconstructed == global_pairs


def test_none_task_type_plan_preserves_legacy_sampling(mixed_module):
    kwargs = {
        "dataset_lengths": [8, 5],
        "weights": [2.0, 1.0],
        "epoch_size": 200,
        "num_replicas": 1,
        "rank": 0,
        "seed": 41,
        "drop_last": True,
    }
    legacy_pairs = list(mixed_module.DistributedWeightedMixtureSampler(**kwargs))
    explicit_disabled_pairs = list(
        mixed_module.DistributedWeightedMixtureSampler(
            **kwargs,
            task_type_plans=[None, None],
        )
    )

    assert explicit_disabled_pairs == legacy_pairs


def test_weighted_sampler_can_log_batch_source_counts(mixed_module, monkeypatch):
    class CaptureLogger:
        def __init__(self):
            self.calls = []

        def info(self, *args):
            self.calls.append(args)

    capture_logger = CaptureLogger()
    monkeypatch.setattr(mixed_module, "logger", capture_logger)
    monkeypatch.setattr(mixed_module.random, "random", lambda: 0.0)

    sampler = mixed_module.DistributedWeightedMixtureSampler(
        [4, 4],
        [1.0, 1.0],
        epoch_size=4,
        num_replicas=1,
        rank=0,
        seed=5,
        drop_last=True,
        dataset_names=["robotwin", "noe_umi"],
        batch_size=2,
    )

    pairs = list(iter(sampler))

    assert len(pairs) == 4
    assert len(capture_logger.calls) == 2
    for call in capture_logger.calls:
        assert call[0] == (
            "[DREAMZERO_MIXED_BATCH_SOURCE] rank=%s epoch=%s batch_idx=%s counts=%s"
        )
        assert sum(call[-1].values()) == 2


def test_validate_mixed_contract_accepts_robotics_datasets(mixed_module):
    model_cfg = OmegaConf.create(
        {
            "action_horizon": 48,
            "max_action_dim": 16,
            "max_state_dim": 16,
            "image_resolution_width": 160,
            "image_resolution_height": 128,
        }
    )
    dataset_cfgs = [
        {"name": "robotwin", "embodiment_tag": "robotwin", "weight": 1.0},
        {"name": "noe", "embodiment_tag": "noe_umi_16d", "weight": 1.0},
        {
            "name": "robotics",
            "embodiment_tag": "noe_robotics_16d",
            "weight": 1.0,
        },
        {"name": "droid", "embodiment_tag": "oxe_droid", "weight": 1.0},
        {"name": "egodex", "embodiment_tag": "egodex", "weight": 1.0},
    ]

    mixed_module._validate_mixed_contract(model_cfg, dataset_cfgs)


def test_validate_mixed_contract_rejects_mismatched_horizon(mixed_module):
    model_cfg = OmegaConf.create(
        {
            "action_horizon": 48,
            "max_action_dim": 16,
            "max_state_dim": 16,
            "image_resolution_width": 160,
            "image_resolution_height": 128,
        }
    )
    dataset_cfgs = [
        {"name": "robotwin", "embodiment_tag": "robotwin", "weight": 1.0},
        {
            "name": "noe",
            "embodiment_tag": "noe_umi_16d",
            "weight": 1.0,
            "action_horizon": 24,
        },
    ]

    with pytest.raises(ValueError, match="action_horizon"):
        mixed_module._validate_mixed_contract(model_cfg, dataset_cfgs)


def test_validate_mixed_contract_requires_explicit_video_only_action_flag(
    mixed_module,
):
    model_cfg = OmegaConf.create(
        {
            "action_horizon": 48,
            "max_action_dim": 16,
            "max_state_dim": 16,
            "image_resolution_width": 320,
            "image_resolution_height": 256,
        }
    )
    valid = [
        {
            "name": "ego",
            "embodiment_tag": "ego4d",
            "has_real_action": False,
        }
    ]
    mixed_module._validate_mixed_contract(model_cfg, valid)
    mixed_module._validate_mixed_contract(
        model_cfg,
        [{"name": "openvid", "embodiment_tag": "openvid", "has_real_action": False}],
    )

    for invalid in (
        {"name": "ego", "embodiment_tag": "ego4d"},
        {
            "name": "ego",
            "embodiment_tag": "ego4d",
            "has_real_action": True,
        },
    ):
        with pytest.raises(ValueError, match="has_real_action"):
            mixed_module._validate_mixed_contract(model_cfg, [invalid])
    for invalid in (
        {"name": "openvid", "embodiment_tag": "openvid"},
        {"name": "openvid", "embodiment_tag": "openvid", "has_real_action": True},
    ):
        with pytest.raises(ValueError, match="has_real_action"):
            mixed_module._validate_mixed_contract(model_cfg, [invalid])


def test_legacy_dataset_does_not_gain_has_real_action_override(mixed_module):
    model_cfg = OmegaConf.create(
        {
            "embodiment_tag": "robotwin",
            "action_horizon": 48,
            "max_action_dim": 16,
            "max_state_dim": 16,
        }
    )
    dataset_cfg = OmegaConf.create(
        {
            "name": "robotwin",
            "embodiment_tag": "robotwin",
            "data_paths": "/tmp/robotwin",
        }
    )

    child_cfg = mixed_module._mixed_model_cfg(
        model_cfg,
        dataset_cfg,
        {"robotwin": 33},
    )

    assert "has_real_action" not in child_cfg


def test_ego_dataset_can_override_view_layout(mixed_module):
    model_cfg = OmegaConf.create(
        {
            "embodiment_tag": "ego4d",
            "ego_view_layout": "single",
        }
    )
    dataset_cfg = OmegaConf.create(
        {
            "name": "ego4d",
            "embodiment_tag": "ego4d",
            "ego_view_layout": "top_gray",
        }
    )

    child_cfg = mixed_module._mixed_model_cfg(
        model_cfg,
        dataset_cfg,
        {"ego4d": 35},
    )

    assert child_cfg.ego_view_layout == "top_gray"


def test_robotics_dataset_does_not_override_fixed_view_contract(mixed_module):
    model_cfg = OmegaConf.create(
        {
            "embodiment_tag": "noe_robotics_16d",
        }
    )
    dataset_cfg = OmegaConf.create(
        {
            "name": "robotics",
            "embodiment_tag": "noe_robotics_16d",
            "view_resize_mode": "contain",
            "view_pad_value": 23,
            "video_crop_scale": 0.9,
        }
    )

    child_cfg = mixed_module._mixed_model_cfg(
        model_cfg,
        dataset_cfg,
        {"noe_robotics_16d": 39},
    )

    for key in ("view_resize_mode", "view_pad_value", "video_crop_scale"):
        assert key not in child_cfg


@pytest.mark.parametrize(
    ("suffix", "use_subtask", "use_scene"),
    [
        ("+subtask", True, False),
        ("+scene", False, True),
    ],
)
def test_umi_relative_egoverse_config_uses_final_canvas_resolution(
    suffix,
    use_subtask,
    use_scene,
):
    config_path = (
        Path(__file__).resolve().parent
        / "fixtures/dreamzero_mixed"
        / (
            "dreamzero_robotwin_baseline_v0a0_wo_img+umi_qualified_relative+egoverse"
            f"{suffix}.yaml"
        )
    )
    cfg = OmegaConf.load(config_path)
    egoverse_datasets = [
        dataset
        for dataset in cfg.data.train_datasets
        if dataset.embodiment_tag == "egoverse"
    ]

    assert len(egoverse_datasets) == 2
    assert all(dataset.ego_view_layout == "single" for dataset in egoverse_datasets)
    assert all(
        (dataset.image_resolution_width, dataset.image_resolution_height) == (320, 256)
        for dataset in egoverse_datasets
    )
    assert all(
        dataset.use_subtask_instruction is use_subtask for dataset in egoverse_datasets
    )
    assert all(
        dataset.use_scene_instruction is use_scene for dataset in egoverse_datasets
    )


@pytest.mark.parametrize(
    ("tag", "mode", "embodiment_id"),
    [
        ("noe_umi_16d", "noe_umi_eef_se3_s0_wxyz", 34),
        ("noe_robotics_16d", "noe_robotics_eef_se3_s0_wxyz", 39),
    ],
)
def test_dataset_can_override_relative_action_contract(mixed_module, tag, mode, embodiment_id):
    model_cfg = OmegaConf.create(
        {
            "embodiment_tag": "robotwin",
            "relative_action": True,
            "relative_action_mode": "robotwin_eef_se3_s0_wxyz",
            "relative_action_keys": ["eef_position"],
        }
    )
    dataset_cfg = OmegaConf.create(
        {
            "name": tag,
            "embodiment_tag": tag,
            "relative_action": True,
            "relative_action_mode": mode,
            "relative_action_keys": ["eef_position"],
        }
    )

    child_cfg = mixed_module._mixed_model_cfg(
        model_cfg,
        dataset_cfg,
        {"robotwin": 33, tag: embodiment_id},
    )

    assert child_cfg.embodiment_tag == tag
    assert child_cfg.relative_action is True
    assert child_cfg.relative_action_mode == mode
    assert list(child_cfg.relative_action_keys) == ["eef_position"]
    assert model_cfg.embodiment_tag == "robotwin"
    assert model_cfg.relative_action_mode == "robotwin_eef_se3_s0_wxyz"


def test_robotics_relative_config_changes_only_supervised_robotics_representation():
    config_root = Path(__file__).resolve().parents[2] / "examples/sft/config"
    absolute = OmegaConf.to_container(OmegaConf.load(
        config_root / "15_1-v0a0_wo_img+umi_robo_20k+robo_abs.yaml"
    ))
    relative = OmegaConf.to_container(OmegaConf.load(
        config_root / "15_2-v0a0_wo_img+umi_robo_20k+robo_rel.yaml"
    ))

    def check_relative_fields(old, new):
        assert new["relative_action"] is True
        assert new["relative_action_mode"] == "noe_robotics_eef_se3_s0_wxyz"
        assert new["relative_action_keys"] == ["eef_position"]
        assert new.pop("relative_action_pose_reference") == "per_arm"
        assert new["metadata_json_path"] == old["metadata_json_path"].replace(
            "_metadata.json", "_s0_relative_metadata.json"
        )
        for key in ("relative_action", "relative_action_mode", "relative_action_keys", "metadata_json_path"):
            new[key] = old[key]

    changed = 0
    for old, new in zip(absolute["data"]["train_datasets"], relative["data"]["train_datasets"], strict=True):
        if old["embodiment_tag"] == "noe_robotics_16d" and old["has_real_action"]:
            check_relative_fields(old, new)
            changed += 1
    assert changed == 7
    # 默认模型配置也随 AgiBot 切换；UMI 与两个无动作监督子集保持原设置。
    check_relative_fields(absolute["actor"]["model"], relative["actor"]["model"])
    relative["runner"]["logger"]["experiment_name"] = absolute["runner"]["logger"]["experiment_name"]
    assert relative == absolute


def test_noe_subtask_config_prefers_per_dataset_value(mixed_module):
    data_cfg = OmegaConf.create(
        {
            "noe_subtask": {"mode": "off"},
        }
    )
    dataset_cfg = OmegaConf.create(
        {
            "name": "noe_umi",
            "noe_subtask": {
                "mode": "prefer",
                "instruction_key": "subtask_en",
            },
        }
    )

    value = mixed_module._data_option(
        data_cfg,
        dataset_cfg,
        "noe_subtask",
        None,
    )

    assert OmegaConf.to_container(value, resolve=True) == {
        "mode": "prefer",
        "instruction_key": "subtask_en",
    }


@pytest.mark.parametrize(
    ("postfix_override", "expected_postfix"),
    [
        ({}, "Global context."),
        ({"prompt_postfix": None}, "Global context."),
        ({"prompt_postfix": ""}, ""),
        (
            {"prompt_postfix": "The robot model is ARX Lift-2."},
            "The robot model is ARX Lift-2.",
        ),
    ],
)
def test_bundle_passes_dataset_video_key_mapping_to_each_root(
    mixed_module, monkeypatch, postfix_override, expected_postfix
):
    class DummyTransform:
        def set_metadata(self, _metadata):
            pass

        def train(self):
            pass

    transform_module = types.ModuleType("groot.vla.data.transform")
    transform_module.ComposedModalityTransform = DummyTransform
    monkeypatch.setitem(sys.modules, "groot.vla.data.transform", transform_module)
    transforms = sys.modules["rlinf.data.datasets.dreamzero.data_transforms"]
    monkeypatch.setattr(
        transforms, "load_dreamzero_dataset_metadata", lambda _: {}, raising=False
    )
    monkeypatch.setattr(
        transforms, "build_dreamzero_composed_transform",
        lambda *_: DummyTransform(), raising=False,
    )
    monkeypatch.setattr(
        transforms, "collect_dreamzero_dataset_keys",
        lambda *_: (["video.cam_high"], [], [], ["annotation.language.action_text"]),
        raising=False,
    )

    created = []

    class RecordingDataset(Dataset):
        def __init__(self, data_path, **kwargs):
            created.append((data_path, kwargs))

        def __len__(self):
            return 1

    monkeypatch.setattr(mixed_module, "DreamZeroLeRobotDataset", RecordingDataset)
    cfg = OmegaConf.create(
        {
            "data": {"prompt_postfix": "Global context."},
            "actor": {
                "model": {
                    "embodiment_tag": "egoverse",
                    "action_horizon": 48,
                    "action_head_cfg": {
                        "config": {
                            "num_frames": 33,
                            "diffusion_model_cfg": {"max_chunk_size": 4},
                        }
                    },
                }
            },
        }
    )
    dataset_cfg = OmegaConf.create(
        {
            **postfix_override,
            "name": "ego_suite",
            "embodiment_tag": "egoverse",
            "data_paths": ["/tmp/shard-0", "/tmp/shard-1"],
            "video_key_mapping": {
                "video.cam_high": "observation.images.head_left"
            },
            "episode_exclusion_manifest": "meta/episode_exclusions.jsonl",
            "prompt_style": "robotics",
        }
    )

    dataset, info = mixed_module._build_dataset_bundle(
        cfg,
        dataset_cfg,
        dataset_idx=0,
        embodiment_tag_mapping={"egoverse": 36},
        eval_dataset=False,
    )

    assert all(kwargs["prompt_postfix"] == expected_postfix for _, kwargs in created)
    assert all(kwargs["prompt_style"] == "robotics" for _, kwargs in created)
    assert len(dataset) == 2
    assert info["num_roots"] == 2
    assert [path for path, _ in created] == ["/tmp/shard-0", "/tmp/shard-1"]
    assert all(
        dict(kwargs["video_key_mapping"])
        == {"video.cam_high": "observation.images.head_left"}
        for _, kwargs in created
    )
    assert all(
        kwargs["episode_exclusion_manifest"] == "meta/episode_exclusions.jsonl"
        for _, kwargs in created
    )

    sibling_cfg = OmegaConf.create(
        {"name": "egoverse", "embodiment_tag": "egoverse", "data_paths": "/tmp/egoverse"}
    )
    mixed_module._build_dataset_bundle(
        cfg,
        sibling_cfg,
        dataset_idx=1,
        embodiment_tag_mapping={"egoverse": 36},
        eval_dataset=False,
    )
    assert created[-1][1]["prompt_postfix"] == "Global context."
    assert created[-1][1]["prompt_style"] is None
    assert cfg.data.prompt_postfix == "Global context."
    assert created[-1][1]["video_key_mapping"] is None
    assert created[-1][1]["episode_exclusion_manifest"] is None


def test_build_mixed_dataloader_returns_weighted_loader(mixed_module, monkeypatch):
    datasets = [
        _ListDataset([{"source": "robotwin-0"}, {"source": "robotwin-1"}]),
        _ListDataset(
            [
                {"source": "noe-0"},
                {"source": "noe-1"},
                {"source": "noe-2"},
            ]
        ),
        _ListDataset([{"source": "droid-0"}]),
    ]
    infos = [
        {
            "name": "robotwin",
            "embodiment_tag": "robotwin",
            "weight": 3.0,
            "length": len(datasets[0]),
        },
        {
            "name": "noe_umi",
            "embodiment_tag": "noe_umi_16d",
            "weight": 1.0,
            "length": len(datasets[1]),
        },
        {
            "name": "oxe_droid",
            "embodiment_tag": "oxe_droid",
            "weight": 2.0,
            "length": len(datasets[2]),
        },
    ]

    def fake_build_dataset_bundle(
        cfg,
        dataset_cfg,
        *,
        dataset_idx,
        embodiment_tag_mapping,
        eval_dataset,
    ):
        assert embodiment_tag_mapping == {
            "robotwin": 33,
            "noe_umi_16d": 34,
            "oxe_droid": 17,
        }
        assert not eval_dataset
        return datasets[dataset_idx], infos[dataset_idx]

    monkeypatch.setattr(
        mixed_module,
        "_build_dataset_bundle",
        fake_build_dataset_bundle,
    )

    cfg = OmegaConf.create(
        {
            "data": {
                "train_datasets": [
                    {
                        "name": "robotwin",
                        "data_paths": "/tmp/robotwin",
                        "embodiment_tag": "robotwin",
                        "weight": 3.0,
                    },
                    {
                        "name": "noe_umi",
                        "data_paths": "/tmp/noe",
                        "embodiment_tag": "noe_umi_16d",
                        "weight": 1.0,
                    },
                    {
                        "name": "oxe_droid",
                        "data_paths": "/tmp/droid",
                        "embodiment_tag": "oxe_droid",
                        "weight": 2.0,
                    },
                ],
                "mixture": {"epoch_size": 8, "seed": 123},
                "num_workers": 0,
                "prefetch_factor": 2,
                "dataloader_profile": False,
                "dataloader_profile_first_n": 1,
                "dataloader_profile_slow_s": 1.0,
            },
            "actor": {
                "micro_batch_size": 2,
                "seed": 123,
                "model": {
                    "action_horizon": 48,
                    "state_horizon": 1,
                    "max_action_dim": 16,
                    "max_state_dim": 16,
                    "image_resolution_width": 160,
                    "image_resolution_height": 128,
                    "tokenizer_path": "dummy-tokenizer",
                    "max_seq_len": 64,
                    "action_head_cfg": {
                        "config": {
                            "num_frames": 33,
                            "diffusion_model_cfg": {"max_chunk_size": 4},
                        }
                    },
                    "embodiment_tag_mapping": {
                        "robotwin": 33,
                        "noe_umi_16d": 34,
                        "oxe_droid": 17,
                    },
                },
            },
        }
    )

    data_loader, info = mixed_module.build_mixed_dreamzero_sft_dataloader(
        cfg,
        world_size=1,
        rank=0,
        eval_dataset=False,
    )
    batch = next(iter(data_loader))

    assert isinstance(
        data_loader.sampler,
        mixed_module.DistributedWeightedMixtureSampler,
    )
    assert data_loader.sampler.seed == 123
    assert data_loader.generator.initial_seed() == 123
    assert info["num_samples"] == 8
    assert info["mixture_names"] == ["robotwin", "noe_umi", "oxe_droid"]
    assert info["mixture_lengths"] == [2, 3, 1]
    assert info["mixture_weights"] == [3.0, 1.0, 2.0]
    assert batch["num_items"] == 2
    assert all("source" in item for item in batch["items"])
    assert data_loader.collate_fn.kwargs["temporal_pad_lengths"] == {
        "images": 33,
        "state": 4,
        "action": 192,
    }


def test_noe_robotics_company_training_config_contract():
    config_path = (
        Path(__file__).resolve().parent
        / "fixtures/dreamzero_mixed/dreamzero_pretrain_v0a0_wo_img_ff+robotics.yaml"
    )
    cfg = OmegaConf.load(config_path)
    datasets = list(cfg.data.train_datasets)

    expected_entries = [
        (
            "agibot_world_alpha",
            "../data/noe_data_robotics/agibot_world-alpha",
            ".assets/agibot_world_alpha_metadata.json",
            0.3932,
            True,
        ),
        (
            "robocoin_cobot_magic_640x480",
            "../data/noe_data_robotics/robocoin/cobot_magic__3cam_front_640x480_wrist1_640x480_wrist2_640x480",
            ".assets/robocoin_cobot_magic_640x480_metadata.json",
            0.1861,
            True,
        ),
        (
            "robocoin_cobot_magic_1280x720",
            "../data/noe_data_robotics/robocoin/cobot_magic__3cam_front_1280x720_wrist1_1280x720_wrist2_1280x720",
            ".assets/robocoin_cobot_magic_1280x720_metadata.json",
            0.0332,
            True,
        ),
        (
            "robocoin_galaxea_r1_lite_1280x720",
            "../data/noe_data_robotics/robocoin/galaxea_r1_lite__3cam_front_1280x720_wrist1_1280x720_wrist2_1280x720",
            ".assets/robocoin_galaxea_r1_lite_1280x720_metadata.json",
            0.0579,
            True,
        ),
        (
            "robocoin_galbot_g1_mixed",
            "../data/noe_data_robotics/robocoin/galbot_g1__3cam_front_640x480_wrist1_640x368_wrist2_640x368",
            ".assets/robocoin_galbot_g1_mixed_metadata.json",
            0.0358,
            True,
        ),
        (
            "robomind_v1_agilex_640x480",
            "../data/noe_data_robotics/robomind_v1/group_03_image_640x480_wrist1_640x480_wrist2_640x480_agilex",
            ".assets/robomind_v1_agilex_640x480_metadata.json",
            0.0353,
            True,
        ),
        (
            "robomind_v2_ark_640x480",
            "../data/noe_data_robotics/robomind_v2-part1/group_04_image_640x480_wrist1_640x480_wrist2_640x480_ark",
            ".assets/robomind_v2_ark_640x480_cleaned_full_metadata.json",
            0.1377,
            False,
        ),
        (
            "robomind_v2_agilex_640x480",
            "../data/noe_data_robotics/robomind_v2-part2/group_02_agilex_image_640x480_wrist1_640x480_wrist2_640x480_rgb_cleaned_h264",
            ".assets/robomind_v2_agilex_640x480_cleaned_full_metadata.json",
            0.0686,
            False,
        ),
        (
            "robomind_v2_franka_main_1280x720",
            "../data/noe_data_robotics/robomind_v2-part2/group_06_franka_image_1280x720_wrist1_640x480_wrist2_640x480",
            ".assets/robomind_v2_franka_main_1280x720_metadata.json",
            0.0522,
            True,
        ),
    ]

    assert len(datasets) == 9
    assert len({dataset.name for dataset in datasets}) == 9
    assert "robomind_v2_franka_wrist1_1280x720" not in {
        dataset.name for dataset in datasets
    }
    assert [
        (
            dataset.name,
            dataset.data_paths,
            dataset.metadata_json_path,
            float(dataset.weight),
            bool(dataset.has_real_action),
        )
        for dataset in datasets
    ] == expected_entries
    assert sum(float(dataset.weight) for dataset in datasets) == pytest.approx(1.0)
    for dataset in datasets:
        assert dataset.embodiment_tag == "noe_robotics_16d"
        assert dataset.metadata_embodiment_tag == "noe_robotics_16d"
        assert dataset.tasks_filename == "tasks_v1.jsonl"
        assert dataset.relative_action is False
        assert dataset.relative_action_mode == ""
        assert list(dataset.relative_action_keys) == []
        assert dataset.image_resolution_width == 160
        assert dataset.image_resolution_height == 128
        assert "concat_method" not in dataset
        assert "view_resize_mode" not in dataset
        assert "view_pad_value" not in dataset
        assert "video_crop_scale" not in dataset

    assert cfg.data.train_data_paths == "../data/noe_data_robotics"
    assert cfg.data.mixture.epoch_size == 200000
    assert cfg.data.mixture.seed == 42
    assert "episode_instruction_key" not in cfg.data
    assert "filter_noe_umi_qualified_episodes" not in cfg.data
    assert cfg.data.return_episode_first_images is True
    assert cfg.actor.model.embodiment_tag == "noe_robotics_16d"
    assert cfg.actor.model.metadata_embodiment_tag == "noe_robotics_16d"
    assert cfg.actor.model.embodiment_tag_mapping.noe_robotics_16d == 39
    assert cfg.actor.model.max_action_dim == 16
    assert cfg.actor.model.max_state_dim == 16
    assert cfg.actor.model.relative_action is False
    assert cfg.actor.model.relative_action_mode is None
    assert list(cfg.actor.model.relative_action_keys) == []
    action_head_cfg = cfg.actor.model.action_head_cfg.config
    assert action_head_cfg.use_episode_first_frame_token is True
    assert action_head_cfg.episode_first_frame_rope_index == 1000
