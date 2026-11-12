# Copyright 2026 The RLinf Authors.

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rlinf.data.datasets.dreamzero.dreamzero import (
    DreamZeroCollator,
    DreamZeroLeRobotDataset,
)
from rlinf.data.datasets.dreamzero.noe_robotics_contract import (
    NOE_ROBOTICS_EEF_CHANNELS,
    validate_noe_robotics_modality,
)
from rlinf.data.datasets.dreamzero.relative_actions import (
    RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ,
    ensure_relative_action_metadata_compatible,
    robotwin_eef_rel_to_abs_wxyz,
)
from rlinf.data.datasets.dreamzero.utils import load_task_texts, safe_lang_text

VIDEO_KEYS = [
    "video.image",
    "video.wrist_image_1",
    "video.wrist_image_2",
]


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _packed_modality(end_dim=30):
    state = {
        name: {"start": i, "end": i + 1, "canonical_id": i}
        for i, name in enumerate(NOE_ROBOTICS_EEF_CHANNELS)
    }
    action = {
        name: {"start": i, "end": i + 1, "canonical_id": i}
        for i, name in enumerate(NOE_ROBOTICS_EEF_CHANNELS)
    }
    for i in range(16, end_dim):
        state[f"extra_{i}"] = {
            "start": i,
            "end": i + 1,
            "canonical_id": i + 100,
        }
        action[f"extra_{i}"] = {
            "start": i,
            "end": i + 1,
            "canonical_id": i + 100,
        }
    return {"state": state, "action": action}


def _write_robotics_tree(root, *, vector_dim=30, modality=None):
    meta_dir = root / "meta"
    data_dir = root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    video_features = {
        "observation.images.image": {
            "dtype": "image",
            "shape": [480, 640, 3],
        },
        "observation.images.wrist_image_1": {
            "dtype": "image",
            "shape": [480, 640, 3],
        },
        "observation.images.wrist_image_2": {
            "dtype": "image",
            "shape": [480, 640, 3],
        },
    }
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": 1,
        "total_videos": 0,
        "chunks_size": 1000,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "features": {
            **video_features,
            "observation.state": {"dtype": "float32", "shape": [vector_dim]},
            "action": {"dtype": "float32", "shape": [vector_dim]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta_dir / "modality.json").write_text(
        json.dumps(modality if modality is not None else _packed_modality(vector_dim)),
        encoding="utf-8",
    )
    _write_jsonl(meta_dir / "tasks_v1.jsonl", [{"task_index": 0, "task": "move"}])
    _write_jsonl(meta_dir / "episodes.jsonl", [{"episode_index": 0, "length": 1}])

    vector_type = pa.list_(pa.float32(), vector_dim)
    table = pa.table(
        {
            **{
                source_key: pa.array([b"unused"], type=pa.binary())
                for source_key in video_features
            },
            "observation.state": pa.array(
                [[float(i) for i in range(vector_dim)]],
                type=vector_type,
            ),
            "action": pa.array(
                [[float(i) for i in range(vector_dim)]],
                type=vector_type,
            ),
            "task_index": pa.array([0], type=pa.int64()),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")


def _build_robotics_dataset(root, **kwargs):
    return DreamZeroLeRobotDataset(
        data_path=str(root),
        video_keys=VIDEO_KEYS,
        state_keys=["state.eef_position"],
        action_keys=["action.eef_position"],
        language_keys=["annotation.language.action_text"],
        data_transform=lambda raw: raw,
        lazy_load=True,
        num_frames=25,
        state_horizon=1,
        action_horizon=48,
        max_chunk_size=4,
        embodiment_tag="noe_robotics_16d",
        tasks_filename="tasks_v1.jsonl",
        **kwargs,
    )


def test_load_task_texts_keeps_tasks_jsonl_as_default(tmp_path):
    _write_jsonl(tmp_path / "tasks.jsonl", [{"task_index": 0, "task": "old"}])
    _write_jsonl(tmp_path / "tasks_v1.jsonl", [{"task_index": 0, "task": "new"}])

    assert load_task_texts(tmp_path) == {0: "old"}


def test_load_task_texts_uses_explicit_tasks_v1(tmp_path):
    _write_jsonl(tmp_path / "tasks.jsonl", [{"task_index": 0, "task": "old"}])
    _write_jsonl(tmp_path / "tasks_v1.jsonl", [{"task_index": 0, "task": "new"}])

    assert load_task_texts(tmp_path, tasks_filename="tasks_v1.jsonl") == {0: "new"}


def test_load_task_texts_samples_explicit_variants(tmp_path, monkeypatch):
    _write_jsonl(
        tmp_path / "tasks_v1_variants_100.jsonl",
        [
            {
                "task_index": 0,
                "task": "base",
                "variant_count": 2,
                "variants": ["first", "second"],
            }
        ],
    )
    task_map = load_task_texts(
        tmp_path,
        tasks_filename="tasks_v1_variants_100.jsonl",
    )
    assert task_map == {0: ("first", "second")}
    monkeypatch.setattr(np.random, "choice", lambda values: values[1])
    assert safe_lang_text(0, task_map) == "second"


def test_load_task_texts_rejects_missing_explicit_tasks_file(tmp_path):
    _write_jsonl(tmp_path / "tasks.jsonl", [{"task_index": 0, "task": "old"}])

    with pytest.raises(FileNotFoundError, match="tasks_v1.jsonl"):
        load_task_texts(tmp_path, tasks_filename="tasks_v1.jsonl")


def test_load_task_texts_rejects_empty_explicit_tasks_file(tmp_path):
    (tmp_path / "tasks_v1.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Configured task file .* not found"):
        load_task_texts(tmp_path, tasks_filename="tasks_v1.jsonl")


def test_load_task_texts_rejects_nested_explicit_tasks_path(tmp_path):
    with pytest.raises(ValueError, match="must be a basename"):
        load_task_texts(tmp_path, tasks_filename="../tasks_v1.jsonl")


def test_load_task_texts_rejects_blank_explicit_caption(tmp_path):
    _write_jsonl(tmp_path / "tasks_v1.jsonl", [{"task_index": 0, "task": "  "}])

    with pytest.raises(ValueError, match="Empty task caption"):
        load_task_texts(tmp_path, tasks_filename="tasks_v1.jsonl")


def test_noe_robotics_modality_accepts_first_16_canonical_eef_channels():
    validate_noe_robotics_modality(_packed_modality())


def test_noe_robotics_modality_rejects_quaternion_order_change():
    modality = _packed_modality()
    modality["action"]["left_quat_w"]["canonical_id"] = 4

    with pytest.raises(ValueError, match="action.left_quat_w"):
        validate_noe_robotics_modality(modality)


def test_noe_robotics_modality_rejects_noncontiguous_first_16_slice():
    modality = _packed_modality()
    modality["state"]["right_gripper"]["end"] = 17

    with pytest.raises(ValueError, match="state.right_gripper"):
        validate_noe_robotics_modality(modality)


@pytest.mark.parametrize("reverse", [False, True])
def test_prompt_postfix_survives_dataset_transform_and_mixed_collation(
    tmp_path, monkeypatch, reverse
):
    _write_robotics_tree(tmp_path)
    features = []
    for postfix in (
        "  The robot model is Cobot Magic.  ",
        "The robot model is ARX Lift-2.",
        "",
    ):
        dataset = _build_robotics_dataset(
            tmp_path, prompt_postfix=postfix, sampling_mode="fixed_window"
        )
        monkeypatch.setattr(
            dataset,
            "_decode_v2_image",
            lambda _: np.zeros((2, 2, 3), dtype=np.uint8),
        )
        dataset.data_transform = lambda raw: {
            "text": raw["annotation.language.action_text"],
            "text_negative": "Negative prompt.",
            "embodiment_id": 39,
        }
        features.append(dataset[0])
    if reverse:
        features.reverse()

    captured = []

    def tokenizer(texts, **kwargs):
        captured.append(list(texts))
        return np.zeros((len(texts), 8), dtype=np.int64), np.ones(
            (len(texts), 8), dtype=bool
        )

    batch = DreamZeroCollator.collate_batch(
        features,
        tokenizer,
        {"noe_robotics_16d": 39},
    )
    from rlinf.data.datasets.dreamzero.data_transforms import format_training_prompt

    base_prompt = format_training_prompt("move.", 39, {"noe_robotics_16d": 39})
    expected = [
        base_prompt + " The robot model is Cobot Magic.",
        base_prompt + " The robot model is ARX Lift-2.",
        base_prompt,
    ]
    assert captured[0] == (expected[::-1] if reverse else expected)
    assert captured[1] == ["Negative prompt."] * 3
    assert "prompt_postfix" not in batch
    assert batch["text"].shape == (3, 8)
    assert batch["embodiment_id"].tolist() == [39, 39, 39]


@pytest.mark.parametrize("instruction", ["", "   ", "['']"])
def test_prompt_postfix_does_not_reintroduce_empty_language(instruction):
    captured = []

    def tokenizer(texts, **kwargs):
        captured.extend(texts)
        return np.zeros((len(texts), 8)), np.ones((len(texts), 8))

    batch = DreamZeroCollator.collate_batch(
        [
            {
                "text": instruction,
                "embodiment_id": 39,
                "prompt_postfix": "The robot model is Cobot Magic.",
            }
        ],
        tokenizer,
        {"noe_robotics_16d": 39},
    )
    assert captured == [""]
    assert "prompt_postfix" not in batch


@pytest.mark.parametrize("postfix", [None, "", "The robot model is Genie-1."])
def test_single_dataset_loader_passes_prompt_postfix(tmp_path, monkeypatch, postfix):
    import groot.vla.data.transform as groot_transform
    from omegaconf import OmegaConf

    import rlinf.data.datasets.dreamzero.data_transforms as registry
    import rlinf.data.datasets.dreamzero.dreamzero as dataset_module

    _write_robotics_tree(tmp_path)

    class TextTransform:
        def set_metadata(self, metadata):
            pass

        def train(self):
            pass

        def __call__(self, raw):
            return {"text": raw["annotation.language.action_text"], "embodiment_id": 39}

    monkeypatch.setattr(groot_transform, "ComposedModalityTransform", TextTransform)
    monkeypatch.setattr(registry, "load_dreamzero_dataset_metadata", lambda _: None)
    monkeypatch.setattr(
        registry, "build_dreamzero_composed_transform", lambda *_: TextTransform()
    )
    monkeypatch.setattr(
        registry,
        "collect_dreamzero_dataset_keys",
        lambda *_: (
            VIDEO_KEYS,
            ["state.eef_position"],
            ["action.eef_position"],
            ["annotation.language.action_text"],
        ),
    )
    monkeypatch.setattr(
        DreamZeroLeRobotDataset,
        "_decode_v2_image",
        staticmethod(lambda _: np.zeros((2, 2, 3), dtype=np.uint8)),
    )
    captured = []

    def tokenizer(texts, **kwargs):
        captured.extend(texts)
        return np.zeros((len(texts), 8)), np.ones((len(texts), 8))

    def init_collator(self, **kwargs):
        self.tokenizer = tokenizer
        self.embodiment_tag_mapping = kwargs["embodiment_tag_mapping"]
        self.temporal_pad_lengths = None
        self.profile = False
        self.profile_seen = 0

    monkeypatch.setattr(DreamZeroCollator, "__init__", init_collator)
    cfg = OmegaConf.create(
        {
            "data": {
                "num_workers": 0,
                "sampling_mode": "fixed_window",
                "tasks_filename": "tasks_v1.jsonl",
                "prompt_postfix": postfix,
            },
            "actor": {
                "micro_batch_size": 1,
                "model": {
                    "embodiment_tag": "noe_robotics_16d",
                    "action_horizon": 48,
                    "action_head_cfg": {
                        "config": {
                            "num_frames": 25,
                            "diffusion_model_cfg": {"max_chunk_size": 4},
                        }
                    },
                },
            },
        }
    )
    loader, _ = dataset_module.build_dreamzero_sft_dataloader(
        cfg,
        data_paths=str(tmp_path),
        world_size=1,
        rank=0,
    )
    batch = next(iter(loader))
    base_prompt = registry.format_training_prompt("move.", 39, {"noe_robotics_16d": 39})
    assert captured == [base_prompt + (" " + postfix if postfix else "")]
    assert "prompt_postfix" not in batch


def test_prompt_postfix_rejects_non_string_values(tmp_path):
    _write_robotics_tree(tmp_path)
    with pytest.raises(TypeError, match="prompt_postfix"):
        _build_robotics_dataset(tmp_path, prompt_postfix=123)


def test_noe_robotics_dataset_maps_and_slices_first_16_eef_channels(tmp_path):
    _write_robotics_tree(tmp_path)

    dataset = _build_robotics_dataset(tmp_path)

    assert dataset._source_video_key == {
        "video.image": "observation.images.image",
        "video.wrist_image_1": "observation.images.wrist_image_1",
        "video.wrist_image_2": "observation.images.wrist_image_2",
    }
    assert dataset._state_components == {
        "state.eef_position": ("observation.state", slice(0, 16))
    }
    assert dataset._action_components == {
        "action.eef_position": ("action", slice(0, 16))
    }
    assert dataset.tasks_filename == "tasks_v1.jsonl"

    sample = {
        **{
            key: np.zeros((1, 2, 2, 3), dtype=np.uint8)
            for key in dataset._source_video_key.values()
        },
        "observation.state": np.arange(30, dtype=np.float32)[None, :],
        "action": np.arange(60, dtype=np.float32).reshape(2, 30),
        "task_index": 0,
    }
    mapped = dataset._build_modality_dict(sample)

    np.testing.assert_array_equal(
        mapped["state.eef_position"], sample["observation.state"][:, :16]
    )
    np.testing.assert_array_equal(
        mapped["action.eef_position"], sample["action"][:, :16]
    )
    assert mapped["annotation.language.action_text"] == "move"


def test_noe_robotics_dataset_validates_modality_during_construction(tmp_path):
    modality = _packed_modality()
    modality["action"]["left_quat_w"]["canonical_id"] = 4
    _write_robotics_tree(tmp_path, modality=modality)

    with pytest.raises(ValueError, match="action.left_quat_w"):
        _build_robotics_dataset(tmp_path)


def test_robotics_relative_maps_first_16d_with_one_sample_anchor(tmp_path):
    _write_robotics_tree(tmp_path)
    dataset = _build_robotics_dataset(
        tmp_path,
        relative_action=True,
        relative_action_mode=RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ,
        relative_action_keys=["eef_position"],
    )
    state = np.zeros((2, 30), dtype=np.float32)
    state[:, [3, 11]] = 1
    state[:, 0] = [10, 15]
    state[:, 8] = [-10, -3]
    state[:, [7, 15]] = [[0.2, 0.8], [0.4, 0.6]]
    action = np.repeat(state, 48, axis=0)
    action[:, 0] += 1
    action[:, 8] += 2
    action[:, [7, 15]] = [0.1, 0.9]
    action[::2, 3:7] *= -1
    action[1::2, 11:15] *= -1
    # 后续附加通道不参与 EEF；第二个 chunk 仍参考 sample 的第一条 state。
    action[:, 16:] = 1234
    sample = {
        **{key: np.zeros((1, 2, 2, 3), dtype=np.uint8) for key in dataset._source_video_key.values()},
        "observation.state": state,
        "action": action,
        "task_index": 0,
    }
    mapped = dataset._build_modality_dict(sample)
    rel_state = mapped["state.eef_position"]
    rel_action = mapped["action.eef_position"]
    np.testing.assert_allclose(rel_state[:, [0, 8]], [[0, 0], [5, 7]])
    np.testing.assert_allclose(rel_action[:, 0], np.repeat([1, 6], 48))
    np.testing.assert_allclose(rel_action[:, 8], np.repeat([2, 9], 48))
    np.testing.assert_array_equal(rel_state[:, [7, 15]], state[:, [7, 15]])
    np.testing.assert_array_equal(rel_action[:, [7, 15]], action[:, [7, 15]])
    assert rel_action.shape == (96, 16)
    np.testing.assert_array_equal(rel_action[:, [3, 11]], np.ones((96, 2)))
    restored = robotwin_eef_rel_to_abs_wxyz(rel_action, state[:1, :16])
    np.testing.assert_allclose(restored[:, [0, 1, 2, 8, 9, 10]], action[:, [0, 1, 2, 8, 9, 10]])


@pytest.mark.parametrize("metadata_mode", [None, "noe_umi_eef_se3_s0_wxyz", "robotwin_eef_se3_s0_wxyz"])
def test_robotics_relative_rejects_wrong_metadata(metadata_mode):
    with pytest.raises(ValueError, match="requires metadata relative_action_mode"):
        ensure_relative_action_metadata_compatible(
            embodiment_tag="noe_robotics_16d",
            relative_action=True,
            relative_action_mode=RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ,
            metadata_relative_action_mode=metadata_mode,
        )
