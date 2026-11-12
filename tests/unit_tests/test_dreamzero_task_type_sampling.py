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

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.dreamzero.task_type_sampling import (
    ACTION_TYPE_VALUES,
    TaskTypeSamplingConfig,
    TaskTypeSamplingPlan,
    build_task_type_frame_index,
    merge_task_type_frame_indices,
)


def _write_sidecar(root, rows, filename="task_type_v2.jsonl"):
    metadata_dir = root / "meta" / "task_type"
    metadata_dir.mkdir(parents=True)
    path = metadata_dir / filename
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_frame_index_does_not_fallback_to_flat_sidecar_path(tmp_path):
    metadata_dir = tmp_path / "meta"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "task_type_v2.jsonl").write_text(
        json.dumps({"episode_index": 0, "task_type": "Cleaning"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="meta/task_type/task_type_v2.jsonl",
    ):
        build_task_type_frame_index(
            tmp_path,
            episodes=[0],
            episode_starts=[0, 4],
            config=_enabled_config(),
        )


def _enabled_config(weights=None):
    value = {"enabled": True}
    if weights is not None:
        value["weights"] = weights
    return TaskTypeSamplingConfig.from_value(value)


def _action_weights(**overrides):
    weights = dict.fromkeys(ACTION_TYPE_VALUES, 1.0)
    weights.update(overrides)
    return weights


def test_config_accepts_omegaconf_mapping():
    config = TaskTypeSamplingConfig.from_value(
        OmegaConf.create(
            {
                "enabled": True,
                "metadata_file": "meta/task_type/task_type_v2.jsonl",
                "weights": {"Cleaning": 2.0},
            }
        )
    )

    assert config.enabled is True
    assert config.metadata_file == "meta/task_type/task_type_v2.jsonl"
    assert config.weights == {"Cleaning": 2.0}


def test_config_accepts_zero_weight():
    config = _enabled_config({"Cleaning": 1.0, "Other/Unclear": 0.0})

    assert config.weights == {"Cleaning": 1.0, "Other/Unclear": 0.0}


def test_action_type_config_requires_explicit_weights():
    with pytest.raises(ValueError, match="requires explicit weights"):
        TaskTypeSamplingConfig.from_value(
            {"enabled": True, "label_space": "action_type"}
        )


def test_action_type_config_uses_action_sidecar_by_default():
    config = TaskTypeSamplingConfig.from_value(
        {
            "enabled": True,
            "label_space": "action_type",
            "weights": _action_weights(),
        }
    )

    assert config.metadata_file == "meta/task_type/action_type.jsonl"
    assert len(config.label_values) == 28
    assert config.label_value_set == frozenset(config.label_values)


def test_action_type_config_requires_all_weights():
    with pytest.raises(ValueError, match="must list all action types"):
        TaskTypeSamplingConfig.from_value(
            {
                "enabled": True,
                "label_space": "action_type",
                "weights": {"organize": 1.0},
            }
        )


@pytest.mark.parametrize(
    "invalid_weight",
    [-1.0, float("nan"), float("inf"), float("-inf")],
)
def test_config_rejects_invalid_weight(invalid_weight):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _enabled_config({"Cleaning": invalid_weight})


def test_config_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="at least one positive weight"):
        _enabled_config({"Cleaning": 0.0, "Other/Unclear": 0.0})


def test_build_frame_index_uses_frame_ranges_and_counts_extra_rows(tmp_path):
    _write_sidecar(
        tmp_path,
        [
            {"episode_index": 0, "task_type": "Basic Manipulation"},
            {"episode_index": 1, "task_type": "Cleaning"},
            {"episode_index": 9, "task_type": "Organization"},
        ],
    )

    index = build_task_type_frame_index(
        tmp_path,
        episodes=[0, 1],
        episode_starts=[0, 100, 110],
        config=_enabled_config(),
    )

    assert index.episode_count == 2
    assert index.frame_count == 110
    assert index.sidecar_rows == 3
    assert index.extra_rows == 1
    assert index.episode_counts == {
        "Basic Manipulation": 1,
        "Cleaning": 1,
    }
    assert index.frame_counts == {
        "Basic Manipulation": 100,
        "Cleaning": 10,
    }
    cleaning = index.ranges["Cleaning"]
    np.testing.assert_array_equal(
        cleaning.sample_from_offsets(np.asarray([0, 9])),
        np.asarray([100, 109]),
    )


def test_action_type_frame_index_and_sampling_plan(tmp_path):
    _write_sidecar(
        tmp_path,
        [
            {"episode_index": 0, "task_type": "organize"},
            {"episode_index": 1, "task_type": "place"},
        ],
        filename="action_type.jsonl",
    )
    config = TaskTypeSamplingConfig.from_value(
        {
            "enabled": True,
            "label_space": "action_type",
            "weights": _action_weights(organize=3.0),
        }
    )
    index = build_task_type_frame_index(
        tmp_path,
        episodes=[0, 1],
        episode_starts=[0, 100, 110],
        config=config,
    )
    plan = TaskTypeSamplingPlan.from_config(index, config)
    sample_ids = plan.sample_many(20_000, torch.Generator().manual_seed(7))

    assert plan.task_types == ("organize", "place")
    assert 0.72 <= float((sample_ids < 100).double().mean()) <= 0.78
    assert int(sample_ids.min()) >= 0
    assert int(sample_ids.max()) < 110


def test_task_type_config_rejects_action_label(tmp_path):
    _write_sidecar(
        tmp_path,
        [{"episode_index": 0, "task_type": "organize"}],
    )

    with pytest.raises(ValueError, match="Invalid task_type"):
        build_task_type_frame_index(
            tmp_path,
            episodes=[0],
            episode_starts=[0, 4],
            config=_enabled_config(),
        )


def test_action_type_config_rejects_task_label(tmp_path):
    _write_sidecar(
        tmp_path,
        [{"episode_index": 0, "task_type": "Organization"}],
        filename="action_type.jsonl",
    )
    config = TaskTypeSamplingConfig.from_value(
        {
            "enabled": True,
            "label_space": "action_type",
            "weights": _action_weights(),
        }
    )

    with pytest.raises(ValueError, match="Invalid action_type"):
        build_task_type_frame_index(
            tmp_path,
            episodes=[0],
            episode_starts=[0, 4],
            config=config,
        )


def test_frame_offsets_are_uniform_across_multiple_episode_ranges(tmp_path):
    _write_sidecar(
        tmp_path,
        [
            {"episode_index": 0, "task_type": "Cleaning"},
            {"episode_index": 1, "task_type": "Cleaning"},
        ],
    )
    index = build_task_type_frame_index(
        tmp_path,
        episodes=[0, 1],
        episode_starts=[0, 100, 110],
        config=_enabled_config(),
    )

    cleaning = index.ranges["Cleaning"]
    np.testing.assert_array_equal(
        cleaning.sample_from_offsets(np.asarray([0, 99, 100, 109])),
        np.asarray([0, 99, 100, 109]),
    )


def test_frame_index_ignores_zero_length_episodes(tmp_path):
    _write_sidecar(
        tmp_path,
        [
            {"episode_index": 0, "task_type": "Cleaning"},
            {"episode_index": 1, "task_type": "Organization"},
            {"episode_index": 2, "task_type": "Basic Manipulation"},
        ],
    )

    index = build_task_type_frame_index(
        tmp_path,
        episodes=[0, 1, 2],
        episode_starts=[0, 4, 4, 10],
        config=_enabled_config(),
    )

    assert index.episode_count == 2
    assert index.frame_count == 10
    assert index.extra_rows == 1
    assert index.episode_counts == {
        "Basic Manipulation": 1,
        "Cleaning": 1,
    }
    assert "Organization" not in index.ranges


def test_frame_index_rejects_missing_duplicate_and_invalid_rows(tmp_path):
    missing_root = tmp_path / "missing"
    _write_sidecar(
        missing_root,
        [{"episode_index": 0, "task_type": "Cleaning"}],
    )
    with pytest.raises(ValueError, match="missing 1 usable episodes"):
        build_task_type_frame_index(
            missing_root,
            episodes=[0, 1],
            episode_starts=[0, 4, 8],
            config=_enabled_config(),
        )

    duplicate_root = tmp_path / "duplicate"
    _write_sidecar(
        duplicate_root,
        [
            {"episode_index": 0, "task_type": "Cleaning"},
            {"episode_index": 0, "task_type": "Cleaning"},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate episode_index=0"):
        build_task_type_frame_index(
            duplicate_root,
            episodes=[0],
            episode_starts=[0, 4],
            config=_enabled_config(),
        )

    invalid_root = tmp_path / "invalid"
    _write_sidecar(
        invalid_root,
        [{"episode_index": 0, "task_type": "Painting"}],
    )
    with pytest.raises(ValueError, match="Invalid task_type"):
        build_task_type_frame_index(
            invalid_root,
            episodes=[0],
            episode_starts=[0, 4],
            config=_enabled_config(),
        )


def test_merge_indices_scopes_duplicate_episode_ids_by_root(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    for root in (root_a, root_b):
        _write_sidecar(
            root,
            [{"episode_index": 0, "task_type": "Cleaning"}],
        )
    index_a = build_task_type_frame_index(
        root_a,
        episodes=[0],
        episode_starts=[0, 10],
        config=_enabled_config(),
    )
    index_b = build_task_type_frame_index(
        root_b,
        episodes=[0],
        episode_starts=[0, 5],
        config=_enabled_config(),
    )

    merged = merge_task_type_frame_indices([index_a, index_b], offsets=[0, 10])

    assert merged.episode_count == 2
    assert merged.frame_count == 15
    np.testing.assert_array_equal(
        merged.ranges["Cleaning"].starts,
        np.asarray([0, 10]),
    )
    np.testing.assert_array_equal(
        merged.ranges["Cleaning"].sample_from_offsets(np.asarray([0, 9, 10, 14])),
        np.asarray([0, 9, 10, 14]),
    )


def test_sampling_plan_respects_custom_type_weights(tmp_path):
    _write_sidecar(
        tmp_path,
        [
            {"episode_index": 0, "task_type": "Basic Manipulation"},
            {"episode_index": 1, "task_type": "Cleaning"},
        ],
    )
    config = _enabled_config(
        {
            "Basic Manipulation": 3.0,
            "Cleaning": 1.0,
        }
    )
    index = build_task_type_frame_index(
        tmp_path,
        episodes=[0, 1],
        episode_starts=[0, 100, 110],
        config=config,
    )
    plan = TaskTypeSamplingPlan.from_config(index, config)
    generator = torch.Generator().manual_seed(7)

    sample_ids = plan.sample_many(20_000, generator)
    basic_ratio = float((sample_ids < 100).double().mean())

    assert 0.72 <= basic_ratio <= 0.78
    assert int(sample_ids.min()) >= 0
    assert int(sample_ids.max()) < 110


def test_sampling_plan_excludes_zero_weight_types(tmp_path):
    _write_sidecar(
        tmp_path,
        [
            {"episode_index": 0, "task_type": "Basic Manipulation"},
            {"episode_index": 1, "task_type": "Cleaning"},
        ],
    )
    config = _enabled_config(
        {
            "Basic Manipulation": 1.0,
            "Cleaning": 0.0,
            "Gardening": 1.0,
        }
    )
    index = build_task_type_frame_index(
        tmp_path,
        episodes=[0, 1],
        episode_starts=[0, 100, 110],
        config=config,
    )
    plan = TaskTypeSamplingPlan.from_config(index, config)

    sample_ids = plan.sample_many(10_000, torch.Generator().manual_seed(7))

    assert plan.task_types == ("Basic Manipulation",)
    assert plan.effective_weights == {"Basic Manipulation": 1.0}
    assert plan.unavailable_types == ("Gardening",)
    assert plan.disabled_types == ("Cleaning",)
    assert int(sample_ids.min()) >= 0
    assert int(sample_ids.max()) < 100


def test_sampling_plan_requires_positive_weight_for_an_available_type(tmp_path):
    _write_sidecar(
        tmp_path,
        [{"episode_index": 0, "task_type": "Cleaning"}],
    )
    config = _enabled_config({"Cleaning": 0.0, "Gardening": 1.0})
    index = build_task_type_frame_index(
        tmp_path,
        episodes=[0],
        episode_starts=[0, 4],
        config=config,
    )

    with pytest.raises(ValueError, match="no enabled available types"):
        TaskTypeSamplingPlan.from_config(index, config)


def test_sampling_plan_requires_weights_for_every_present_type(tmp_path):
    _write_sidecar(
        tmp_path,
        [{"episode_index": 0, "task_type": "Cleaning"}],
    )
    index = build_task_type_frame_index(
        tmp_path,
        episodes=[0],
        episode_starts=[0, 4],
        config=_enabled_config(),
    )
    incomplete = _enabled_config({"Organization": 1.0})

    with pytest.raises(ValueError, match="missing labels"):
        TaskTypeSamplingPlan.from_config(index, incomplete)
