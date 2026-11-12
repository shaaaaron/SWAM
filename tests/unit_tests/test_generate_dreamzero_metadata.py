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
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from rlinf.data.datasets.dreamzero.noe_robotics_contract import (  # noqa: E402
    NOE_ROBOTICS_EEF_CHANNELS,
)
from rlinf.data.datasets.dreamzero.relative_actions import (  # noqa: E402
    ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS,
    RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
    robotwin_eef_abs_to_rel_wxyz,
    temporal_continuous_robotwin_eef_action_wxyz,
)
from toolkits.lerobot import (
    generate_dreamzero_metadata as metadata_module,  # noqa: E402
)
from toolkits.lerobot.generate_dreamzero_metadata import (  # noqa: E402
    _build_split_modalities,
    _compute_stats,
    _feature_resolution,
    _resolve_metadata_dataset_roots,
    _robotwin_eef_relative_samples,
    _video_metadata_from_feature,
    build_metadata,
)


def test_compute_stats_bounds_quantile_samples_only():
    values = np.arange(40, dtype=np.float32).reshape(20, 2)

    stats = _compute_stats(values, quantile_sample_limit=5)
    quantile_values = values[::4]

    np.testing.assert_allclose(stats["min"], np.min(values, axis=0))
    np.testing.assert_allclose(stats["max"], np.max(values, axis=0))
    np.testing.assert_allclose(stats["mean"], np.mean(values, axis=0))
    np.testing.assert_allclose(stats["std"], np.std(values, axis=0))
    np.testing.assert_allclose(stats["q01"], np.quantile(quantile_values, 0.01, axis=0))
    np.testing.assert_allclose(stats["q99"], np.quantile(quantile_values, 0.99, axis=0))


def test_compute_stats_rejects_nonpositive_quantile_sample_limit():
    with pytest.raises(ValueError, match="quantile_sample_limit must be positive"):
        _compute_stats(np.ones((2, 1)), quantile_sample_limit=0)


def test_metadata_parent_expansion_is_scoped(tmp_path):
    child = tmp_path / "child"
    info_path = child / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text("{}\n", encoding="utf-8")

    assert _resolve_metadata_dataset_roots(
        tmp_path, embodiment_tag="robotwin"
    ) == [child]
    assert _resolve_metadata_dataset_roots(
        tmp_path, embodiment_tag="noe_umi_16d"
    ) == [child]
    with pytest.raises(FileNotFoundError, match="missing meta/info.json"):
        _resolve_metadata_dataset_roots(tmp_path, embodiment_tag="oxe_droid")


def test_split_modalities_preserve_explicit_absolute_semantics():
    modalities = _build_split_modalities(
        {
            "state": {
                "position": {"start": 0, "end": 3, "absolute": True},
            },
            "action": {
                "delta": {"start": 0, "end": 3, "absolute": False},
            },
        },
        kind="action",
        full_dim=3,
    )

    assert modalities["delta"]["absolute"] is False


def test_feature_resolution_uses_width_height_order_for_named_lerobot_images():
    feature = {
        "shape": [180, 320, 3],
        "names": ["height", "width", "channel"],
    }

    assert _feature_resolution(feature) == [320, 180]


def test_feature_resolution_uses_width_height_order_for_plain_image_shape():
    feature = {"shape": [180, 320, 3]}

    assert _feature_resolution(feature) == [320, 180]


@pytest.mark.parametrize(
    ("shape", "names"),
    [
        ([180, 320, 3], ["height", "width", "channel"]),
        ([3, 180, 320], ["channels", "height", "width"]),
    ],
)
def test_video_metadata_supports_channel_dimension_names(shape, names):
    info = {
        "features": {
            "observation.images.cam_high": {
                "shape": shape,
                "names": names,
                "info": {"video.fps": 50},
            }
        }
    }

    metadata = _video_metadata_from_feature(info, "observation.images.cam_high")

    assert metadata == {
        "resolution": [320, 180],
        "channels": 3,
        "fps": 50.0,
    }


def test_video_metadata_rejects_mismatched_shape_and_names():
    info = {
        "features": {
            "observation.images.cam_high": {
                "shape": [3, 180, 320],
                "names": ["height", "width"],
            }
        }
    }

    with pytest.raises(ValueError, match="mismatched shape and names"):
        _video_metadata_from_feature(info, "observation.images.cam_high")


def test_video_metadata_requires_named_channel_dimension():
    info = {
        "features": {
            "observation.images.cam_high": {
                "shape": [3, 180, 320],
                "names": ["depth", "height", "width"],
            }
        }
    }

    with pytest.raises(ValueError, match="must declare 'channel' or 'channels'"):
        _video_metadata_from_feature(info, "observation.images.cam_high")


def _quat_z(theta):
    half = theta / 2.0
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


def _pose16(xyz, quat, left_grip=0.0, right_xyz=None, right_quat=None, right_grip=0.0):
    right_xyz = right_xyz if right_xyz is not None else xyz
    right_quat = right_quat if right_quat is not None else quat
    return [
        *xyz,
        *quat,
        left_grip,
        *right_xyz,
        *right_quat,
        right_grip,
    ]


def test_build_metadata_can_generate_robotwin_eef_relative_action_stats(tmp_path):
    meta_dir = tmp_path / "meta"
    data_dir = tmp_path / "data" / "chunk-000"
    meta_dir.mkdir()
    data_dir.mkdir(parents=True)

    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "features": {
                    "observation.images.cam_high": {
                        "dtype": "video",
                        "shape": [64, 80, 3],
                        "names": ["height", "width", "channel"],
                        "video_info": {"video.fps": 10},
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0}) + "\n",
        encoding="utf-8",
    )
    (meta_dir / "modality.json").write_text(
        json.dumps(
            {
                "video": {"cam_high": {"original_key": "observation.images.cam_high"}},
                "state": {"eef_position": {"start": 0, "end": 16}},
                "action": {"eef_position": {"start": 0, "end": 16}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = np.asarray(
        [
            _pose16([0.0, 0.0, 0.0], _quat_z(0.0)),
            _pose16([1.0, 0.0, 0.0], _quat_z(np.pi / 2)),
            _pose16([2.0, 0.0, 0.0], _quat_z(np.pi / 2)),
            _pose16([3.0, 0.0, 0.0], _quat_z(np.pi)),
        ],
        dtype=np.float32,
    )
    action = np.asarray(
        [
            _pose16([0.5, 0.0, 0.0], _quat_z(np.pi / 4), left_grip=0.1),
            _pose16([1.5, 0.0, 0.0], _quat_z(np.pi / 2), left_grip=0.2),
            _pose16([2.5, 0.0, 0.0], _quat_z(np.pi), left_grip=0.3),
            _pose16([3.5, 0.0, 0.0], _quat_z(np.pi), left_grip=0.4),
        ],
        dtype=np.float32,
    )
    table = pa.Table.from_pydict(
        {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")

    metadata, stats = build_metadata(
        tmp_path,
        embodiment_tag="robotwin",
        state_key="observation.state",
        action_key="action",
        video_keys=None,
        use_modality_json=True,
        max_episodes=None,
        relative_action_mode="robotwin_eef_se3_s0_wxyz",
        relative_action_keys=["eef_position"],
        action_horizon=2,
        max_chunk_size=2,
    )

    expected_state_relative, expected_action_relative = (
        _robotwin_eef_relative_samples(
            state,
            action,
            {
                "state": {"eef_position": {"start": 0, "end": 16}},
                "action": {"eef_position": {"start": 0, "end": 16}},
            },
            relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
            action_horizon=2,
            max_chunk_size=2,
            episode_index=0,
        )
    )
    expected_state_stats = _compute_stats(expected_state_relative)
    expected_action_stats = _compute_stats(expected_action_relative)
    actual_state_stats = metadata["robotwin"]["statistics"]["state"]["eef_position"]
    actual_action_stats = metadata["robotwin"]["statistics"]["action"]["eef_position"]

    assert metadata["robotwin"]["relative_action_pose_reference"] == "per_arm"
    np.testing.assert_allclose(actual_state_stats["q99"], expected_state_stats["q99"])
    np.testing.assert_allclose(actual_action_stats["q99"], expected_action_stats["q99"])
    np.testing.assert_allclose(
        stats["state.eef_position"]["q01"], expected_state_stats["q01"]
    )
    np.testing.assert_allclose(
        stats["action.eef_position"]["q01"], expected_action_stats["q01"]
    )
    assert not np.allclose(
        expected_state_relative,
        state[: len(expected_state_relative)],
    )

    abs_metadata, _ = build_metadata(
        tmp_path,
        embodiment_tag="robotwin",
        state_key="observation.state",
        action_key="action",
        video_keys=None,
        use_modality_json=True,
        max_episodes=None,
    )
    assert "relative_action_mode" not in abs_metadata["robotwin"]


def test_build_metadata_applies_temporal_action_quaternion_continuity(tmp_path):
    meta_dir = tmp_path / "meta"
    data_dir = tmp_path / "data" / "chunk-000"
    meta_dir.mkdir()
    data_dir.mkdir(parents=True)

    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "chunks_size": 1000,
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/"
                    "episode_{episode_index:06d}.parquet"
                ),
                "features": {
                    "observation.images.cam_high": {
                        "dtype": "video",
                        "shape": [64, 80, 3],
                        "names": ["height", "width", "channel"],
                        "video_info": {"video.fps": 10},
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0}) + "\n",
        encoding="utf-8",
    )
    (meta_dir / "modality.json").write_text(
        json.dumps(
            {
                "video": {
                    "cam_high": {
                        "original_key": "observation.images.cam_high"
                    }
                },
                "state": {"eef_position": {"start": 0, "end": 16}},
                "action": {"eef_position": {"start": 0, "end": 16}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    positive = [1.0, 0.0, 0.0, 0.0]
    negative = [-1.0, 0.0, 0.0, 0.0]
    state = np.asarray(
        [
            _pose16([0.0, 0.0, 0.0], positive),
            _pose16([0.0, 0.0, 0.0], positive),
            _pose16([0.0, 0.0, 0.0], negative),
            _pose16([0.0, 0.0, 0.0], negative),
        ],
        dtype=np.float32,
    )
    action = np.asarray(
        [
            _pose16([0.0, 0.0, 0.0], negative),
            _pose16([0.0, 0.0, 0.0], positive),
            _pose16([0.0, 0.0, 0.0], positive),
            _pose16([0.0, 0.0, 0.0], positive),
        ],
        dtype=np.float32,
    )
    pq.write_table(
        pa.Table.from_pydict(
            {
                "observation.state": pa.array(
                    state.tolist(), type=pa.list_(pa.float32())
                ),
                "action": pa.array(
                    action.tolist(), type=pa.list_(pa.float32())
                ),
            }
        ),
        data_dir / "episode_000000.parquet",
    )

    metadata, _ = build_metadata(
        tmp_path,
        embodiment_tag="robotwin",
        state_key="observation.state",
        action_key="action",
        video_keys=None,
        use_modality_json=True,
        max_episodes=None,
        action_horizon=2,
        action_quaternion_mode=ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS,
    )

    expected_action = temporal_continuous_robotwin_eef_action_wxyz(
        action,
        state,
        action_horizon=2,
    )
    expected_stats = _compute_stats(expected_action)
    actual_entry = metadata["robotwin"]
    actual_stats = actual_entry["statistics"]["action"]["eef_position"]

    assert (
        actual_entry["action_quaternion_mode"]
        == ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS
    )
    for name in ("min", "max", "mean", "std", "q01", "q99"):
        np.testing.assert_allclose(actual_stats[name], expected_stats[name])
    assert not np.allclose(actual_stats["mean"], _compute_stats(action)["mean"])


def test_temporal_action_quaternion_metadata_rejects_relative_mode(tmp_path):
    with pytest.raises(ValueError, match="only supports absolute action metadata"):
        build_metadata(
            tmp_path,
            embodiment_tag="robotwin",
            state_key="observation.state",
            action_key="action",
            video_keys=None,
            use_modality_json=True,
            max_episodes=None,
            relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
            relative_action_keys=["eef_position"],
            action_horizon=2,
            max_chunk_size=2,
            action_quaternion_mode=ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS,
        )


def test_build_metadata_can_generate_qualified_noe_umi_relative_stats(tmp_path):
    dataset_root = tmp_path / "noe_umi_part"
    meta_dir = dataset_root / "meta"
    data_dir = dataset_root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    video_features = {
        key: {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
            "info": {"video.fps": 30},
        }
        for key, shape in {
            "observation.images.image": [64, 80, 3],
            "observation.images.wrist_image_1": [80, 64, 3],
            "observation.images.wrist_image_2": [80, 64, 3],
        }.items()
    }
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "chunks_size": 1000,
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
                ),
                "features": {
                    "observation.state": {"shape": [23]},
                    "action": {"shape": [23]},
                    **video_features,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    qualification = {
        "slam_validity_all": True,
        "width_detection": True,
        "width_calibration": True,
    }
    (meta_dir / "episodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 0, **qualification}),
                json.dumps(
                    {
                        "episode_index": 1,
                        **qualification,
                        "width_detection": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state_eef = np.asarray(
        [
            _pose16([0.0, 0.0, 0.0], _quat_z(0.0)),
            _pose16([1.0, 0.0, 0.0], _quat_z(np.pi / 2)),
            _pose16([2.0, 0.0, 0.0], _quat_z(np.pi / 2)),
            _pose16([3.0, 0.0, 0.0], _quat_z(np.pi)),
        ],
        dtype=np.float32,
    )
    action_eef = np.asarray(
        [
            _pose16([0.5, 0.0, 0.0], _quat_z(np.pi / 4)),
            _pose16([1.5, 0.0, 0.0], _quat_z(np.pi / 2)),
            _pose16([2.5, 0.0, 0.0], _quat_z(np.pi)),
            _pose16([3.5, 0.0, 0.0], _quat_z(np.pi)),
        ],
        dtype=np.float32,
    )
    ego = np.arange(28, dtype=np.float32).reshape(4, 7)
    state = np.concatenate([state_eef, ego], axis=1)
    action = np.concatenate([action_eef, ego + 100.0], axis=1)
    table = pa.Table.from_pydict(
        {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")

    metadata, stats = build_metadata(
        tmp_path,
        embodiment_tag="noe_umi_16d",
        state_key="observation.state",
        action_key="action",
        video_keys=list(video_features),
        use_modality_json=False,
        max_episodes=None,
        relative_action_mode=RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ,
        relative_action_keys=["eef_position"],
        action_horizon=2,
        max_chunk_size=2,
        filter_noe_umi_qualified_episodes=True,
    )

    expected_state, expected_action = _robotwin_eef_relative_samples(
        state,
        action,
        {
            "state": {"eef_position": {"start": 0, "end": 16}},
            "action": {"eef_position": {"start": 0, "end": 16}},
        },
        relative_action_mode=RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ,
        action_horizon=2,
        max_chunk_size=2,
        episode_index=0,
    )
    entry = metadata["noe_umi_16d"]
    assert entry["relative_action_mode"] == (
        RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ
    )
    np.testing.assert_allclose(
        entry["statistics"]["state"]["eef_position"]["q99"],
        _compute_stats(expected_state)["q99"],
    )
    np.testing.assert_allclose(
        entry["statistics"]["action"]["eef_position"]["q01"],
        _compute_stats(expected_action)["q01"],
    )
    assert len(stats["observation.state"]["mean"]) == 23
    assert len(stats["action"]["mean"]) == 23


def test_robotwin_eef_relative_stats_rejects_length_mismatch():
    modality_cfg = {
        "state": {"eef_position": {"start": 0, "end": 16}},
        "action": {"eef_position": {"start": 0, "end": 16}},
    }
    state = np.zeros((3, 16), dtype=np.float32)
    action = np.zeros((2, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="lengths to match"):
        _robotwin_eef_relative_samples(
            state,
            action,
            modality_cfg,
            relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
            action_horizon=2,
            max_chunk_size=2,
            episode_index=7,
        )


def test_robotwin_eef_chunk_relative_stats_use_each_chunk_anchor():
    modality_cfg = {
        "state": {"eef_position": {"start": 0, "end": 16}},
        "action": {"eef_position": {"start": 0, "end": 16}},
    }
    state = np.asarray(
        [
            _pose16([0.0, 0.0, 0.0], _quat_z(0.0)),
            _pose16([1.0, 0.0, 0.0], _quat_z(np.pi / 4)),
            _pose16([2.0, 0.0, 0.0], _quat_z(np.pi / 2)),
            _pose16([3.0, 0.0, 0.0], _quat_z(np.pi)),
        ],
        dtype=np.float32,
    )
    action = np.asarray(
        [
            _pose16([0.5, 0.0, 0.0], _quat_z(np.pi / 4)),
            _pose16([1.5, 0.0, 0.0], _quat_z(np.pi / 2)),
            _pose16([2.5, 0.0, 0.0], _quat_z(np.pi)),
            _pose16([3.5, 0.0, 0.0], _quat_z(np.pi)),
        ],
        dtype=np.float32,
    )

    state_samples, action_samples = _robotwin_eef_relative_samples(
        state,
        action,
        modality_cfg,
        relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
        action_horizon=2,
        max_chunk_size=2,
        anchor_stride=4,
        episode_index=10,
    )

    anchors = state[[0, 2]]
    np.testing.assert_allclose(
        state_samples,
        robotwin_eef_abs_to_rel_wxyz(anchors, anchors),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        action_samples,
        robotwin_eef_abs_to_rel_wxyz(
            action.reshape(2, 2, 16), anchors[:, None, :]
        ).reshape(4, 16),
        atol=1e-6,
    )


def test_robotwin_eef_relative_stats_warns_and_skips_short_episode():
    modality_cfg = {
        "state": {"eef_position": {"start": 0, "end": 16}},
        "action": {"eef_position": {"start": 0, "end": 16}},
    }
    state = np.zeros((1, 16), dtype=np.float32)
    action = np.zeros((1, 16), dtype=np.float32)

    with pytest.warns(UserWarning, match="Skipping this episode"):
        state_samples, action_samples = _robotwin_eef_relative_samples(
            state,
            action,
            modality_cfg,
            relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
            action_horizon=2,
            max_chunk_size=2,
            episode_index=8,
        )

    assert state_samples.shape == (0, 16)
    assert action_samples.shape == (0, 16)


def test_robotwin_eef_relative_stats_use_action_horizon_as_macro_stride():
    modality_cfg = {
        "state": {"eef_position": {"start": 0, "end": 16}},
        "action": {"eef_position": {"start": 0, "end": 16}},
    }
    state = np.asarray(
        [_pose16([float(t), 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]) for t in range(5)],
        dtype=np.float32,
    )
    action = state.copy()

    state_samples, action_samples = _robotwin_eef_relative_samples(
        state,
        action,
        modality_cfg,
        relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
        action_horizon=2,
        max_chunk_size=2,
        episode_index=9,
    )

    np.testing.assert_allclose(
        state_samples[:, 0], [0.0, 2.0, 0.0, 2.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        action_samples[:, 0],
        [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0],
    )


def _write_packed_robotics_dataset(root):
    meta_dir = root / "meta"
    data_dir = root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    video_shapes = {
        "observation.images.image": [720, 1280, 3],
        "observation.images.wrist_image_1": [480, 640, 3],
        "observation.images.wrist_image_2": [368, 640, 3],
    }
    video_features = {
        key: {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
            "video_info": {"video.fps": 30},
        }
        for key, shape in video_shapes.items()
    }
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "chunks_size": 1000,
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
                ),
                "features": {
                    "observation.state": {"shape": [30]},
                    "action": {"shape": [30]},
                    **video_features,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0}) + "\n",
        encoding="utf-8",
    )

    state = np.arange(180, dtype=np.float32).reshape(6, 30)
    action = state + 1000.0
    table = pa.Table.from_pydict(
        {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")
    modality = {
        kind: {name: {"start": i, "end": i + 1, "canonical_id": i}
               for i, name in enumerate(NOE_ROBOTICS_EEF_CHANNELS)}
        for kind in ("state", "action")
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality), encoding="utf-8")
    return list(video_shapes)


def test_build_metadata_generates_streaming_absolute_robotics_16d(tmp_path):
    video_keys = _write_packed_robotics_dataset(tmp_path)

    metadata, stats = build_metadata(
        tmp_path,
        embodiment_tag="noe_robotics_16d",
        state_key="observation.state",
        action_key="action",
        video_keys=video_keys,
        use_modality_json=False,
        max_episodes=None,
        quantile_sample_limit=4,
        streaming_stats=True,
    )

    entry = metadata["noe_robotics_16d"]
    assert entry["embodiment_tag"] == "noe_robotics_16d"
    assert entry["modalities"]["state"]["eef_position"]["shape"] == [16]
    assert entry["modalities"]["action"]["eef_position"]["shape"] == [16]
    assert len(entry["statistics"]["state"]["eef_position"]["q99"]) == 16
    assert len(entry["statistics"]["action"]["eef_position"]["q99"]) == 16
    assert len(stats["observation.state"]["mean"]) == 30
    assert len(stats["state.eef_position"]["mean"]) == 16
    assert entry["modalities"]["video"]["image"]["resolution"] == [1280, 720]
    assert entry["modalities"]["video"]["wrist_image_1"]["resolution"] == [
        640,
        480,
    ]


def test_streaming_stats_matches_materialized_exact_moments_and_is_deterministic():
    chunks = [
        np.arange(15, dtype=np.float64).reshape(3, 5),
        np.linspace(-3.0, 7.0, 25, dtype=np.float64).reshape(5, 5),
        np.full((2, 5), 11.5, dtype=np.float64),
    ]
    expected = _compute_stats(np.concatenate(chunks, axis=0))

    accumulators = [
        metadata_module._StreamingStatsAccumulator(
            quantile_sample_limit=4,
            seed=123,
        )
        for _ in range(2)
    ]
    results = []
    for accumulator in accumulators:
        for chunk in chunks:
            accumulator.update(chunk)
        results.append(accumulator.finalize())
        assert accumulator.count == 10
        assert accumulator.quantile_sample_rows <= 4

    for key in ("min", "max", "mean", "std"):
        np.testing.assert_allclose(
            results[0][key],
            expected[key],
            rtol=1e-10,
            atol=1e-12,
        )
    np.testing.assert_array_equal(results[0]["q01"], results[1]["q01"])
    np.testing.assert_array_equal(results[0]["q99"], results[1]["q99"])


def test_streaming_stats_rejects_nonfinite_and_feature_dimension_changes():
    accumulator = metadata_module._StreamingStatsAccumulator(
        quantile_sample_limit=4,
        seed=42,
    )

    with pytest.raises(ValueError, match="non-finite"):
        accumulator.update(np.asarray([[1.0, np.nan]]))

    accumulator.update(np.ones((2, 3)))
    with pytest.raises(ValueError, match="feature dimension"):
        accumulator.update(np.ones((1, 4)))


def test_streaming_metadata_requires_positive_quantile_limit(tmp_path):
    with pytest.raises(ValueError, match="quantile_sample_limit must be positive"):
        build_metadata(
            tmp_path,
            embodiment_tag="noe_robotics_16d",
            state_key="observation.state",
            action_key="action",
            video_keys=[],
            use_modality_json=False,
            max_episodes=None,
            quantile_sample_limit=None,
            streaming_stats=True,
        )


@pytest.mark.parametrize(
    ("tag", "mode"),
    [
        ("noe_umi_16d", RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ),
        ("noe_robotics_16d", RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ),
    ],
)
def test_streaming_relative_metadata_matches_materialized_samples(tmp_path, tag, mode):
    video_keys = _write_packed_robotics_dataset(tmp_path)
    state = np.zeros((6, 30), dtype=np.float32)
    state[:, 0] = 10 + np.arange(6)
    state[:, [3, 11]] = 1
    state[:, [7, 15]] = [0.2, 0.8]
    action = state.copy()
    action[:, 0] += 1
    action[:, [7, 15]] = [0.1, 0.9]
    action[::2, 3:7] *= -1
    parquet_path = tmp_path / "data/chunk-000/episode_000000.parquet"
    pq.write_table(pa.table({"observation.state": state.tolist(), "action": action.tolist()}), parquet_path)
    args = dict(
        embodiment_tag=tag,
        state_key="observation.state",
        action_key="action",
        video_keys=video_keys,
        use_modality_json=False,
        max_episodes=None,
        relative_action_mode=mode,
        relative_action_keys=["eef_position"],
        action_horizon=2,
        max_chunk_size=2,
        quantile_sample_limit=1000,
    )
    materialized, raw_stats = build_metadata(tmp_path, **args)
    streamed, streamed_stats = build_metadata(tmp_path, **args, streaming_stats=True)
    for kind in ("state", "action"):
        for stat in ("min", "max", "mean", "std", "q01", "q99"):
            np.testing.assert_allclose(
                streamed[tag]["statistics"][kind]["eef_position"][stat],
                materialized[tag]["statistics"][kind]["eef_position"][stat],
                atol=1e-12,
            )
    for key in raw_stats:
        for stat in raw_stats[key]:
            np.testing.assert_allclose(streamed_stats[key][stat], raw_stats[key][stat], atol=1e-12)
    # 五个合法起点，前三个包含两个 chunk，后两个只有一个 chunk。
    expected_x = np.array([1, 2, 3, 4] * 3 + [1, 2] * 2)
    action_stats = streamed[tag]["statistics"]["action"]["eef_position"]
    assert action_stats["mean"][0] == expected_x.mean()
    assert action_stats["max"][0] == 4
    np.testing.assert_allclose(np.array(action_stats["mean"])[[7, 15]], [0.1, 0.9])
    assert action_stats["min"][3] == action_stats["max"][3] == 1
    assert streamed[tag]["relative_action_mode"] == mode
    assert streamed[tag]["relative_action_pose_reference"] == "per_arm"
    assert streamed[tag]["relative_action_stats"]["action_horizon"] == 2
    assert streamed[tag]["relative_action_stats"]["streaming_stats"] is True


def test_relative_sample_batches_preserve_order_and_bound_memory():
    state = np.array([_pose16([float(t), 0, 0], [1, 0, 0, 0]) for t in range(6)])
    modality = metadata_module._NOE_UMI_EEF_MODALITY_CFG
    args = dict(
        relative_action_mode=RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ,
        action_horizon=2,
        max_chunk_size=2,
        episode_index=0,
    )
    batches = list(metadata_module._iter_robotwin_eef_relative_samples(
        state, state, modality, batch_size=2, **args,
    ))
    assert [len(actions) for _, actions in batches] == [8, 6, 2]
    np.testing.assert_array_equal(
        np.concatenate([actions[:, 0] for _, actions in batches]),
        [0, 1, 2, 3] * 3 + [0, 1] * 2,
    )


def test_robotics_relative_metadata_validates_source_contract(tmp_path):
    video_keys = _write_packed_robotics_dataset(tmp_path)
    path = tmp_path / "meta/modality.json"
    modality = json.loads(path.read_text())
    modality["action"]["left_quat_w"]["canonical_id"] = 4
    path.write_text(json.dumps(modality), encoding="utf-8")
    with pytest.raises(ValueError, match="action.left_quat_w"):
        build_metadata(
            tmp_path,
            embodiment_tag="noe_robotics_16d",
            state_key="observation.state",
            action_key="action",
            video_keys=video_keys,
            use_modality_json=False,
            max_episodes=None,
            relative_action_mode=RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ,
            relative_action_keys=["eef_position"],
            action_horizon=2,
            max_chunk_size=2,
        )


@pytest.mark.parametrize("relative", [False, True])
def test_episode_exclusions_precede_reads_and_sampling(tmp_path, monkeypatch, relative):
    video_keys = _write_packed_robotics_dataset(tmp_path)
    args = {
        "embodiment_tag": "noe_robotics_16d",
        "state_key": "observation.state",
        "action_key": "action",
        "video_keys": video_keys,
        "use_modality_json": False,
        "max_episodes": 1,
        "episode_stride": 2,
        "streaming_stats": True,
        "quantile_sample_limit": 1000,
    }
    if relative:
        args.update(
            relative_action_mode=RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ,
            relative_action_keys=["eef_position"],
            action_horizon=2,
            max_chunk_size=2,
        )
    expected = build_metadata(tmp_path, **args)
    data_dir = tmp_path / "data/chunk-000"
    (data_dir / "episode_000000.parquet").rename(data_dir / "episode_000001.parquet")
    extreme = np.full((6, 30), 1e9).tolist()
    pq.write_table(
        pa.table({"observation.state": extreme, "action": extreme}),
        data_dir / "episode_000000.parquet",
    )
    (tmp_path / "meta/episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": i}) + "\n" for i in range(3)),
        encoding="utf-8",
    )
    manifest = tmp_path / "meta/episode_exclusions.jsonl"
    manifest.write_text(
        '{"episode_index": 0}\n\n{"episode_index": 0}\n', encoding="utf-8"
    )
    read_table = pq.read_table
    reads = []

    def checked_read(path, *args, **kwargs):
        assert path.name == "episode_000001.parquet"
        reads.append(path.name)
        return read_table(path, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", checked_read)
    actual = build_metadata(
        tmp_path,
        **args,
        episode_exclusion_manifest=Path("meta/episode_exclusions.jsonl"),
    )
    assert actual == expected
    assert reads == ["episode_000001.parquet"]
    # Exclusion is opt-in; without it the first, extreme episode is read.
    monkeypatch.setattr(pq, "read_table", read_table)
    _, unfiltered = build_metadata(tmp_path, **args)
    assert unfiltered["observation.state"]["max"][0] == 1e9


@pytest.mark.parametrize(
    ("content", "error", "match"),
    [
        ('{"episode_index": -1}\n', ValueError, "non-negative integer"),
        ('{"episode_index": true}\n', ValueError, "non-negative integer"),
        ('{"episode_index": 0.0}\n', ValueError, "non-negative integer"),
        ('{"episode_index": "0"}\n', ValueError, "non-negative integer"),
        ('{"episode_index": 999}\n', ValueError, "Unknown excluded"),
        ('{"episode_index": 0}\n', ValueError, "No episodes"),
        ("{}\n", KeyError, "episode_index"),
        ("not-json\n", json.JSONDecodeError, "Expecting value"),
        (None, FileNotFoundError, "missing.jsonl"),
    ],
)
def test_episode_exclusions_reject_invalid_manifests(tmp_path, content, error, match):
    _write_packed_robotics_dataset(tmp_path)
    manifest = tmp_path / "meta/missing.jsonl"
    if content is not None:
        manifest.write_text(content, encoding="utf-8")
    with pytest.raises(error, match=match):
        metadata_module._load_episodes(
            tmp_path / "meta", episode_exclusion_manifest=manifest
        )


def test_episode_exclusions_resolve_per_root_and_compose_with_qualification(tmp_path):
    for name, excluded in [("a", 0), ("b", 1)]:
        root = tmp_path / name
        _write_packed_robotics_dataset(root)
        (root / "meta/episodes.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "episode_index": i,
                        **dict.fromkeys(
                            metadata_module._NOE_UMI_QUALIFICATION_FIELDS, i != 0
                        ),
                    }
                )
                + "\n"
                for i in range(3)
            ),
            encoding="utf-8",
        )
        (root / "meta/episode_exclusions.jsonl").write_text(
            json.dumps({"episode_index": excluded}) + "\n", encoding="utf-8"
        )
        episodes = metadata_module._load_episodes(
            root / "meta",
            filter_noe_umi_qualified_episodes=True,
            episode_exclusion_manifest=Path("meta/episode_exclusions.jsonl"),
        )
        assert episodes == ([1, 2] if name == "a" else [2])


def test_robotics_video_resolution_override_preserves_statistics(tmp_path):
    video_keys = _write_packed_robotics_dataset(tmp_path)
    args = {
        "embodiment_tag": "noe_robotics_16d",
        "state_key": "observation.state",
        "action_key": "action",
        "video_keys": video_keys,
        "use_modality_json": False,
        "max_episodes": None,
        "quantile_sample_limit": 1000,
        "streaming_stats": True,
    }
    expected, expected_stats = build_metadata(tmp_path, **args)
    info_path = tmp_path / "meta/info.json"
    info = json.loads(info_path.read_text())
    for key in video_keys:
        info["features"][key]["shape"] = [None, None, 3]
        info["features"][key]["video_shape_varies_by_episode"] = True
    info_path.write_text(json.dumps(info), encoding="utf-8")
    metadata, stats = build_metadata(tmp_path, **args, video_resolution=[320, 256])
    assert stats == expected_stats
    assert (
        metadata["noe_robotics_16d"]["statistics"]
        == expected["noe_robotics_16d"]["statistics"]
    )
    for video in metadata["noe_robotics_16d"]["modalities"]["video"].values():
        assert video == {"resolution": [320, 256], "channels": 3, "fps": 30.0}
    with pytest.raises(ValueError, match="positive integer"):
        build_metadata(tmp_path, **args, video_resolution=[0, 256])
