import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from rlinf.data.datasets.dreamzero import dreamzero as dreamzero_module  # noqa: E402
from rlinf.data.datasets.dreamzero.data_transforms.egodex import (  # noqa: E402
    ACTION_KEYS,
    LANGUAGE_KEYS,
    STATE_KEYS,
    VIDEO_KEYS,
)
from rlinf.data.datasets.dreamzero.dreamzero import (  # noqa: E402
    DreamZeroLeRobotDataset,
    _resolve_dreamzero_data_paths,
)
from rlinf.data.datasets.dreamzero.egodex_confidence import (  # noqa: E402
    EGODEX_CONFIDENCE_SIDECAR_NAME,
    EGODEX_CONFIDENCE_STATUS_CONTAINS_ZERO,
    EGODEX_CONFIDENCE_STATUS_MISSING_ALL_ONE,
)


class _EgoDexReadTransform:
    """读取 loader 构造出的 absolute state/action，不做模型侧归一化。"""

    def __call__(self, raw):
        return {
            "images": raw[VIDEO_KEYS[0]],
            "state": np.concatenate([raw[key] for key in STATE_KEYS], axis=-1),
            "action": np.concatenate([raw[key] for key in ACTION_KEYS], axis=-1),
            "text": raw[LANGUAGE_KEYS[0]],
        }


def _write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_egodex_tree(
    root,
    episode_length: int = 20,
    confidence: np.ndarray | None = None,
    confidence_status: str = "valid",
    write_confidence_sidecar: bool = True,
) -> np.ndarray:
    meta_dir = root / "meta"
    data_dir = root / "data" / "chunk-000"
    video_dir = root / "videos" / "chunk-000" / "observation.images.egocentric"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "human_hands",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": episode_length,
        "total_videos": 1,
        "chunks_size": 1,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.egocentric": {
                "dtype": "video",
                "shape": [3, 16, 20],
            },
            "observation.state": {"dtype": "float32", "shape": [6]},
            "observation.state_confidence": {
                "dtype": "float32",
                "shape": [2],
            },
            "action": {"dtype": "float32", "shape": [6]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    modality = {
        "video": {
            "egocentric": {
                "original_key": "observation.images.egocentric",
            }
        },
        "state": {
            "left_hand_position": {
                "start": 0,
                "end": 3,
                "original_key": "observation.state",
                "absolute": True,
            },
            "right_hand_position": {
                "start": 3,
                "end": 6,
                "original_key": "observation.state",
                "absolute": True,
            },
        },
        "action": {
            "left_hand_delta": {
                "start": 0,
                "end": 3,
                "original_key": "action",
                "absolute": False,
            },
            "right_hand_delta": {
                "start": 3,
                "end": 6,
                "original_key": "action",
                "absolute": False,
            },
        },
        "annotation": {"task": {"original_key": "task_index"}},
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality), encoding="utf-8")
    _write_jsonl(
        meta_dir / "episodes.jsonl",
        [{"episode_index": 0, "episode_length": episode_length}],
    )
    if write_confidence_sidecar:
        _write_jsonl(
            meta_dir / EGODEX_CONFIDENCE_SIDECAR_NAME,
            [
                {
                    "episode_index": 0,
                    "confidence_status": confidence_status,
                    "error_reason": "",
                }
            ],
        )
    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": "move hands"}])

    base = np.arange(episode_length, dtype=np.float32)[:, None]
    offsets = np.arange(6, dtype=np.float32)[None, :]
    state = base + offsets
    delta = np.zeros_like(state)
    delta[:-1] = state[1:] - state[:-1]
    if confidence is None:
        confidence = np.full((episode_length, 2), 0.9, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    if confidence.shape != (episode_length, 2):
        raise ValueError(f"Invalid test confidence shape {confidence.shape}")
    table = pa.Table.from_pydict(
        {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(delta.tolist(), type=pa.list_(pa.float32())),
            "observation.state_confidence": pa.array(
                confidence.tolist(), type=pa.list_(pa.float32())
            ),
            "task_index": [0] * episode_length,
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")
    (video_dir / "episode_000000.mp4").write_bytes(b"")
    return state


def test_egodex_loader_outputs_absolute_hand_actions(tmp_path, monkeypatch):
    state = _write_egodex_tree(tmp_path)

    def fake_decode_video_frames(_path, timestamps, **_kwargs):
        return np.zeros((len(timestamps), 16, 20, 3), dtype=np.uint8)

    from lerobot.datasets import video_utils

    monkeypatch.setattr(
        dreamzero_module, "probe_video_container_fps", lambda _path: 30.0
    )
    monkeypatch.setattr(video_utils, "decode_video_frames", fake_decode_video_frames)
    dataset = DreamZeroLeRobotDataset(
        data_path=str(tmp_path),
        video_keys=VIDEO_KEYS,
        state_keys=STATE_KEYS,
        action_keys=ACTION_KEYS,
        language_keys=LANGUAGE_KEYS,
        data_transform=_EgoDexReadTransform(),
        lazy_load=True,
        num_frames=9,
        state_horizon=1,
        action_horizon=8,
        max_chunk_size=1,
        sampling_mode="multi_anchor",
        multi_anchor_macro_stride=8,
        multi_anchor_video_offsets=list(range(8)),
        multi_anchor_resample_attempts=1,
        embodiment_tag="egodex",
    )

    assert dataset._use_lazy_video_tree
    assert not dataset._use_image_parquet_tree
    assert not getattr(dataset, "_use_v2_image_parquet", False)
    sample = dataset[0]

    np.testing.assert_allclose(sample["state"], state[[0]])
    np.testing.assert_allclose(sample["action"], state[1:9])
    assert sample["images"].shape == (9, 16, 20, 3)
    assert sample["text"] == "move hands"
    assert (
        "observation.state_confidence" not in dataset._get_episode_table(0).column_names
    )


@pytest.mark.parametrize(
    "confidence_status",
    [
        EGODEX_CONFIDENCE_STATUS_MISSING_ALL_ONE,
        EGODEX_CONFIDENCE_STATUS_CONTAINS_ZERO,
    ],
)
def test_egodex_confidence_sidecar_filters_non_valid_episode(
    tmp_path, confidence_status
):
    _write_egodex_tree(tmp_path, confidence_status=confidence_status)

    with pytest.raises(ValueError, match="kept no episodes"):
        DreamZeroLeRobotDataset(
            data_path=str(tmp_path),
            video_keys=VIDEO_KEYS,
            state_keys=STATE_KEYS,
            action_keys=ACTION_KEYS,
            language_keys=LANGUAGE_KEYS,
            data_transform=_EgoDexReadTransform(),
            lazy_load=True,
            num_frames=9,
            state_horizon=1,
            action_horizon=8,
            max_chunk_size=1,
            sampling_mode="multi_anchor",
            multi_anchor_macro_stride=8,
            multi_anchor_video_offsets=list(range(8)),
            multi_anchor_resample_attempts=1,
            embodiment_tag="egodex",
        )


def test_egodex_requires_confidence_sidecar(tmp_path):
    _write_egodex_tree(tmp_path, write_confidence_sidecar=False)

    with pytest.raises(FileNotFoundError, match="confidence sidecar not found"):
        DreamZeroLeRobotDataset(
            data_path=str(tmp_path),
            video_keys=VIDEO_KEYS,
            state_keys=STATE_KEYS,
            action_keys=ACTION_KEYS,
            language_keys=LANGUAGE_KEYS,
            data_transform=_EgoDexReadTransform(),
            lazy_load=True,
            num_frames=9,
            state_horizon=1,
            action_horizon=8,
            max_chunk_size=1,
            sampling_mode="multi_anchor",
            multi_anchor_macro_stride=8,
            multi_anchor_video_offsets=list(range(8)),
            multi_anchor_resample_attempts=1,
            embodiment_tag="egodex",
        )


def test_egodex_rejects_action_without_next_state(tmp_path):
    _write_egodex_tree(tmp_path, episode_length=20)
    dataset = DreamZeroLeRobotDataset(
        data_path=str(tmp_path),
        video_keys=VIDEO_KEYS,
        state_keys=STATE_KEYS,
        action_keys=ACTION_KEYS,
        language_keys=LANGUAGE_KEYS,
        data_transform=_EgoDexReadTransform(),
        lazy_load=True,
        num_frames=9,
        state_horizon=1,
        action_horizon=8,
        max_chunk_size=1,
        sampling_mode="fixed_window",
        embodiment_tag="egodex",
    )
    table = dataset._get_episode_table(0)

    with pytest.raises(IndexError, match=r"requires action_idx \+ 1"):
        dataset._materialize_parquet_sample(
            frame_in_ep=15,
            episode_index=0,
            ep_len=20,
            table=table,
            decode_video=False,
        )


def test_egodex_collection_root_discovers_all_tasks(tmp_path):
    _write_egodex_tree(tmp_path / "part1" / "task_a")
    _write_egodex_tree(tmp_path / "part2" / "task_b")

    roots = _resolve_dreamzero_data_paths(str(tmp_path), "egodex")

    assert roots == [
        str(tmp_path / "part1" / "task_a"),
        str(tmp_path / "part2" / "task_b"),
    ]


def test_egodex_collection_root_skips_task_without_valid_episode(tmp_path):
    _write_egodex_tree(tmp_path / "part1" / "task_a")
    _write_egodex_tree(
        tmp_path / "part2" / "task_b",
        confidence_status=EGODEX_CONFIDENCE_STATUS_MISSING_ALL_ONE,
    )

    roots = _resolve_dreamzero_data_paths(str(tmp_path), "egodex")

    assert roots == [str(tmp_path / "part1" / "task_a")]
