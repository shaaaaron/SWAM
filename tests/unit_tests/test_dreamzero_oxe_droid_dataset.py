import io
import json

import numpy as np
import pytest

pytest.importorskip("groot")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
Image = pytest.importorskip("PIL.Image")

from rlinf.data.datasets.dreamzero.data_transforms.oxe_droid import (  # noqa: E402
    OxeDroidDataTransform,
)
from rlinf.data.datasets.dreamzero.dreamzero import (  # noqa: E402
    DreamZeroLeRobotDataset,
)

VIDEO_KEYS = [
    "video.exterior_image_1_left",
    "video.exterior_image_2_left",
    "video.wrist_image_left",
]
STATE_KEYS = ["state.joint_position", "state.gripper_position"]
ACTION_KEYS = ["action.joint_position", "action.gripper_position"]
LANGUAGE_KEYS = ["annotation.language.language_instruction"]


class _OxeDroidReadTransform:
    def __call__(self, raw):
        views = np.stack([raw[key] for key in VIDEO_KEYS], axis=0)
        views = np.transpose(views, (0, 1, 4, 2, 3))
        concat = OxeDroidDataTransform.concat_multiview_video(views)[0]
        images = np.transpose(concat, (0, 2, 3, 1))

        return {
            "images": images,
            "state": np.concatenate([raw[key] for key in STATE_KEYS], axis=-1),
            "action": np.concatenate([raw[key] for key in ACTION_KEYS], axis=-1),
            "text": raw[LANGUAGE_KEYS[0]],
        }


def _png_bytes(value: int, height: int = 128, width: int = 160) -> bytes:
    image = np.full((height, width, 3), int(value), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_synthetic_oxe_droid_tree(root, episode_length: int = 32) -> None:
    meta_dir = root / "meta"
    data_dir = root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    image_features = {
        "observation.images.exterior_image_1_left": {
            "dtype": "image",
            "shape": [128, 160, 3],
        },
        "observation.images.exterior_image_2_left": {
            "dtype": "image",
            "shape": [128, 160, 3],
        },
        "observation.images.wrist_image_left": {
            "dtype": "image",
            "shape": [128, 160, 3],
        },
    }
    info = {
        "codebase_version": "v2.1",
        "fps": 15,
        "total_episodes": 1,
        "total_frames": episode_length,
        "total_videos": 0,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            **image_features,
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [8]},
            "annotation.language.language_instruction": {"dtype": "string"},
            "task_index": {"dtype": "int64"},
        },
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f)

    modality = {
        "video": {
            "exterior_image_1_left": {
                "original_key": "observation.images.exterior_image_1_left"
            },
            "exterior_image_2_left": {
                "original_key": "observation.images.exterior_image_2_left"
            },
            "wrist_image_left": {
                "original_key": "observation.images.wrist_image_left"
            },
        },
        "state": {
            "joint_position": {
                "original_key": "observation.state",
                "start": 0,
                "end": 7,
            },
            "gripper_position": {
                "original_key": "observation.state",
                "start": 7,
                "end": 8,
            },
        },
        "action": {
            "joint_position": {
                "original_key": "action",
                "start": 0,
                "end": 7,
            },
            "gripper_position": {
                "original_key": "action",
                "start": 7,
                "end": 8,
            },
        },
        "annotation": {
            "language.language_instruction": {
                "original_key": "annotation.language.language_instruction"
            }
        },
    }
    with open(meta_dir / "modality.json", "w", encoding="utf-8") as f:
        json.dump(modality, f)

    _write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {
                "episode_index": 0,
                "episode_length": episode_length,
                "tasks": ["pick object"],
            }
        ],
    )
    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": "pick object"}])

    frames = np.arange(episode_length, dtype=np.float32)
    state = [[5.0 + float(frame)] * 7 + [float(frame % 2)] for frame in frames]
    action = [
        [105.0 + float(frame)] * 7 + [float((frame + 1) % 2)]
        for frame in frames
    ]
    table = pa.Table.from_pydict(
        {
            "observation.images.exterior_image_1_left": [
                _png_bytes(20 + int(frame)) for frame in frames
            ],
            "observation.images.exterior_image_2_left": [
                _png_bytes(80 + int(frame)) for frame in frames
            ],
            "observation.images.wrist_image_left": [
                _png_bytes(140 + int(frame)) for frame in frames
            ],
            "observation.state": pa.array(state, type=pa.list_(pa.float32())),
            "action": pa.array(action, type=pa.list_(pa.float32())),
            "annotation.language.language_instruction": ["pick object"] * episode_length,
            "task_index": [0] * episode_length,
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")


def _write_lazy_video_tree_with_episode_length(root, episode_length: int = 5) -> None:
    meta_dir = root / "meta"
    data_dir = root / "data" / "chunk-000"
    video_dir = root / "videos" / "chunk-000" / "observation.images.image"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)

    info = {
        "codebase_version": "v2.1",
        "fps": 15,
        "total_episodes": 1,
        "total_frames": episode_length,
        "total_videos": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.image": {"dtype": "video", "shape": [128, 160, 3]},
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [8]},
            "task_index": {"dtype": "int64"},
        },
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f)

    modality = {
        "video": {"image": {"original_key": "observation.images.image"}},
        "state": {
            "eef_position": {
                "original_key": "observation.state",
                "start": 0,
                "end": 8,
            }
        },
        "action": {
            "eef_position": {
                "original_key": "action",
                "start": 0,
                "end": 8,
            }
        },
    }
    with open(meta_dir / "modality.json", "w", encoding="utf-8") as f:
        json.dump(modality, f)

    _write_jsonl(
        meta_dir / "episodes.jsonl",
        [{"episode_index": 0, "length": episode_length}],
    )
    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": "pick object"}])
    (video_dir / "episode_000000.mp4").write_bytes(b"")

    table = pa.Table.from_pydict(
        {
            "observation.state": pa.array(
                [[0.0] * 8 for _ in range(episode_length)],
                type=pa.list_(pa.float32()),
            ),
            "action": pa.array(
                [[0.0] * 8 for _ in range(episode_length)],
                type=pa.list_(pa.float32()),
            ),
            "task_index": [0] * episode_length,
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")




def test_lazy_video_tree_uses_episode_lengths_without_parquet_fallback(
    tmp_path, monkeypatch
):
    _write_lazy_video_tree_with_episode_length(tmp_path)

    def fail_infer_episode_length(self, episode_index):
        raise AssertionError("episode length should come from episodes.jsonl")

    monkeypatch.setattr(
        DreamZeroLeRobotDataset,
        "_infer_episode_length_from_parquet",
        fail_infer_episode_length,
    )

    dataset = DreamZeroLeRobotDataset(
        data_path=str(tmp_path),
        video_keys=["video.image"],
        state_keys=["state.eef_position"],
        action_keys=["action.eef_position"],
        language_keys=["annotation.language.action_text"],
        data_transform=lambda raw: raw,
        lazy_load=True,
        num_frames=1,
        state_horizon=1,
        action_horizon=1,
        max_chunk_size=1,
        sampling_mode="multi_anchor",
        multi_anchor_macro_stride=1,
        multi_anchor_video_offsets=[0],
        multi_anchor_resample_attempts=1,
        embodiment_tag="robotwin",
    )

    assert len(dataset) == 5
    assert dataset._episode_lengths == [5]


def test_bad_video_manifest_is_ignored_and_runtime_quarantine_stays_in_memory(tmp_path):
    _write_lazy_video_tree_with_episode_length(tmp_path)
    manifest_path = tmp_path / "meta" / "bad_video_episodes.jsonl"
    original_manifest = '{"episode_index": 0, "error": "historical failure"}\n'
    manifest_path.write_text(original_manifest, encoding="utf-8")

    dataset = DreamZeroLeRobotDataset(
        data_path=str(tmp_path),
        video_keys=["video.image"],
        state_keys=["state.eef_position"],
        action_keys=["action.eef_position"],
        language_keys=["annotation.language.action_text"],
        data_transform=lambda raw: raw,
        lazy_load=True,
        num_frames=1,
        state_horizon=1,
        action_horizon=1,
        max_chunk_size=1,
        sampling_mode="multi_anchor",
        multi_anchor_macro_stride=1,
        multi_anchor_video_offsets=[0],
        multi_anchor_resample_attempts=1,
        embodiment_tag="robotwin",
    )

    assert len(dataset) == 5
    assert dataset._bad_video_episodes == set()

    episode_index = dataset._record_bad_video_episode(
        0,
        RuntimeError("Could not push packet to decoder"),
    )

    assert episode_index == 0
    assert dataset._bad_video_episodes == {0}
    assert manifest_path.read_text(encoding="utf-8") == original_manifest
