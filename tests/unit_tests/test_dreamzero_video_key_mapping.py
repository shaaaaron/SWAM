# Copyright 2026 The RLinf Authors.

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from rlinf.data.datasets.dreamzero import dreamzero as dreamzero_module  # noqa: E402
from rlinf.data.datasets.dreamzero.dreamzero import (  # noqa: E402
    DreamZeroLeRobotDataset,
)

EPISODE_LENGTH = 128


def _write_ego_tree(root, *, include_canonical=False):
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    features = {
        "observation.images.head_left": {"dtype": "video", "shape": [3, 2, 2]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    if include_canonical:
        features["observation.images.cam_high"] = {
            "dtype": "video",
            "shape": [3, 2, 2],
        }
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": EPISODE_LENGTH,
        "total_videos": int(include_canonical) + 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "arrange tableware"}) + "\n",
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": EPISODE_LENGTH}) + "\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.table({"task_index": [0] * EPISODE_LENGTH}),
        data / "episode_000000.parquet",
    )
    for source_key in features:
        if not source_key.startswith("observation.images."):
            continue
        video_dir = root / "videos" / "chunk-000" / source_key
        video_dir.mkdir(parents=True)
        (video_dir / "episode_000000.mp4").write_bytes(b"")


def _dataset(root, **kwargs):
    return DreamZeroLeRobotDataset(
        data_path=str(root),
        video_keys=["video.cam_high"],
        state_keys=[],
        action_keys=[],
        language_keys=["annotation.language.action_text"],
        data_transform=lambda raw: {
            "images": raw["video.cam_high"],
            "text": raw["annotation.language.action_text"],
        },
        lazy_load=True,
        num_frames=9,
        state_horizon=1,
        action_horizon=8,
        max_chunk_size=1,
        sampling_mode="multi_anchor",
        multi_anchor_video_offsets=list(range(8)),
        embodiment_tag="egoverse",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("include_canonical", "use_mapping", "expected_source"),
    [
        (False, True, "observation.images.head_left"),
        (True, True, "observation.images.head_left"),
        (True, False, "observation.images.cam_high"),
    ],
)
def test_video_key_mapping_selects_the_expected_camera(
    tmp_path, monkeypatch, include_canonical, use_mapping, expected_source
):
    _write_ego_tree(tmp_path, include_canonical=include_canonical)
    from lerobot.datasets import video_utils

    decoded_paths = []

    def fake_decode(path, timestamps, **_kwargs):
        decoded_paths.append(path)
        return np.full((len(timestamps), 2, 2, 3), 17, dtype=np.uint8)

    monkeypatch.setattr(dreamzero_module, "probe_video_container_fps", lambda _: 30.0)
    monkeypatch.setattr(video_utils, "decode_video_frames", fake_decode)
    mapping = {"video.cam_high": "observation.images.head_left"}
    dataset = _dataset(
        tmp_path,
        video_key_mapping=OmegaConf.create(mapping) if use_mapping else None,
    )

    sample = dataset[0]

    assert dataset._source_video_key == {"video.cam_high": expected_source}
    assert decoded_paths and all(p.parent.name == expected_source for p in decoded_paths)
    assert np.all(sample["images"] == 17)
    assert sample["text"] == "arrange tableware"


@pytest.mark.parametrize(
    ("mapping", "error", "message"),
    [
        ({"video.wrist": "observation.images.head_left"}, ValueError, "not in video_keys"),
        ({"video.cam_high": "observation.images.missing"}, ValueError, "video/image feature"),
        ({"video.cam_high": "task_index"}, ValueError, "video/image feature"),
        ({"video.cam_high": " "}, ValueError, "non-empty feature key"),
        (["observation.images.head_left"], TypeError, "must be a mapping"),
        ({}, KeyError, "Could not map transform video keys"),
    ],
)
def test_invalid_video_key_mapping_fails_at_construction(
    tmp_path, mapping, error, message
):
    _write_ego_tree(tmp_path)

    with pytest.raises(error, match=message):
        _dataset(tmp_path, video_key_mapping=mapping)


def test_mapped_video_path_is_checked_only_when_sampling(tmp_path):
    _write_ego_tree(tmp_path)
    video_path = (
        tmp_path
        / "videos/chunk-000/observation.images.head_left/episode_000000.mp4"
    )
    video_path.unlink()

    dataset = _dataset(
        tmp_path,
        video_key_mapping={"video.cam_high": "observation.images.head_left"},
    )
    with pytest.raises(FileNotFoundError, match="observation.images.head_left"):
        dataset[0]


def _write_two_episode_tree(root):
    _write_ego_tree(root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info.update(total_episodes=2, total_frames=2 * EPISODE_LENGTH, total_videos=2)
    info_path.write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": episode, "length": EPISODE_LENGTH}) + "\n"
            for episode in range(2)
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table({"task_index": [0] * EPISODE_LENGTH}),
        root / "data/chunk-000/episode_000001.parquet",
    )
    (
        root / "videos/chunk-000/observation.images.head_left/episode_000001.mp4"
    ).write_bytes(b"")


@pytest.mark.parametrize("relative_path", [True, False])
def test_explicit_exclusions_remove_frames_and_do_not_modify_metadata(
    tmp_path, monkeypatch, relative_path
):
    _write_two_episode_tree(tmp_path)
    manifest = tmp_path / "meta/episode_exclusions.jsonl"
    manifest.write_text(
        json.dumps({"episode_index": 0, "reason": "invalid_action"}) + "\n",
        encoding="utf-8",
    )
    paths = sorted((tmp_path / "meta").iterdir())
    before = {path: path.read_bytes() for path in paths}
    from lerobot.datasets import video_utils

    def fake_decode(path, timestamps, **_kwargs):
        assert path.name == "episode_000001.mp4"
        return np.zeros((len(timestamps), 2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(dreamzero_module, "probe_video_container_fps", lambda _: 30.0)
    monkeypatch.setattr(video_utils, "decode_video_frames", fake_decode)
    dataset = _dataset(
        tmp_path,
        video_key_mapping={"video.cam_high": "observation.images.head_left"},
        episode_exclusion_manifest=(
            "meta/episode_exclusions.jsonl" if relative_path else str(manifest)
        ),
    )
    assert dataset._episodes == [1]
    assert len(dataset) == EPISODE_LENGTH
    assert {dataset._episode_index_for_idx(i) for i in range(len(dataset))} == {1}
    assert dataset[0]["text"] == "arrange tableware"
    assert {path: path.read_bytes() for path in paths} == before


def test_relative_exclusions_are_scoped_to_each_dataset_root(tmp_path, monkeypatch):
    roots = [tmp_path / "shard-0", tmp_path / "shard-1"]
    for episode_index, root in enumerate(roots):
        _write_two_episode_tree(root)
        (root / "meta/episode_exclusions.jsonl").write_text(
            json.dumps({"episode_index": episode_index}) + "\n", encoding="utf-8"
        )
    monkeypatch.chdir(roots[0])
    datasets = [
        _dataset(
            root,
            video_key_mapping={"video.cam_high": "observation.images.head_left"},
            episode_exclusion_manifest="meta/episode_exclusions.jsonl",
        )
        for root in roots
    ]
    assert [dataset._episodes for dataset in datasets] == [[1], [0]]
    assert [len(dataset) for dataset in datasets] == [EPISODE_LENGTH] * 2


@pytest.mark.parametrize(
    "record, message",
    [
        ({"episode_index": 99}, "Unknown excluded"),
        ({"episode_index": -1}, "non-negative integer"),
        ({"episode_index": True}, "non-negative integer"),
        ({"episode_index": "0"}, "non-negative integer"),
    ],
)
def test_invalid_exclusion_records_fail_at_startup(tmp_path, record, message):
    _write_ego_tree(tmp_path)
    manifest = tmp_path / "exclusions.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _dataset(
            tmp_path,
            video_key_mapping={"video.cam_high": "observation.images.head_left"},
            episode_exclusion_manifest=str(manifest),
        )


def test_missing_exclusion_manifest_fails_at_startup(tmp_path):
    _write_ego_tree(tmp_path)
    with pytest.raises(FileNotFoundError):
        _dataset(
            tmp_path,
            video_key_mapping={"video.cam_high": "observation.images.head_left"},
            episode_exclusion_manifest="meta/missing.jsonl",
        )


@pytest.mark.parametrize("use_exclusions", [True, False])
def test_runtime_decode_failure_stays_temporary_and_other_errors_propagate(
    tmp_path, monkeypatch, use_exclusions
):
    _write_two_episode_tree(tmp_path)
    (tmp_path / "meta/episode_exclusions.jsonl").write_text("", encoding="utf-8")
    options = {
        "episode_exclusion_manifest": (
            "meta/episode_exclusions.jsonl" if use_exclusions else None
        )
    }
    legacy_manifest = tmp_path / "meta/bad_video_episodes.jsonl"
    legacy_manifest.write_text('{"episode_index": 1}\n', encoding="utf-8")
    before = {path: path.read_bytes() for path in (tmp_path / "meta").iterdir()}
    from lerobot.datasets import video_utils

    def fake_decode(path, timestamps, **_kwargs):
        if path.name == "episode_000000.mp4":
            raise RuntimeError("decode_video_frames failed")
        return np.zeros((len(timestamps), 2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(dreamzero_module, "probe_video_container_fps", lambda _: 30.0)
    monkeypatch.setattr(video_utils, "decode_video_frames", fake_decode)
    monkeypatch.setattr(dreamzero_module.random, "randint", lambda *_: EPISODE_LENGTH)
    mapping = {"video.cam_high": "observation.images.head_left"}
    dataset = _dataset(tmp_path, video_key_mapping=mapping, **options)
    assert dataset._bad_video_episodes == set()
    assert dataset[0]["text"] == "arrange tableware"
    assert dataset._bad_video_episodes == {0}
    assert {path: path.read_bytes() for path in before} == before
    restarted = _dataset(tmp_path, video_key_mapping=mapping, **options)
    assert restarted._bad_video_episodes == set()
    assert restarted._episodes == [0, 1]

    def programming_error(*_args, **_kwargs):
        raise RuntimeError("unexpected tensor shape")

    monkeypatch.setattr(video_utils, "decode_video_frames", programming_error)
    with pytest.raises(RuntimeError, match="unexpected tensor shape"):
        restarted[0]
    assert restarted._bad_video_episodes == set()
