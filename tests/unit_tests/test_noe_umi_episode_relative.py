import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rlinf.data.datasets.dreamzero.noe_umi_episode_reference import (
    NOE_UMI_FRAME_QUALITY_FIELDS,
    read_noe_umi_episode_reference,
    read_noe_umi_frame_quality,
)
from rlinf.data.datasets.dreamzero.relative_actions import (
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ,
    robotwin_eef_abs_to_rel_wxyz,
)
from toolkits.lerobot.noe_umi_episode_metadata import build_noe_umi_episode_metadata


def _write_episode_root(root, *, reference_valid=True):
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {
        "observation.state": {"shape": [23]},
        "action": {"shape": [23]},
        **{
            key: {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
                "info": {"video.fps": 30},
            }
            for key, shape in [
                ("observation.images.image", [64, 80, 3]),
                ("observation.images.wrist_image_1", [80, 64, 3]),
                ("observation.images.wrist_image_2", [80, 64, 3]),
            ]
        },
    }
    info = {
        "fps": 30,
        "chunks_size": 1000,
        "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    ep = {
        "episode_index": 0,
        "length": 4,
        "slam_validity_all": True,
        "width_detection": True,
        "width_calibration": True,
    }
    (root / "meta/episodes.jsonl").write_text(json.dumps(ep) + "\n")
    state = np.zeros((4, 23), dtype=np.float32)
    state[:, 0] = [1000, 1000.1, 1000.2, 1000.3]
    state[:, 8] = [-3000, -2999.8, -2999.6, -2999.4]
    state[:, [3, 11, 19]] = 1
    state[:, 7] = 0.05
    state[:, 15] = 0.06
    action = state[np.minimum(np.arange(4) + 1, 3)]
    quality = [reference_valid, False, True, True]
    table = pa.table(
        {
            "observation.state": state.tolist(),
            "action": action.tolist(),
            **dict.fromkeys(NOE_UMI_FRAME_QUALITY_FIELDS, quality),
        }
    )
    pq.write_table(table, root / "data/chunk-000/episode_000000.parquet")
    return table, state, action


def test_episode_reference_always_reads_row_zero_and_existing_quality(tmp_path):
    table, state, _ = _write_episode_root(tmp_path / "root", reference_valid=False)
    reference, valid = read_noe_umi_episode_reference(table, (True, True))
    np.testing.assert_array_equal(reference, state[:1, :16])
    assert valid is False
    assert read_noe_umi_frame_quality(table, np.array([2]), (False, False))[0]


def test_shared_quality_preserves_null_and_episode_fallback():
    table = pa.table({key: [1, None, 0, 1] for key in NOE_UMI_FRAME_QUALITY_FIELDS})
    assert read_noe_umi_frame_quality(table, np.arange(4)).tolist() == [
        True,
        False,
        False,
        True,
    ]
    assert read_noe_umi_frame_quality(
        pa.table({"other": [1, 2]}), np.array([0, 1]), (True, True)
    ).all()
    assert not read_noe_umi_frame_quality(
        pa.table({"other": [1, 2]}), np.array([0, 1]), (True, False)
    ).any()


def test_episode_metadata_masks_reference_and_aligns_action_quality(tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"
    table, state, action = _write_episode_root(good)
    _write_episode_root(bad, reference_valid=False)
    meta, stats = build_noe_umi_episode_metadata(
        [good, bad], apply_frame_quality=True, quantile_sample_limit=100, stats_seed=42
    )
    entry = meta["noe_umi_16d"]
    assert (
        entry["relative_action_mode"]
        == RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ
    )
    assert entry["relative_action_reference"] == "episode_frame0"
    assert entry["relative_action_stats"]["state_rows"] == 3
    assert entry["relative_action_stats"]["action_rows"] == 3
    assert stats["audit"]["roots"][1]["counts"]["state_rows"] == 0
    assert stats["audit"]["roots"][1]["counts"]["action_rows"] == 0
    expected = {
        "state": robotwin_eef_abs_to_rel_wxyz(state[[0, 2, 3], :16], state[:1, :16]),
        "action": robotwin_eef_abs_to_rel_wxyz(action[[1, 2, 3], :16], state[:1, :16]),
    }
    for kind, values in expected.items():
        st = entry["statistics"][kind]["eef_position"]
        np.testing.assert_allclose(
            st["mean"], values.astype(float).mean(axis=0), atol=1e-12
        )
        np.testing.assert_allclose(
            st["std"], values.astype(float).std(axis=0), atol=1e-12
        )
        np.testing.assert_allclose(
            st["q01"], np.quantile(values.astype(float), 0.01, axis=0), atol=1e-12
        )


def test_episode_metadata_quality_switch_off_does_not_filter(tmp_path):
    root = tmp_path / "bad"
    _write_episode_root(root, reference_valid=False)
    meta, _ = build_noe_umi_episode_metadata(
        [root], apply_frame_quality=False, quantile_sample_limit=100
    )
    assert meta["noe_umi_16d"]["relative_action_stats"]["state_rows"] == 4
    assert meta["noe_umi_16d"]["relative_action_stats"]["action_rows"] == 4


def test_episode_metadata_records_missing_files_when_explicitly_enabled(tmp_path, capsys):
    good, missing = tmp_path / "good", tmp_path / "missing"
    _write_episode_root(good)
    _write_episode_root(missing)
    (missing / "data/chunk-000/episode_000000.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        build_noe_umi_episode_metadata([good, missing], apply_frame_quality=True)
    metadata, stats = build_noe_umi_episode_metadata(
        [good, missing], apply_frame_quality=True, skip_missing_episodes=True
    )
    assert metadata["noe_umi_16d"]["relative_action_stats"]["state_rows"] == 3
    assert stats["audit"]["counts"]["processed_episodes"] == 2
    assert stats["audit"]["counts"]["missing_episodes"] == 1
    assert stats["audit"]["roots"][1]["counts"]["missing_episodes"] == 1
    assert str(missing / "data/chunk-000/episode_000000.parquet") in capsys.readouterr().out


def test_episode_cli_generates_basic_stats_and_preserves_existing_overwrite_behavior(tmp_path, monkeypatch):
    from toolkits.lerobot.generate_dreamzero_metadata import main

    root = tmp_path / "root"
    _write_episode_root(root)
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"roots": [str(root)]}))
    metadata, stats = tmp_path / "metadata.json", tmp_path / "stats.json"
    metadata.write_text("previous metadata")
    stats.write_text("previous stats")
    monkeypatch.setattr("sys.argv", [
        "generate_dreamzero_metadata", "--preset", "noe_umi_16d",
        "--relative-action-mode", RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ,
        "--relative-action-keys", "eef_position", "--apply-noe-umi-frame-quality",
        "--dataset-roots-file", str(roots), "--output-metadata", str(metadata),
        "--output-stats", str(stats),
    ])
    main()
    assert json.loads(metadata.read_text())["noe_umi_16d"]["relative_action_stats"]["state_rows"] == 3
    audit = json.loads(stats.read_text())["audit"]
    assert "episodes" not in audit
    assert all("info_sha256" not in source for source in audit["sources"])


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"anchor_stride": 40}, "anchor_stride"),
        ({"relative_action_keys": []}, "relative_action_keys"),
        ({"quantile_sample_limit": 0}, "positive"),
        ({"relative_action_pose_reference": "left_hand"}, "per_arm"),
        ({"action_quaternion_mode": "temporal_continuous"}, "preserve"),
    ],
)
def test_episode_generator_api_rejects_inapplicable_options(tmp_path, overrides, message):
    from toolkits.lerobot.generate_dreamzero_metadata import build_metadata

    root = tmp_path / "root"
    _write_episode_root(root)
    kwargs = {
        "embodiment_tag": "noe_umi_16d",
        "state_key": "observation.state",
        "action_key": "action",
        "video_keys": None,
        "use_modality_json": False,
        "max_episodes": None,
        "relative_action_mode": RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ,
        "relative_action_keys": ["eef_position"],
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        build_metadata(root, **kwargs)


@pytest.mark.parametrize(
    "options, message",
    [
        (["--anchor-stride", "40"], "anchor_stride"),
        (["--relative-action-keys"], "relative_action_keys"),
        (["--relative-action-pose-reference", "left_hand"], "per_arm"),
        (["--action-quaternion-mode", "temporal_continuous"], "preserve"),
    ],
)
def test_episode_generator_cli_uses_the_same_validation(tmp_path, monkeypatch, options, message):
    from toolkits.lerobot.generate_dreamzero_metadata import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_dreamzero_metadata",
            "--preset", "noe_umi_16d",
            "--relative-action-mode", RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ,
            "--relative-action-keys", "eef_position",
            "--output-metadata", str(tmp_path / "metadata.json"),
            *options,
        ],
    )
    with pytest.raises(ValueError, match=message):
        main()


@pytest.mark.parametrize("reference_valid", [True, False])
@pytest.mark.parametrize("cache_size", [0, 1])
def test_real_loader_windows_share_frame_zero_and_gate_whole_episode(
    tmp_path, reference_valid, cache_size
):
    from test_dreamzero_noe_umi_dataset import _build_dataset, _write_noe_umi_tree

    from rlinf.data.datasets.dreamzero.dreamzero import _NOE_UMI_EPISODE_REFERENCE_KEY

    _write_noe_umi_tree(tmp_path)
    n = 300
    state = np.zeros((n, 23), dtype=np.float32)
    state[:, 0] = 1000 + np.arange(n) / 100
    state[:, 8] = -3000 + np.arange(n) / 200
    state[:, [3, 11, 19]] = 1
    state[:, 7] = 0.05
    state[:, 15] = 0.06
    action = state[np.minimum(np.arange(n) + 1, n - 1)]
    quality = np.ones(n, dtype=bool)
    quality[0] = reference_valid
    quality[60] = False
    table = pa.table(
        {
            "observation.state": state.tolist(),
            "action": action.tolist(),
            **{k: quality.tolist() for k in NOE_UMI_FRAME_QUALITY_FIELDS},
        }
    )
    pq.write_table(table, tmp_path / "data/chunk-000/episode_000000.parquet")
    dataset = _build_dataset(
        tmp_path,
        filter_qualified=True,
        quality_mode="frame",
        relative_action=True,
        relative_action_mode=RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ,
        pq_cache_max_episodes=cache_size,
    )
    for frame in (0, 48, 96):
        loaded = dataset._get_episode_table(0)
        sample = dataset._materialize_parquet_sample(
            frame, 0, n, loaded, decode_video=False
        )
        np.testing.assert_array_equal(
            sample[_NOE_UMI_EPISODE_REFERENCE_KEY], state[:1, :16]
        )
        for video_key in dataset._source_video_key:
            sample[video_key] = np.zeros((33, 4, 4, 3), dtype=np.uint8)
        out = dataset._build_modality_dict(sample)
        for kind in ("state", "action"):
            raw = sample["observation.state" if kind == "state" else "action"][:, :16]
            np.testing.assert_allclose(
                out[kind + ".eef_position"],
                robotwin_eef_abs_to_rel_wxyz(raw, state[:1, :16]),
            )
            assert out["__noe_umi_" + kind + "_quality"].any() == reference_valid
        assert _NOE_UMI_EPISODE_REFERENCE_KEY not in out
        assert all(k in out for k in dataset._source_video_key)
    assert len(dataset._pq_cache) == cache_size


def test_episode_loader_preserves_existing_quality_switches(tmp_path):
    from test_dreamzero_noe_umi_dataset import _build_dataset, _write_noe_umi_tree

    _write_noe_umi_tree(tmp_path)
    dataset = _build_dataset(
        tmp_path,
        filter_qualified=False,
        relative_action=True,
        relative_action_mode=RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ,
    )
    assert dataset._uses_noe_umi_frame_quality() is False
    assert dataset.noe_umi_quality_mode == "episode"
