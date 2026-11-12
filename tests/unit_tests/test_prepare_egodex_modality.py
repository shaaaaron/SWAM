import json

import pytest

from toolkits.lerobot.prepare_egodex_modality import (
    ACTION_NAMES,
    MODALITY_JSON,
    STATE_NAMES,
    prepare_egodex_modality,
)


def _valid_info():
    return {
        "codebase_version": "v2.1",
        "robot_type": "human_hands",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": [STATE_NAMES],
            },
            "observation.state_confidence": {
                "dtype": "float32",
                "shape": [2],
                "names": ["left_hand", "right_hand"],
            },
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": [ACTION_NAMES],
            },
            "observation.images.egocentric": {
                "dtype": "video",
                "shape": [3, 1080, 1920],
                "names": ["channels", "height", "width"],
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        },
    }


def _make_task(collection, part, task, info=None):
    meta_dir = collection / part / task / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(
        json.dumps(info or _valid_info()) + "\n",
        encoding="utf-8",
    )
    return meta_dir


def test_prepare_writes_missing_sidecars_after_validating_all_roots(tmp_path):
    first = _make_task(tmp_path, "part1", "task_a")
    second = _make_task(tmp_path, "part2", "task_b")

    summary = prepare_egodex_modality(tmp_path)

    assert summary.found == 2
    assert summary.missing == 2
    assert summary.matching == 0
    assert summary.written == 2
    assert json.loads((first / "modality.json").read_text()) == MODALITY_JSON
    assert json.loads((second / "modality.json").read_text()) == MODALITY_JSON
    assert MODALITY_JSON["state"]["left_hand_position"]["absolute"] is True
    assert MODALITY_JSON["action"]["left_hand_delta"]["absolute"] is False


def test_prepare_dry_run_does_not_write(tmp_path, capsys):
    meta_dir = _make_task(tmp_path, "extra", "task_a")

    summary = prepare_egodex_modality(tmp_path, dry_run=True)

    assert summary.missing == 1
    assert summary.written == 0
    assert not (meta_dir / "modality.json").exists()
    assert "[WRITE]" in capsys.readouterr().out


def test_prepare_keeps_matching_sidecar(tmp_path):
    meta_dir = _make_task(tmp_path, "part1", "task_a")
    modality_path = meta_dir / "modality.json"
    modality_path.write_text(json.dumps(MODALITY_JSON), encoding="utf-8")

    summary = prepare_egodex_modality(tmp_path)

    assert summary.matching == 1
    assert summary.written == 0
    assert json.loads(modality_path.read_text()) == MODALITY_JSON


def test_prepare_rejects_mismatched_sidecar_before_writing_anything(tmp_path):
    bad_meta = _make_task(tmp_path, "part1", "task_a")
    missing_meta = _make_task(tmp_path, "part1", "task_b")
    bad_path = bad_meta / "modality.json"
    bad_path.write_text(json.dumps({"state": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="No files were written"):
        prepare_egodex_modality(tmp_path)

    assert json.loads(bad_path.read_text()) == {"state": {}}
    assert not (missing_meta / "modality.json").exists()


def test_prepare_overwrite_repairs_mismatched_sidecar(tmp_path):
    meta_dir = _make_task(tmp_path, "part1", "task_a")
    modality_path = meta_dir / "modality.json"
    modality_path.write_text(json.dumps({"state": {}}), encoding="utf-8")

    summary = prepare_egodex_modality(tmp_path, overwrite=True)

    assert summary.mismatched == 1
    assert summary.written == 1
    assert json.loads(modality_path.read_text()) == MODALITY_JSON


def test_prepare_rejects_incompatible_schema_before_writing(tmp_path):
    valid_meta = _make_task(tmp_path, "part1", "task_a")
    invalid_info = _valid_info()
    invalid_info["features"]["action"]["names"] = [list(reversed(ACTION_NAMES))]
    _make_task(tmp_path, "part1", "task_b", invalid_info)

    with pytest.raises(ValueError, match="action.names must be"):
        prepare_egodex_modality(tmp_path)

    assert not (valid_meta / "modality.json").exists()


def test_prepare_requires_two_hand_confidence(tmp_path):
    invalid_info = _valid_info()
    invalid_info["features"]["observation.state_confidence"]["shape"] = [1]
    _make_task(tmp_path, "part1", "task_a", invalid_info)

    with pytest.raises(ValueError, match=r"state_confidence.shape must be \[2\]"):
        prepare_egodex_modality(tmp_path)


def test_prepare_requires_at_least_one_lerobot_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="No EgoDex LeRobot roots"):
        prepare_egodex_modality(tmp_path)
