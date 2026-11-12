import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from rlinf.data.datasets.dreamzero.egodex_confidence import (  # noqa: E402
    EGODEX_CONFIDENCE_SIDECAR_NAME,
)
from toolkits.lerobot.generate_egodex_metadata import (  # noqa: E402
    build_metadata,
    compute_statistics,
    episode_paths,
)


def _write_episode(path, state):
    state = np.asarray(state, dtype=np.float32)
    delta = np.zeros_like(state)
    delta[:-1] = state[1:] - state[:-1]
    pq.write_table(
        pa.Table.from_pydict(
            {
                "observation.state": pa.array(
                    state.tolist(), type=pa.list_(pa.float32())
                ),
                "action": pa.array(delta.tolist(), type=pa.list_(pa.float32())),
            }
        ),
        path,
    )
    return state, delta


def test_egodex_metadata_uses_absolute_state_and_action(tmp_path):
    state, delta = _write_episode(
        tmp_path / "episode.parquet",
        [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7, 8],
        ],
    )
    path = tmp_path / "episode.parquet"

    state_stats, action_stats, sample_count = compute_statistics(
        [path], total_frames=3, quantile_sample_limit=100
    )
    metadata = build_metadata(state_stats, action_stats)["egodex"]

    np.testing.assert_allclose(state_stats["mean"], state.mean(axis=0))
    np.testing.assert_allclose(
        action_stats["mean"],
        (state[:-1] + delta[:-1]).mean(axis=0),
    )
    assert sample_count == 3
    assert metadata["modalities"]["state"]["left_hand_position"]["absolute"]
    assert metadata["modalities"]["action"]["left_hand_position"]["absolute"]


def test_egodex_metadata_filters_episode_paths_from_sidecar(tmp_path):
    meta_dir = tmp_path / "meta"
    data_dir = tmp_path / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    for episode_index in (0, 1):
        _write_episode(
            data_dir / f"episode_{episode_index:06d}.parquet",
            [[0] * 6, [1] * 6, [2] * 6],
        )
    (meta_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps({"episode_index": index, "length": 3}) for index in (0, 1))
        + "\n",
        encoding="utf-8",
    )
    (meta_dir / EGODEX_CONFIDENCE_SIDECAR_NAME).write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "episode_index": 0,
                        "confidence_status": "valid",
                        "error_reason": "",
                    }
                ),
                json.dumps(
                    {
                        "episode_index": 1,
                        "confidence_status": "contains_zero",
                        "error_reason": "",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    paths, total_frames = episode_paths(tmp_path)

    assert paths == [data_dir / "episode_000000.parquet"]
    assert total_frames == 3


def test_egodex_metadata_rejects_inconsistent_delta(tmp_path):
    path = tmp_path / "episode.parquet"
    state, delta = _write_episode(
        path,
        [[0] * 6, [1] * 6, [2] * 6],
    )
    delta[0, 0] = 10.0
    pq.write_table(
        pa.Table.from_pydict(
            {
                "observation.state": pa.array(
                    state.tolist(), type=pa.list_(pa.float32())
                ),
                "action": pa.array(delta.tolist(), type=pa.list_(pa.float32())),
            }
        ),
        path,
    )

    with pytest.raises(ValueError, match="does not reconstruct"):
        compute_statistics([path], total_frames=3, quantile_sample_limit=100)
