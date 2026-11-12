import numpy as np

from rlinf.data.datasets.dreamzero.sampling_strategy import (
    MultiAnchorTemporalConfig,
    require_multi_anchor_temporal_indices,
)


def test_multi_anchor_temporal_indices_are_relative_offsets():
    language = np.zeros(240, dtype=np.int64)
    cfg = MultiAnchorTemporalConfig(
        max_chunk_size=2,
        macro_stride=24,
        action_horizon=24,
        video_in_chunk_offsets=(0, 3, 6, 9, 12, 15, 18, 21),
    )

    temporal = require_multi_anchor_temporal_indices(
        72,
        language,
        len(language),
        cfg,
        episode_index=0,
    )

    np.testing.assert_array_equal(temporal.video, np.arange(-24, 25, 3))
    np.testing.assert_array_equal(temporal.state, np.asarray([-24, 0]))
    np.testing.assert_array_equal(temporal.action, np.arange(-24, 24))


def test_multi_anchor_legacy_absolute_offsets_reproduce_old_double_add_contract():
    language = np.zeros(240, dtype=np.int64)
    cfg = MultiAnchorTemporalConfig(
        max_chunk_size=2,
        macro_stride=24,
        action_horizon=24,
        video_in_chunk_offsets=(0, 3, 6, 9, 12, 15, 18, 21),
        legacy_absolute_offsets=True,
    )

    temporal = require_multi_anchor_temporal_indices(
        72,
        language,
        len(language),
        cfg,
        episode_index=0,
    )

    np.testing.assert_array_equal(temporal.video, np.arange(48, 97, 3))
    np.testing.assert_array_equal(temporal.state, np.asarray([48, 72]))
    np.testing.assert_array_equal(temporal.action, np.arange(48, 96))
