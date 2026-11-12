"""Shared UMI quality reads and fixed episode-frame-zero reference.

Quality semantics match DreamZeroLeRobotDataset: five frame flags, with the
existing conservative episode SLAM/width fallback when any column is missing.
No coordinate-magnitude or quaternion-norm quality rules are introduced here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NOE_UMI_FRAME_QUALITY_FIELDS = (
    "slam_diagnostics_valid_mask.right",
    "slam_diagnostics_valid_mask.left",
    "slam_diagnostics_valid_mask.ego",
    "width_valid_mask.right",
    "width_valid_mask.left",
)
NOE_UMI_EPISODE_QUALITY_FIELDS = (
    "slam_validity_all",
    "width_detection",
    "width_calibration",
)


def noe_umi_episode_quality(record: dict[str, Any]) -> tuple[bool, bool]:
    return (
        record.get("slam_validity_all") is True,
        record.get("width_detection") is True
        and record.get("width_calibration") is True,
    )


def read_noe_umi_frame_quality(
    table: Any,
    indices: np.ndarray,
    episode_quality: tuple[bool, bool] = (False, False),
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if not set(NOE_UMI_FRAME_QUALITY_FIELDS).issubset(table.column_names):
        return np.full(indices.shape, all(episode_quality), dtype=bool)
    quality = np.ones(indices.shape, dtype=bool)
    for name in NOE_UMI_FRAME_QUALITY_FIELDS:
        col = table.column(name)
        values = np.asarray(col.to_numpy(zero_copy_only=False))
        if col.null_count:
            values = values.copy()
            values[np.asarray(col.is_null().to_numpy(zero_copy_only=False))] = 0
        quality &= np.asarray(values, dtype=bool)[indices]
    return quality


def read_noe_umi_episode_reference(
    table: Any,
    episode_quality: tuple[bool, bool] = (False, False),
    *,
    state_key: str = "observation.state",
) -> tuple[np.ndarray, bool]:
    """Read row zero regardless of quality; return [1,16] pose and its quality."""
    if table.num_rows == 0:
        raise ValueError("UMI episode reference requires a non-empty episode")
    if state_key in table.column_names:
        row = table.column(state_key)[0].as_py()
    elif state_key == "observation.state" and "observation" in table.column_names:
        row = table.column("observation")[0].as_py()["state"]
    else:
        raise KeyError(f"UMI episode missing reference state column {state_key!r}")
    pose = np.asarray(row, dtype=np.float32).reshape(-1)
    if pose.size < 16:
        raise ValueError(
            f"UMI episode reference must contain at least 16D, got {pose.shape}"
        )
    valid = read_noe_umi_frame_quality(
        table, np.asarray([0], dtype=np.int64), episode_quality
    )[0]
    return pose[None, :16].copy(), bool(valid)
