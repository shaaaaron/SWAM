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

"""EgoDex confidence 离线分类和 episode sidecar 筛选。"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

EGODEX_CONFIDENCE_SOURCE_KEY = "observation.state_confidence"
EGODEX_CONFIDENCE_SIDECAR_NAME = "egodex_confidence_labels.jsonl"
EGODEX_CONFIDENCE_STATUS_VALID = "valid"
EGODEX_CONFIDENCE_STATUS_MISSING_ALL_ONE = "missing_all_one"
EGODEX_CONFIDENCE_STATUS_CONTAINS_ZERO = "contains_zero"
EGODEX_CONFIDENCE_STATUS_ERROR = "error"
EGODEX_CONFIDENCE_STATUSES = {
    EGODEX_CONFIDENCE_STATUS_VALID,
    EGODEX_CONFIDENCE_STATUS_MISSING_ALL_ONE,
    EGODEX_CONFIDENCE_STATUS_CONTAINS_ZERO,
    EGODEX_CONFIDENCE_STATUS_ERROR,
}


def _as_confidence_array(raw_confidence: Any, *, source: str) -> np.ndarray:
    """将原始 confidence 转为 float32 NumPy 数组。"""
    values = (
        raw_confidence.to_pylist()
        if hasattr(raw_confidence, "to_pylist")
        else raw_confidence
    )
    try:
        confidence = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to read EgoDex confidence from {source}.") from exc
    if not np.issubdtype(confidence.dtype, np.number):
        raise TypeError(
            f"EgoDex confidence must be numeric, got {confidence.dtype} in {source}."
        )
    return confidence.astype(np.float32, copy=False)


def classify_egodex_confidence(
    raw_confidence: Any,
    *,
    expected_length: int,
    source: str,
) -> str:
    """校验完整 episode confidence 并返回互斥的离线分类状态。"""
    confidence = _as_confidence_array(raw_confidence, source=source)
    expected_shape = (int(expected_length), 2)
    if confidence.shape != expected_shape:
        raise ValueError(
            "EgoDex confidence must have shape [T,2], got "
            f"{confidence.shape}, expected {expected_shape} in {source}."
        )
    if (
        not np.isfinite(confidence).all()
        or np.any(confidence < 0.0)
        or np.any(confidence > 1.0)
    ):
        raise ValueError(
            f"EgoDex confidence must contain finite values in [0,1]: {source}."
        )
    if np.all(confidence == 1.0):
        return EGODEX_CONFIDENCE_STATUS_MISSING_ALL_ONE
    if np.any(confidence == 0.0):
        return EGODEX_CONFIDENCE_STATUS_CONTAINS_ZERO
    return EGODEX_CONFIDENCE_STATUS_VALID


def load_egodex_confidence_labels(
    meta_dir: Path,
) -> dict[int, str]:
    """读取统一 sidecar，返回 episode index 到 confidence 状态的映射。"""
    sidecar_path = Path(meta_dir) / EGODEX_CONFIDENCE_SIDECAR_NAME
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"EgoDex confidence sidecar not found: {sidecar_path}. "
            "Run toolkits/lerobot/prepare_egodex_confidence_filter.py first."
        )

    labels: dict[int, str] = {}
    invalid_indices: set[int] = set()
    with open(sidecar_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            episode_index = record.get("episode_index")
            if type(episode_index) is not int or episode_index < 0:
                continue
            if episode_index in labels or episode_index in invalid_indices:
                labels.pop(episode_index, None)
                invalid_indices.add(episode_index)
                continue
            status = record.get("confidence_status")
            if type(status) is not str or status not in EGODEX_CONFIDENCE_STATUSES:
                invalid_indices.add(episode_index)
                continue
            labels[episode_index] = status
    return labels


def filter_egodex_episode_indices(
    meta_dir: Path,
    episode_indices: Iterable[int],
) -> set[int]:
    """按 confidence sidecar 返回状态为 valid 的已声明 episode。"""
    declared = {int(index) for index in episode_indices}
    labels = load_egodex_confidence_labels(meta_dir)
    return {
        index
        for index in declared
        if labels.get(index) == EGODEX_CONFIDENCE_STATUS_VALID
    }
