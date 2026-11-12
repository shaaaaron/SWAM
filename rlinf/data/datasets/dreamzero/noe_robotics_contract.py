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

"""Canonical action/state contract for NOE robotics datasets."""

from __future__ import annotations

from typing import Any

NOE_ROBOTICS_TAG = "noe_robotics_16d"
NOE_ROBOTICS_EMBODIMENT_ID = 39
NOE_ROBOTICS_EEF_CHANNELS = (
    "left_x",
    "left_y",
    "left_z",
    "left_quat_w",
    "left_quat_x",
    "left_quat_y",
    "left_quat_z",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_quat_w",
    "right_quat_x",
    "right_quat_y",
    "right_quat_z",
    "right_gripper",
)


def validate_noe_robotics_modality(modality: dict[str, Any]) -> None:
    """Validate the canonical first-16 dual-arm EEF state/action mapping."""
    for kind in ("state", "action"):
        entries = modality.get(kind)
        if not isinstance(entries, dict):
            raise ValueError(f"meta/modality.json missing {kind} mapping")
        for index, channel in enumerate(NOE_ROBOTICS_EEF_CHANNELS):
            spec = entries.get(channel)
            expected = {
                "start": index,
                "end": index + 1,
                "canonical_id": index,
            }
            if not isinstance(spec, dict):
                raise ValueError(f"meta/modality.json missing {kind}.{channel}")
            actual = {
                "start": int(spec.get("start", -1)),
                "end": int(spec.get("end", -1)),
                "canonical_id": int(spec.get("canonical_id", -1)),
            }
            if actual != expected:
                raise ValueError(
                    f"Invalid {kind}.{channel} EEF mapping: "
                    f"expected={expected}, actual={actual}"
                )
