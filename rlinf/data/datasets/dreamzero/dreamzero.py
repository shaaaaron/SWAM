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

import bisect
import json
import os
import random
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.datasets.dreamzero.data_transforms import (
    format_training_prompt,
    normalize_instruction_text,
)
from rlinf.data.datasets.dreamzero.dense_subtasks import DenseSubtaskIndex
from rlinf.data.datasets.dreamzero.ego_instructions import (
    EgoInstructionIndex,
    append_scene_instruction,
)
from rlinf.data.datasets.dreamzero.egodex_confidence import (
    filter_egodex_episode_indices,
)
from rlinf.data.datasets.dreamzero.noe_robotics_contract import (
    NOE_ROBOTICS_TAG,   # noe_robotics_16d
    validate_noe_robotics_modality,
)
from rlinf.data.datasets.dreamzero.noe_subtasks import (
    NoeSubtaskConfig,
    NoeSubtaskIndex,
    SubtaskResolution,
)
from rlinf.data.datasets.dreamzero.noe_umi_episode_reference import (
    NOE_UMI_EPISODE_QUALITY_FIELDS as _NOE_UMI_QUALIFICATION_FIELDS,
    NOE_UMI_FRAME_QUALITY_FIELDS as _NOE_UMI_FRAME_QUALITY_FIELDS,
    noe_umi_episode_quality as _noe_umi_episode_quality,
    read_noe_umi_episode_reference,
    read_noe_umi_frame_quality,
)
from rlinf.data.datasets.dreamzero.relative_actions import (
    ACTION_QUATERNION_MODE_PRESERVE,
    ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS,
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ,
    ROBOTWIN_EEF_ACTION_KEY,
    ROBOTWIN_EEF_STATE_KEY,
    ensure_robotwin_eef_dim,
    is_dual_arm_eef_se3_relative_action,
    is_dual_arm_eef_se3_relative_state,
    is_robotwin_eef_se3_chunk_relative,
    relative_action_mode_embodiment_tag,
    resolve_action_quaternion_mode,
    resolve_relative_action_pose_reference,
    robotwin_eef_abs_to_rel_wxyz,
    temporal_continuous_robotwin_eef_action_wxyz,
)
from rlinf.data.datasets.dreamzero.sampling_strategy import (
    EmptyTemporalSampleError,
    MultiAnchorTemporalConfig,
    SamplingMode,
    TemporalIndices,
    build_fixed_window_offsets,
    require_multi_anchor_temporal_indices,
)
from rlinf.data.datasets.dreamzero.task_type_sampling import (
    TaskTypeSamplingConfig,
    build_task_type_frame_index,
)
from rlinf.data.datasets.dreamzero.utils import (
    collate_ready_sample,
    discover_local_lerobot_episode_indices,
    droid_default_state_action_slices,
    infer_modality_json_from_features,
    infer_named_component_slices,
    load_modality_json,
    load_task_texts,
    probe_video_container_fps,
    safe_lang_text,
)
from rlinf.utils.logging import get_logger

logger = get_logger()

_NOE_UMI_16D_TAG = "noe_umi_16d"
_NOE_UMI_QUALITY_MODE_EPISODE = "episode"
_NOE_UMI_QUALITY_MODE_FRAME = "frame"
_NOE_UMI_STATE_QUALITY_KEY = "__noe_umi_state_quality"
_NOE_UMI_ACTION_QUALITY_KEY = "__noe_umi_action_quality"
_NOE_UMI_EPISODE_REFERENCE_KEY = "__noe_umi_episode_reference"
_EGODEX_TAG = "egodex"
_VIDEO_ONLY_TAGS = {"ego4d", "egoverse", "egopro", "openvid"}
_DENSE_SUBTASK_SUPPORTED_TAGS = {
    _NOE_UMI_16D_TAG,
    NOE_ROBOTICS_TAG,
    "ego4d",
    "egoverse",
    "egopro",
    _EGODEX_TAG,
    "robotwin",
}
_DEFAULT_DENSE_SUBTASK_FILE = "meta/subtask/subtasks_dense_seed.parquet"
_SUBTASK_MODE_LEGACY = "legacy"
_SUBTASK_MODE_UNIFIED_DENSE = "unified_dense"
_PACKED_EEF_16D_TAGS = {_NOE_UMI_16D_TAG, NOE_ROBOTICS_TAG} # noe_umi_16d, noe_robotics_16d
_PACKED_EEF_16D_VIDEO_SOURCES = {
    "video.image": "observation.images.image",
    "video.wrist_image_1": "observation.images.wrist_image_1",
    "video.wrist_image_2": "observation.images.wrist_image_2",
}
_PACKED_EEF_16D_COMPONENT_SOURCES = {
    "state": {
        "state.eef_position": ("observation.state", slice(0, 16)),
    },
    "action": {
        "action.eef_position": ("action", slice(0, 16)),
    },
}
_EGODEX_ACTION_COMPONENT_SOURCES = {
    "action.left_hand_position": ("action", slice(0, 3)),
    "action.right_hand_position": ("action", slice(3, 6)),
}
_EGODEX_ACTION_STATE_KEY = "__egodex_action_state"
_EPISODE_FIRST_VIDEO_PREFIX = "__episode_first_video."
_EPISODE_FIRST_NUM_VIDEO_FRAMES_KEY = "__episode_first_num_video_frames"
_NOE_SUBTASK_INSTRUCTION_KEY = "__noe_subtask_instruction"
_DENSE_SUBTASK_INSTRUCTION_KEY = "__dense_subtask_instruction"
_EGO_INSTRUCTION_WINDOW_KEY = "__ego_instruction_window"
_NOE_SUBTASK_STATS_WARMUP = 1_000
_NOE_SUBTASK_STATS_INTERVAL = 10_000


def _check_noe_umi_episode_qualification(
    record: dict[str, Any],
) -> tuple[bool, str | None]:
    """检查 NOE/UMI episode 质量字段，返回是否合格和 metadata 错误。"""
    missing = [
        field for field in _NOE_UMI_QUALIFICATION_FIELDS if field not in record
    ]
    if missing:
        return False, f"missing fields={missing}"

    invalid_types = {
        field: type(record[field]).__name__
        for field in _NOE_UMI_QUALIFICATION_FIELDS
        if type(record[field]) is not bool
    }
    if invalid_types:
        return False, f"invalid field types={invalid_types}"

    return (
        all(record[field] is True for field in _NOE_UMI_QUALIFICATION_FIELDS),
        None,
    )


def _suppress_ffmpeg_log(worker_id: int = 0) -> None:
    """Suppress FFmpeg/libdav1d INFO banners in DataLoader worker subprocesses.

    torchcodec bundles its own FFmpeg (separate from system/PyAV), so
    ``av.logging.set_level()`` does not reach it.  We call ``av_log_set_level``
    on every loaded copy of libavutil via ctypes to silence the decoder init
    banner (``AV_LOG_ERROR=16`` suppresses INFO / VERBOSE / DEBUG).
    """
    import ctypes
    import ctypes.util as _cutils

    candidates: set[str] = set()
    for _soname in (
        "avutil",
        "libavutil",
        "libavutil.so",
        "libavutil.so.60",
        "libavutil.so.59",
        "libavutil.so.58",
        "libavutil.so.57",
        "libavutil.so.56",
    ):
        _path = _cutils.find_library(_soname)
        if _path:
            candidates.add(_path)

    try:
        with open("/proc/self/maps", encoding="utf-8") as maps_file:
            for line in maps_file:
                path = line.rstrip().split()[-1]
                if "/" in path and "libavutil" in path:
                    candidates.add(path)
    except Exception:
        pass

    for _path in candidates:
        try:
            _lib = ctypes.CDLL(_path)
            _lib.av_log_set_level(16)  # AV_LOG_ERROR
        except Exception:
            pass


_suppress_ffmpeg_log()


def _is_video_decode_error(exc: Exception) -> bool:
    message = str(exc)
    return any(
        pattern in message
        for pattern in (
            "Could not push packet to decoder",
            "Invalid data found when processing input",
            "decode_video_frames",
            "torchcodec",
            "decord",
            "Invalid frame index=",
        )
    )


def _read_bad_video_episode_manifest(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    bad_episodes: set[int] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                bad_episodes.add(int(record["episode_index"]))
            except Exception:
                logger.warning("Ignoring malformed bad video manifest row in %s: %s", path, line.rstrip())
    return bad_episodes


def _normalize_dreamzero_data_paths(data_paths: str | list[str]) -> list[str]:
    if isinstance(data_paths, (str, Path)):
        return [str(data_paths)]
    if isinstance(data_paths, Iterable):
        paths = [str(path) for path in data_paths]
        if paths:
            return paths
    return [str(data_paths)]


def _resolve_dreamzero_data_paths(
    data_paths: str | list[str],
    embodiment_tag: str,
) -> list[str]:
    """Resolve configured DreamZero data paths into LeRobot dataset roots.

    RoboTwin training configs often point at subset directories containing many
    task datasets. Each task is its own LeRobot root with ``meta/info.json``.
    Non-RoboTwin embodiments keep the configured shape for backward
    compatibility.
    """

    paths = _normalize_dreamzero_data_paths(data_paths)
    if str(embodiment_tag) not in {
        "robotwin",
        _NOE_UMI_16D_TAG,
        _EGODEX_TAG,
        "egoverse",
        "egopro",
    }:
        return paths

    task_dirs: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        root = Path(raw_path).expanduser()
        candidates: list[Path]
        if (root / "meta" / "info.json").is_file():
            candidates = [root]
        else:
            info_paths = set(root.rglob("meta/info.json"))
            if root.is_dir():
                # rglob does not descend into directory symlinks used by link farms.
                info_paths.update(
                    child / "meta" / "info.json"
                    for child in root.iterdir() # 枚举一级子目录
                    if (child / "meta" / "info.json").is_file()
                )
            candidates = sorted(
                info_json.parent.parent for info_json in info_paths
            )
        for candidate in candidates:
            if str(embodiment_tag) == _EGODEX_TAG:
                episodes_path = candidate / "meta" / "episodes.jsonl"
                with open(episodes_path, encoding="utf-8") as f:
                    episode_indices = [
                        int(json.loads(line)["episode_index"])
                        for line in f
                        if line.strip()
                    ]
                kept = filter_egodex_episode_indices(
                    candidate / "meta", episode_indices
                )
                # 跳过 confidence 过滤后为空的 EgoDex task，避免后续构造空 Dataset 报错。
                if not kept:
                    logger.warning(
                        "Skipping EgoDex task root with no valid confidence episodes: "
                        "path=%s",
                        candidate,
                    )
                    continue
            resolved = str(candidate)
            if resolved not in seen:
                seen.add(resolved)
                task_dirs.append(resolved)

    if not task_dirs:
        raise FileNotFoundError(
            "No DreamZero LeRobot datasets found. Expected each dataset directory "
            "to contain meta/info.json under one of data.train_data_paths: "
            f"{paths}"
        )
    return sorted(task_dirs)


def _prepare_noe_subtask_configs(
    data_paths: list[str],
    noe_subtask: Any,
    *,
    embodiment_tag: str,
    sampling_mode: str,
) -> list[Any]:
    """
    [
        {
            "mode": "prefer",
            "metadata_file": "meta/subtask_segments.parquet",
            "missing_metadata_policy": "fallback",
            "instruction_key": "subtask_en",
            "required_quality": 1,
        },
        {
            "mode": "off",
        },
        ......
    ]
    mode：prefer 和 off，表示是否使用sutbask
    missing_metadata_policy: fallback 和 error，表示部分subtask缺失的处理方式
    """
    config = NoeSubtaskConfig.from_value(noe_subtask)
    if config.mode != "prefer": # 默认不适用subtask
        return [noe_subtask] * len(data_paths)

    # subtask 只支持 noe_umi_16 和 multi anchor
    if str(embodiment_tag) != _NOE_UMI_16D_TAG:
        return [noe_subtask] * len(data_paths)
    if sampling_mode != "multi_anchor":
        return [noe_subtask] * len(data_paths)

    # 将路径标准化
    configured_path = Path(config.metadata_file)
    metadata_paths: list[Path] = []
    for data_path in data_paths:
        root = Path(data_path).resolve()
        source_path = (
            configured_path if configured_path.is_absolute() else root / configured_path
        ).resolve()
        metadata_paths.append(source_path)

    available = [path.is_file() for path in metadata_paths]
    if not any(available):
        expected = "\n".join(f"  - {path}" for path in metadata_paths)
        raise FileNotFoundError(
            "NOE subtask mode='prefer' 未找到任何元数据 parquet。"
            f"期望路径：\n{expected}"
        )

    missing_paths = [
        path for path, present in zip(metadata_paths, available) if not present
    ]
    if missing_paths:
        if config.missing_metadata_policy == "error":
            missing = "\n".join(f"  - {path}" for path in missing_paths)
            raise FileNotFoundError(
                "NOE subtask mode='prefer' 缺少部分元数据 parquet； "
                "missing_metadata_policy='error' 不允许静默降级。缺失路径：\n"
                f"{missing}"
            )
        logger.warning(
            "NOE subtask metadata is partially available: %s/%s roots have "
            "metadata; missing roots use mode='off' because "
            "missing_metadata_policy='fallback'. Missing paths: %s",
            sum(available),
            len(available),
            [str(path) for path in missing_paths],
        )

    return [noe_subtask if present else {"mode": "off"} for present in available]


class DreamZeroLeRobotDataset(Dataset):
    """Generic LeRobot-backed DreamZero SFT dataset.

    Loads LeRobot modalities, applies ``ComposedModalityTransform`` per sample
    (typically in DataLoader workers), and returns collate-ready numpy dicts.
    """

    def __init__(
        self,
        data_path: str | list[str],
        video_keys: list[str],
        state_keys: list[str],
        action_keys: list[str],
        language_keys: list[str],
        data_transform: Any,
        lazy_load: bool,
        num_frames: int,
        state_horizon: int,
        action_horizon: int,
        max_chunk_size: int,
        relative_action: bool = False,
        relative_action_keys: list[str] | None = None,
        relative_action_mode: str | None = None,
        relative_action_pose_reference: str | None = None,
        action_quaternion_mode: str | None = None,
        pq_cache_max_episodes: int = 128,
        video_tolerance_s: float = 0.1,
        video_backend: str = "pyav",
        sampling_mode: SamplingMode = "multi_anchor",
        multi_anchor_resample_attempts: int = 32,
        multi_anchor_macro_stride: int = 24,
        multi_anchor_legacy_absolute_offsets: bool = False,
        bad_video_manifest_name: str = "bad_video_episodes.jsonl",
        episode_exclusion_manifest: str | None = None,
        dataloader_profile: bool = False,
        dataloader_profile_first_n: int = 4,
        dataloader_profile_slow_s: float = 2.0,
        return_episode_first_images: bool = False,
        multi_anchor_video_offsets: list[int] | tuple[int, ...] = (
            0,
            3,
            6,
            9,
            12,
            15,
            18,
            21,
        ),
        embodiment_tag: str | None = None,
        episode_instruction_key: str | None = None,
        subtask_mode: str = _SUBTASK_MODE_LEGACY,
        subtask_file: str = _DEFAULT_DENSE_SUBTASK_FILE,
        use_subtask_instruction: bool = False,
        use_dense_caption_instruction: bool = False,
        dense_caption_include_object: bool = True,
        ego_caption_segment_mode: str | None = None,
        use_scene_instruction: bool = False,
        filter_noe_umi_qualified_episodes: bool = False,
        noe_umi_quality_mode: str = _NOE_UMI_QUALITY_MODE_EPISODE,
        noe_subtask: Any | None = None,
        episode_discovery_mode: str = "metadata",
        task_type_sampling: Any | None = None,
        tasks_filename: str = "tasks.jsonl",
        subtask_caption_field: str = "action_detail_en",
        caption_granularity: str = "subtask",
        subtask_caption_extra: str = "none",
        video_key_mapping: Mapping[str, str] | None = None,
        prompt_postfix: str | None = "",
        prompt_style: str | None = None,
    ):
        if isinstance(data_path, (list, tuple)):
            if len(data_path) == 0:
                raise ValueError(
                    "DreamZeroLeRobotDataset requires at least one data path."
                )
            data_path = data_path[0]
        if prompt_postfix is not None and not isinstance(prompt_postfix, str):
            raise TypeError("prompt_postfix must be a string or None.")
        self.prompt_postfix = (prompt_postfix or "").strip()
        self.data_path = str(data_path)
        self.embodiment_tag = str(embodiment_tag or "")
        self.prompt_style = str(prompt_style or "").strip().lower()
        if self.prompt_style not in {"", "robotics"}:
            raise ValueError(
                "prompt_style must be empty or 'robotics', "
                f"got {prompt_style!r}."
            )
        self.episode_instruction_key = (
            str(episode_instruction_key) if episode_instruction_key else None
        )
        self.subtask_mode = str(subtask_mode or _SUBTASK_MODE_LEGACY).strip().lower()
        if self.subtask_mode not in {
            _SUBTASK_MODE_LEGACY,
            _SUBTASK_MODE_UNIFIED_DENSE,
        }:
            raise ValueError(
                "subtask_mode must be 'legacy' or 'unified_dense', "
                f"got {subtask_mode!r}."
            )
        self.subtask_file = str(subtask_file).strip()
        if not self.subtask_file:
            raise ValueError("subtask_file must not be empty.")
        self.subtask_caption_field = subtask_caption_field
        self.subtask_caption_extra = subtask_caption_extra
        self.caption_granularity = str(caption_granularity or "subtask").strip().lower()
        if self.caption_granularity not in {"subtask", "episode"}:
            raise ValueError(
                "caption_granularity must be 'subtask' or 'episode', "
                f"got {caption_granularity!r}."
            )
        self.use_subtask_instruction = bool(use_subtask_instruction)
        if (
            self.subtask_caption_extra != "none"
            and self.subtask_mode != _SUBTASK_MODE_UNIFIED_DENSE
        ):
            raise ValueError(
                "subtask_caption_extra requires subtask_mode='unified_dense'."
            )
        self.use_dense_caption_instruction = bool(use_dense_caption_instruction)
        self.dense_caption_include_object = bool(dense_caption_include_object)
        self.ego_caption_segment_mode = ego_caption_segment_mode
        self.use_scene_instruction = bool(use_scene_instruction)
        if self.use_subtask_instruction and self.use_dense_caption_instruction:
            raise ValueError(
                "use_subtask_instruction and use_dense_caption_instruction "
                "are mutually exclusive."
            )
        if (
            (self.use_subtask_instruction or self.use_scene_instruction)
            and self.embodiment_tag not in {"ego4d", "egoverse"}
        ):
            raise ValueError(
                "Ego instruction flags currently require embodiment_tag='ego4d' "
                "or 'egoverse'; "
                f"got {self.embodiment_tag!r}."
            )
        if self.use_dense_caption_instruction and self.embodiment_tag not in {
            "ego4d",
            "egoverse",
            _EGODEX_TAG,
        }:
            raise ValueError(
                "use_dense_caption_instruction currently requires "
                "embodiment_tag='ego4d', 'egoverse', or 'egodex'; "
                f"got {self.embodiment_tag!r}."
            )
        self.filter_noe_umi_qualified_episodes = bool(
            filter_noe_umi_qualified_episodes
        )
        quality_mode = str(noe_umi_quality_mode).strip().lower()
        if quality_mode not in {
            _NOE_UMI_QUALITY_MODE_EPISODE,
            _NOE_UMI_QUALITY_MODE_FRAME,
        }:
            raise ValueError(
                "noe_umi_quality_mode must be 'episode' or 'frame', "
                f"got {noe_umi_quality_mode!r}."
            )
        self.noe_umi_quality_mode = quality_mode
        self._noe_umi_episode_quality: dict[int, tuple[bool, bool]] = {}
        self._noe_umi_quality_fallback_warned = False
        self.task_type_sampling_config = TaskTypeSamplingConfig.from_value(
            task_type_sampling
        )
        self.task_type_frame_index = None
        self.noe_subtask_config = NoeSubtaskConfig.from_value(noe_subtask)
        self._noe_subtask_resolution_counts = { # 简单的情况记录
            "total": 0,
            "matched": 0,
            "ambiguous_random": 0,
            "no_match": 0,
            "mixed": 0,
        }
        # 若配置该字段，则使用 episodes.jsonl 中的 episode-level instruction。
        self.lazy_load = bool(lazy_load)
        if self.subtask_mode == _SUBTASK_MODE_UNIFIED_DENSE:
            if self.embodiment_tag not in _DENSE_SUBTASK_SUPPORTED_TAGS:
                raise ValueError(
                    "subtask_mode='unified_dense' requires a supported "
                    f"embodiment, got {self.embodiment_tag!r}."
                )
            if not self.lazy_load:
                raise ValueError(
                    "subtask_mode='unified_dense' requires lazy_load=True."
                )
            if (
                self.use_subtask_instruction
                or self.use_dense_caption_instruction
                or self.use_scene_instruction
                or self.noe_subtask_config.mode != "off"
            ):
                raise ValueError(
                    "subtask_mode='unified_dense' cannot be combined with legacy "
                    "subtask, dense-caption, or scene instruction settings."
                )
        discovery_mode = str(episode_discovery_mode).strip().lower()
        if discovery_mode not in {"scan", "metadata"}:
            raise ValueError(
                "episode_discovery_mode must be 'scan' or 'metadata', "
                f"got {episode_discovery_mode!r}."
            )
        self.episode_discovery_mode = discovery_mode
        self.sampling_mode: SamplingMode = sampling_mode
        self.state_horizon = int(state_horizon)
        self.action_horizon = int(action_horizon)
        self.max_chunk_size = int(max_chunk_size)
        self.num_frames = int(num_frames)
        self.multi_anchor_resample_attempts = max(
            1, int(multi_anchor_resample_attempts)
        )
        if self.noe_subtask_config.mode == "prefer":
            if not self._is_noe_umi_16d():
                raise ValueError(
                    "data.noe_subtask.mode='prefer' currently requires "
                    f"embodiment_tag={_NOE_UMI_16D_TAG!r}, got "
                    f"{self.embodiment_tag!r}."
                )
            if self.sampling_mode != "multi_anchor":
                raise ValueError(
                    "data.noe_subtask.mode='prefer' currently requires "
                    "data.sampling_mode='multi_anchor'."
                )
            if not self.episode_instruction_key:
                raise ValueError(
                    "data.noe_subtask.mode='prefer' requires "
                    "data.episode_instruction_key for fallback instructions."
                )
        if self.state_horizon <= 0:
            raise ValueError(f"state_horizon must be positive, got {state_horizon!r}")
        if self.action_horizon <= 0:
            raise ValueError(
                f"action_horizon must be positive, got {action_horizon!r}"
            )
        if self.max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size!r}")
        self.relative_action = bool(relative_action)
        self.relative_action_keys = [
            self._short_modality_key(str(key)) for key in (relative_action_keys or [])
        ]
        self.relative_action_mode = (
            "" if relative_action_mode is None else str(relative_action_mode)
        )
        self.relative_action_pose_reference = resolve_relative_action_pose_reference(
            relative_action_pose_reference,
            relative_action_mode=(
                self.relative_action_mode if self.relative_action else None
            ),
        )
        self.action_quaternion_mode = resolve_action_quaternion_mode(
            action_quaternion_mode
        )
        if (
            self.relative_action
            and self.action_quaternion_mode != ACTION_QUATERNION_MODE_PRESERVE
        ):
            raise ValueError(
                "action_quaternion_mode='temporal_continuous' currently supports "
                "absolute actions only."
            )
        self.data_transform = data_transform
        self._profile_enabled = bool(dataloader_profile)
        self._profile_first_n = max(0, int(dataloader_profile_first_n))
        self._profile_slow_s = max(0.0, float(dataloader_profile_slow_s))
        self._profile_seen = 0
        self.return_episode_first_images = bool(return_episode_first_images)

        self.video_keys = list(video_keys)
        self.state_keys = list(state_keys)
        self.action_keys = list(action_keys)
        self.language_keys = list(language_keys)

        # 检查key maping是否合法
        if video_key_mapping is not None and not isinstance(video_key_mapping, Mapping):
            raise TypeError(
                "video_key_mapping must be a mapping of video keys to feature keys."
            )
        self.video_key_mapping = dict(video_key_mapping or {})
        unknown_keys = [
            key for key in self.video_key_mapping if key not in self.video_keys
        ]
        if unknown_keys:
            raise ValueError(
                f"video_key_mapping contains keys not in video_keys: {unknown_keys!r}."
            )
        for key, source in self.video_key_mapping.items():
            if not isinstance(source, str) or not source or source != source.strip():
                raise ValueError(
                    f"video_key_mapping[{key!r}] must be a non-empty feature key."
                )


        if self.embodiment_tag in _VIDEO_ONLY_TAGS:
            if not self.video_keys or self.state_keys or self.action_keys:
                raise ValueError(
                    f"{self.embodiment_tag} requires video keys and no source "
                    "state/action keys; "
                    f"got video={self.video_keys}, state={self.state_keys}, "
                    f"action={self.action_keys}"
                )
        elif not self.video_keys or not self.state_keys:
            raise ValueError(
                "DreamZeroLeRobotDataset requires video/state modality keys; "
                f"got video={self.video_keys}, state={self.state_keys}, action={self.action_keys}"
            )
        if not self.language_keys:
            raise ValueError(
                "DreamZeroLeRobotDataset requires at least one language key."
            )
        if self.relative_action and not is_dual_arm_eef_se3_relative_action(
            self.relative_action_mode
        ):
            raise ValueError(
                "relative_action=True requires "
                "a supported dual-arm EEF relative action mode."
            )
        if not self.relative_action and is_dual_arm_eef_se3_relative_action(
            self.relative_action_mode
        ):
            raise ValueError(
                "Dual-arm EEF relative action modes require "
                "relative_action=True."
            )
        if self.relative_action and is_dual_arm_eef_se3_relative_action(
            self.relative_action_mode
        ):
            expected_tag = relative_action_mode_embodiment_tag(
                self.relative_action_mode
            )
            if self.embodiment_tag != expected_tag:
                raise ValueError(
                    f"relative_action_mode={self.relative_action_mode!r} requires "
                    f"embodiment_tag={expected_tag!r}, got {self.embodiment_tag!r}."
                )
            if set(self.relative_action_keys) != {"eef_position"}:
                raise ValueError(
                    "Dual-arm EEF relative action modes require "
                    "relative_action_keys=['eef_position']."
                )
            if ROBOTWIN_EEF_STATE_KEY not in self.state_keys:
                raise ValueError(
                    "Dual-arm EEF relative action modes require "
                    f"{ROBOTWIN_EEF_STATE_KEY!r} in state_keys."
                )
            if ROBOTWIN_EEF_ACTION_KEY not in self.action_keys:
                raise ValueError(
                    "Dual-arm EEF relative action modes require "
                    f"{ROBOTWIN_EEF_ACTION_KEY!r} in action_keys."
                )
            if is_robotwin_eef_se3_chunk_relative(self.relative_action_mode):
                if self.sampling_mode != "multi_anchor" or self.state_horizon != 1:
                    raise ValueError(
                        "RoboTwin chunk-relative mode requires "
                        "sampling_mode='multi_anchor' and state_horizon=1; "
                        f"got sampling_mode={self.sampling_mode!r}, "
                        f"state_horizon={self.state_horizon}."
                    )

        self._root = Path(self.data_path).resolve()
        self._meta_dir = self._root / "meta"

        # Runtime decode failures stay in memory; only explicit exclusions are read.
        self._bad_video_manifest_path = self._meta_dir / str(bad_video_manifest_name)
        self._bad_video_episodes: set[int] = set()
        self._episode_exclusion_manifest = (    # 储存排除文件的jsonl
            (self._root / Path(episode_exclusion_manifest).expanduser()).resolve()
            if episode_exclusion_manifest is not None
            else None
        )
        with open(self._meta_dir / "info.json") as f:
            self._info = json.load(f)
        self._fps = float(self._info.get("fps", 10))
        self._version = str(self._info.get("codebase_version", "v3.0"))
        self.tasks_filename = str(tasks_filename).strip()
        self._tasks = load_task_texts(
            self._meta_dir,
            tasks_filename=self.tasks_filename,
        )
        # 未配置 episode_instruction_key 时，普通数据集仍走 tasks.jsonl/task_index。
        self._episode_instruction_texts = self._load_episode_instruction_texts()
        self._ego_instruction_index = self._load_ego_instruction_index()
        self._noe_subtask_index = self._load_noe_subtask_index()
        self._dense_subtask_index = self._load_dense_subtask_index()
        self._features = self._info.get("features") or {}
        self._modality_meta = load_modality_json(self._meta_dir)
        if not self._modality_meta:
            self._modality_meta = infer_modality_json_from_features(self._features)
            logger.info(
                "meta/modality.json not found under %s; inferred modality mapping from info.json features.",
                self._meta_dir,
            )
        if self._is_noe_robotics_16d():
            validate_noe_robotics_modality(self._modality_meta)
        self._source_video_key = self._build_source_video_key_map()
        self._state_components = self._build_component_sources("state", self.state_keys)
        self._action_components = self._build_component_sources(
            "action", self.action_keys
        )
        self._language_sources = self._build_language_sources()
        self._vector_source_keys = sorted(
            {
                source
                for source, _ in [
                    *self._state_components.values(),
                    *self._action_components.values(),
                ]
            }
        )

        self._use_image_parquet_tree = self._uses_image_parquet_storage()
        self._use_lazy_video_tree = bool(
            self.lazy_load
            and not self._use_image_parquet_tree
            and self._info.get("video_path")
            and self._root.exists()
        )
        if self._episode_exclusion_manifest is not None and not self._use_lazy_video_tree:
            raise ValueError(
                "episode_exclusion_manifest requires lazy local MP4 datasets."
            )
        if self._is_egodex() and not self._use_lazy_video_tree:
            raise ValueError(
                "EgoDex confidence sidecar filtering requires the lazy local MP4 "
                "dataset path so episode filtering cannot be bypassed."
            )
        if self.filter_noe_umi_qualified_episodes:
            if not self._is_noe_umi_16d():
                raise ValueError(
                    "filter_noe_umi_qualified_episodes=True requires "
                    f"embodiment_tag={_NOE_UMI_16D_TAG!r}, got {self.embodiment_tag!r}."
                )
            if not self._use_lazy_video_tree:
                raise ValueError(
                    "filter_noe_umi_qualified_episodes=True requires the lazy local "
                    "MP4 dataset path so episode filtering cannot be bypassed."
                )
        if self.return_episode_first_images and not self._use_lazy_video_tree:
            raise NotImplementedError(
                "return_episode_first_images=True requires lazy local mp4 video "
                "datasets so the original episode frame 0 can be decoded explicitly."
            )
        self._video_backend = str(video_backend)
        if (
            self.sampling_mode == "multi_anchor"
            and not self.lazy_load
            and not self._use_image_parquet_tree
        ):
            raise ValueError(
                "DreamZeroLeRobotDataset sampling_mode='multi_anchor' requires lazy_load=True "
                "or a v2 image-parquet layout (no mp4 video_path)."
            )
        if self.sampling_mode == "fixed_window":
            self._fixed_window_temporal: TemporalIndices = build_fixed_window_offsets(
                self.num_frames,
                self.state_horizon,
                self.action_horizon,
                self.max_chunk_size,
            )
            self._multi_anchor_cfg = None
        else:
            self._fixed_window_temporal = None
            self._multi_anchor_cfg = MultiAnchorTemporalConfig(
                max_chunk_size=self.max_chunk_size,
                macro_stride=int(multi_anchor_macro_stride),
                action_horizon=self.action_horizon,
                video_in_chunk_offsets=tuple(int(x) for x in multi_anchor_video_offsets),
                legacy_absolute_offsets=bool(multi_anchor_legacy_absolute_offsets),
            )
        if self._use_lazy_video_tree:
            self._init_lazy_map_style(pq_cache_max_episodes, video_tolerance_s)
        else:
            self._init_lerobot_or_v2_parquet()
        if self.task_type_sampling_config.enabled:
            self._init_task_type_frame_index()

    @staticmethod
    def _short_modality_key(key: str) -> str:
        return key.split(".", 1)[1] if "." in key else key

    def _is_noe_umi_16d(self) -> bool:
        return self.embodiment_tag == _NOE_UMI_16D_TAG

    def _is_noe_robotics_16d(self) -> bool:
        return self.embodiment_tag == NOE_ROBOTICS_TAG

    def _uses_packed_eef_16d_contract(self) -> bool:
        return self.embodiment_tag in _PACKED_EEF_16D_TAGS

    def _uses_noe_umi_frame_quality(self) -> bool:
        return (
            self.filter_noe_umi_qualified_episodes
            and self.noe_umi_quality_mode == _NOE_UMI_QUALITY_MODE_FRAME
            and self._is_noe_umi_16d()
        )

    def _warn_noe_umi_quality_fallback(
        self,
        episode_index: int,
        missing_fields: Iterable[str],
    ) -> None:
        if self._noe_umi_quality_fallback_warned:
            return
        self._noe_umi_quality_fallback_warned = True
        logger.warning(
            "NOE/UMI frame quality columns are incomplete under %s; falling back "
            "to episode-level quality fields. episode=%s missing=%s",
            self.data_path,
            episode_index,
            sorted(missing_fields),
        )

    def _is_egodex(self) -> bool:
        """返回当前数据集是否使用 EgoDex 规则。"""
        return self.embodiment_tag == _EGODEX_TAG

    def _load_episode_instruction_texts(self) -> dict[int, Any]:
        if not self.episode_instruction_key:
            if self._is_noe_umi_16d():
                raise ValueError(
                    "NOE/UMI requires data.episode_instruction_key, e.g. "
                    "'task_annotation', because parquet task_index is not reliable."
                )
            return {}
        episodes_path = self._meta_dir / "episodes.jsonl"
        if not episodes_path.is_file():
            raise FileNotFoundError(
                f"episode_instruction_key={self.episode_instruction_key!r} is "
                f"configured but episodes.jsonl is missing: {episodes_path}"
            )
        texts: dict[int, Any] = {}
        with open(episodes_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    episode_index = int(obj["episode_index"])
                except Exception:
                    logger.warning(
                        "Ignoring malformed episodes.jsonl row in %s: %s",
                        episodes_path,
                        line.rstrip(),
                    )
                    continue
                texts[episode_index] = obj.get(self.episode_instruction_key)
        empty_count = 0
        empty_examples = []
        for episode_index, raw_text in texts.items():
            if raw_text is not None and str(raw_text).strip():
                continue
            empty_count += 1
            if len(empty_examples) < 5:
                empty_examples.append(episode_index)
        if empty_count:
            total = len(texts)
            logger.warning(
                "Episode instructions contain missing/empty values: key=%r "
                "dataset_path=%s empty=%s total=%s rate=%.2f%% "
                "example_episode_indices=%s; affected samples return empty strings.",
                self.episode_instruction_key,
                self.data_path,
                empty_count,
                total,
                100.0 * empty_count / total,
                empty_examples,
            )
        return texts

    def _load_noe_subtask_index(self) -> NoeSubtaskIndex | None:
        # 仅在 prefer 模式加载当前 dataset root 的 subtask parquet 索引。
        if self.noe_subtask_config.mode == "off":
            return None
        index = NoeSubtaskIndex.load(self._root, self.noe_subtask_config)
        logger.info(
            "Loaded NOE subtask index: path=%s episodes=%s segments=%s texts=%s "
            "instruction_key=%s quality=%s",
            index.source_path,
            index.num_episodes,
            index.num_segments,
            index.num_texts,
            self.noe_subtask_config.instruction_key,
            self.noe_subtask_config.required_quality,
        )
        return index

    def _load_dense_subtask_index(self) -> DenseSubtaskIndex | None:
        if self.subtask_mode != _SUBTASK_MODE_UNIFIED_DENSE:
            return None
        index = DenseSubtaskIndex.load(
            self._root,
            self.subtask_file,
            caption_field=self.subtask_caption_field,
            caption_granularity=self.caption_granularity,
            caption_extra=self.subtask_caption_extra,
        )
        logger.info(
            "Loaded unified dense captions: path=%s granularity=%s caption_field=%s rows=%s episodes=%s "
            "segments=%s filtered=%s caption_extra=%s",
            index.source_path,
            index.caption_granularity,
            self.subtask_caption_field,
            index.num_rows,
            index.num_episodes,
            index.num_segments,
            index.num_filtered_segments,
            self.subtask_caption_extra,
        )
        return index

    def _load_ego_instruction_index(self) -> EgoInstructionIndex | None:
        if not (
            self.use_subtask_instruction
            or self.use_dense_caption_instruction
            or self.use_scene_instruction
        ):
            return None
        index = EgoInstructionIndex.load(
            self._root,
            use_subtask_instruction=self.use_subtask_instruction,
            use_dense_caption_instruction=self.use_dense_caption_instruction,
            dense_caption_include_object=self.dense_caption_include_object,
            ego_caption_segment_mode=self.ego_caption_segment_mode,
            use_scene_instruction=self.use_scene_instruction,
        )
        logger.info(
            "Loaded Ego instruction sidecars: path=%s subtask=%s dense=%s "
            "dense_object=%s segment_mode=%s scene=%s",
            self.data_path,
            self.use_subtask_instruction,
            self.use_dense_caption_instruction,
            self.dense_caption_include_object,
            index.ego_caption_segment_mode,
            self.use_scene_instruction,
        )
        return index

    def _resolve_episode_instruction_text(self, sample: dict[str, Any]) -> str:
        if not self.episode_instruction_key:
            raise RuntimeError(
                "_resolve_episode_instruction_text requires episode_instruction_key."
            )
        if "episode_index" not in sample:
            raise KeyError(
                f"Missing episode_index in sample from dataset_path={self.data_path}."
            )
        try:
            episode_index = int(sample["episode_index"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid episode_index={sample['episode_index']!r} in sample "
                f"from dataset_path={self.data_path}."
            ) from None
        raw_text = self._episode_instruction_texts.get(episode_index)
        return "" if raw_text is None else str(raw_text).strip()

    def _uses_image_parquet_storage(self) -> bool:
        """Detect LeRobot trees where image frames live in parquet, not mp4 files."""
        source_keys = set(self._source_video_key.values())
        if not source_keys:
            return False
        source_features = [self._features.get(key) or {} for key in source_keys]
        has_video_feature = any(
            feature.get("dtype") == "video" for feature in source_features
        )
        if has_video_feature:
            return False
        all_image_features = all(
            feature.get("dtype") == "image" for feature in source_features
        )
        if not all_image_features:
            return False
        return (
            self._version.startswith("v2")
            or int(self._info.get("total_videos") or 0) == 0
            or not self._info.get("video_path")
        )

    def _modality_entry(self, modality: str, key: str) -> dict[str, Any] | None:
        entries = self._modality_meta.get(modality)
        if not isinstance(entries, dict):
            return None
        return entries.get(self._short_modality_key(key))

    def _build_source_video_key_map(self) -> dict[str, str]:
        for transform_key, source_key in self.video_key_mapping.items():
            feature = self._features.get(source_key)
            if not isinstance(feature, dict) or feature.get("dtype") not in {
                "video", "image"
            }:
                raise ValueError(
                    f"video_key_mapping[{transform_key!r}]={source_key!r} must "
                    f"refer to a video/image feature in {self.data_path}."
                )
        if self._uses_packed_eef_16d_contract():
            if self.video_key_mapping:
                raise ValueError(
                    "video_key_mapping is not supported for packed EEF datasets."
                )
            mapping = {
                key: _PACKED_EEF_16D_VIDEO_SOURCES[key]
                for key in self.video_keys
                if key in _PACKED_EEF_16D_VIDEO_SOURCES
            }
            missing_features = [
                source for source in mapping.values() if source not in self._features
            ]
            if len(mapping) != len(self.video_keys) or missing_features:
                raise KeyError(
                    f"Could not map packed EEF video keys for {self.video_keys} under "
                    f"{self.data_path}; mapping={mapping}, "
                    f"missing_features={missing_features}, "
                    f"available={sorted(k for k in self._features if k.startswith('observation.images.'))}"
                )
            return mapping

        image_features = [
            k for k in self._features if k.startswith("observation.images.")
        ]
        mapping: dict[str, str] = {}
        for transform_key in self.video_keys:
            if transform_key in self.video_key_mapping:
                mapping[transform_key] = self.video_key_mapping[transform_key]
                continue
            short = self._short_modality_key(transform_key)
            entry = self._modality_entry("video", transform_key)
            if entry is not None and entry.get("original_key"):
                mapping[transform_key] = str(entry["original_key"])
                continue
            canonical = f"observation.images.{short}"
            if canonical in self._features:
                mapping[transform_key] = canonical
                continue
            # LeRobot v2 LIBERO stores image columns without the observation.images prefix.
            if short in self._features or short in ("image", "wrist_image"):
                mapping[transform_key] = short
                continue
            match = [k for k in image_features if k.endswith(f".{short}")]
            if match:
                mapping[transform_key] = match[0]
        if len(mapping) != len(self.video_keys):
            missing = [k for k in self.video_keys if k not in mapping]
            raise KeyError(
                f"Could not map transform video keys {missing} to LeRobot image features "
                f"under {self.data_path}; available={sorted(image_features)}"
            )
        return mapping

    def _default_vector_source_key(self, modality: str) -> str:
        candidates = (
            ("observation.state", "state")
            if modality == "state"
            else ("action", "actions")
        )
        for candidate in candidates:
            if candidate in self._features:
                return candidate
        return candidates[0]

    def _build_component_sources(
        self, modality: str, transform_keys: list[str]
    ) -> dict[str, tuple[str, slice]]:
        if self._is_egodex() and modality == "action":
            if transform_keys != list(_EGODEX_ACTION_COMPONENT_SOURCES):
                raise ValueError(
                    "EgoDex action keys must be "
                    f"{list(_EGODEX_ACTION_COMPONENT_SOURCES)}, got {transform_keys}."
                )
            return dict(_EGODEX_ACTION_COMPONENT_SOURCES)

        if self._uses_packed_eef_16d_contract():
            # NOE 原始 state/action 是 23D；训练只接前 16D EEF。
            preset = _PACKED_EEF_16D_COMPONENT_SOURCES.get(modality, {})
            sources = {
                key: preset[key]
                for key in transform_keys
                if key in preset and preset[key][0] in self._features
            }
            if len(sources) != len(transform_keys):
                missing = [key for key in transform_keys if key not in sources]
                raise KeyError(
                    f"Could not map packed EEF {modality} keys {missing} under "
                    f"{self.data_path}; available={sorted(self._features)}"
                )
            return dict(sources)

        sources: dict[str, tuple[str, slice]] = {}
        missing_keys: list[str] = []
        for key in transform_keys:
            entry = self._modality_entry(modality, key)
            if entry is None:
                missing_keys.append(key)
                continue
            source = str(
                entry.get("original_key") or self._default_vector_source_key(modality)
            )
            start = int(entry.get("start", 0))
            end = entry.get("end")
            sources[key] = (source, slice(start, None if end is None else int(end)))
        if not missing_keys:
            return sources

        missing_short_keys = [self._short_modality_key(k) for k in missing_keys]
        if modality == "state":
            feature = (
                self._features.get("observation.state")
                or self._features.get("state")
                or {}
            )
        else:
            feature = (
                self._features.get("action") or self._features.get("actions") or {}
            )
        dim = int((feature.get("shape") or [0])[0] or 0)
        inferred = infer_named_component_slices(
            feature.get("names"), dim, missing_short_keys
        )
        if inferred is None and len(missing_short_keys) == 1:
            inferred = {missing_short_keys[0]: slice(0, dim or None)}
        if inferred is None:
            # Backward-compatible DROID fallback when meta names are missing.
            if set(missing_short_keys).issubset({"joint_position", "gripper_position"}):
                st_j, st_g, ac_j, ac_g = droid_default_state_action_slices()
                inferred = (
                    {"joint_position": st_j, "gripper_position": st_g}
                    if modality == "state"
                    else {"joint_position": ac_j, "gripper_position": ac_g}
                )
            else:
                raise ValueError(
                    f"Cannot infer {modality} component slices for keys={transform_keys} "
                    f"from feature names={feature.get('names')!r} dim={dim}"
                )
        source = self._default_vector_source_key(modality)
        for key in missing_keys:
            sources[key] = (source, inferred[self._short_modality_key(key)])
        return sources

    def _build_language_sources(self) -> dict[str, str]:
        sources: dict[str, str] = {}
        annotations = self._modality_meta.get("annotation")
        for key in self.language_keys:
            subkey = self._short_modality_key(key)
            entry = annotations.get(subkey) if isinstance(annotations, dict) else None
            source = entry.get("original_key") if entry else None
            sources[key] = str(source or f"annotation.{subkey}")
        return sources

    def _init_lazy_map_style(
        self, pq_cache_max_episodes: int, video_tolerance_s: float
    ) -> None:
        if not self._root.exists():
            raise FileNotFoundError(
                f"DreamZero data_path must be local: {self.data_path}"
            )
        self._chunks_size = int(self._info.get("chunks_size") or 1000)
        self._data_tmpl = str(self._info.get("data_path") or "")
        self._video_tmpl = str(self._info.get("video_path") or "")
        if not self._data_tmpl:
            raise ValueError("meta/info.json missing data_path")

        episode_records: list[dict[str, Any]] = []
        episodes_path = self._meta_dir / "episodes.jsonl"
        with open(episodes_path, encoding="utf-8") as epf:
            for line in epf:
                if line.strip():
                    episode_records.append(json.loads(line))

        if self._episode_exclusion_manifest is not None:
            with self._episode_exclusion_manifest.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    episode_index = record["episode_index"]
                    if type(episode_index) is not int or episode_index < 0:
                        raise ValueError(
                            "Exclusion episode_index must be a non-negative integer."
                        )
                    self._bad_video_episodes.add(episode_index)
            unknown = self._bad_video_episodes - {
                int(record["episode_index"]) for record in episode_records
            }
            if unknown:
                raise ValueError(
                    f"Unknown excluded episode indices: {sorted(unknown)[:5]}."
                )
            logger.info(
                "Loaded %s episode exclusions from %s for %s",
                len(self._bad_video_episodes),
                self._episode_exclusion_manifest,
                self._root,
            )

        egodex_allowed_indices: set[int] | None = None
        if self._is_egodex():
            egodex_allowed_indices = filter_egodex_episode_indices(
                self._meta_dir,
                (
                    int(record.get("episode_index", 0))
                    for record in episode_records
                ),
            )
            logger.info(
                "EgoDex confidence filter: kept=%s skipped=%s path=%s",
                len(egodex_allowed_indices),
                len(episode_records) - len(egodex_allowed_indices),
                self.data_path,
            )
            if not egodex_allowed_indices:
                raise ValueError(
                    "EgoDex confidence filter kept no episodes under "
                    f"{episodes_path}."
                )

        meta_episode_indices: set[int] = set()
        episode_lengths: dict[int, int] = {}
        self._noe_umi_episode_quality = {
            int(obj["episode_index"]): _noe_umi_episode_quality(obj)
            for obj in episode_records
            if self._is_noe_umi_16d()
        }
        qualification_filtered_count = 0
        qualification_invalid_count = 0
        qualification_invalid_examples: list[dict[str, Any]] = []
        for obj in episode_records:
            ep_idx = int(obj["episode_index"])
            if (
                self.filter_noe_umi_qualified_episodes
                and self.noe_umi_quality_mode == _NOE_UMI_QUALITY_MODE_EPISODE
            ):
                qualified, metadata_error = _check_noe_umi_episode_qualification(obj)
                if metadata_error is not None:
                    qualification_invalid_count += 1
                    if len(qualification_invalid_examples) < 5:
                        qualification_invalid_examples.append(
                            {
                                "episode_index": ep_idx,
                                "error": metadata_error,
                            }
                        )
                    continue
                if not qualified:
                    qualification_filtered_count += 1
                    continue
            if (
                egodex_allowed_indices is not None
                and ep_idx not in egodex_allowed_indices
            ):
                continue
            meta_episode_indices.add(ep_idx)
            for k in ("episode_length", "length", "num_frames", "num_steps"):
                if obj.get(k) is not None:
                    episode_lengths[ep_idx] = int(obj[k])
                    break

        if (
            self.filter_noe_umi_qualified_episodes
            and self.noe_umi_quality_mode == _NOE_UMI_QUALITY_MODE_EPISODE
        ):
            if qualification_invalid_count:
                logger.warning(
                    "Skipped %s NOE/UMI episodes with invalid qualification metadata "
                    "under %s; examples=%s",
                    qualification_invalid_count,
                    self.data_path,
                    qualification_invalid_examples,
                )
            logger.info(
                "NOE/UMI qualification filter: kept=%s filtered=%s invalid=%s path=%s",
                len(meta_episode_indices),
                qualification_filtered_count,
                qualification_invalid_count,
                self.data_path,
            )
            if not meta_episode_indices:
                raise ValueError(
                    "NOE/UMI qualification filter kept no episodes under "
                    f"{episodes_path}."
                )

        self._episodes = discover_local_lerobot_episode_indices(
            self._root,
            self._info,
            allowed_episode_indices=meta_episode_indices,
            discovery_mode=self.episode_discovery_mode,
        )
        if self._bad_video_episodes:
            before = len(self._episodes)
            self._episodes = [
                ep for ep in self._episodes if ep not in self._bad_video_episodes
            ]
            logger.warning(
                "Filtered %s excluded episodes from %s using %s",
                before - len(self._episodes),
                self.data_path,
                self._episode_exclusion_manifest,
            )
        if not self._episodes:
            raise FileNotFoundError(
                f"No usable episodes remain after filtering under {self._root}"
            )
        self._episode_lengths = []
        for ep in self._episodes:
            episode_length = episode_lengths.get(ep)
            if episode_length is None:
                episode_length = self._infer_episode_length_from_parquet(ep)
            self._episode_lengths.append(int(episode_length))
        self._episode_starts = [0]
        total = 0
        for n in self._episode_lengths:
            total += int(n)
            self._episode_starts.append(total)
        self._total_frames = int(total)
        self._pq_cache: "OrderedDict[int, Any]" = OrderedDict()
        self._pq_cache_max_episodes = int(pq_cache_max_episodes)
        if self._pq_cache_max_episodes < 0:
            raise ValueError(
                "pq_cache_max_episodes must be non-negative; "
                f"got {pq_cache_max_episodes!r}."
            )
        self._video_decode_fps_cache: "OrderedDict[str, float]" = OrderedDict()
        self._video_decode_fps_cache_max = 512
        self._video_tolerance_s = float(video_tolerance_s)
        if self._video_tolerance_s <= 0:
            raise ValueError(
                f"video_tolerance_s must be positive, got {video_tolerance_s!r}"
            )

    def _init_task_type_frame_index(self) -> None:
        """根据实际存储布局建立 task-type 的 episode/frame 索引。

        task-type sampling 只接入 lazy local MP4；使用初始化时已经过滤的
        episode 列表及其累计 frame 边界。v2 image-parquet 仍由基础 dataset
        读取逻辑支持，但不属于 task-type sampling 的支持范围。
        """
        if self._use_lazy_video_tree:
            episodes = self._episodes
            episode_starts = self._episode_starts
        else:
            raise ValueError(
                "data.task_type_sampling.enabled requires a lazy local MP4 "
                "dataset; v2 image-parquet is not supported by task-type sampling."
            )
        self.task_type_frame_index = build_task_type_frame_index(
            self._root,
            episodes,
            episode_starts,
            self.task_type_sampling_config,
        )

    def _init_lerobot_or_v2_parquet(self) -> None:
        if self._use_image_parquet_tree:
            self._init_v2_image_parquet()
            return
        import lerobot.datasets.lerobot_dataset as lerobot_dataset

        if self.sampling_mode == "multi_anchor":
            delta_timestamps = {
                self._source_video_key[k]: [0.0] for k in self.video_keys
            }
            state_sources = {source for source, _ in self._state_components.values()}
            action_sources = {source for source, _ in self._action_components.values()}
            for source in state_sources | action_sources:
                delta_timestamps[source] = [0.0]
        else:
            fixed_window = self._fixed_window_temporal
            assert fixed_window is not None
            delta_timestamps = {
                self._source_video_key[k]: [t / self._fps for t in fixed_window.video]
                for k in self.video_keys
            }
            state_sources = {source for source, _ in self._state_components.values()}
            action_sources = {source for source, _ in self._action_components.values()}
            for source in state_sources:
                delta_timestamps[source] = [t / self._fps for t in fixed_window.state]
            for source in action_sources:
                delta_timestamps[source] = [t / self._fps for t in fixed_window.action]
        self.dataset = lerobot_dataset.LeRobotDataset(
            self.data_path,
            delta_timestamps=delta_timestamps,
            video_backend=self._video_backend,
        )
        self._use_v2_image_parquet = False

    def _init_v2_image_parquet(self) -> None:
        import pyarrow.parquet as pq

        data_root = self._root / "data"
        episodes_path = self._meta_dir / "episodes.jsonl"
        self._episodes_meta = []
        with open(episodes_path) as f:
            for line in f:
                if line.strip():
                    self._episodes_meta.append(json.loads(line))
        self._ep_frames = []
        self._ep_parquet_paths = []
        for ep in self._episodes_meta:
            ep_idx = int(ep["episode_index"])
            pq_path = (
                data_root
                / f"chunk-{ep_idx // 1000:03d}"
                / f"episode_{ep_idx:06d}.parquet"
            )
            table = pq.read_table(pq_path)
            self._ep_frames.append(len(table))
            self._ep_parquet_paths.append(pq_path)
        self._cumulative = np.cumsum(self._ep_frames)
        self._total_frames = int(self._cumulative[-1])
        self._pq_cache = {}
        self._use_v2_image_parquet = True

    def _read_v2_episode(self, ep_pos: int):
        if ep_pos not in self._pq_cache:
            import pyarrow.parquet as pq

            self._pq_cache[ep_pos] = pq.read_table(str(self._ep_parquet_paths[ep_pos]))
            if len(self._pq_cache) > 50:
                del self._pq_cache[next(iter(self._pq_cache))]
        return self._pq_cache[ep_pos]

    def _decode_v2_image(self, cell) -> np.ndarray:
        from io import BytesIO

        from PIL import Image

        raw = cell.as_py()
        if isinstance(raw, dict):
            raw = raw.get("bytes", raw)
        if isinstance(raw, bytes):
            return np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
        return np.asarray(raw)

    def _get_v2_image_sample(self, idx: int) -> dict[str, Any]:
        # NOE/UMI subtask 在该 v2 image-parquet 分支中目前弃用，仅在 lazy MP4 路径接入。
        # EgoDex 默认不使用该 image-Parquet 分支，而是通过 lazy MP4 路径读取视频。
        frame_in_ep, episode_index, ep_len = self._resolve_index_context(idx)
        ep_pos = int(np.searchsorted(self._cumulative, idx, side="right"))
        table = self._read_v2_episode(ep_pos)
        video_offsets, state_offsets, action_offsets = self._temporal_offsets_for_frame(
            frame_in_ep, episode_index, ep_len, episode_table=table
        )

        def clamp(offset: int) -> int:
            return min(max(frame_in_ep + int(offset), 0), ep_len - 1)

        sample: dict[str, Any] = {
            "episode_index": episode_index,
            "frame_index": frame_in_ep,
        }
        for transform_key, source_key in self._source_video_key.items():
            sample[transform_key] = np.stack(
                [
                    self._decode_v2_image(table.column(source_key)[clamp(o)])
                    for o in video_offsets
                ],
                axis=0,
            )
        state_rows = [clamp(o) for o in state_offsets]
        action_rows = [clamp(o) for o in action_offsets]
        state_sources = {source for source, _ in self._state_components.values()}
        action_sources = {source for source, _ in self._action_components.values()}
        for source in state_sources:
            if source not in table.column_names:
                continue
            sample[source] = np.asarray(
                [table.column(source)[r].as_py() for r in state_rows], dtype=np.float32
            )
        for source in action_sources:
            if source not in table.column_names:
                continue
            sample[source] = np.asarray(
                [table.column(source)[r].as_py() for r in action_rows], dtype=np.float32
            )
        for key, source in self._language_sources.items():
            if source in table.column_names:
                sample[key] = table.column(source)[frame_in_ep].as_py()
        if "task" in table.column_names:
            sample["task"] = table.column("task")[frame_in_ep].as_py()
        elif "task_index" in table.column_names:
            sample["task_index"] = table.column("task_index")[frame_in_ep].as_py()
        return sample

    def _infer_episode_length_from_parquet(self, episode_index: int) -> int:
        import pyarrow.parquet as pq

        return int(
            pq.read_metadata(str(self._get_parquet_path(episode_index))).num_rows
        )

    def _get_parquet_path(self, episode_index: int) -> Path:
        ep_chunk = int(episode_index) // self._chunks_size
        rel = Path(
            self._data_tmpl.format(
                episode_chunk=ep_chunk, episode_index=int(episode_index)
            )
        )
        p = (self._root / rel).resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"Parquet file not found for episode {episode_index}: {p}"
            )
        return p

    def _get_video_path(self, episode_index: int, video_key: str) -> Path:
        ep_chunk = int(episode_index) // self._chunks_size
        rel = Path(
            self._video_tmpl.format(
                episode_chunk=ep_chunk,
                video_key=video_key,
                episode_index=int(episode_index),
            )
        )
        p = (self._root / rel).resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"Video file not found for episode {episode_index} key {video_key}: {p}"
            )
        return p

    def _decode_fps_for_video_file(self, video_path: Path) -> float:
        key = str(video_path.resolve())
        if key in self._video_decode_fps_cache:
            fps = self._video_decode_fps_cache.pop(key)
            self._video_decode_fps_cache[key] = fps
            return fps
        fps = float(probe_video_container_fps(video_path) or self._fps)
        self._video_decode_fps_cache[key] = fps
        if len(self._video_decode_fps_cache) > self._video_decode_fps_cache_max:
            self._video_decode_fps_cache.popitem(last=False)
        return fps

    def _get_episode_table(self, episode_index: int):
        episode_index = int(episode_index)
        if episode_index in self._pq_cache:
            tbl = self._pq_cache.pop(episode_index)
            self._pq_cache[episode_index] = tbl
            return tbl
        import pyarrow.parquet as pq

        p = self._get_parquet_path(episode_index)
        schema = set(pq.read_schema(str(p)).names)
        cols = [
            "observation",
            *self._vector_source_keys,
            "task",
            "task_index",
            *self._language_sources.values(),
        ]
        if self._uses_noe_umi_frame_quality():
            missing_quality = set(_NOE_UMI_FRAME_QUALITY_FIELDS) - schema
            if missing_quality:
                self._warn_noe_umi_quality_fallback(episode_index, missing_quality)
            else:
                cols.extend(_NOE_UMI_FRAME_QUALITY_FIELDS)
        cols = [c for c in cols if c in schema]
        tbl = pq.read_table(str(p), columns=list(dict.fromkeys(cols)))
        if self._pq_cache_max_episodes > 0:
            self._pq_cache[episode_index] = tbl
            if len(self._pq_cache) > self._pq_cache_max_episodes:
                self._pq_cache.popitem(last=False)
        return tbl

    @staticmethod
    def _clip_indices(indices: np.ndarray, length: int) -> np.ndarray:
        return np.clip(indices.astype(np.int64), 0, max(0, int(length) - 1))

    @staticmethod
    def _video_to_thwc_uint8(frames: Any) -> np.ndarray:
        """Match Groot VideoTransform: numpy (T, H, W, C) uint8.

        LeRobot ``decode_video_frames`` returns float32 (T, C, H, W) in [0, 1].
        Parquet/PIL paths usually already yield uint8 (T, H, W, C).
        """
        if torch.is_tensor(frames):
            arr = frames.detach().cpu().numpy()
        else:
            arr = np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None, ...]
        elif arr.ndim == 5:
            # (B, T, C, H, W) -> (B, T, H, W, C)
            if arr.shape[2] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (0, 1, 3, 4, 2))
        elif arr.ndim == 4:
            if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32, copy=False)
            max_v = float(arr_f.max()) if arr_f.size else 0.0
            if max_v <= 1.0 + 1e-3:
                arr = np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr_f, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _col_exists(table, name: str) -> bool:
        return hasattr(table, "column_names") and name in table.column_names

    @staticmethod
    def _read_list_column(table, name: str, indices: np.ndarray) -> np.ndarray:
        col = table.column(name)
        return np.asarray(
            [col[int(i)].as_py() for i in indices.tolist()], dtype=np.float32
        )

    def _noe_umi_quality_masks(
        self,
        episode_index: int,
        table: Any,
        state_idx: np.ndarray,
        action_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Build simple state/action validity masks for the configured UMI frame mode."""
        if not self._uses_noe_umi_frame_quality():
            return None

        columns = set(getattr(table, "column_names", ()))
        missing_quality = set(_NOE_UMI_FRAME_QUALITY_FIELDS) - columns
        if missing_quality:
            self._warn_noe_umi_quality_fallback(episode_index, missing_quality)
        episode_quality = self._noe_umi_episode_quality.get(
            int(episode_index), (False, False)
        )
        next_action_idx = np.clip(action_idx + 1, 0, max(0, table.num_rows - 1))
        state_valid = read_noe_umi_frame_quality(table, state_idx, episode_quality)
        action_valid = read_noe_umi_frame_quality(
            table, next_action_idx, episode_quality
        )
        return state_valid, action_valid

    @staticmethod
    def _read_struct_list_field(
        table, struct_col: str, field: str, indices: np.ndarray
    ) -> np.ndarray:
        col = table.column(struct_col)
        arr = col.chunk(0) if hasattr(col, "num_chunks") and col.num_chunks > 0 else col
        field_arr = arr.field(field)
        return np.asarray(
            [field_arr[int(i)].as_py() for i in indices.tolist()], dtype=np.float32
        )

    def _resolve_index_context(self, idx: int) -> tuple[int, int, int]:
        """Return ``(frame_in_ep, episode_index, episode_length)`` for a global index."""
        if self._use_lazy_video_tree:
            if idx < 0 or idx >= self._total_frames:
                raise IndexError(
                    f"Index {idx} out of range for dataset of len {self._total_frames}"
                )
            ep_pos = bisect.bisect_right(self._episode_starts, int(idx)) - 1
            ep_pos = max(0, min(ep_pos, len(self._episodes) - 1))
            frame_in_ep = int(idx) - int(self._episode_starts[ep_pos])
            episode_index = int(self._episodes[ep_pos])
            ep_len = int(self._episode_lengths[ep_pos])
            return frame_in_ep, episode_index, ep_len
        if getattr(self, "_use_v2_image_parquet", False):
            if idx < 0 or idx >= self._total_frames:
                raise IndexError(
                    f"Index {idx} out of range for dataset of len {self._total_frames}"
                )
            ep_pos = int(np.searchsorted(self._cumulative, idx, side="right"))
            frame_in_ep = (
                int(idx) if ep_pos == 0 else int(idx - self._cumulative[ep_pos - 1])
            )
            episode_index = int(
                self._episodes_meta[ep_pos].get("episode_index", ep_pos)
            )
            ep_len = int(self._ep_frames[ep_pos])
            return frame_in_ep, episode_index, ep_len
        raise RuntimeError(
            "_resolve_index_context requires lazy video tree or v2 image parquet in multi_anchor mode"
        )

    def _language_column_for_episode_table(self, table) -> str | None:
        for source in self._language_sources.values():
            if self._col_exists(table, source):
                return source
        for key in ("task_index", "task"):
            if self._col_exists(table, key):
                return key
        return None

    def _read_episode_language_labels(
        self, episode_index: int, episode_table: Any | None = None
    ) -> np.ndarray:
        if episode_table is None:
            table = self._get_episode_table(episode_index)
        else:
            table = episode_table
        if self.episode_instruction_key:
            return np.full(table.num_rows, int(episode_index), dtype=np.int64)
        col = self._language_column_for_episode_table(table)
        if col is None:
            return np.zeros(table.num_rows, dtype=object)
        if col == "task_index":
            return np.asarray(
                [table.column(col)[i].as_py() for i in range(table.num_rows)],
                dtype=np.int64,
            )
        return np.asarray(
            [table.column(col)[i].as_py() for i in range(table.num_rows)],
            dtype=object,
        )

    def _temporal_offsets_for_frame(
        self,
        frame_in_ep: int,
        episode_index: int,
        ep_len: int,
        episode_table: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.sampling_mode == "fixed_window":
            assert self._fixed_window_temporal is not None
            t = self._fixed_window_temporal
            return t.video, t.state, t.action
        assert self._multi_anchor_cfg is not None
        language = self._read_episode_language_labels(
            episode_index, episode_table=episode_table
        )
        temporal = require_multi_anchor_temporal_indices(
            frame_in_ep,
            language,
            ep_len,
            self._multi_anchor_cfg,
            episode_index=episode_index,
        )
        return temporal.video, temporal.state, temporal.action

    def _resolve_noe_subtask(
        self,
        *,
        episode_index: int,
        frame_index: int,
        action_indices: np.ndarray,
    ) -> SubtaskResolution | None:
        # 将 sampling 后的 action indices 拆成 horizon chunks，再查询共同 subtask。
        if self._noe_subtask_index is None:
            return None
        indices = np.asarray(action_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0 or indices.size % self.action_horizon != 0:
            raise ValueError(
                "Sampled action indices must contain complete action chunks: "
                f"size={indices.size}, action_horizon={self.action_horizon}."
            )

        action_chunks: list[tuple[int, int]] = []
        for chunk in indices.reshape(-1, self.action_horizon):
            if np.any(np.diff(chunk) < 0):
                raise ValueError(
                    "Sampled action indices must be non-decreasing within each "
                    f"chunk, got {chunk.tolist()}."
                )
            action_chunks.append((int(chunk[0]), int(chunk[-1]) + 1))

        return self._noe_subtask_index.resolve(
            episode_index=episode_index,
            frame_index=frame_index,
            action_chunks=action_chunks,
        )

    def _record_noe_subtask_resolution(
        self,
        resolution: SubtaskResolution,
    ) -> None:
        # 在当前 worker 内累计解析状态并按固定间隔输出命中率。
        counts = self._noe_subtask_resolution_counts
        counts["total"] += 1
        counts[resolution.status] += 1
        total = counts["total"]
        if total != _NOE_SUBTASK_STATS_WARMUP and (
            total % _NOE_SUBTASK_STATS_INTERVAL != 0
        ):
            return

        selected = counts["matched"] + counts["ambiguous_random"]
        fallback = counts["no_match"] + counts["mixed"]
        logger.info(
            "NOE subtask resolution stats: path=%s pid=%s total=%s "
            "matched=%s ambiguous_random=%s no_match=%s mixed=%s "
            "subtask_rate=%.2f%% fallback_rate=%.2f%%",
            self.data_path,
            os.getpid(),
            total,
            counts["matched"],
            counts["ambiguous_random"],
            counts["no_match"],
            counts["mixed"],
            100.0 * selected / total,
            100.0 * fallback / total,
        )
        if total == _NOE_SUBTASK_STATS_WARMUP and selected == 0:
            logger.warning(
                "NOE subtask prefer mode produced zero matches after %s samples: "
                "path=%s pid=%s",
                total,
                self.data_path,
                os.getpid(),
            )

    def _attach_noe_subtask_instruction(
        self,
        sample: dict[str, Any],
        *,
        episode_index: int,
        frame_index: int,
        action_indices: np.ndarray,
    ) -> None:
        # 仅在成功解析共同文本时写入 sample，未命中时不添加任何内容。
        resolution = self._resolve_noe_subtask(
            episode_index=episode_index,
            frame_index=frame_index,
            action_indices=action_indices,
        )
        if resolution is None:
            return
        self._record_noe_subtask_resolution(resolution)
        if resolution.text is not None:
            sample[_NOE_SUBTASK_INSTRUCTION_KEY] = resolution.text

    def _attach_dense_subtask_instruction(
        self,
        sample: dict[str, Any],
        *,
        episode_index: int,
        video_indices: np.ndarray,
    ) -> None:
        if self._dense_subtask_index is None:
            return
        text = self._dense_subtask_index.resolve(episode_index, video_indices)
        if text is not None:
            sample[_DENSE_SUBTASK_INSTRUCTION_KEY] = text

    def _materialize_parquet_sample(
        self,
        frame_in_ep: int,
        episode_index: int,
        ep_len: int,
        table: Any,
        *,
        decode_video: bool,
    ) -> dict[str, Any]:
        video_offsets, state_offsets, action_offsets = self._temporal_offsets_for_frame(
            frame_in_ep, episode_index, ep_len, episode_table=table
        )
        video_idx = self._clip_indices(frame_in_ep + video_offsets, ep_len)
        state_idx = self._clip_indices(frame_in_ep + state_offsets, ep_len)
        action_idx = self._clip_indices(frame_in_ep + action_offsets, ep_len)

        sample: dict[str, Any] = {
            "episode_index": episode_index,
            "frame_index": frame_in_ep,
        }
        if self._ego_instruction_index is not None:
            sample[_EGO_INSTRUCTION_WINDOW_KEY] = (
                int(action_idx.min()),
                int(action_idx.max()) + 1,
            )
        self._attach_dense_subtask_instruction(
            sample,
            episode_index=episode_index,
            video_indices=video_idx,
        )
        self._attach_noe_subtask_instruction(
            sample,
            episode_index=episode_index,
            frame_index=frame_in_ep,
            action_indices=action_idx,
        )
        if decode_video:
            from lerobot.datasets.video_utils import decode_video_frames

            for transform_key, source_key in self._source_video_key.items():
                video_path = self._get_video_path(episode_index, source_key)
                fps = self._decode_fps_for_video_file(video_path)
                sample[transform_key] = decode_video_frames(
                    video_path,
                    [float(int(i)) / fps for i in video_idx.tolist()],
                    tolerance_s=self._video_tolerance_s,
                    backend=self._video_backend,
                )
                if self.return_episode_first_images:
                    sample[f"{_EPISODE_FIRST_VIDEO_PREFIX}{transform_key}"] = (
                        decode_video_frames(
                            video_path,
                            [0.0],
                            tolerance_s=self._video_tolerance_s,
                            backend=self._video_backend,
                        )
                    )

        for key in ("task", "task_index"):
            if self._col_exists(table, key):
                sample[key] = table.column(key)[int(frame_in_ep)].as_py()
        for key, source in self._language_sources.items():
            if self._col_exists(table, source):
                sample[key] = table.column(source)[int(frame_in_ep)].as_py()

        for source, _ in self._state_components.values():
            if source in sample:
                continue
            if self._col_exists(table, source):
                sample[source] = self._read_list_column(table, source, state_idx)
            elif source == "observation.state" and self._col_exists(
                table, "observation"
            ):
                sample[source] = self._read_struct_list_field(
                    table, "observation", "state", state_idx
                )
            else:
                raise KeyError(
                    f"episode parquet missing state source column {source!r}"
                )
        for source, _ in self._action_components.values():
            if source in sample:
                continue
            if not self._col_exists(table, source):
                raise KeyError(
                    f"episode parquet missing action source column {source!r}"
                )
            sample[source] = self._read_list_column(table, source, action_idx)
        quality_masks = self._noe_umi_quality_masks(
            episode_index,
            table,
            state_idx,
            action_idx,
        )
        if self.relative_action_mode == RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ:
            reference, reference_valid = read_noe_umi_episode_reference(
                table, self._noe_umi_episode_quality.get(int(episode_index), (False, False))
            )
            sample[_NOE_UMI_EPISODE_REFERENCE_KEY] = reference
            if quality_masks is not None:
                quality_masks = tuple(mask & reference_valid for mask in quality_masks)
        if quality_masks is not None:
            sample[_NOE_UMI_STATE_QUALITY_KEY] = quality_masks[0]
            sample[_NOE_UMI_ACTION_QUALITY_KEY] = quality_masks[1]
        # 读取与各 action 时间步对齐的 state，用于将 EgoDex delta action 重建为 absolute action。
        if self._is_egodex():
            if not self._col_exists(table, "observation.state"):
                raise KeyError(
                    "EgoDex episode parquet missing 'observation.state' for "
                    "absolute action construction."
                )
            sample[_EGODEX_ACTION_STATE_KEY] = self._read_list_column(
                table, "observation.state", action_idx
            )
            next_action_idx = action_idx + 1
            if np.any(next_action_idx >= table.num_rows):
                raise IndexError(
                    "EgoDex absolute action construction requires action_idx + 1 "
                    "to remain "
                    f"inside episode {episode_index}, got action_idx={action_idx.tolist()} "
                    f"for {table.num_rows} rows."
                )
        return sample

    def _put_egodex_absolute_actions(
        self,
        out: dict[str, Any],
        sample: dict[str, Any],
    ) -> None:
        """输入同时间步 6D state/delta，输出左右手 3D absolute action。"""
        state = np.asarray(sample[_EGODEX_ACTION_STATE_KEY], dtype=np.float32)
        delta = np.asarray(sample["action"], dtype=np.float32)
        if state.shape != delta.shape or state.ndim != 2 or state.shape[1] != 6:
            raise ValueError(
                "EgoDex absolute action requires state/delta [T,6], got "
                f"state={state.shape}, delta={delta.shape}."
            )
        action = state + delta
        out["action.left_hand_position"] = action[:, :3]
        out["action.right_hand_position"] = action[:, 3:6]

    def _get_lazy_sample(self, idx: int) -> dict[str, Any]:
        frame_in_ep, episode_index, ep_len = self._resolve_index_context(idx)
        table = self._get_episode_table(episode_index)
        return self._materialize_parquet_sample(
            frame_in_ep,
            episode_index,
            ep_len,
            table,
            decode_video=True,
        )

    def __len__(self) -> int:
        if self._use_lazy_video_tree or getattr(self, "_use_v2_image_parquet", False):
            return int(self._total_frames)
        return len(self.dataset)

    def _resolve_task_text(self, sample: dict[str, Any]) -> str:
        task_text = sample.get("task")
        if task_text is None:
            task_text = safe_lang_text(sample.get("task_index", 0), self._tasks)
        candidates = []
        for key in self.language_keys:
            if key in sample:
                text = safe_lang_text(sample[key], self._tasks)
                if text:
                    candidates.append(text)
        if candidates:
            return str(np.random.choice(candidates))
        return str(task_text or "")

    def _resolve_default_instruction_text(self, sample: dict[str, Any]) -> str:
        if self.episode_instruction_key:
            return self._resolve_episode_instruction_text(sample)
        return self._resolve_task_text(sample)

    def _resolve_ego_instruction(self, sample: dict[str, Any]) -> str:
        start_frame, end_frame = sample[_EGO_INSTRUCTION_WINDOW_KEY]
        episode_index = int(sample["episode_index"])

        text = self._ego_instruction_index.get_ego_block_prompt(
            episode_index,
            int(start_frame),
            int(end_frame),
            episode_instruction=self._resolve_default_instruction_text(sample),
        )
        if self.use_scene_instruction:
            scene = self._ego_instruction_index.get_ego_scenario(episode_index).get(
                "scenario_caption", ""
            )
            text = append_scene_instruction(text, str(scene or ""))
        if random.random() < 0.001:
            logger.info(
                "[EGO_INSTRUCTION_PROMPT] dataset=%s episode=%s window=[%s,%s) "
                "subtask=%s dense=%s dense_object=%s scene=%s prompt=%r",
                self.data_path,
                episode_index,
                start_frame,
                end_frame,
                self.use_subtask_instruction,
                self.use_dense_caption_instruction,
                self.dense_caption_include_object,
                self.use_scene_instruction,
                text,
            )
        return text

    def _state_key_for_relative_action(self, action_key: str) -> str | None:
        """Pair ``action.<sub>`` with ``state.<sub>`` (groot convention for relative_action)."""
        short = self._short_modality_key(action_key)
        for sk in self.state_keys:
            if self._short_modality_key(sk) == short:
                return sk
        return None

    def _is_relative_action_key(self, action_key: str) -> bool:
        if not self.relative_action:
            return False
        if not self.relative_action_keys:
            raise ValueError(
                "relative_action=True requires explicit relative_action_keys; "
                "empty keys would make relative action selection ambiguous."
            )
        short = self._short_modality_key(action_key)
        if short not in self.relative_action_keys:
            return False
        return self._state_key_for_relative_action(action_key) is not None

    def _prepare_dual_arm_eef_relative_state(
        self, out: dict[str, Any], episode_reference: np.ndarray | None = None
    ) -> np.ndarray | None:
        # 选定 absolute 参考，供 state 和后续 action 转换共用。
        if ROBOTWIN_EEF_STATE_KEY not in out:
            raise ValueError(
                "Dual-arm EEF relative action modes require "
                f"{ROBOTWIN_EEF_STATE_KEY!r} before converting state/action."
            )
        state_arr = np.asarray(out[ROBOTWIN_EEF_STATE_KEY], dtype=np.float32)
        if state_arr.ndim != 2:
            raise ValueError(
                "DreamZeroLeRobotDataset expects 2D state.eef_position before "
                f"relative conversion, got shape={state_arr.shape}."
            )
        ensure_robotwin_eef_dim(ROBOTWIN_EEF_STATE_KEY, state_arr)
        if state_arr.shape[0] == 0:
            raise ValueError(
                "Dual-arm EEF relative action modes require at least one "
                "state.eef_position row to define an anchor."
            )
        absolute_state = state_arr.copy()
        reference_state = absolute_state
        if self.relative_action_mode == RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_EPISODE_WXYZ:
            if episode_reference is None:
                raise ValueError("UMI episode-relative sample is missing its episode row-zero reference")
            reference_state = np.asarray(episode_reference, dtype=np.float32)
            if reference_state.shape != (1, 16):
                raise ValueError(f"Expected episode reference [1,16], got {reference_state.shape}")
        if is_dual_arm_eef_se3_relative_state(self.relative_action_mode):
            state_anchor = (
                reference_state
                if is_robotwin_eef_se3_chunk_relative(self.relative_action_mode)
                else reference_state[0:1]
            )
            out[ROBOTWIN_EEF_STATE_KEY] = robotwin_eef_abs_to_rel_wxyz(
                absolute_state,
                state_anchor,
                self.relative_action_pose_reference,
            )
        return reference_state

    def _convert_action_abs_to_relative(
        self,
        value: np.ndarray,
        action_key: str,
        absolute_state: np.ndarray | None,
    ) -> np.ndarray:
        sk = self._state_key_for_relative_action(action_key)
        if action_key != ROBOTWIN_EEF_ACTION_KEY or sk != ROBOTWIN_EEF_STATE_KEY:
            raise ValueError(
                "Dual-arm EEF relative action modes only support "
                "action.eef_position paired with state.eef_position."
            )
        if absolute_state is None:
            raise ValueError(
                "Dual-arm EEF relative action modes require "
                f"absolute {sk!r} to be available before converting {action_key!r}. "
                "DreamZeroLeRobotDataset should put state components before action components."
            )
        assert value.ndim == 2, (
            "DreamZeroLeRobotDataset expects 2D action components before relative "
            f"conversion, got {action_key} shape={value.shape}."
        )
        ensure_robotwin_eef_dim(action_key, value)
        ensure_robotwin_eef_dim(sk, absolute_state)
        t_act = value.shape[0]
        if t_act == 0:
            return value.astype(np.float32, copy=False)
        if np.random.random() < 0.001:
            print(
                "[Dual-arm EEF SE3 relative action] "
                f"mode={self.relative_action_mode}, key={action_key}, "
                f"action_shape={value.shape}, "
                f"state_key={sk}, embodiment={self.embodiment_tag}, "
                f"sampling_mode={self.sampling_mode}",
                flush=True,
            )
        if not is_robotwin_eef_se3_chunk_relative(self.relative_action_mode):
            return robotwin_eef_abs_to_rel_wxyz(
                value,
                absolute_state[0:1],
                self.relative_action_pose_reference,
            )

        num_chunks = absolute_state.shape[0]
        expected_actions = num_chunks * self.action_horizon
        if t_act != expected_actions:
            raise ValueError(
                "RoboTwin chunk-relative mode requires complete action chunks: "
                f"action rows={t_act}, state anchors={num_chunks}, "
                f"action_horizon={self.action_horizon}, "
                f"expected action rows={expected_actions}."
            )
        action_chunks = value.reshape(num_chunks, self.action_horizon, -1)
        anchors = absolute_state[:, None, :]
        return robotwin_eef_abs_to_rel_wxyz(
            action_chunks,
            anchors,
            self.relative_action_pose_reference,
        ).reshape(t_act, -1)

    def _put_components(
        self,
        out: dict[str, Any],
        sample: dict[str, Any],
        components: dict[str, tuple[str, slice]],
        *,
        is_action: bool,
        robotwin_eef_absolute_state: np.ndarray | None = None,
    ) -> None:
        for key, (source, sl) in components.items():
            raw = np.asarray(sample[source], dtype=np.float32)
            if raw.ndim == 1:
                raw = raw[None, :]
            value = raw[:, sl].astype(np.float32)
            if is_action and self._is_relative_action_key(key):
                value = self._convert_action_abs_to_relative(
                    value,
                    key,
                    robotwin_eef_absolute_state,
                )
            out[key] = value

    def _load_raw_sample(self, idx: int) -> dict[str, Any]:
        if self._use_lazy_video_tree:
            return self._get_lazy_sample(idx)
        if getattr(self, "_use_v2_image_parquet", False):
            return self._get_v2_image_sample(idx)
        return self.dataset[idx]

    def _should_log_profile(self, total_s: float) -> bool:
        if not self._profile_enabled:
            return False
        self._profile_seen += 1
        return (
            self._profile_seen <= self._profile_first_n
            or total_s >= self._profile_slow_s
        )

    def _episode_index_for_idx(self, idx: int) -> int | None:
        if self._use_lazy_video_tree or getattr(self, "_use_v2_image_parquet", False):
            try:
                _, episode_index, _ = self._resolve_index_context(idx)
                return int(episode_index)
            except Exception:
                return None
        return None

    def _record_bad_video_episode(self, idx: int, exc: Exception) -> int | None:
        """将解码失败的 episode 加入当前 worker 的内存集合，不修改数据集文件。"""
        episode_index = self._episode_index_for_idx(idx)
        if episode_index is None:
            return None
        if episode_index in self._bad_video_episodes:
            return episode_index
        self._bad_video_episodes.add(episode_index)
        return episode_index

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.multi_anchor_resample_attempts):
            original_idx = int(idx)
            t0 = time.perf_counter()
            t_skip = t0
            episode_index = self._episode_index_for_idx(idx)
            if episode_index in self._bad_video_episodes:
                last_error = RuntimeError(
                    f"episode {episode_index} is quarantined in this DataLoader worker"
                )
                idx = random.randint(0, max(0, len(self) - 1))
                if self._should_log_profile(time.perf_counter() - t0):
                    logger.info(
                        "[DATALOADER_PROFILE] pid=%s path=%s idx=%s episode=%s "
                        "attempt=%s/%s skipped_bad_episode=1 next_idx=%s total_s=%.4f",
                        os.getpid(),
                        self.data_path,
                        original_idx,
                        episode_index,
                        attempt + 1,
                        self.multi_anchor_resample_attempts,
                        idx,
                        time.perf_counter() - t0,
                    )
                continue
            try:
                raw_sample = self._load_raw_sample(idx)
                t_raw = time.perf_counter()
                raw = self._build_modality_dict(raw_sample)
                actual_num_video_frames = None
                if self.return_episode_first_images:
                    actual_num_video_frames = raw.pop(
                        _EPISODE_FIRST_NUM_VIDEO_FRAMES_KEY
                    )
                t_build = time.perf_counter()
                transformed = self.data_transform(raw)
                if self.return_episode_first_images:
                    self._split_episode_first_images(
                        transformed,
                        actual_num_video_frames=actual_num_video_frames,
                    )
                t_transform = time.perf_counter()
                sample = collate_ready_sample(transformed)
                sample["prompt_style"] = self.prompt_style
                if self.prompt_postfix:
                    sample["prompt_postfix"] = self.prompt_postfix
                t_ready = time.perf_counter()
                total_s = t_ready - t0
                if self._should_log_profile(total_s):
                    images = sample.get("images")
                    image_shape = tuple(images.shape) if hasattr(images, "shape") else None
                    logger.info(
                        "[DATALOADER_PROFILE] pid=%s path=%s mode=%s backend=%s "
                        "idx=%s episode=%s attempt=%s/%s "
                        "raw_s=%.4f build_s=%.4f transform_s=%.4f ready_s=%.4f total_s=%.4f "
                        "image_shape=%s",
                        os.getpid(),
                        self.data_path,
                        self.sampling_mode,
                        self._video_backend,
                        original_idx,
                        episode_index,
                        attempt + 1,
                        self.multi_anchor_resample_attempts,
                        t_raw - t_skip,
                        t_build - t_raw,
                        t_transform - t_build,
                        t_ready - t_transform,
                        total_s,
                        image_shape,
                    )
                return sample
            except EmptyTemporalSampleError as exc:
                last_error = exc
                if self.sampling_mode != "multi_anchor":
                    raise
                if attempt + 1 >= self.multi_anchor_resample_attempts:
                    break
                idx = random.randint(0, max(0, len(self) - 1))
            except RuntimeError as exc:
                if not _is_video_decode_error(exc):
                    raise
                last_error = exc
                episode_index = self._record_bad_video_episode(idx, exc)
                logger.warning(
                    "Skipping DreamZero sample idx=%s episode=%s after video decode failure "
                    "(attempt %s/%s); quarantined for this DataLoader worker only: %s",
                    idx,
                    episode_index,
                    attempt + 1,
                    self.multi_anchor_resample_attempts,
                    exc,
                )
                if attempt + 1 >= self.multi_anchor_resample_attempts:
                    break
                idx = random.randint(0, max(0, len(self) - 1))
        raise EmptyTemporalSampleError(
            f"Failed to sample a valid multi_anchor index after "
            f"{self.multi_anchor_resample_attempts} attempts: {last_error}"
        )

    def _build_modality_dict(self, sample: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        num_video_frames: int | None = None
        for transform_key, source_key in self._source_video_key.items():
            if transform_key in sample:
                raw_frames = sample[transform_key]
            else:
                raw_frames = sample[source_key]
            video = self._video_to_thwc_uint8(raw_frames)
            if num_video_frames is None:
                num_video_frames = int(video.shape[0])
            elif int(video.shape[0]) != num_video_frames:
                raise ValueError(
                    "All video views must have the same temporal length before transform; "
                    f"got {transform_key} length={video.shape[0]}, expected={num_video_frames}."
                )

            if self.return_episode_first_images:
                first_key = f"{_EPISODE_FIRST_VIDEO_PREFIX}{transform_key}"
                if first_key not in sample:
                    raise KeyError(
                        f"Missing decoded episode first frame for {transform_key!r}."
                    )
                # 输入是原始 mp4 的 timestamp=0.0 首帧，shape [1,H,W,C] uint8；临时拼到采样视频最前面，使它和当前视频共享同一套 view 拼接/增强。
                first_video = self._video_to_thwc_uint8(sample[first_key])
                if first_video.shape[0] != 1:
                    raise ValueError(
                        "Decoded episode first frame must contain exactly one frame; "
                        f"got {first_key} shape={first_video.shape}."
                    )
                video = np.concatenate([first_video, video], axis=0)
            out[transform_key] = video
        if self.return_episode_first_images:
            if num_video_frames is None:
                raise ValueError("return_episode_first_images=True requires at least one video view.")
            # 记录当前样本实际视频长度 T；multi-anchor 边界样本可能短于配置 num_frames。
            out[_EPISODE_FIRST_NUM_VIDEO_FRAMES_KEY] = num_video_frames

        self._put_components(out, sample, self._state_components, is_action=False)
        if self._is_egodex():
            self._put_egodex_absolute_actions(out, sample)
        elif self.relative_action and is_dual_arm_eef_se3_relative_action(
            self.relative_action_mode
        ):
            robotwin_eef_absolute_state = (
                self._prepare_dual_arm_eef_relative_state(
                    out, sample.get(_NOE_UMI_EPISODE_REFERENCE_KEY)
                )
            )
            self._put_components(
                out,
                sample,
                self._action_components,
                is_action=True,
                robotwin_eef_absolute_state=robotwin_eef_absolute_state,
            )
        else:
            self._put_components(out, sample, self._action_components, is_action=True)

        if (
            self.action_quaternion_mode == ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS
            and ROBOTWIN_EEF_ACTION_KEY in out
        ):
            if ROBOTWIN_EEF_STATE_KEY not in out:
                raise ValueError(
                    "Temporal action quaternion continuity requires state.eef_position."
                )
            out[ROBOTWIN_EEF_ACTION_KEY] = temporal_continuous_robotwin_eef_action_wxyz(
                out[ROBOTWIN_EEF_ACTION_KEY],
                np.asarray(out[ROBOTWIN_EEF_STATE_KEY], dtype=np.float32),
                action_horizon=self.action_horizon,
            )

        for quality_key in (
            _NOE_UMI_STATE_QUALITY_KEY,
            _NOE_UMI_ACTION_QUALITY_KEY,
        ):
            if quality_key in sample:
                out[quality_key] = np.asarray(sample[quality_key], dtype=bool)

        # 选择原始文本，后续进行Transform进行模板替换
        dense_subtask_instruction = sample.get(_DENSE_SUBTASK_INSTRUCTION_KEY)
        if dense_subtask_instruction is not None:
            out[self.language_keys[0]] = str(dense_subtask_instruction)
            return out

        if self._ego_instruction_index is not None:
            out[self.language_keys[0]] = self._resolve_ego_instruction(sample)
            return out

        subtask_instruction = sample.get(_NOE_SUBTASK_INSTRUCTION_KEY)
        if subtask_instruction is not None:
            text = str(subtask_instruction).strip()
            if not text:
                raise ValueError(
                    "Resolved NOE subtask instruction must not be empty."
                )
            out[self.language_keys[0]] = text
            return out

        if self.episode_instruction_key:
            out[self.language_keys[0]] = self._resolve_episode_instruction_text(sample)
            return out

        fallback_text = self._resolve_task_text(sample)
        wrote_language = False
        for key in self.language_keys:
            source = self._language_sources.get(key, key)
            value = sample.get(key, sample.get(source, ""))
            text = safe_lang_text(value, self._tasks) if value != "" else ""
            if text:
                out[key] = text
                wrote_language = True
        if not wrote_language:
            out[self.language_keys[0]] = fallback_text
        return out

    def _split_episode_first_images(
        self,
        transformed: dict[str, Any],
        *,
        actual_num_video_frames: int,
    ) -> None:
        """从 transform 后的 images 里切出原始 episode 首帧。

        输入 transformed["images"] 是 DreamTransform/VLM processor 后的
        [(T * V_out), H, W, C]；因为首帧被 prepend 到每个 view，前 V_out
        张图对应原始 episode 第 0 帧。输出新增 episode_first_images，并把
        images 还原成原采样视频长度。
        """
        images = transformed.get("images")
        if images is None:
            raise KeyError("return_episode_first_images=True requires transformed['images'].")
        if not isinstance(images, np.ndarray):
            images = np.asarray(images)
        expected_total = int(actual_num_video_frames) + 1
        if expected_total <= 1:
            raise ValueError(f"Invalid actual video frame count {actual_num_video_frames}.")
        if images.ndim < 1 or images.shape[0] % expected_total != 0:
            raise ValueError(
                "Cannot split episode_first_images from transformed images; expected "
                f"leading dim to be divisible by actual_num_frames+1={expected_total}, "
                f"got shape={images.shape}."
            )
        v_out = images.shape[0] // expected_total
        if v_out <= 0:
            raise ValueError(f"Invalid transformed view count v_out={v_out}.")
        transformed["episode_first_images"] = images[:v_out]
        transformed["images"] = images[v_out:]


class DreamZeroCollator:
    """Stack transformed samples and tokenize text (Groot ``DefaultDataCollator``-style)."""

    TEMPORAL_SOURCE_KEYS = {
        "images": "image_valid_mask",
        "state": "state_valid_mask",
        "action": "action_valid_mask",
    }
    TEMPORAL_PAD_KEYS = {
        "images",
        "state",
        "state_mask",
        "action",
        "action_mask",
    }
    TEMPORAL_PAD_LENGTH_KEYS = {
        "images": "images",
        "state": "state",
        "state_mask": "state",
        "action": "action",
        "action_mask": "action",
    }
    # Keys stored as list-of-lists (not stacked), e.g. per-view images for latent concat mode.
    LIST_KEYS = {"video_by_view"}

    def __init__(
        self,
        tokenizer_path: str,
        max_seq_len: int,
        embodiment_tag_mapping: dict[str, int],
        profile: bool = False,
        profile_first_n: int = 4,
        profile_slow_s: float = 1.0,
        temporal_pad_lengths: dict[str, int] | None = None,
    ):
        from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
            HuggingfaceTokenizer,
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=max_seq_len,
            clean="whitespace",
        )
        self.embodiment_tag_mapping = embodiment_tag_mapping
        self.profile = bool(profile)
        self.profile_first_n = max(0, int(profile_first_n))
        self.profile_slow_s = max(0.0, float(profile_slow_s))
        self.profile_seen = 0
        self.temporal_pad_lengths = self._normalize_temporal_pad_lengths(
            temporal_pad_lengths
        )

    @staticmethod
    def _normalize_temporal_pad_lengths(
        temporal_pad_lengths: dict[str, int] | None,
    ) -> dict[str, int]:
        if temporal_pad_lengths is None:
            return {}

        unknown_keys = set(temporal_pad_lengths) - set(
            DreamZeroCollator.TEMPORAL_SOURCE_KEYS
        )
        if unknown_keys:
            raise ValueError(
                "Unsupported temporal padding targets: "
                f"{sorted(unknown_keys)}."
            )

        normalized = {
            key: int(length) for key, length in temporal_pad_lengths.items()
        }
        invalid = {key: length for key, length in normalized.items() if length <= 0}
        if invalid:
            raise ValueError(
                f"Temporal padding targets must be positive, got {invalid}."
            )
        return normalized

    @staticmethod
    def collate_batch(
        features: list[dict[str, Any]],
        tokenizer: Any,
        embodiment_tag_mapping: dict[str, int],
        temporal_pad_lengths: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        temporal_pad_lengths = DreamZeroCollator._normalize_temporal_pad_lengths(
            temporal_pad_lengths
        )
        batch: dict[str, Any] = {}
        for key in features[0]:
            if key in {"prompt_postfix", "prompt_style"}:
                continue
            if key == "text":
                texts = []
                for elem in features:
                    instruction = normalize_instruction_text(elem[key])
                    if not instruction:
                        prompt = ""
                    elif elem.get("prompt_style") == "robotics":
                        prompt = (
                            f"A multi-view video shows that a dual-arm robot {instruction}"
                            " The video is split into three views: The top view shows the "
                            "main camera, the bottom-left view shows the left wrist camera, "
                            "and the bottom-right view shows the right wrist camera."
                        )
                    else:
                        prompt = format_training_prompt(
                            instruction,
                            int(elem["embodiment_id"]),
                            embodiment_tag_mapping,
                        )
                    postfix = (elem.get("prompt_postfix") or "").strip()
                    if prompt and postfix:
                        prompt = f"{prompt} {postfix}"
                    texts.append(prompt)
                ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
                batch[key] = ids
                batch["text_attention_mask"] = mask
            elif key == "text_negative":
                values = [elem[key] for elem in features]
                ids, mask = tokenizer(values, return_mask=True, add_special_tokens=True)
                batch[key] = ids
                batch["text_attention_mask_negative"] = mask
            else:
                values = [elem[key] for elem in features]
                # --- list keys: keep as list-of-lists, no stacking ---
                if key in DreamZeroCollator.LIST_KEYS:
                    batch[key] = values
                    continue

                if key in DreamZeroCollator.TEMPORAL_PAD_KEYS and all(
                    isinstance(value, np.ndarray) and value.ndim >= 1
                    for value in values
                ):
                    lengths = [value.shape[0] for value in values]
                    max_len = max(lengths)
                    length_key = DreamZeroCollator.TEMPORAL_PAD_LENGTH_KEYS[key]
                    target_len = temporal_pad_lengths.get(length_key)
                    if target_len is not None:
                        if max_len > target_len:
                            raise ValueError(
                                f"Temporal length for key={key!r} exceeds configured "
                                f"target: observed={max_len}, target={target_len}."
                            )
                        max_len = target_len
                    padded = []
                    for value in values:
                        if value.shape[0] < max_len:
                            pad_count = max_len - value.shape[0]
                            if key == "images":
                                pad_values = np.repeat(value[-1:], pad_count, axis=0)
                                value = np.concatenate([value, pad_values], axis=0)
                            else:
                                pad_width = [(0, pad_count)] + [
                                    (0, 0)
                                ] * (value.ndim - 1)
                                value = np.pad(value, pad_width, mode="constant")
                        padded.append(value)
                    values = padded
                    if key in DreamZeroCollator.TEMPORAL_SOURCE_KEYS:
                        valid_mask = np.zeros((len(lengths), max_len), dtype=bool)
                        for i, length in enumerate(lengths):
                            valid_mask[i, :length] = True
                        batch[DreamZeroCollator.TEMPORAL_SOURCE_KEYS[key]] = (
                            torch.from_numpy(valid_mask)
                        )
                try:
                    batch[key] = torch.from_numpy(np.stack(values))
                except ValueError as e:
                    shapes = [np.asarray(v).shape for v in values]
                    raise ValueError(
                        f"Shape mismatch in collate for key='{key}': shapes={shapes}"
                    ) from e
                if (
                    key in DreamZeroCollator.TEMPORAL_SOURCE_KEYS
                    and DreamZeroCollator.TEMPORAL_SOURCE_KEYS[key] not in batch
                ):
                    mask_key = DreamZeroCollator.TEMPORAL_SOURCE_KEYS[key]
                    batch[mask_key] = torch.ones(
                        batch[key].shape[:2],
                        dtype=torch.bool,
                    )
        return batch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        t0 = time.perf_counter()
        batch = self.collate_batch(
            features,
            self.tokenizer,
            self.embodiment_tag_mapping,
            self.temporal_pad_lengths,
        )
        total_s = time.perf_counter() - t0
        self.profile_seen += 1
        if self.profile and (
            self.profile_seen <= self.profile_first_n or total_s >= self.profile_slow_s
        ):
            images = batch.get("images")
            image_shape = tuple(images.shape) if hasattr(images, "shape") else None
            logger.info(
                "[DATALOADER_COLLATE_PROFILE] pid=%s batch_size=%s total_s=%.4f "
                "image_shape=%s keys=%s",
                os.getpid(),
                len(features),
                total_s,
                image_shape,
                sorted(batch.keys()),
            )
        return batch


def _build_dreamzero_temporal_pad_lengths(
    *,
    num_frames: int,
    max_chunk_size: int,
    state_horizon: int,
    action_horizon: int,
) -> dict[str, int]:
    lengths = {
        "images": int(num_frames),
        "state": int(max_chunk_size) * int(state_horizon),
        "action": int(max_chunk_size) * int(action_horizon),
    }
    invalid = {key: length for key, length in lengths.items() if length <= 0}
    if invalid:
        raise ValueError(
            f"DreamZero temporal padding lengths must be positive: {invalid}."
        )
    return lengths


def build_dreamzero_sft_dataloader(
    cfg,
    world_size: int,
    rank: int,
    data_paths: str,
    eval_dataset: bool = False,
):
    """Build DreamZero SFT dataloader -- callable from FSDPVlaSftWorker.

    Uses DistributedSampler to shard data across GPUs:
      - Each of the 8 GPUs sees 1/8 of the dataset per epoch
      - micro_batch_size samples are returned per iteration per GPU
      - Global effective batch size = micro_batch_size * world_size * grad_accum_steps
    """
    if cfg.data.get("train_datasets", None) is not None and not eval_dataset:
        from rlinf.data.datasets.dreamzero.mixed import (
            build_mixed_dreamzero_sft_dataloader,
        )

        return build_mixed_dreamzero_sft_dataloader(
            cfg,
            world_size,
            rank,
            eval_dataset=eval_dataset,
        )
    task_type_config = TaskTypeSamplingConfig.from_value(
        cfg.data.get("task_type_sampling", None)
    )
    if task_type_config.enabled and not eval_dataset:
        raise ValueError(
            "data.task_type_sampling.enabled requires data.train_datasets; "
            "data.train_data_paths is a legacy compatibility field."
        )

    from groot.vla.data.transform import ComposedModalityTransform

    from rlinf.data.datasets.dreamzero.data_transforms import (
        build_dreamzero_composed_transform,
        collect_dreamzero_dataset_keys,
        embodiment_tag_mapping_for_embodiment,
        load_dreamzero_dataset_metadata,
    )

    data_cfg = cfg.data
    model_cfg = cfg.actor.model
    tokenizer_path = model_cfg.get("tokenizer_path", "google/umt5-xxl")
    embodiment_tag = model_cfg.embodiment_tag

    metadata = load_dreamzero_dataset_metadata(model_cfg)
    data_transform = build_dreamzero_composed_transform(model_cfg, tokenizer_path)
    assert isinstance(data_transform, ComposedModalityTransform), f"{data_transform=}"
    data_transform.set_metadata(metadata)
    if eval_dataset:
        data_transform.eval()
    else:
        data_transform.train()

    video_keys, state_keys, action_keys, language_keys = collect_dreamzero_dataset_keys(
        data_transform, embodiment_tag
    )

    sampling_mode = data_cfg.get("sampling_mode", "multi_anchor")
    if sampling_mode not in ("multi_anchor", "fixed_window"):
        raise ValueError(
            f"Unsupported data.sampling_mode {sampling_mode!r}; "
            "use 'multi_anchor' or 'fixed_window'."
        )
    max_chunk_size = model_cfg.action_head_cfg.config.diffusion_model_cfg.max_chunk_size
    num_frames = model_cfg.action_head_cfg.config.num_frames
    state_horizon = model_cfg.get("state_horizon", 1)
    action_horizon = model_cfg.action_horizon
    max_seq_len = int(model_cfg.get("max_seq_len", 512))
    embodiment_tag_mapping = embodiment_tag_mapping_for_embodiment(
        embodiment_tag, model_cfg.get("embodiment_tag_mapping")
    )
    temporal_pad_lengths = None
    if sampling_mode == "multi_anchor":
        temporal_pad_lengths = _build_dreamzero_temporal_pad_lengths(
            num_frames=num_frames,
            max_chunk_size=max_chunk_size,
            state_horizon=state_horizon,
            action_horizon=action_horizon,
        )

    resolved_data_paths = _resolve_dreamzero_data_paths(data_paths, embodiment_tag)
    dataset_kwargs = dict(
        video_keys=video_keys,
        state_keys=state_keys,
        action_keys=action_keys,
        language_keys=language_keys,
        video_key_mapping=data_cfg.get("video_key_mapping", None),
        prompt_postfix=data_cfg.get("prompt_postfix", ""),
        data_transform=data_transform,
        lazy_load=cfg.data.get("lazy_load", True),
        num_frames=num_frames,
        state_horizon=state_horizon,
        action_horizon=action_horizon,
        relative_action=bool(model_cfg.get("relative_action", False)),
        relative_action_keys=list(model_cfg.get("relative_action_keys", [])),
        relative_action_mode=model_cfg.get("relative_action_mode"),
        relative_action_pose_reference=model_cfg.get(
            "relative_action_pose_reference"
        ),
        action_quaternion_mode=model_cfg.get("action_quaternion_mode"),
        pq_cache_max_episodes=cfg.data.get("parquet_cache_size", 128),
        video_tolerance_s=cfg.data.get("video_tolerance_s", 0.1),
        video_backend=data_cfg.get("video_backend", "pyav"),
        max_chunk_size=max_chunk_size,
        sampling_mode=sampling_mode,
        multi_anchor_resample_attempts=data_cfg.get(
            "multi_anchor_resample_attempts", 32
        ),
        multi_anchor_macro_stride=data_cfg.get("multi_anchor_macro_stride", 24),
        multi_anchor_legacy_absolute_offsets=data_cfg.get(
            "multi_anchor_legacy_absolute_offsets", False
        ),
        bad_video_manifest_name=data_cfg.get(
            "bad_video_manifest_name", "bad_video_episodes.jsonl"
        ),
        episode_exclusion_manifest=data_cfg.get("episode_exclusion_manifest", None),
        dataloader_profile=data_cfg.get("dataloader_profile", False),
        dataloader_profile_first_n=data_cfg.get("dataloader_profile_first_n", 4),
        dataloader_profile_slow_s=data_cfg.get("dataloader_profile_slow_s", 2.0),
        return_episode_first_images=data_cfg.get("return_episode_first_images", False),
        multi_anchor_video_offsets=data_cfg.get(
            "multi_anchor_video_offsets", [0, 3, 6, 9, 12, 15, 18, 21]
        ),
        embodiment_tag=embodiment_tag,
        episode_instruction_key=data_cfg.get("episode_instruction_key", None),
        subtask_mode=data_cfg.get("subtask_mode", _SUBTASK_MODE_LEGACY),
        subtask_file=data_cfg.get(
            "subtask_file", _DEFAULT_DENSE_SUBTASK_FILE
        ),
        subtask_caption_field=data_cfg.get("subtask_caption_field", "action_detail_en"),
        caption_granularity=data_cfg.get("caption_granularity", "subtask"),
        subtask_caption_extra=data_cfg.get("subtask_caption_extra", "none"),
        use_subtask_instruction=bool(
            data_cfg.get("use_subtask_instruction", False)
        ),
        use_dense_caption_instruction=bool(
            data_cfg.get("use_dense_caption_instruction", False)
        ),
        dense_caption_include_object=bool(
            data_cfg.get("dense_caption_include_object", True)
        ),
        ego_caption_segment_mode=data_cfg.get("ego_caption_segment_mode", None),
        use_scene_instruction=bool(data_cfg.get("use_scene_instruction", False)),
        filter_noe_umi_qualified_episodes=bool(
            data_cfg.get("filter_noe_umi_qualified_episodes", False)
        ),
        noe_umi_quality_mode=data_cfg.get(
            "noe_umi_quality_mode", _NOE_UMI_QUALITY_MODE_EPISODE
        ),
        noe_subtask=data_cfg.get("noe_subtask", None),
        tasks_filename=data_cfg.get("tasks_filename", "tasks.jsonl"),
        episode_discovery_mode=data_cfg.get(
            "episode_discovery_mode", "metadata"
        ),
    )
    subtask_configs = _prepare_noe_subtask_configs(
        resolved_data_paths,
        dataset_kwargs["noe_subtask"],
        embodiment_tag=embodiment_tag,
        sampling_mode=sampling_mode,
    )
    datasets = [
        DreamZeroLeRobotDataset(
            data_path=data_path,
            **{**dataset_kwargs, "noe_subtask": subtask_config},
        )
        for data_path, subtask_config in zip(
            resolved_data_paths, subtask_configs
        )
    ]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    logger.info(
        "DreamZero LeRobot dataset: embodiment=%s sampling_mode=%s max_chunk_size=%s "
        "action_horizon(transform)=%s num_datasets=%s num_samples=%s data_paths=%s "
        "video_keys=%s state_keys=%s action_keys=%s language_keys=%s",
        embodiment_tag,
        sampling_mode,
        max_chunk_size,
        action_horizon,
        len(datasets),
        len(dataset),
        resolved_data_paths[:5],
        video_keys,
        state_keys,
        action_keys,
        language_keys,
    )
    sampler_seed = int(data_cfg.get("seed", cfg.actor.get("seed", 0)))
    dataloader_seed = sampler_seed + rank
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(dataloader_seed)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=sampler_seed,
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
    )
    num_workers = int(cfg.data.get("num_workers", 4))
    prefetch_factor = int(cfg.data.get("prefetch_factor", 4))
    dataloader_profile = bool(data_cfg.get("dataloader_profile", False))
    dataloader_profile_first_n = int(data_cfg.get("dataloader_profile_first_n", 4))
    dataloader_profile_slow_s = float(data_cfg.get("dataloader_profile_slow_s", 2.0))
    logger.info(
        "DreamZero DataLoader config: rank=%s/%s batch_size=%s num_workers=%s "
        "prefetch_factor=%s persistent_workers=%s pin_memory=%s sampling_mode=%s "
        "video_backend=%s sampler_seed=%s dataloader_seed=%s profile=%s "
        "profile_first_n=%s profile_slow_s=%s",
        rank,
        world_size,
        cfg.actor.micro_batch_size,
        num_workers,
        prefetch_factor if num_workers > 0 else None,
        num_workers > 0,
        True,
        sampling_mode,
        data_cfg.get("video_backend", "pyav"),
        sampler_seed,
        dataloader_seed,
        dataloader_profile,
        dataloader_profile_first_n,
        dataloader_profile_slow_s,
    )
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,  # samples per GPU per step
        sampler=sampler,
        drop_last=not eval_dataset,
        num_workers=num_workers,
        pin_memory=True,  # faster CPU->GPU transfer
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=_suppress_ffmpeg_log,
        generator=dataloader_generator,
        collate_fn=DreamZeroCollator(
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            embodiment_tag_mapping=dict(embodiment_tag_mapping),
            profile=dataloader_profile,
            profile_first_n=dataloader_profile_first_n,
            profile_slow_s=dataloader_profile_slow_s,
            temporal_pad_lengths=temporal_pad_lengths,
        ),
    )
    return data_loader, {"num_samples": len(dataset)}
