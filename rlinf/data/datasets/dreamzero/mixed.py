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

from __future__ import annotations

import math
import random
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.datasets.dreamzero.dreamzero import (
    DreamZeroCollator,
    DreamZeroLeRobotDataset,
    _build_dreamzero_temporal_pad_lengths,
    _prepare_noe_subtask_configs,
    _resolve_dreamzero_data_paths,
    _suppress_ffmpeg_log,
)
from rlinf.data.datasets.dreamzero.task_type_sampling import (
    TaskTypeSamplingConfig,
    TaskTypeSamplingPlan,
    merge_task_type_frame_indices,
)
from rlinf.utils.logging import get_logger

logger = get_logger()

_SUPPORTED_MIXED_EMBODIMENTS = {
    "egodex",
    "ego4d",
    "egoverse",
    "egopro",
    "openvid",
    "robotwin",
    "noe_umi_16d",
    "noe_robotics_16d",
    "oxe_droid",
}
_VIDEO_ONLY_EMBODIMENTS = {"ego4d", "egoverse", "egopro", "openvid"}
_DATASET_MODEL_OVERRIDE_KEYS = {
    "embodiment_tag",
    "metadata_json_path",
    "metadata_embodiment_tag",
    "image_resolution_width",
    "image_resolution_height",
    "concat_method",
    "default_instruction",
    "language_dropout_prob",
    "always_use_default_instruction",
    "has_real_action",
    "ego_view_layout",
    "relative_action",
    "relative_action_keys",
    "relative_action_mode",
    "relative_action_pose_reference",
    "action_quaternion_mode",
}


def _plain_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _plain_dict(value: Any) -> dict[str, Any]:
    value = _plain_container(value)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping config, got {type(value)!r}: {value!r}")
    return dict(value)


def _plain_list(value: Any) -> list[Any]:
    value = _plain_container(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Expected list config, got {type(value)!r}: {value!r}")


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    return cfg.get(key, default)


def _dataset_name(dataset_cfg: Any, idx: int) -> str:
    name = _cfg_get(dataset_cfg, "name", None)
    return str(name) if name else f"dataset_{idx}"


def _dataset_data_paths(dataset_cfg: Any) -> Any:
    data_paths = _cfg_get(dataset_cfg, "data_paths", None)
    if data_paths is None:
        raise ValueError("Each data.train_datasets entry must set data_paths.")
    return data_paths


def _data_option(data_cfg: Any, dataset_cfg: Any, key: str, default: Any) -> Any:
    value = _cfg_get(dataset_cfg, key, None)
    if value is not None:
        return value
    return _cfg_get(data_cfg, key, default)


def _merge_default_embodiment_mapping(
    model_cfg: Any,
    dataset_cfgs: Sequence[Any],
) -> dict[str, int]:
    from rlinf.data.datasets.dreamzero.data_transforms import (
        embodiment_tag_mapping_for_embodiment,
    )

    override = _cfg_get(model_cfg, "embodiment_tag_mapping", None)
    if override is not None:
        mapping = _plain_dict(override)
    else:
        mapping: dict[str, int] = {}
        for dataset_cfg in dataset_cfgs:
            tag = str(_cfg_get(dataset_cfg, "embodiment_tag", "")).strip()
            if not tag:
                raise ValueError("Each mixed dataset must set embodiment_tag.")
            mapping.update(embodiment_tag_mapping_for_embodiment(tag))

    missing = [
        str(_cfg_get(dataset_cfg, "embodiment_tag", ""))
        for dataset_cfg in dataset_cfgs
        if str(_cfg_get(dataset_cfg, "embodiment_tag", "")) not in mapping
    ]
    if missing:
        raise ValueError(
            "actor.model.embodiment_tag_mapping is missing mixed dataset tags: "
            f"{missing}. Current mapping: {mapping!r}"
        )
    return {str(k): int(v) for k, v in mapping.items()}


def _mixed_model_cfg(
    model_cfg: Any,
    dataset_cfg: Any,
    embodiment_tag_mapping: dict[str, int],
) -> Any:
    # 给每个子数据集复制一份全局 actor.model，只覆盖自身的 metadata/embodiment/transform 字段。
    model_dict = _plain_dict(model_cfg)
    dataset_dict = _plain_dict(dataset_cfg)
    for key in _DATASET_MODEL_OVERRIDE_KEYS:
        if key in dataset_dict and dataset_dict[key] is not None:
            model_dict[key] = dataset_dict[key]
    model_dict["embodiment_tag_mapping"] = dict(embodiment_tag_mapping)
    return OmegaConf.create(model_dict)


def _validate_mixed_contract(
    model_cfg: Any,
    dataset_cfgs: Sequence[Any],
) -> None:
    if not dataset_cfgs:
        raise ValueError("data.train_datasets must contain at least one dataset.")

    global_action_horizon = int(_cfg_get(model_cfg, "action_horizon"))
    global_max_action_dim = int(_cfg_get(model_cfg, "max_action_dim"))
    global_max_state_dim = int(_cfg_get(model_cfg, "max_state_dim"))
    for idx, dataset_cfg in enumerate(dataset_cfgs):
        name = _dataset_name(dataset_cfg, idx)
        tag = str(_cfg_get(dataset_cfg, "embodiment_tag", "")).strip()
        if tag not in _SUPPORTED_MIXED_EMBODIMENTS:
            raise ValueError(
                f"Mixed DreamZero SFT v0 supports only "
                f"{sorted(_SUPPORTED_MIXED_EMBODIMENTS)}, got {tag!r} for {name!r}."
            )
        if tag in _VIDEO_ONLY_EMBODIMENTS and (
            "has_real_action" not in dataset_cfg
            or bool(_cfg_get(dataset_cfg, "has_real_action", True))
        ):
            raise ValueError(
                f"Mixed video-only dataset {name!r} must explicitly set "
                "has_real_action: false."
            )
        weight = float(_cfg_get(dataset_cfg, "weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Mixed dataset {name!r} has invalid weight={weight!r}.")

        action_horizon = int(_cfg_get(dataset_cfg, "action_horizon", global_action_horizon))
        max_action_dim = int(_cfg_get(dataset_cfg, "max_action_dim", global_max_action_dim))
        max_state_dim = int(_cfg_get(dataset_cfg, "max_state_dim", global_max_state_dim))
        if action_horizon != global_action_horizon:
            raise ValueError(
                f"Mixed dataset {name!r} action_horizon={action_horizon} does not "
                f"match actor.model.action_horizon={global_action_horizon}."
            )
        if max_action_dim != global_max_action_dim:
            raise ValueError(
                f"Mixed dataset {name!r} max_action_dim={max_action_dim} does not "
                f"match actor.model.max_action_dim={global_max_action_dim}."
            )
        if max_state_dim != global_max_state_dim:
            raise ValueError(
                f"Mixed dataset {name!r} max_state_dim={max_state_dim} does not "
                f"match actor.model.max_state_dim={global_max_state_dim}."
            )

class DreamZeroMixtureDataset(Dataset):
    """Dataset wrapper whose sampler chooses a source dataset per sample."""

    def __init__(self, datasets: Sequence[Dataset], names: Sequence[str]):
        if len(datasets) != len(names):
            raise ValueError(
                f"datasets and names length mismatch: {len(datasets)} vs {len(names)}"
            )
        if not datasets:
            raise ValueError("DreamZeroMixtureDataset requires at least one dataset.")
        self.datasets = list(datasets)
        self.names = [str(name) for name in names]
        self.lengths = [len(dataset) for dataset in self.datasets]
        if any(length <= 0 for length in self.lengths):
            raise ValueError(
                f"All mixed datasets must be non-empty, got lengths={self.lengths}."
            )
        self._cumulative = []
        total = 0
        for length in self.lengths:
            total += int(length)
            self._cumulative.append(total)

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, index: Any) -> dict[str, Any]:
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        if isinstance(index, (tuple, list)) and len(index) == 2:
            dataset_idx = int(index[0])
            sample_idx = int(index[1])
            return self.datasets[dataset_idx][sample_idx]

        flat_idx = int(index)
        if flat_idx < 0 or flat_idx >= len(self):
            raise IndexError(
                f"Index {flat_idx} out of range for mixed dataset of len {len(self)}."
            )
        dataset_idx = bisect_right(self._cumulative, flat_idx)
        prev = self._cumulative[dataset_idx - 1] if dataset_idx > 0 else 0
        sample_idx = flat_idx - prev
        return self.datasets[dataset_idx][sample_idx]


class DistributedWeightedMixtureSampler(Sampler[tuple[int, int]]):
    """Distributed weighted sampler that yields ``(dataset_idx, sample_idx)`` pairs."""

    def __init__(
        self,
        dataset_lengths: Sequence[int],
        weights: Sequence[float],
        *,
        epoch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        drop_last: bool = True,
        # dataset_names, batch_size: 仅用于低概率打印 mix train batch 来源，不参与采样逻辑。
        dataset_names: Sequence[str] | None = None,
        batch_size: int | None = None,
        task_type_plans: Sequence[TaskTypeSamplingPlan | None] | None = None,
    ):
        if len(dataset_lengths) != len(weights):
            raise ValueError(
                "dataset_lengths and weights length mismatch: "
                f"{len(dataset_lengths)} vs {len(weights)}"
            )
        if not dataset_lengths:
            raise ValueError("DistributedWeightedMixtureSampler requires datasets.")
        if any(int(length) <= 0 for length in dataset_lengths):
            raise ValueError(f"All dataset lengths must be positive: {dataset_lengths}")
        if any(float(weight) <= 0.0 for weight in weights):
            raise ValueError(f"All mixture weights must be positive: {weights}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas!r}")
        if not 0 <= rank < num_replicas:
            raise ValueError(
                f"rank must be in [0, {num_replicas}), got rank={rank!r}"
            )

        self.dataset_lengths = [int(length) for length in dataset_lengths]
        self.weights = torch.as_tensor([float(weight) for weight in weights], dtype=torch.double)
        self.weights = self.weights / self.weights.sum()
        self.epoch_size = int(epoch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.dataset_names = (
            [str(name) for name in dataset_names] if dataset_names is not None else None
        )

        # task type检验
        if task_type_plans is not None and len(task_type_plans) != len(
            self.dataset_lengths
        ):
            raise ValueError(
                "task_type_plans and dataset_lengths length mismatch: "
                f"{len(task_type_plans)} vs {len(self.dataset_lengths)}"
            )
        self.task_type_plans = (
            list(task_type_plans) if task_type_plans is not None else None
        )
        if self.task_type_plans is not None:
            for dataset_idx, (dataset_len, task_type_plan) in enumerate(
                zip(self.dataset_lengths, self.task_type_plans)
            ):
                if (
                    task_type_plan is not None
                    and task_type_plan.index.frame_count != dataset_len
                ):
                    raise ValueError(
                        f"Task type frame count mismatch for dataset {dataset_idx}: "
                        f"{task_type_plan.index.frame_count} vs {dataset_len}."
                    )
        self.batch_size = int(batch_size or 0)
        self.epoch = 0

        if self.epoch_size <= 0:
            raise ValueError(f"epoch_size must be positive, got {epoch_size!r}")
        if self.drop_last:
            self.num_samples = self.epoch_size // self.num_replicas
            if self.num_samples <= 0:
                raise ValueError(
                    f"epoch_size={self.epoch_size} is smaller than "
                    f"num_replicas={self.num_replicas} with drop_last=True."
                )
        else:
            self.num_samples = math.ceil(self.epoch_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[tuple[int, int]]:
        # 所有 rank 用同一随机种子生成全局采样序列，再各自按 rank 切片。
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        dataset_ids = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        sample_ids = torch.empty(self.total_size, dtype=torch.long)
        for dataset_idx, dataset_len in enumerate(self.dataset_lengths):
            # 每个 dataset 随机抽取对应数目的 idx
            mask = dataset_ids == dataset_idx   # 每次处理一个数据集
            count = int(mask.sum().item())
            if count:
                task_type_plan = (
                    None
                    if self.task_type_plans is None
                    else self.task_type_plans[dataset_idx]
                )
                if task_type_plan is None:
                    sample_ids[mask] = torch.randint(
                        dataset_len,
                        (count,),
                        generator=generator,
                        dtype=torch.long,
                    )
                else:
                    sample_ids[mask] = task_type_plan.sample_many(count, generator)

        pairs = list(zip(dataset_ids.tolist(), sample_ids.tolist()))
        pairs = pairs[self.rank : self.total_size : self.num_replicas]
        # 仅用于低概率打印 mix train 当前 batch 的来源 dataset name，不影响采样结果。
        if self.dataset_names is not None and self.batch_size > 0:
            for start in range(0, len(pairs), self.batch_size):
                if random.random() < 0.001:
                    batch_pairs = pairs[start : start + self.batch_size]
                    counts = Counter(
                        self.dataset_names[int(dataset_idx)]
                        for dataset_idx, _ in batch_pairs
                    )
                    logger.info(
                        "[DREAMZERO_MIXED_BATCH_SOURCE] "
                        "rank=%s epoch=%s batch_idx=%s counts=%s",
                        self.rank,
                        self.epoch,
                        start // self.batch_size,
                        dict(sorted(counts.items())),
                    )
        return iter((int(dataset_idx), int(sample_idx)) for dataset_idx, sample_idx in pairs)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _build_dataset_bundle(
    cfg: Any,
    dataset_cfg: Any,
    *,
    dataset_idx: int,
    embodiment_tag_mapping: dict[str, int],
    eval_dataset: bool,
) -> tuple[Dataset, dict[str, Any]]:
    assert not eval_dataset, "DreamZero mixed dataloader is train-only."

    from groot.vla.data.transform import ComposedModalityTransform

    from rlinf.data.datasets.dreamzero.data_transforms import (
        build_dreamzero_composed_transform,
        collect_dreamzero_dataset_keys,
        load_dreamzero_dataset_metadata,
    )

    data_cfg = cfg.data
    model_cfg = cfg.actor.model
    name = _dataset_name(dataset_cfg, dataset_idx)
    ds_model_cfg = _mixed_model_cfg(model_cfg, dataset_cfg, embodiment_tag_mapping)
    tokenizer_path = ds_model_cfg.get("tokenizer_path", "google/umt5-xxl")
    embodiment_tag = str(ds_model_cfg.embodiment_tag)

    metadata = load_dreamzero_dataset_metadata(ds_model_cfg)
    data_transform = build_dreamzero_composed_transform(ds_model_cfg, tokenizer_path)
    assert isinstance(data_transform, ComposedModalityTransform), f"{data_transform=}"
    data_transform.set_metadata(metadata)
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
    data_paths = _dataset_data_paths(dataset_cfg)
    resolved_data_paths = _resolve_dreamzero_data_paths(data_paths, embodiment_tag)
    task_type_config = TaskTypeSamplingConfig.from_value(
        _data_option(data_cfg, dataset_cfg, "task_type_sampling", None)
    )

    dataset_kwargs = dict(
        video_keys=video_keys,
        state_keys=state_keys,
        action_keys=action_keys,
        language_keys=language_keys,
        video_key_mapping=_data_option(
            data_cfg, dataset_cfg, "video_key_mapping", None
        ),
        prompt_postfix=_data_option(data_cfg, dataset_cfg, "prompt_postfix", ""),
        prompt_style=_data_option(data_cfg, dataset_cfg, "prompt_style", None),
        data_transform=data_transform,
        lazy_load=bool(_data_option(data_cfg, dataset_cfg, "lazy_load", True)),
        num_frames=num_frames,
        state_horizon=state_horizon,
        action_horizon=action_horizon,
        relative_action=bool(ds_model_cfg.get("relative_action", False)),
        relative_action_keys=list(ds_model_cfg.get("relative_action_keys", [])),
        relative_action_mode=ds_model_cfg.get("relative_action_mode"),
        relative_action_pose_reference=ds_model_cfg.get("relative_action_pose_reference"),
        action_quaternion_mode=ds_model_cfg.get("action_quaternion_mode"),
        pq_cache_max_episodes=_data_option(data_cfg, dataset_cfg, "parquet_cache_size", 128),
        video_tolerance_s=_data_option(data_cfg, dataset_cfg, "video_tolerance_s", 0.1),
        video_backend=_data_option(data_cfg, dataset_cfg, "video_backend", "pyav"),
        max_chunk_size=max_chunk_size,
        sampling_mode=sampling_mode,
        multi_anchor_resample_attempts=_data_option(
            data_cfg, dataset_cfg, "multi_anchor_resample_attempts", 32
        ),
        multi_anchor_macro_stride=_data_option(
            data_cfg, dataset_cfg, "multi_anchor_macro_stride", 24
        ),
        multi_anchor_legacy_absolute_offsets=_data_option(
            data_cfg,
            dataset_cfg,
            "multi_anchor_legacy_absolute_offsets",
            False,
        ),
        bad_video_manifest_name=_data_option(
            data_cfg,
            dataset_cfg,
            "bad_video_manifest_name",
            "bad_video_episodes.jsonl",
        ),
        episode_exclusion_manifest=_data_option(
            data_cfg, dataset_cfg, "episode_exclusion_manifest", None
        ),
        dataloader_profile=_data_option(data_cfg, dataset_cfg, "dataloader_profile", False),
        dataloader_profile_first_n=_data_option(
            data_cfg, dataset_cfg, "dataloader_profile_first_n", 4
        ),
        dataloader_profile_slow_s=_data_option(
            data_cfg, dataset_cfg, "dataloader_profile_slow_s", 2.0
        ),
        return_episode_first_images=bool(
            _data_option(data_cfg, dataset_cfg, "return_episode_first_images", False)
        ),
        multi_anchor_video_offsets=_data_option(
            data_cfg,
            dataset_cfg,
            "multi_anchor_video_offsets",
            [0, 3, 6, 9, 12, 15, 18, 21],
        ),
        embodiment_tag=embodiment_tag,
        tasks_filename=_data_option(
            data_cfg,
            dataset_cfg,
            "tasks_filename",
            "tasks.jsonl",
        ),
        episode_instruction_key=_data_option(
            data_cfg, dataset_cfg, "episode_instruction_key", None
        ),
        subtask_mode=_cfg_get(dataset_cfg, "subtask_mode", "legacy"),
        subtask_file=_cfg_get(
            dataset_cfg,
            "subtask_file",
            "meta/subtask/subtasks_dense_seed.parquet",
        ),
        subtask_caption_field=_cfg_get(
            dataset_cfg, "subtask_caption_field", "action_detail_en"
        ),
        caption_granularity=_cfg_get(
            dataset_cfg, "caption_granularity", "subtask"
        ),
        subtask_caption_extra=_cfg_get(dataset_cfg, "subtask_caption_extra", "none"),
        use_subtask_instruction=bool(
            _data_option(data_cfg, dataset_cfg, "use_subtask_instruction", False)
        ),
        use_dense_caption_instruction=bool(
            _data_option(
                data_cfg,
                dataset_cfg,
                "use_dense_caption_instruction",
                False,
            )
        ),
        dense_caption_include_object=bool(
            _data_option(
                data_cfg,
                dataset_cfg,
                "dense_caption_include_object",
                True,
            )
        ),
        ego_caption_segment_mode=_data_option(
            data_cfg,
            dataset_cfg,
            "ego_caption_segment_mode",
            None,
        ),
        use_scene_instruction=bool(
            _data_option(data_cfg, dataset_cfg, "use_scene_instruction", False)
        ),
        filter_noe_umi_qualified_episodes=bool(
            _data_option(
                data_cfg,
                dataset_cfg,
                "filter_noe_umi_qualified_episodes",
                False,
            )
        ),
        noe_umi_quality_mode=_data_option(
            data_cfg,
            dataset_cfg,
            "noe_umi_quality_mode",
            "episode",
        ),
        noe_subtask=_data_option(data_cfg, dataset_cfg, "noe_subtask", None),
        episode_discovery_mode=_data_option(
            data_cfg,
            dataset_cfg,
            "episode_discovery_mode",
            "metadata",
        ),
        task_type_sampling=task_type_config,
    )
    subtask_configs = _prepare_noe_subtask_configs(
        resolved_data_paths,
        dataset_kwargs["noe_subtask"],
        embodiment_tag=embodiment_tag,
        sampling_mode=sampling_mode,
    )   # 只有umi数据会进入处理
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

    task_type_plan = None
    task_type_index = None
    if task_type_config.enabled:    # 将子集的frame index进行合并
        child_indices = []  # 保存每个子 dataset 的 task-type 索引。
        offsets = []        # 保存每个子集的全局索引起点
        offset = 0          # 保存当前累计的帧数
        for child_dataset in datasets:
            child_index = child_dataset.task_type_frame_index
            if child_index is None:
                raise RuntimeError(
                    "Enabled task type sampling did not build a child frame index."
                )
            child_indices.append(child_index)
            # The index belongs to the main-process sampler, not DataLoader workers.
            child_dataset.task_type_frame_index = None
            offsets.append(offset)
            offset += len(child_dataset)
        task_type_index = (
            child_indices[0]
            if len(child_indices) == 1
            else merge_task_type_frame_indices(child_indices, offsets)
        )
        if task_type_index.frame_count != len(dataset):
            raise ValueError(
                f"Task type frame count mismatch for {name!r}: "
                f"{task_type_index.frame_count} vs {len(dataset)}."
            )
        task_type_plan = TaskTypeSamplingPlan.from_config(
            task_type_index,
            task_type_config,
        )

    info = {
        "name": name,
        "embodiment_tag": embodiment_tag,
        "weight": float(_cfg_get(dataset_cfg, "weight", 1.0)),
        "length": len(dataset),
        "num_roots": len(datasets),
        "data_paths_preview": resolved_data_paths[:5],
        "video_keys": video_keys,
        "state_keys": state_keys,
        "action_keys": action_keys,
        "language_keys": language_keys,
        "_task_type_plan": task_type_plan,
        "task_type_sampling": {
            "enabled": task_type_config.enabled,
            "label_space": task_type_config.label_space,
            "episode_counts": (
                task_type_index.episode_counts if task_type_index else {}
            ),
            "frame_counts": task_type_index.frame_counts if task_type_index else {},
            "effective_weights": (
                task_type_plan.effective_weights if task_type_plan else {}
            ),
            "missing_rows": 0,
            "extra_rows": task_type_index.extra_rows if task_type_index else 0,
            "unavailable_types": (
                list(task_type_plan.unavailable_types) if task_type_plan else []
            ),
            "disabled_types": (
                list(task_type_plan.disabled_types) if task_type_plan else []
            ),
        },
    }
    logger.info(
        "DreamZero mixed dataset[%s]: name=%s embodiment=%s weight=%s length=%s "
        "num_roots=%s paths=%s video_keys=%s state_keys=%s action_keys=%s language_keys=%s",
        dataset_idx,
        name,
        embodiment_tag,
        info["weight"],
        info["length"],
        info["num_roots"],
        info["data_paths_preview"],
        video_keys,
        state_keys,
        action_keys,
        language_keys,
    )
    task_type_report = info["task_type_sampling"]
    logger.info(
        "[DREAMZERO_TASK_TYPE_SAMPLING] dataset=%s enabled=%s label_space=%s "
        "roots=%s paths=%s "
        "episode_counts=%s frame_counts=%s effective_weights=%s "
        "missing=%s extra=%s unavailable=%s disabled=%s",
        name,
        task_type_report["enabled"],
        task_type_report["label_space"],
        info["num_roots"],
        info["data_paths_preview"],
        task_type_report["episode_counts"],
        task_type_report["frame_counts"],
        task_type_report["effective_weights"],
        task_type_report["missing_rows"],
        task_type_report["extra_rows"],
        task_type_report["unavailable_types"],
        task_type_report["disabled_types"],
    )
    return dataset, info


def build_mixed_dreamzero_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    eval_dataset: bool = False,
):
    """Build a weighted mixed DreamZero SFT dataloader.

    The v0 mixture path is intentionally narrow: it supports explicitly
    whitelisted 16D action contracts while keeping the legacy single-dataset
    builder intact.
    """

    if eval_dataset:
        raise ValueError(
            "DreamZero mixed dataloader is train-only. Keep eval/inference on a "
            "single explicit embodiment such as robotwin."
        )

    data_cfg = cfg.data
    model_cfg = cfg.actor.model
    dataset_cfgs = _plain_list(data_cfg.get("train_datasets", None))
    _validate_mixed_contract(model_cfg, dataset_cfgs)
    if any(
        str(_cfg_get(ds, "embodiment_tag", "")) in _VIDEO_ONLY_EMBODIMENTS
        for ds in dataset_cfgs
    ):
        action_cfg = model_cfg.action_head_cfg.config
        if bool(action_cfg.get("video_condition_on_action_state", True)):
            raise ValueError(
                "Video-only training requires "
                "video_condition_on_action_state: false."
            )
        if bool(action_cfg.get("use_past_action_context", False)):
            raise ValueError(
                "Video-only training requires use_past_action_context: false."
            )

    embodiment_tag_mapping = _merge_default_embodiment_mapping(model_cfg, dataset_cfgs)
    tokenizer_path = model_cfg.get("tokenizer_path", "google/umt5-xxl")
    max_seq_len = int(model_cfg.get("max_seq_len", 512))
    sampling_mode = data_cfg.get("sampling_mode", "multi_anchor")
    temporal_pad_lengths = None
    if sampling_mode == "multi_anchor":
        temporal_pad_lengths = _build_dreamzero_temporal_pad_lengths(
            num_frames=model_cfg.action_head_cfg.config.num_frames,
            max_chunk_size=(
                model_cfg.action_head_cfg.config.diffusion_model_cfg.max_chunk_size
            ),
            state_horizon=model_cfg.get("state_horizon", 1),
            action_horizon=model_cfg.action_horizon,
        )

    datasets: list[Dataset] = []
    infos: list[dict[str, Any]] = []
    for dataset_idx, dataset_cfg in enumerate(dataset_cfgs):
        dataset, info = _build_dataset_bundle(
            cfg,
            dataset_cfg,
            dataset_idx=dataset_idx,
            embodiment_tag_mapping=embodiment_tag_mapping,
            eval_dataset=eval_dataset,
        )
        datasets.append(dataset)
        infos.append(info)

    names = [info["name"] for info in infos]
    weights = [float(info["weight"]) for info in infos]
    lengths = [int(info["length"]) for info in infos]
    dataset = DreamZeroMixtureDataset(datasets, names)

    mixture_cfg = data_cfg.get("mixture", {}) or {}
    epoch_size = int(_cfg_get(mixture_cfg, "epoch_size", sum(lengths)))
    seed = int(
        _cfg_get(mixture_cfg, "seed", data_cfg.get("seed", cfg.actor.get("seed", 0)))
    )
    dataloader_seed = seed + rank
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(dataloader_seed)
    task_type_plans = [info.get("_task_type_plan") for info in infos]
    if not any(plan is not None for plan in task_type_plans):
        task_type_plans = None
    sampler = DistributedWeightedMixtureSampler(
        lengths,
        weights,
        epoch_size=epoch_size,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        drop_last=True,
        dataset_names=names,
        batch_size=cfg.actor.micro_batch_size,
        task_type_plans=task_type_plans,
    )

    num_workers = int(data_cfg.get("num_workers", 4))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 4))
    dataloader_profile = bool(data_cfg.get("dataloader_profile", False))
    dataloader_profile_first_n = int(data_cfg.get("dataloader_profile_first_n", 4))
    dataloader_profile_slow_s = float(data_cfg.get("dataloader_profile_slow_s", 2.0))

    logger.info(
        "DreamZero mixed dataloader: rank=%s/%s batch_size=%s epoch_size=%s "
        "per_rank_samples=%s sampler_seed=%s dataloader_seed=%s names=%s "
        "lengths=%s weights=%s mapping=%s",
        rank,
        world_size,
        cfg.actor.micro_batch_size,
        epoch_size,
        len(sampler),
        seed,
        dataloader_seed,
        names,
        lengths,
        weights,
        embodiment_tag_mapping,
    )
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
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
    return data_loader, {
        "num_samples": int(epoch_size),
        "mixture_names": names,
        "mixture_lengths": lengths,
        "mixture_weights": weights,
        "task_type_sampling": [
            info.get("task_type_sampling", {"enabled": False}) for info in infos
        ],
    }
