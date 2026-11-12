# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DMuon integration helpers for the FSDP2 engine."""

from __future__ import annotations

import importlib
import math
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.config import torch_dtype_from_precision
from rlinf.hybrid_engines.fsdp.utils import (
    clip_grad_by_total_norm_,
    get_grad_norm,
)
from rlinf.utils.utils import get_rng_state, set_rng_state

_CHECKPOINT_FORMAT = "rlinf-dmuon-sharded-v2"
_CHECKPOINT_DIR = "dmuon_checkpoint"
_CHECKPOINT_MANIFEST = "manifest.pt"
_CHECKPOINT_RANK_FILE = "rank_{rank:05d}.pt"
_OPTIMIZER_FQN_ATTR = "_rlinf_fsdp_param_fqns"


@dataclass(frozen=True)
class DMuonSelection:
    """Summary of parameters selected for DMuon ownership."""

    names: tuple[str, ...]
    numel: int


@dataclass(frozen=True)
class CombinedGradNorm:
    """Regular and DMuon gradient norms before clipping."""

    regular: float
    muon: float
    total: float
    coefficient: float


def get_optimizer_type(cfg: DictConfig) -> str:
    """Normalize the actor optimizer name without changing legacy Adam configs."""

    raw = str(cfg.optim.get("optimizer", "adamw")).strip().lower()
    if raw in {"adam", "adamw"}:
        return "adamw"
    if raw == "dmuon":
        return "dmuon"
    raise ValueError(
        f"Unsupported FSDP optimizer {raw!r}; expected 'adam', 'adamw', or 'dmuon'."
    )


def _load_dmuon():
    try:
        return importlib.import_module("dmuon")
    except ImportError as exc:
        raise ImportError(
            "optimizer=dmuon requires the DMuon package on PYTHONPATH or in the "
            "active Python environment."
        ) from exc


def _dmuon_cfg(cfg: DictConfig) -> DictConfig:
    dmuon_cfg = cfg.optim.get("dmuon", None)
    if dmuon_cfg is None:
        raise ValueError("optimizer=dmuon requires actor.optim.dmuon configuration")
    return dmuon_cfg


def _require_fields(config: DictConfig, fields: tuple[str, ...]) -> None:
    missing = [name for name in fields if config.get(name, None) is None]
    if missing:
        raise ValueError(
            f"actor.optim.dmuon is missing required field(s): {', '.join(missing)}"
        )


def validate_dmuon_config(cfg: DictConfig) -> None:
    """Fail fast for engine features that do not cover DMuon-owned tensors."""

    if get_optimizer_type(cfg) != "dmuon":
        return

    strategy = str(cfg.fsdp_config.get("strategy", "fsdp2")).lower()
    if strategy != "fsdp2":
        raise ValueError("DMuon is supported only with FSDP2")
    if bool(cfg.get("enable_offload", False)) or bool(
        cfg.fsdp_config.get("cpu_offload", False)
    ):
        raise ValueError("DMuon offload is not supported")
    if bool(cfg.fsdp_config.get("grad_scaler", {}).get("enabled", False)):
        raise ValueError("DMuon requires fsdp_config.grad_scaler.enabled=false")
    if bool(cfg.fsdp_config.get("save_full_model_weights", True)):
        raise ValueError(
            "DMuon requires fsdp_config.save_full_model_weights=false; "
            "merge the sharded checkpoint explicitly for model export"
        )
    if int(cfg.optim.get("critic_warmup_steps", 0) or 0) > 0:
        raise ValueError("DMuon does not support critic_warmup_steps")

    dmuon_cfg = _dmuon_cfg(cfg)
    _require_fields(
        dmuon_cfg,
        (
            "lr",
            "momentum",
            "weight_decay",
            "ns_steps",
            "nesterov",
            "param_name_pattern",
            "expected_param_count",
            "master_dtype",
            "optim_dtype",
        ),
    )
    try:
        re.compile(str(dmuon_cfg.param_name_pattern))
    except re.error as exc:
        raise ValueError(
            f"Invalid actor.optim.dmuon.param_name_pattern: {exc}"
        ) from exc
    expected = int(dmuon_cfg.expected_param_count)
    if expected <= 0:
        raise ValueError("actor.optim.dmuon.expected_param_count must be positive")

    mixed_precision = cfg.fsdp_config.mixed_precision
    torch_dtype_from_precision(mixed_precision.param_dtype)
    torch_dtype_from_precision(mixed_precision.reduce_dtype)
    torch_dtype_from_precision(dmuon_cfg.master_dtype)
    torch_dtype_from_precision(dmuon_cfg.optim_dtype)


def dedicate_dmuon_params(
    model: nn.Module,
    cfg: DictConfig,
    device_mesh: DeviceMesh,
    logger: Any,
    device: Optional[torch.device] = None,
) -> DMuonSelection:
    """Select and dedicate configured matrix parameters before FSDP2 wrapping."""

    validate_dmuon_config(cfg)
    dmuon = _load_dmuon()
    dmuon_cfg = _dmuon_cfg(cfg)
    pattern = re.compile(str(dmuon_cfg.param_name_pattern))

    def predicate(name: str, param: nn.Parameter) -> bool:
        return (
            param.requires_grad
            and param.ndim == 2
            and pattern.fullmatch(name) is not None
        )

    matched = tuple(
        (name, param)
        for name, param in model.named_parameters()
        if predicate(name, param)
    )
    expected = int(dmuon_cfg.expected_param_count)
    if len(matched) != expected:
        examples = [name for name, _ in matched[:8]]
        raise ValueError(
            "DMuon parameter selection count mismatch: "
            f"expected={expected}, matched={len(matched)}, examples={examples}"
        )

    if device is not None:
        for name, param in matched:
            if param.is_meta:
                raise ValueError(
                    f"DMuon parameter {name!r} is still on the meta device"
                )
            if param.device != device:
                param.data = param.data.to(device=device)

    mesh_names = tuple(device_mesh.mesh_dim_names or ())
    shard_mesh = device_mesh["fsdp"] if "fsdp" in mesh_names else device_mesh
    replicate_mesh: Optional[DeviceMesh] = (
        device_mesh["ddp"] if "ddp" in mesh_names else None
    )

    mixed_precision = cfg.fsdp_config.mixed_precision
    param_dtype = torch_dtype_from_precision(mixed_precision.param_dtype)
    grad_dtype = torch_dtype_from_precision(mixed_precision.reduce_dtype)
    master_dtype = torch_dtype_from_precision(dmuon_cfg.master_dtype)
    optim_dtype = torch_dtype_from_precision(dmuon_cfg.optim_dtype)

    dmuon.dedicate_params(
        model,
        mesh=shard_mesh,
        replicate_mesh=replicate_mesh,
        predicate=predicate,
        compute_dtype=param_dtype,
        reshard_after_forward=bool(cfg.fsdp_config.reshard_after_forward),
        param_policy={
            "defaults": {
                "route": "muon",
                "param_dtype": param_dtype,
                "grad_dtype": grad_dtype,
                "master_dtype": master_dtype,
                "optim_dtype": optim_dtype,
            }
        },
        process_group_policy="isolated",
    )

    if replicate_mesh is not None:  # 在2d mesh中同步
        dmuon.broadcast_all_updates(model)

    selection = DMuonSelection(
        names=tuple(name for name, _ in matched),
        numel=sum(param.numel() for _, param in matched),
    )
    logger.info(
        "[FSDP][DMuon] dedicated %d matrices with %.3fB parameters",
        len(selection.names),
        selection.numel / 1.0e9,
    )
    return selection


def build_dmuon_optimizer(model: nn.Module, cfg: DictConfig) -> Optimizer:
    """Build DMuon for dedicated matrices and AdamW for remaining parameters."""

    dmuon = _load_dmuon()
    dmuon_cfg = _dmuon_cfg(cfg)
    value_head_params = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and "value_head" in name
    ]
    if value_head_params and float(cfg.optim.value_lr) != float(cfg.optim.lr):
        raise ValueError(
            "DMuon does not yet support a separate value_lr parameter group"
        )

    optimizer = dmuon.Muon(
        model,
        lr=float(dmuon_cfg.lr),
        momentum=float(dmuon_cfg.momentum),
        weight_decay=float(dmuon_cfg.weight_decay),
        ns_steps=int(dmuon_cfg.ns_steps),
        nesterov=bool(dmuon_cfg.nesterov),
        adamw_lr=float(cfg.optim.lr),
        adamw_betas=(
            float(cfg.optim.adam_beta1),
            float(cfg.optim.adam_beta2),
        ),
        adamw_weight_decay=float(cfg.optim.get("weight_decay", 1.0e-2)),
        adamw_eps=float(cfg.optim.get("adam_eps", 1.0e-8)),
    )

    model_fqns = {id(param): name for name, param in model.named_parameters()}
    fsdp_param_fqns = {}
    for param in optimizer._fsdp_params:
        fqn = model_fqns.get(id(param))
        if fqn is None:
            raise RuntimeError(
                "DMuon FSDP parameter is absent from model.named_parameters() "
                "while building the optimizer"
            )
        if fqn in fsdp_param_fqns.values():
            raise RuntimeError(f"Duplicate DMuon FSDP parameter FQN: {fqn}")
        fsdp_param_fqns[id(param)] = fqn
    setattr(optimizer, _OPTIMIZER_FQN_ATTR, fsdp_param_fqns)
    return optimizer


def dmuon_no_sync(
    model: nn.Module,
    *,
    enabled: bool,
    is_last_micro_batch: bool,
) -> ContextManager:
    """Disable dedicated gradient communication on non-final micro-batches."""

    if not enabled or is_last_micro_batch:
        return nullcontext()
    return _load_dmuon().no_sync(model)


def clip_dmuon_combined_grad_norm_(
    model: nn.Module,
    optimizer: Optimizer,
    *,
    dp_group: Optional[dist.ProcessGroup],
    max_grad_norm: float,
) -> CombinedGradNorm:
    """Clip regular and dedicated gradients with one global coefficient."""

    dmuon = _load_dmuon()
    muon_stats = dmuon.clip_grad_norm_(optimizer, None)
    muon_norm = float(muon_stats.total_norm)
    regular_norm = float(
        get_grad_norm(model.parameters(), dp_group=dp_group, norm_type=2.0)
    )
    total_norm = math.hypot(regular_norm, muon_norm)
    finite = math.isfinite(total_norm)
    coefficient = (
        min(1.0, float(max_grad_norm) / (total_norm + 1.0e-6))
        if finite
        else 0.0
    )

    if coefficient < 1.0 and finite:
        clip_grad_by_total_norm_(
            model.parameters(),
            max_grad_norm=float(max_grad_norm),
            total_norm=total_norm,
        )
        dmuon.clip_grad_norm_(
            optimizer,
            coefficient * (muon_norm + 1.0e-6),
        )

    return CombinedGradNorm(
        regular=regular_norm,
        muon=muon_norm,
        total=total_norm,
        coefficient=coefficient,
    )


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return value


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return getattr(tensor, "_local_tensor", tensor)


def _dedicated_params_by_fqn(model: nn.Module, dmuon: Any) -> dict[str, Any]:
    module_fqns = {id(module): name for name, module in model.named_modules()}
    result: dict[str, Any] = {}
    for param in dmuon.get_dedicated_params(model):
        module_id = id(param.module)
        if module_id not in module_fqns:
            raise RuntimeError(
                f"DMuon parameter parent module is absent from model: {param.param_name}"
            )
        prefix = module_fqns[module_id]
        fqn = f"{prefix}.{param.param_name}" if prefix else param.param_name
        if fqn in result:
            raise RuntimeError(f"Duplicate DMuon parameter FQN: {fqn}")
        result[fqn] = param
    return result


def _optimizer_regular_params_by_fqn(
    optimizer: Optimizer,
) -> dict[str, nn.Parameter]:
    param_fqns = getattr(optimizer, _OPTIMIZER_FQN_ATTR, None)
    if param_fqns is None:
        raise RuntimeError(
            "DMuon optimizer is missing the stable FSDP parameter FQN mapping"
        )

    result = {}
    for param in optimizer._fsdp_params:
        fqn = param_fqns.get(id(param))
        if fqn is None:
            raise RuntimeError(
                "DMuon optimizer contains an FSDP parameter without a stable FQN"
            )
        if fqn in result:
            raise RuntimeError(f"Duplicate DMuon optimizer FSDP parameter FQN: {fqn}")
        result[fqn] = param
    if len(result) != len(param_fqns):
        raise RuntimeError(
            "DMuon optimizer stable FQN mapping does not match its FSDP parameters"
        )
    return result


def _regular_params_by_fqn(
    model: nn.Module, dedicated_fqns: set[str]
) -> dict[str, nn.Parameter]:
    return {
        name: param
        for name, param in model.named_parameters()
        if name not in dedicated_fqns
    }


def _copy_checkpoint_tensor(
    target: torch.Tensor, source: torch.Tensor, *, label: str
) -> None:
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(
            f"DMuon checkpoint shape mismatch for {label}: "
            f"saved={tuple(source.shape)}, current={tuple(target.shape)}"
        )
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def _require_same_keys(label: str, saved: dict, current: dict) -> None:
    saved_keys = set(saved)
    current_keys = set(current)
    if saved_keys != current_keys:
        missing = sorted(current_keys - saved_keys)[:8]
        unexpected = sorted(saved_keys - current_keys)[:8]
        raise ValueError(
            f"DMuon checkpoint {label} keys mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _local_model_state(
    model: nn.Module,
    optimizer: Optimizer,
    dedicated: dict[str, Any],
) -> dict[str, dict]:
    dedicated_state: dict[str, dict[str, torch.Tensor]] = {}
    for fqn, param in dedicated.items():
        entry = {}
        for attr in ("_owned_data", "_sharded_adamw_data"):
            tensor = getattr(param, attr, None)
            if tensor is not None:
                entry[attr] = _clone_to_cpu(tensor)
        if entry:
            dedicated_state[fqn] = entry

    regular = _optimizer_regular_params_by_fqn(optimizer)
    return {
        "dedicated": dedicated_state,
        "regular": {
            fqn: _clone_to_cpu(_local_tensor(param))
            for fqn, param in regular.items()
        },
        "buffers": {
            fqn: _clone_to_cpu(_local_tensor(buffer))
            for fqn, buffer in model.named_buffers()
        },
    }


def _optimizer_param_groups(optimizer: Optimizer) -> list[dict]:
    return [
        _clone_to_cpu({key: value for key, value in group.items() if key != "params"})
        for group in optimizer.param_groups
    ]


def _local_optimizer_state(
    optimizer: Optimizer,
    dedicated: dict[str, Any],
) -> dict[str, dict]:
    dedicated_ids = {id(param): fqn for fqn, param in dedicated.items()}
    regular = _optimizer_regular_params_by_fqn(optimizer)
    regular_ids = {id(param): fqn for fqn, param in regular.items()}
    dedicated_state = {}
    regular_state = {}

    for key, state in optimizer.state.items():
        if isinstance(key, int) and key in dedicated_ids:
            dedicated_state[dedicated_ids[key]] = _clone_to_cpu(state)
        elif id(key) in regular_ids:
            regular_state[regular_ids[id(key)]] = _clone_to_cpu(state)
        elif state:
            raise RuntimeError(
                "DMuon optimizer contains state for an unknown parameter key: "
                f"{key!r}"
            )

    return {
        "dedicated": dedicated_state,
        "regular": regular_state,
    }


def _checkpoint_model_metadata(
    model: nn.Module,
    optimizer: Optimizer,
    dedicated: dict[str, Any],
) -> dict[str, dict]:
    regular = _optimizer_regular_params_by_fqn(optimizer)
    return {
        "dedicated": {
            fqn: {
                "shape": tuple(param._orig_size),
                "owner_rank": tuple(param.owner_rank),
                "route": getattr(param, "_dmuon_route", None),
            }
            for fqn, param in dedicated.items()
        },
        "regular": {
            fqn: {
                "shape": tuple(param.shape),
                "is_sharded": hasattr(param, "_local_tensor"),
            }
            for fqn, param in regular.items()
        },
        "buffers": {
            fqn: {"shape": tuple(buffer.shape)}
            for fqn, buffer in model.named_buffers()
        },
    }


def _checkpoint_device_mesh(device_mesh: DeviceMesh) -> dict[str, Any]:
    return {
        "shape": tuple(device_mesh.mesh.shape),
        "dim_names": tuple(device_mesh.mesh_dim_names or ()),
        "rank_layout": device_mesh.mesh.tolist(),
    }


def _validate_checkpoint_device_mesh(
    saved_device_mesh: Any,
    device_mesh: DeviceMesh,
) -> None:
    if saved_device_mesh is None:
        raise ValueError("DMuon checkpoint manifest is missing device_mesh")
    current_device_mesh = _checkpoint_device_mesh(device_mesh)
    if saved_device_mesh != current_device_mesh:
        raise ValueError(
            "DMuon checkpoint device mesh mismatch: "
            f"saved={saved_device_mesh}, current={current_device_mesh}"
        )


def _restore_state_value(
    value: Any,
    *,
    device: torch.device,
    reference: Any = None,
) -> Any:
    if isinstance(value, torch.Tensor):
        if isinstance(reference, torch.Tensor):
            return value.to(device=reference.device, dtype=reference.dtype)
        return value.to(device=device)
    if isinstance(value, dict):
        reference_dict = reference if isinstance(reference, dict) else {}
        return {
            key: _restore_state_value(
                item,
                device=device,
                reference=reference_dict.get(key),
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _restore_state_value(item, device=device) for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _restore_state_value(item, device=device) for item in value
        )
    return value


def _restore_model_state(
    model: nn.Module,
    optimizer: Optimizer,
    dedicated: dict[str, Any],
    saved: dict[str, dict],
) -> None:
    expected_dedicated = {}
    for fqn, param in dedicated.items():
        attrs = {
            attr: tensor
            for attr in ("_owned_data", "_sharded_adamw_data")
            if (tensor := getattr(param, attr, None)) is not None
        }
        if attrs:
            expected_dedicated[fqn] = attrs

    _require_same_keys(
        "local dedicated model", saved["dedicated"], expected_dedicated
    )
    regular = _optimizer_regular_params_by_fqn(optimizer)
    buffers = dict(model.named_buffers())
    _require_same_keys("local regular model", saved["regular"], regular)
    _require_same_keys("model buffers", saved["buffers"], buffers)

    with torch.no_grad():
        for fqn, attrs in expected_dedicated.items():
            _require_same_keys(
                f"dedicated model parameter {fqn}",
                saved["dedicated"][fqn],
                attrs,
            )
            for attr, target in attrs.items():
                _copy_checkpoint_tensor(
                    target,
                    saved["dedicated"][fqn][attr],
                    label=f"{fqn}.{attr}",
                )
        for fqn, param in regular.items():
            _copy_checkpoint_tensor(
                _local_tensor(param), saved["regular"][fqn], label=fqn
            )
        for fqn, buffer in buffers.items():
            _copy_checkpoint_tensor(
                _local_tensor(buffer), saved["buffers"][fqn], label=fqn
            )


def _restore_optimizer_state(
    model: nn.Module,
    optimizer: Optimizer,
    dedicated: dict[str, Any],
    saved: dict[str, dict],
) -> None:
    regular = _optimizer_regular_params_by_fqn(optimizer)
    unexpected_dedicated = set(saved["dedicated"]) - set(dedicated)
    unexpected_regular = set(saved["regular"]) - set(regular)
    if unexpected_dedicated or unexpected_regular:
        raise ValueError(
            "DMuon checkpoint optimizer keys mismatch: "
            f"unexpected_dedicated={sorted(unexpected_dedicated)[:8]}, "
            f"unexpected_regular={sorted(unexpected_regular)[:8]}"
        )

    dedicated_references = {
        fqn: optimizer.state.get(id(param), {})
        for fqn, param in dedicated.items()
    }
    regular_references = {
        fqn: optimizer.state.get(param, {}) for fqn, param in regular.items()
    }
    optimizer.state.clear()

    for fqn, state in saved["dedicated"].items():
        param = dedicated[fqn]
        optimizer.state[id(param)] = _restore_state_value(
            state,
            device=param.device,
            reference=dedicated_references[fqn],
        )
    for fqn, state in saved["regular"].items():
        param = regular[fqn]
        local = _local_tensor(param)
        optimizer.state[param] = _restore_state_value(
            state,
            device=local.device,
            reference=regular_references[fqn],
        )


def _restore_param_groups(optimizer: Optimizer, saved_groups: list[dict]) -> None:
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError(
            "DMuon checkpoint optimizer group count mismatch: "
            f"saved={len(saved_groups)}, current={len(optimizer.param_groups)}"
        )
    for index, (saved, current) in enumerate(
        zip(saved_groups, optimizer.param_groups)
    ):
        for key in ("use_muon", "subgroup_type"):
            if key in saved and key in current and saved[key] != current[key]:
                raise ValueError(
                    "DMuon checkpoint optimizer group structure mismatch at "
                    f"index {index}: {key} saved={saved[key]!r}, "
                    f"current={current[key]!r}"
                )
        current.update(saved)


def save_dmuon_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    save_path: str,
    *,
    device_mesh: Optional[DeviceMesh] = None,
    optimizer_steps: int,
    training_step: int,
    save_full_model_weights: bool,
) -> None:
    """Save a same-topology DMuon checkpoint without gathering full state."""

    if save_full_model_weights:
        raise ValueError(
            "DMuon does not save full model weights during training; merge the "
            "sharded checkpoint explicitly for model export"
        )
    if device_mesh is None:
        raise ValueError("DMuon checkpoint save requires the current device mesh")

    dmuon = _load_dmuon()
    dmuon.wait_all_replicate_broadcasts(model)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    checkpoint_dir = os.path.join(save_path, _CHECKPOINT_DIR)
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    dist.barrier()

    dedicated = _dedicated_params_by_fqn(model, dmuon)
    local_payload = {
        "format": _CHECKPOINT_FORMAT,
        "rank": rank,
        "world_size": world_size,
        "model": _local_model_state(model, optimizer, dedicated),
        "optimizer": _local_optimizer_state(optimizer, dedicated),
        "rng": get_rng_state(),
    }
    rank_file = os.path.join(
        checkpoint_dir, _CHECKPOINT_RANK_FILE.format(rank=rank)
    )
    temporary_rank_file = f"{rank_file}.tmp"
    torch.save(local_payload, temporary_rank_file)
    os.replace(temporary_rank_file, rank_file)
    dist.barrier()

    if rank == 0:
        manifest = {
            "format": _CHECKPOINT_FORMAT,
            "world_size": world_size,
            "dmuon_version": getattr(dmuon, "__version__", "unknown"),
            "rank_files": [
                _CHECKPOINT_RANK_FILE.format(rank=index)
                for index in range(world_size)
            ],
            "device_mesh": _checkpoint_device_mesh(device_mesh),
            "model_metadata": _checkpoint_model_metadata(
                model, optimizer, dedicated
            ),
            "optimizer_param_groups": _optimizer_param_groups(optimizer),
            "lr_scheduler": _clone_to_cpu(lr_scheduler.state_dict()),
            "optimizer_steps": int(optimizer_steps),
            "training_step": int(training_step),
        }
        manifest_file = os.path.join(checkpoint_dir, _CHECKPOINT_MANIFEST)
        temporary_manifest = f"{manifest_file}.tmp"
        torch.save(manifest, temporary_manifest)
        os.replace(temporary_manifest, manifest_file)
    dist.barrier()


def load_dmuon_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    load_path: str,
    *,
    device_mesh: DeviceMesh,
) -> tuple[int, int]:
    """Load a same-topology DMuon checkpoint and return step counters."""

    checkpoint_dir = os.path.join(load_path, _CHECKPOINT_DIR)
    manifest_file = os.path.join(checkpoint_dir, _CHECKPOINT_MANIFEST)
    if not os.path.isfile(manifest_file):
        raise FileNotFoundError(
            f"DMuon checkpoint manifest not found: {manifest_file}"
        )
    manifest = torch.load(manifest_file, map_location="cpu", weights_only=False)
    if manifest.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported DMuon checkpoint format: {manifest.get('format')!r}"
        )
    _validate_checkpoint_device_mesh(manifest.get("device_mesh"), device_mesh)

    world_size = dist.get_world_size()
    saved_world_size = int(manifest.get("world_size", -1))
    if saved_world_size != world_size:
        raise ValueError(
            "DMuon checkpoint world size mismatch: "
            f"saved={saved_world_size}, current={world_size}"
        )

    rank = dist.get_rank()
    rank_files = manifest.get("rank_files", [])
    if len(rank_files) != world_size:
        raise ValueError(
            "DMuon checkpoint rank file count mismatch: "
            f"saved={len(rank_files)}, current={world_size}"
        )
    rank_file = os.path.join(checkpoint_dir, rank_files[rank])
    if not os.path.isfile(rank_file):
        raise FileNotFoundError(f"DMuon rank checkpoint not found: {rank_file}")
    local_payload = torch.load(rank_file, map_location="cpu", weights_only=False)
    if local_payload.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(
            "Unsupported DMuon rank checkpoint format: "
            f"{local_payload.get('format')!r}"
        )
    if int(local_payload.get("rank", -1)) != rank:
        raise ValueError(
            "DMuon rank checkpoint mismatch: "
            f"saved={local_payload.get('rank')}, current={rank}"
        )
    if int(local_payload.get("world_size", -1)) != world_size:
        raise ValueError(
            "DMuon rank checkpoint world size mismatch: "
            f"saved={local_payload.get('world_size')}, current={world_size}"
        )

    dmuon = _load_dmuon()
    dedicated = _dedicated_params_by_fqn(model, dmuon)
    current_metadata = _checkpoint_model_metadata(model, optimizer, dedicated)
    if manifest.get("model_metadata") != current_metadata:
        raise ValueError("DMuon checkpoint model topology or FQN metadata mismatch")

    _restore_model_state(model, optimizer, dedicated, local_payload["model"])
    _restore_optimizer_state(
        model, optimizer, dedicated, local_payload["optimizer"]
    )
    _restore_param_groups(optimizer, manifest["optimizer_param_groups"])
    lr_scheduler.load_state_dict(manifest["lr_scheduler"])
    set_rng_state(local_payload["rng"])
    dist.barrier()
    return int(manifest.get("optimizer_steps", 0)), int(
        manifest.get("training_step", 0)
    )
