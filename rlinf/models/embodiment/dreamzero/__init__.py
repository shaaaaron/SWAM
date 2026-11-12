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

import json
from pathlib import Path

import torch
import torch.nn as nn
from groot.vla.data.transform import ComposedModalityTransform
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from rlinf.data.datasets.dreamzero.data_transforms import (
    build_dreamzero_composed_transform,
    load_dreamzero_dataset_metadata,
)
from rlinf.data.datasets.dreamzero.relative_actions import (
    ensure_relative_action_cache_compatible,
)
from rlinf.models.embodiment.dreamzero.dreamzero_config import DreamZeroConfig
from rlinf.models.embodiment.dreamzero.dreamzero_policy import DreamZeroPolicy
from rlinf.utils.logging import get_logger


def _configure_torch_dynamo_for_dreamzero_inference() -> None:
    """Raise Dynamo limits for DreamZero flow scheduler ``torch.compile`` during rollout.

    ``FlowUniPCMultistepScheduler.multistep_uni_p_bh_update`` is compiled with
    ``fullgraph=True, dynamic=False``. UniPC multistep order / history tensors can change
    rank across denoising steps (e.g. 3D action vs 5D video), which triggers recompiles.
    The default ``recompile_limit`` (8) is too low for embodied eval and raises
    ``FailOnRecompileLimitHit`` — same mitigation as ``dreamzero/eval_utils/serve_dreamzero_wan22.py``.
    """
    _dynamo = torch._dynamo.config
    if hasattr(_dynamo, "cache_size_limit"):
        _dynamo.cache_size_limit = max(getattr(_dynamo, "cache_size_limit", 8), 1000)
    if hasattr(_dynamo, "recompile_limit"):
        _dynamo.recompile_limit = max(getattr(_dynamo, "recompile_limit", 8), 800)
    if hasattr(_dynamo, "accumulated_cache_size_limit"):
        _dynamo.accumulated_cache_size_limit = max(
            getattr(_dynamo, "accumulated_cache_size_limit", 8), 1000
        )
    if hasattr(_dynamo, "accumulated_recompile_limit"):
        _dynamo.accumulated_recompile_limit = max(
            getattr(_dynamo, "accumulated_recompile_limit", 8), 2000
        )


def _promote_scalar_params_to_1d(model):
    """FSDP does not support 0-d parameters, so we promote scalar Parameters to shape=[1]."""
    scalar_param_names = [name for name, p in model.named_parameters() if p.ndim == 0]
    for full_name in scalar_param_names:
        if "." in full_name:
            module_name, param_name = full_name.rsplit(".", 1)
            module = model.get_submodule(module_name)
        else:
            module = model
            param_name = full_name

        old_p = getattr(module, param_name)
        new_p = nn.Parameter(
            old_p.detach().reshape(1),
            requires_grad=old_p.requires_grad,
        )
        setattr(module, param_name, new_p)


def _validate_checkpoint_weight_keys(
    checkpoint_config_mode: str,
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    """
    在yaml only情况下要求model完全一致
    """
    if checkpoint_config_mode != "yaml_only" or not (
        missing_keys or unexpected_keys
    ):
        return

    raise RuntimeError(
        "DreamZero yaml_only config is incompatible with checkpoint weights: "
        f"{len(missing_keys)} missing keys and "
        f"{len(unexpected_keys)} unexpected keys. "
        f"First missing keys: {missing_keys[:20]}; "
        f"first unexpected keys: {unexpected_keys[:20]}."
    )


def _align_checkpoint_tensor_shapes(model, state_dict):
    # 加载 checkpoint 前，检查权重形状
    model_state = model.state_dict()
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None or target.shape == value.shape:
            # - 形状相同，通过
            # - 不存在，交给后续的加载处理
            continue
        if target.numel() == value.numel() == 1:
            # 标量参数，跳过
            state_dict[key] = value.reshape(target.shape)
        else:
            # 形状不同，报错
            raise RuntimeError(
                f"Checkpoint shape mismatch for {key}: "
                f"got {tuple(value.shape)}, expected {tuple(target.shape)}."
            )


def get_model(cfg: DictConfig, torch_dtype=None):
    """Load DreamZero policy from checkpoint."""

    from rlinf.utils.patcher import Patcher

    _configure_torch_dynamo_for_dreamzero_inference()

    Patcher.clear()
    Patcher.add_patch(
        "groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAE",
        "rlinf.models.embodiment.dreamzero.patch.wan_video_vae.WanVideoVAE",
    )
    Patcher.add_patch(
        "groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAE38",
        "rlinf.models.embodiment.dreamzero.patch.wan_video_vae.WanVideoVAE38",
    )
    Patcher.add_patch(
        "groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAEStateDictConverter",
        "rlinf.models.embodiment.dreamzero.patch.wan_video_vae.WanVideoVAEStateDictConverter",
    )
    _dit_chunk = "groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk"
    Patcher.add_wrapper(
        f"{_dit_chunk}.CausalWanSelfAttention._process_clean_image_only",
        torch.compile(mode="reduce-overhead"),
    )
    Patcher.add_wrapper(
        f"{_dit_chunk}.CausalWanSelfAttention._process_state_blocks",
        torch.compile(mode="reduce-overhead"),
    )
    Patcher.add_wrapper(
        f"{_dit_chunk}.CausalWanSelfAttention._process_noisy_image_blocks",
        torch.compile(mode="reduce-overhead"),
    )
    Patcher.add_wrapper(
        f"{_dit_chunk}.CausalWanSelfAttention._process_noisy_action_blocks",
        torch.compile(mode="reduce-overhead"),
    )
    Patcher.add_patch(
        f"{_dit_chunk}.CausalWanModel._forward_train",
        "rlinf.models.embodiment.dreamzero.patch.wan_causal_model_forward_train._forward_train",
    )
    Patcher.apply()

    model_path = cfg.get("model_path", None)

    tokenizer_path = cfg.get("tokenizer_path", "google/umt5-xxl")

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config_dict, dict):
        raise ValueError(
            "DreamZero actor.model must resolve to a mapping after validate_sft_cfg()."
        )

    # RLinf loading policy belongs to the run config, not the HF model config.
    checkpoint_config_mode = config_dict.pop("checkpoint_config_mode", "merge")

    dreamzero_config = DreamZeroConfig(**config_dict)

    has_full_model_weights = False
    st = st_index = None
    if model_path is not None:
        ckpt = Path(model_path)
        st = ckpt / "model.safetensors"
        st_index = ckpt / "model.safetensors.index.json"
        has_full_model_weights = st.exists() or st_index.exists()

    # Disable defer_lora_injection for immediate loading
    if "config" in dreamzero_config.action_head_cfg and isinstance(
        dreamzero_config.action_head_cfg["config"], dict
    ):
        dreamzero_config.action_head_cfg["config"]["defer_lora_injection"] = False
        # If full DreamZero safetensors are absent, fall back to component loading from
        # WAN paths in checkpoint config or preset YAML (diffusion / text / image / vae).
        dreamzero_config.action_head_cfg["config"]["skip_component_loading"] = (
            has_full_model_weights
        )

    metadata = load_dreamzero_dataset_metadata(cfg)
    data_transforms = build_dreamzero_composed_transform(cfg, tokenizer_path)
    assert isinstance(data_transforms, ComposedModalityTransform), f"{data_transforms=}"
    data_transforms.set_metadata(metadata)
    data_transforms.eval()

    embodiment_tag = str(cfg.embodiment_tag)

    dreamzero_config.data_transforms = data_transforms
    dreamzero_config.embodiment_tag = embodiment_tag
    dreamzero_config.relative_action = bool(cfg.get("relative_action", False))
    # Legacy config compatibility: accepted but ignored by the s0-relative path.
    dreamzero_config.relative_action_per_horizon = bool(
        cfg.get("relative_action_per_horizon", False)
    )
    dreamzero_config.relative_action_keys = list(cfg.get("relative_action_keys") or [])
    dreamzero_config.relative_action_mode = cfg.get("relative_action_mode") or ""
    dreamzero_config.relative_action_pose_reference = cfg.get(
        "relative_action_pose_reference", "per_arm"
    )
    action_head_cfg = dreamzero_config.action_head_cfg or {}
    action_head_inner = action_head_cfg.get("config", {}) if hasattr(action_head_cfg, "get") else {}
    use_past_action_context = (
        bool(action_head_inner.get("use_past_action_context", False))
        if hasattr(action_head_inner, "get")
        else False
    )
    ensure_relative_action_cache_compatible(
        relative_action=dreamzero_config.relative_action,
        relative_action_mode=dreamzero_config.relative_action_mode,
        use_past_action_context=use_past_action_context,
    )

    model = DreamZeroPolicy(
        config=dreamzero_config,
    )

    # Load DreamZero full weights if available; otherwise keep component-initialized model.
    if has_full_model_weights and model_path is not None:
        state_dict = {}
        if st_index is not None and st_index.exists():
            with open(st_index, "r") as f:
                index = json.load(f)
            for shard_file in sorted(set(index["weight_map"].values())):
                state_dict.update(load_file(str(ckpt / shard_file)))
        elif st is not None and st.exists():
            state_dict.update(load_file(str(st)))
        if any(".base_layer." in k for k in state_dict):
            state_dict = {
                k.replace(".base_layer.", "."): v for k, v in state_dict.items()
            }

        # 对齐模型和参数形状
        _align_checkpoint_tensor_shapes(model, state_dict)
        incompatible = model.load_state_dict(state_dict, strict=False)
        _validate_checkpoint_weight_keys(
            checkpoint_config_mode,
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )
        del state_dict
    else:
        loc = str(model_path) if model_path is not None else "model_path=null"
        get_logger().warning(
            "No model.safetensors under %s; initializing DreamZero from component weights "
            "in config (set diffusion/text/image/vae paths in checkpoint config or preset).",
            loc,
        )
    if hasattr(model, "action_head"):
        ah = model.action_head
        if not hasattr(ah, "trt_engine"):
            ah.trt_engine = None
        if not hasattr(ah, "trt_context"):
            ah.trt_context = None
    _promote_scalar_params_to_1d(model)
    model = model.to(dtype=torch_dtype)

    return model
