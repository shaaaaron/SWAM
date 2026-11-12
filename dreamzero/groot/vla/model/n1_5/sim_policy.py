import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
from tianshou.data import Batch
from tianshou.policy import BasePolicy as BaseTianshouPolicy
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
import tree
import time

from groot.vla.data.schema import DatasetMetadata, EmbodimentTag
from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.data.relative_actions import (
    ROBOTWIN_EEF_ACTION_KEY,
    ROBOTWIN_EEF_DIM,
    ROBOTWIN_EEF_STATE_KEY,
    ensure_relative_action_cache_compatible,
    ensure_relative_action_metadata_compatible,
    ensure_robotwin_eef_dim,
    is_robotwin_eef_se3_chunk_relative,
    is_robotwin_eef_se3_relative_action,
    is_robotwin_eef_se3_s0_relative_action,
    is_robotwin_eef_se3_s0_relative_state,
    resolve_relative_action_pose_reference,
    robotwin_eef_abs_to_rel_wxyz,
    robotwin_eef_rel_to_abs_wxyz,
)


class ModelManager:
    """
    Manages model loading/offloading to handle memory efficiently when using multiple models.
    Modified to keep VLM always loaded and only manage text_encoder/vae components.
    """
    def __init__(self):
        self.active_components = None  # Track which components are active for action_head
        self.models = {}
        self.vlm_policy = None  # Keep VLM always loaded
        self.action_head_policy = None  # Action head policy for component management

    def register_model(self, name: str, policy_instance):
        """Register a policy instance with the manager."""
        self.models[name] = policy_instance

        if name == "vlm":
            self.vlm_policy = policy_instance
            # Load VLM to GPU and keep it there
            self.load_vlm_model()
        elif name == "action_head":
            self.action_head_policy = policy_instance
            # Ensure vram management is enabled for action head
            self.enable_action_head_vram_management()
            # Initially offload action head components to save memory
            self.offload_action_head_components()

    def load_vlm_model(self):
        """Load VLM model to GPU and keep it there."""
        if self.vlm_policy is None:
            return

        policy = self.vlm_policy
        print(f"Loading VLM model to GPU (keeping it loaded)...")

        # Move VLM model to GPU
        policy.trained_model.to(device=policy.device)

        # Apply bf16 if needed
        if policy.eval_bf16:
            policy.trained_model = policy.trained_model.to(dtype=torch.bfloat16)

        torch.cuda.empty_cache()  # Clear cache after loading

    def activate_model(self, name: str):
        """Activate a model - for VLM this is a no-op, for action_head this manages components."""
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")

        if name == "vlm":
            # VLM is always loaded, just ensure it's ready
            if self.vlm_policy:
                self.vlm_policy.ensure_model_on_gpu()
            return
        elif name == "action_head":
            # For action head, activate the text_encoder and vae components
            self.activate_action_head_components(['text_encoder', 'vae'])
            self.active_components = ['text_encoder', 'vae']

    def activate_action_head_components(self, component_names):
        """Activate specific components of the action head model."""
        if self.action_head_policy is None:
            return

        policy = self.action_head_policy
        print(f"Loading action head components {component_names} to GPU...")

        # Check if the action head has the load_models_to_device method
        if (hasattr(policy.trained_model, "action_head") and
            hasattr(policy.trained_model.action_head, "load_models_to_device")):
            # Use the selective loading method
            policy.trained_model.action_head.load_models_to_device(component_names)
        else:
            # Fallback to loading the entire action head
            if hasattr(policy.trained_model, "action_head") and hasattr(policy.trained_model.action_head, "image_encoder"):
                policy.trained_model.action_head.enable_vram_management()
            else:
                policy.trained_model.to(device=policy.device)

        # Apply bf16 if needed
        if policy.eval_bf16:
            policy.trained_model = policy.trained_model.to(dtype=torch.bfloat16)

        torch.cuda.empty_cache()  # Clear cache after loading

    def offload_action_head_components(self):
        """Offload action head components to CPU."""
        if self.action_head_policy is None:
            return

        policy = self.action_head_policy
        print(f"Offloading action head components to CPU...")

        # Check if the action head has the load_models_to_device method
        if (hasattr(policy.trained_model, "action_head") and
            hasattr(policy.trained_model.action_head, "load_models_to_device")):
            # Use the selective offloading method - pass empty list to offload all
            policy.trained_model.action_head.load_models_to_device([])
        else:
            # Fallback to offloading the entire action head
            if hasattr(policy.trained_model, "action_head") and hasattr(policy.trained_model.action_head, "image_encoder"):
                if hasattr(policy.trained_model.action_head, 'disable_vram_management'):
                    policy.trained_model.action_head.disable_vram_management()
                policy.trained_model.to(device='cpu')
            else:
                policy.trained_model.to(device='cpu')

        torch.cuda.empty_cache()  # Clear cache after offloading
        self.active_components = None

    def load_model(self, name: str):
        """Load a model to GPU. (Deprecated - use activate_model instead)"""
        print(f"Warning: load_model is deprecated. Use activate_model instead.")
        self.activate_model(name)

    def offload_model(self, name: str):
        """Offload a model to CPU. (Deprecated - components are managed automatically)"""
        if name == "action_head":
            self.offload_action_head_components()
        # VLM is never offloaded in this new approach

    def get_status(self):
        """Get current status of model manager for debugging."""
        status = {
            "vlm_loaded": self.vlm_policy is not None,
            "action_head_available": self.action_head_policy is not None,
            "active_components": self.active_components
        }
        return status

    def enable_action_head_vram_management(self):
        """Enable vram management for the action head model."""
        if self.action_head_policy is None:
            return

        policy = self.action_head_policy

        # Check if the action head has vram management capability
        if (hasattr(policy.trained_model, "action_head") and
            hasattr(policy.trained_model.action_head, "enable_vram_management")):
            print("Enabling vram management for action head...")
            policy.trained_model.action_head.enable_vram_management()
        else:
            print("Warning: Action head does not support vram management")


class BaseGrootSimPolicy(BaseTianshouPolicy):
    def __init__(self, embodiment_tag: EmbodimentTag, model_path: str, device: int | str):
        super().__init__()
        self.embodiment_tag = embodiment_tag
        self.model_path = model_path
        self.device = device

    def forward(self, batch, state=None, **kwargs):
        raise NotImplementedError

    @property
    def video_delta_indices(self) -> np.ndarray:
        return np.array([0])

    @property
    def state_delta_indices(self) -> np.ndarray:
        return np.array([0])

    @property
    def raw_data_image_transform(self) -> Callable:
        return lambda x: x

    def on_env_init(self, env: gym.Env):
        pass

    def learn(self, batch: Batch, **kwargs) -> dict:
        """Dummy learn method for BasePolicy.learn as this is an inference-only wrapper.

        Args:
            batch: Input batch of experiences
            **kwargs: Additional arguments

        Returns:
            Empty dict as no learning occurs
        """
        return {}


def _update_tokenizer_path_in_config(cfg, new_path: str) -> None:
    """Update tokenizer_path in the transforms subtree only (avoids trainer.model etc.)."""
    from omegaconf import DictConfig, ListConfig
    if isinstance(cfg, DictConfig):
        if "tokenizer_path" in cfg:
            cfg.tokenizer_path = new_path
        # Only recurse via "transforms" to avoid triggering resolution of trainer, etc.
        if "transforms" not in cfg:
            return
        sub = cfg.transforms
        if isinstance(sub, DictConfig):
            for v in sub.values():
                _update_tokenizer_path_in_config(v, new_path)
        elif isinstance(sub, ListConfig):
            for v in sub:
                _update_tokenizer_path_in_config(v, new_path)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            _update_tokenizer_path_in_config(v, new_path)


class GrootSimPolicy(BaseGrootSimPolicy):
    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        device: int | str,
        model_config_overrides: list[str] | None = [],
        tokenizer_path_override: str | None = None,
        skip_assert_delta_indices: bool = False,
        skip_img_transform: bool = False,
        lazy_load: bool = False,
        device_mesh: DeviceMesh | None = None,
    ):
        """
        Initialize the GrootSimPolicy.

        Args:
            env_name (str): The name of the environment.
            model_path (str): Path to the model checkpoint.
            device (int | str): Device to run the model on.
            lazy_load (bool): If True, don't load model to GPU immediately.
            device_mesh (DeviceMesh | None): Device mesh to parallelize the model across.
        """
        super().__init__(embodiment_tag=embodiment_tag, model_path=model_path, device=device)
        model_dir = Path(model_path)
        self.rank = dist.get_rank()

        exp_cfg_dir = model_dir / "experiment_cfg"
        train_cfg_path = exp_cfg_dir / "conf.yaml"
        train_cfg = OmegaConf.load(train_cfg_path)
        if tokenizer_path_override is not None:
            _update_tokenizer_path_in_config(train_cfg, tokenizer_path_override)
        self.train_cfg = train_cfg
        self.lazy_load = lazy_load

        # Store model loading parameters for lazy loading
        self.model_config_overrides = model_config_overrides
        self.model_dir = model_dir

        # 1. Load the model
        if (
            train_cfg.model._target_.endswith(".from_pretrained")
            or train_cfg.model._target_.endswith(".from_pretrained_for_tuning")
            or train_cfg.model._target_.endswith(".from_pretrained_with_wrapped_action_head")
        ):
            # Compatibility with the finetuned model
            model_target = train_cfg.model._target_.rsplit(".", 1)[0]
        else:
            model_target = train_cfg.model._target_

        self.model_target = model_target

        if model_config_overrides is not None and len(model_config_overrides) != 0:
            print(f"Applying model config overrides: {model_config_overrides}")
            # Only apply new logic if there are config overrides

            # Import the model class from the target module path
            module_path, class_name = model_target.rsplit(".", 1)
            if "lora" in class_name:
                module_path, class_name = module_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            model_config_class = model_class.config_class

            # Load the model config from the model directory
            model_config = json.load(open(model_dir / "config.json", "r"))
            model_config = OmegaConf.create(model_config)
            model_config.merge_with_dotlist(list(model_config_overrides))
            model_config = OmegaConf.to_container(
                model_config, resolve=True
            )  # need dict for later model instantiation
            model_config = model_config_class.from_dict(model_config)

            # Instantiate the model
            if hasattr(train_cfg, "save_lora_only") and train_cfg.save_lora_only is True:
                print(f"Loading LoRA weights from pretrained")
                model = model_class.load_lora(model_path)
            else:
                print(f"Loading model from pretrained directly")
                model = model_class.from_pretrained(model_path, config=model_config)
        else:
            print(f"No model config overrides provided")
            # Otherwise, just call from_pretrained directly
            cls_module, cls_name = model_target.rsplit(".", 1)
            if 'lora' in cls_name:
                cls_module, cls_name = cls_module.rsplit(".", 1)
            if hasattr(train_cfg, "save_lora_only") and train_cfg.save_lora_only is True:
                print(f"Loading LoRA weights from pretrained")
                cls = getattr(importlib.import_module(cls_module), cls_name)
                model = cls.load_lora(model_path)
            else:
                print(f"Loading model from pretrained directly")
                cls = getattr(importlib.import_module(cls_module), cls_name)
                from_pretrained = getattr(cls, "from_pretrained")
                model = from_pretrained(model_path)

        model.eval()
        model.requires_grad_(False)
        if model.action_head.train_architecture == "lora":
            print(f"Merging LoRA weights into main model weights")
            # Merge the LoRA weights into the main model weights, and delete the LoRA
            # weights to save memory. Note that the WanModel is in model.action_head.model.
            model.action_head.model = model.action_head.model.merge_and_unload()
        else:
            print(f"Skipping merging LoRA weights into main model weights")

        self.eval_bf16 = self.train_cfg.get("eval_bf16", False)
        if self.eval_bf16 and not lazy_load:
            model = model.to(dtype=torch.bfloat16)

        # Store model initially on CPU if lazy loading
        if lazy_load:
            model.to(device='cpu')
        else:
            model.to(device=device)

        # Post initialize, move RoPE freqs to cuda.
        model.post_initialize()

        # Parallelize the model across devices.
        try:
            model.parallelize(device_mesh=device_mesh)
        except Exception as e:
            print("Skipping parallelization")

        torch.cuda.empty_cache()

        self.trained_model = model
        self.relative_action_mode = self.train_cfg.get("relative_action_mode") or ""
        self.relative_action_pose_reference = resolve_relative_action_pose_reference(
            self.train_cfg.get("relative_action_pose_reference"),
            relative_action_mode=(
                self.relative_action_mode
                if bool(self.train_cfg.get("relative_action", False))
                else None
            ),
        )
        self.relative_action_keys = list(self.train_cfg.get("relative_action_keys", []) or [])
        if bool(self.train_cfg.get("relative_action", False)) and not is_robotwin_eef_se3_relative_action(
            self.relative_action_mode
        ):
            raise ValueError(
                "relative_action=True requires "
                "a supported RoboTwin relative action mode."
            )
        if is_robotwin_eef_se3_relative_action(self.relative_action_mode):
            if not bool(self.train_cfg.get("relative_action", False)):
                raise ValueError(
                    "RoboTwin relative action modes require "
                    "relative_action=True."
                )
            if self.relative_action_keys != ["eef_position"]:
                raise ValueError(
                    "RoboTwin s0-relative action modes require "
                    "relative_action_keys=['eef_position']."
                )
        use_past_action_context = bool(
            getattr(
                getattr(getattr(self.trained_model, "action_head", None), "config", None),
                "use_past_action_context",
                False,
            )
        )
        ensure_relative_action_cache_compatible(
            relative_action=bool(self.train_cfg.get("relative_action", False)),
            relative_action_mode=self.relative_action_mode,
            use_past_action_context=use_past_action_context,
        )
        self._robotwin_eef_s0_window_anchor: np.ndarray | None = None
        self._robotwin_eef_s0_window_session_id: str | None = None
        self._robotwin_eef_s0_window_prompt: str | None = None

        # 2. Load the action, video, and state transforms
        # 2.1. Load the metadata for normalization stats
        # We have an assumption: one policy is only for rolling out one type of env, i.e., one embodiment_tag
        # metadata_versions = train_cfg.metadata_versions
        # metadata = get_metadata(self.embodiment_tag, metadata_versions[self.embodiment_tag.value])
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)
        if "gr1_unified_offline_rl" in metadatas and self.embodiment_tag.value == "gr1_unified":
            self.embodiment_tag = EmbodimentTag.GR1_UNIFIED_OFFLINE_RL
        metadata_entry = metadatas[self.embodiment_tag.value]
        ensure_relative_action_metadata_compatible(
            embodiment_tag=self.embodiment_tag.value,
            relative_action=bool(self.train_cfg.get("relative_action", False)),
            relative_action_mode=self.relative_action_mode,
            relative_action_pose_reference=self.relative_action_pose_reference,
            metadata_relative_action_mode=metadata_entry.get("relative_action_mode"),
            metadata_relative_action_pose_reference=metadata_entry.get(
                "relative_action_pose_reference"
            ),
            metadata_path=metadata_path,
        )
        metadata = DatasetMetadata.model_validate(metadata_entry)

        # If the model's action head has target_video_height/width (e.g. DreamZero Wan 5B), use that
        # as the expected video resolution so the transform matches the model. metadata.json can
        # otherwise contain a different resolution (e.g. 180x320) from dataset config.
        if hasattr(self.trained_model, "action_head") and hasattr(
            self.trained_model.action_head, "config"
        ):
            cfg = self.trained_model.action_head.config
            target_h = getattr(cfg, "target_video_height", None)
            target_w = getattr(cfg, "target_video_width", None)
            if target_h is not None and target_w is not None and metadata.modalities.video:
                for key in metadata.modalities.video.keys():
                    metadata.modalities.video[key].resolution = (int(target_w), int(target_h))

        # 2.2. Get the eval transforms
        assert (
            self.embodiment_tag.value in train_cfg.transforms
        ), f"{self.embodiment_tag.value=}, {train_cfg.transforms.keys()=}"
        eval_transform_cfg = train_cfg.transforms[self.embodiment_tag.value]
        if skip_img_transform:
            for t in eval_transform_cfg.transforms:
                if t._target_ == "groot.vla.data.transform.VideoCrop":
                    scale = t.scale
                    for key in metadata.modalities.video.keys():
                        metadata.modalities.video[key].resolution = (
                            int(metadata.modalities.video[key].resolution[0] * scale),
                            int(metadata.modalities.video[key].resolution[1] * scale),
                        )
                elif t._target_ == "groot.vla.data.transform.VideoResize":
                    height, width = t.height, t.width
                    for key in metadata.modalities.video.keys():
                        metadata.modalities.video[key].resolution = (height, width)

            # assume it's always ComposedModalityTransform
            assert (
                eval_transform_cfg._target_ == "groot.vla.data.transform.ComposedModalityTransform"
            )
            skipped_transforms = [
                "groot.vla.data.transform.VideoCrop",
                "groot.vla.data.transform.VideoResize",
                "groot.vla.data.transform.VideoColorJitter",
            ]
            eval_transform_cfg.transforms = [
                t for t in eval_transform_cfg.transforms if t._target_ not in skipped_transforms
            ]

        eval_transform = instantiate(train_cfg.transforms[self.embodiment_tag.value])
        assert isinstance(eval_transform, ComposedModalityTransform), f"{eval_transform=}"
        eval_transform.set_metadata(metadata)

        eval_transform.eval()
        self.eval_transform = eval_transform

        # 3. Load horizons needed
        if self.embodiment_tag.value in train_cfg.modality_configs:
            self.modality_configs = instantiate(
                train_cfg.modality_configs[self.embodiment_tag.value]
            )
        else:
            self.modality_configs = instantiate(train_cfg.modality_configs)

        self._video_delta_indices = np.array(self.modality_configs.video.eval_delta_indices)
        # self._video_delta_indices = np.array([0])
        # self.assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)

        # We might not use `state`, which will be a vision-only policy
        if "state" in self.modality_configs:
            self._state_delta_indices = np.array(self.modality_configs.state.eval_delta_indices)
            if not skip_assert_delta_indices:
                self.assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None
        self._raw_data_image_transform = None
        self._episode_first_session_id = None
        self._episode_first_images = None

    def offload_to_cpu(self):
        """Offload the model to CPU to free GPU memory."""
        if hasattr(self.trained_model, "action_head") and hasattr(self.trained_model.action_head, "image_encoder"):
            # For models with vram management, disable it and move to CPU
            if hasattr(self.trained_model.action_head, 'disable_vram_management'):
                self.trained_model.action_head.disable_vram_management()

        self.trained_model.to(device='cpu')
        torch.cuda.empty_cache()
        print(f"Model offloaded to CPU")

    def load_to_gpu(self):
        """Load the model to GPU for inference."""
        print(f"Loading model to GPU...")

        # Move model to GPU
        if hasattr(self.trained_model, "action_head") and hasattr(self.trained_model.action_head, "image_encoder"):
            self.trained_model.action_head.enable_vram_management()
        else:
            self.trained_model.to(device=self.device)

        # Apply bf16 if needed
        if self.eval_bf16:
            self.trained_model = self.trained_model.to(dtype=torch.bfloat16)

        torch.cuda.empty_cache()

    def ensure_model_on_gpu(self):
        """Ensure the model is loaded on GPU before inference."""
        # Check if model is on CPU
        model_device = next(self.trained_model.parameters()).device
        if model_device.type == 'cpu':
            self.load_to_gpu()

    def assert_delta_indices(self, delta_indices: np.ndarray):
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

    def _uses_robotwin_eef_s0_relative_action(self) -> bool:
        return bool(self.train_cfg.get("relative_action", False)) and (
            is_robotwin_eef_se3_s0_relative_action(
                getattr(self, "relative_action_mode", None)
            )
        )

    def _uses_robotwin_eef_s0_relative_state(self) -> bool:
        return self._uses_robotwin_eef_s0_relative_action() and (
            is_robotwin_eef_se3_s0_relative_state(
                getattr(self, "relative_action_mode", None)
            )
        )

    def _uses_robotwin_eef_chunk_relative_action(self) -> bool:
        return bool(self.train_cfg.get("relative_action", False)) and (
            is_robotwin_eef_se3_chunk_relative(
                getattr(self, "relative_action_mode", None)
            )
        )

    def _to_numpy(self, value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def _extract_s0_anchor_from_obs(self, obs: dict[str, Any]) -> np.ndarray:
        if ROBOTWIN_EEF_STATE_KEY not in obs:
            raise KeyError(
                "RoboTwin relative action modes require "
                f"obs[{ROBOTWIN_EEF_STATE_KEY!r}] as the absolute s0 anchor."
            )
        state = self._to_numpy(obs[ROBOTWIN_EEF_STATE_KEY]).astype(
            np.float32,
            copy=False,
        )
        ensure_robotwin_eef_dim(ROBOTWIN_EEF_STATE_KEY, state)
        # 预期输入：normalize 前的 absolute obs["state.eef_position"]，最后一维固定 16D。
        # 形状约定：这里取的是当前 sample/chunk 的第 0 帧 s0，不是 episode 首帧。
        # [T,16] -> [1,16]；[B,T,16] -> [B,1,16]。
        # 标准 RoboTwin server 先给 [1,16]，policy 判定 unbatched 后会扩成 [1,1,16]。
        if state.ndim == 2:
            return state[:1, :]
        if state.ndim == 3:
            return state[:, :1, :]
        raise ValueError(
            "RoboTwin relative action modes expect "
            f"{ROBOTWIN_EEF_STATE_KEY} to be 2D or 3D, got shape={state.shape}."
        )

    def _prepare_robotwin_eef_s0_window_anchor(
        self,
        obs: dict[str, Any],
    ) -> np.ndarray:
        """让 RoboTwin relative anchor 与 action head 的 causal KV window 保持一致。"""
        session_id = obs.get("session_id")
        if not isinstance(session_id, str):
            raise TypeError("RoboTwin relative eval requires string obs['session_id'].")
        prompt = obs.get("annotation.language.action_text")
        if not isinstance(prompt, str):
            raise TypeError("RoboTwin relative eval requires a string action prompt.")

        current_anchor = self._extract_s0_anchor_from_obs(obs)
        if current_anchor.shape != (1, ROBOTWIN_EEF_DIM):
            raise ValueError(
                "RoboTwin causal relative eval expects raw "
                f"{ROBOTWIN_EEF_STATE_KEY} shape=(1,{ROBOTWIN_EEF_DIM}), "
                f"got shape={current_anchor.shape}."
            )
        current_anchor = current_anchor.copy()

        video_key = next(
            (key for key in obs.keys() if key.startswith("video.")),
            None,
        )
        if video_key is None:
            raise KeyError("RoboTwin relative eval requires video.* observations.")
        video_frames = int(obs[video_key].shape[0])
        if video_frames not in (1, 8):
            raise ValueError(
                "RoboTwin relative eval expects 1 frame for a new window "
                f"or 8 frames for a continuation, got {video_frames}."
            )

        action_head = self.trained_model.action_head
        current_start_frame = int(action_head.current_start_frame)
        local_attn_size = int(action_head.model.local_attn_size)
        should_reset = (
            self._robotwin_eef_s0_window_anchor is None
            or session_id != self._robotwin_eef_s0_window_session_id
            or prompt != self._robotwin_eef_s0_window_prompt
            or video_frames == 1
            or current_start_frame == 0
            or current_start_frame >= local_attn_size
        )

        if should_reset:
            self._robotwin_eef_s0_window_anchor = current_anchor
            self._robotwin_eef_s0_window_session_id = session_id
            self._robotwin_eef_s0_window_prompt = prompt

        window_anchor = self._robotwin_eef_s0_window_anchor
        assert window_anchor is not None
        return window_anchor.copy()

    def _convert_obs_state_abs_to_s0_relative(
        self,
        obs: dict[str, Any],
        *,
        s0_anchor: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if not (
            self._uses_robotwin_eef_s0_relative_state()
            or self._uses_robotwin_eef_chunk_relative_action()
        ):
            return obs
        if ROBOTWIN_EEF_STATE_KEY not in obs:
            raise KeyError(
                "RoboTwin full state/action relative mode requires "
                f"obs[{ROBOTWIN_EEF_STATE_KEY!r}] before input normalization."
            )
        if s0_anchor is None and self._uses_robotwin_eef_chunk_relative_action():
            s0_anchor = self._extract_s0_anchor_from_obs(obs)
        if s0_anchor is None:
            raise ValueError(
                "RoboTwin full state/action relative mode requires an "
                "explicit KV-window s0 anchor before input normalization."
            )
        state_value = obs[ROBOTWIN_EEF_STATE_KEY]
        state = self._to_numpy(state_value).astype(np.float32, copy=False)
        rel_state = robotwin_eef_abs_to_rel_wxyz(
            state,
            s0_anchor,
            self.relative_action_pose_reference,
        )
        if torch.is_tensor(state_value):
            rel_state = torch.as_tensor(
                rel_state,
                dtype=state_value.dtype,
                device=state_value.device,
            )
        prepared = dict(obs)
        prepared[ROBOTWIN_EEF_STATE_KEY] = rel_state
        return prepared

    def apply(
        self,
        batch: Batch,
        *,
        robotwin_eef_s0_anchor: np.ndarray | None = None,
        **kwargs,
    ) -> Batch:
        """Normalize inputs"""
        obs = self._convert_obs_state_abs_to_s0_relative(
            batch.obs,
            s0_anchor=robotwin_eef_s0_anchor,
        )

        normalized_input = self.eval_transform(obs)
        batch.normalized_obs = normalized_input
        return batch

    @staticmethod
    def _action_trace_value(value: Any) -> Any:
        if torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {
                str(key): GrootSimPolicy._action_trace_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [GrootSimPolicy._action_trace_value(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _write_robotwin_action_trace(
        self,
        *,
        raw_obs: dict[str, Any],
        normalized_input: dict[str, Any],
        normalized_action: torch.Tensor,
        robotwin_eef_s0_anchor: np.ndarray | None,
    ) -> None:
        trace_path = os.environ.get("DREAMZERO_ACTION_TRACE_PATH")
        if not trace_path or self.rank != 0:
            return

        trace_file = Path(trace_path)
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_index = int(getattr(self, "_robotwin_action_trace_index", 0))
        self._robotwin_action_trace_index = trace_index + 1

        normalized_state = normalized_input.get("state")
        raw_absolute_state = raw_obs.get(ROBOTWIN_EEF_STATE_KEY)
        if raw_absolute_state is None:
            raw_absolute_state = raw_obs.get("state")
        record = {
            "trace_index": trace_index,
            "timestamp": time.time(),
            "session_id": raw_obs.get("session_id"),
            "prompt": raw_obs.get("annotation.language.action_text"),
            "relative_action_mode": getattr(self, "relative_action_mode", None),
            "raw_absolute_state": self._action_trace_value(raw_absolute_state),
            "window_s0_anchor": self._action_trace_value(robotwin_eef_s0_anchor),
            "normalized_state": self._action_trace_value(normalized_state),
            "normalized_action": self._action_trace_value(normalized_action),
            "unnormalized_model_action": self._action_trace_value(
                getattr(self, "_last_unnormalized_model_action", None)
            ),
            "final_absolute_action": self._action_trace_value(
                getattr(self, "_last_final_absolute_action", None)
            ),
        }
        try:
            encoded = json.dumps(
                record,
                ensure_ascii=True,
                default=self._action_trace_value,
            )
            with trace_file.open("a", encoding="utf-8") as f:
                f.write(encoded + "\n")
        except Exception as exc:
            print(f"[ACTION_TRACE] failed to write {trace_file}: {exc!r}", flush=True)

    def _normalize_external_past_actions(self, raw_history) -> torch.Tensor | None:
        """Normalize ee obs-action history to the model's clean action space."""
        if raw_history is None:
            return None
        ensure_relative_action_cache_compatible(
            relative_action=bool(self.train_cfg.get("relative_action", False)),
            relative_action_mode=getattr(self, "relative_action_mode", None),
            use_obs_action_history=True,
        )

        raw_history = np.asarray(raw_history, dtype=np.float32)
        if raw_history.size == 0:
            action_dim = raw_history.shape[-1] if raw_history.ndim > 0 else 0
            return torch.empty((1, 0, action_dim), dtype=torch.float32)
        if raw_history.ndim == 1:
            raw_history = raw_history.reshape(1, -1)
        elif raw_history.ndim == 3:
            if raw_history.shape[0] != 1:
                raise ValueError(
                    "external_past_action.eef_position only supports batch size 1, "
                    f"got {raw_history.shape}"
                )
            raw_history = raw_history[0]
        elif raw_history.ndim != 2:
            raise ValueError(
                "external_past_action.eef_position must be 1D, 2D, or 3D "
                f"(batch size 1), got shape={raw_history.shape}."
            )

        action_data = {"action.eef_position": raw_history.copy()}
        for transform in self.eval_transform.transforms:
            transform_name = transform.__class__.__name__
            if transform_name == "DreamTransform":
                break
            apply_to = getattr(transform, "apply_to", [])
            is_action_transform = any(str(key).startswith("action.") for key in apply_to)
            if transform_name != "ConcatTransform" and not is_action_transform:
                continue
            action_data = transform(action_data)
            if "action" in action_data:
                break

        normalized = action_data.get("action")
        if normalized is None:
            raise ValueError(
                "Failed to normalize external_past_action.eef_position into concatenated action."
            )
        if not torch.is_tensor(normalized):
            normalized = torch.as_tensor(normalized)
        if normalized.ndim == 2:
            normalized = normalized.unsqueeze(0)
        elif normalized.ndim != 3:
            raise ValueError(f"Unexpected normalized external past action shape: {normalized.shape}")

        if self.rank == 0 and not getattr(self, "_external_past_actions_logged", False):
            print(
                "[OBS_ACTION_HISTORY] normalized external past actions "
                f"shape={tuple(normalized.shape)}",
                flush=True,
            )
            self._external_past_actions_logged = True
        return normalized

    def _uses_episode_first_frame_token(self) -> bool:
        """返回当前 action head 是否启用 episode 首帧 ff token。输出为 bool。"""
        action_head = getattr(self.trained_model, "action_head", None)
        return bool(getattr(action_head, "use_episode_first_frame_token", False))

    def _attach_episode_first_images(self, raw_obs: dict[str, Any], normalized_input: dict[str, Any]) -> None:
        """输入 raw_obs["session_id"]: str 和 normalized_input["images"]: uint8 [B,T,H,W,C]。
        输出为原地写入 normalized_input["episode_first_images"]: uint8 [B,1,H,W,C]；同 session 复用首帧，新 session 更新。"""
        if not self._uses_episode_first_frame_token():
            return
        if "images" not in normalized_input:
            raise KeyError("use_episode_first_frame_token=True requires normalized key 'images'.")
        if "session_id" not in raw_obs:
            raise KeyError("use_episode_first_frame_token=True requires raw obs key 'session_id'.")

        session_id = raw_obs["session_id"]
        if not isinstance(session_id, str):
            raise TypeError(
                "use_episode_first_frame_token=True expects session_id to be a string "
                f"from eval, got {type(session_id).__name__}: {session_id!r}."
            )
        if session_id != self._episode_first_session_id:
            self._episode_first_session_id = session_id
            self._episode_first_images = None

        images = normalized_input["images"]
        if not torch.is_tensor(images):
            raise TypeError(
                "use_episode_first_frame_token=True expects normalized images to be a "
                f"torch.Tensor, got {type(images).__name__}."
            )
        if images.dtype != torch.uint8:
            raise TypeError(
                "use_episode_first_frame_token=True expects normalized images to be "
                f"torch.uint8 before video normalization, got {images.dtype}."
            )
        if images.ndim != 5:
            raise ValueError(
                "normalized images must have shape [B,T,H,W,C] before creating "
                f"episode_first_images, got {tuple(images.shape)}."
            )
        if images.shape[1] < 1:
            raise ValueError("normalized images must contain at least one frame.")

        if self._episode_first_images is None:
            first_images = images[:, :1].detach().clone()
            self._episode_first_images = first_images
            if self.rank == 0:
                print(
                    "[FF_TOKEN] cached eval episode first images "
                    f"session_id={session_id!r} shape={tuple(first_images.shape)}",
                    flush=True,
                )

        first_images = self._episode_first_images
        first_images = first_images.to(device=images.device)
        normalized_input["episode_first_images"] = first_images

    def unapply(
        self,
        batch: Batch,
        obs: dict = None,
        *,
        robotwin_eef_s0_anchor: np.ndarray | None = None,
        robotwin_eef_action_anchor: np.ndarray | None = None,
        **kwargs,
    ):
        """Unnormalize actions and convert relative actions to absolute if needed"""
        unnormalized_action = self.eval_transform.unapply(
            dict(action=batch.normalized_action.cpu())
        )
        self._last_unnormalized_model_action = np.asarray(
            unnormalized_action.get(ROBOTWIN_EEF_ACTION_KEY),
            dtype=np.float32,
        ).copy()

        # Check if relative_action is enabled and convert relative to absolute
        relative_action = self.train_cfg.get("relative_action", False)
        relative_action_keys = getattr(self, "relative_action_keys", [])
        relative_action_mode = self.train_cfg.get(
            "relative_action_mode",
            getattr(self, "relative_action_mode", None),
        )
        is_se3_relative = is_robotwin_eef_se3_relative_action(relative_action_mode)
        if relative_action and not is_se3_relative:
            raise ValueError(
                "relative_action currently only supports "
                "RoboTwin relative action modes."
            )
        if is_se3_relative:
            if not relative_action:
                raise ValueError(
                    "RoboTwin relative action modes require "
                    "relative_action=True."
                )
            if relative_action_keys != ["eef_position"]:
                raise ValueError(
                    "RoboTwin relative action modes require "
                    "relative_action_keys=['eef_position']."
                )
            action_anchor = (
                robotwin_eef_action_anchor
                if robotwin_eef_action_anchor is not None
                else robotwin_eef_s0_anchor
            )
            if action_anchor is None and is_robotwin_eef_se3_chunk_relative(
                relative_action_mode
            ):
                if obs is None:
                    raise ValueError(
                        "RoboTwin chunk-relative action restoration requires "
                        "the current absolute state."
                    )
                action_anchor = self._extract_s0_anchor_from_obs(obs)
            if action_anchor is None:
                raise ValueError(
                    "RoboTwin relative action modes require an explicit "
                    "state anchor for action restoration."
                )
            for key in relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"
                if action_key not in unnormalized_action:
                    raise ValueError(
                        "RoboTwin relative action modes require "
                        f"{action_key!r} in unnormalized action output."
                    )
                if (
                    action_key != ROBOTWIN_EEF_ACTION_KEY
                    or state_key != ROBOTWIN_EEF_STATE_KEY
                ):
                    raise ValueError(
                        "RoboTwin relative action modes only support "
                        "action.eef_position paired with state.eef_position."
                    )
                action_shape = ensure_robotwin_eef_dim(
                    action_key,
                    unnormalized_action[action_key],
                )
                anchor_shape = ensure_robotwin_eef_dim(
                    state_key,
                    action_anchor,
                )
                if len(action_shape) != len(anchor_shape):
                    raise ValueError(
                        "RoboTwin relative action modes expect "
                        f"{action_key} and anchor {state_key} to have the same "
                        f"ndim for explicit broadcasting, got action={action_shape}, "
                        f"anchor={anchor_shape}."
                    )
                unnormalized_action[action_key] = robotwin_eef_rel_to_abs_wxyz(
                    unnormalized_action[action_key],
                    action_anchor,
                    self.relative_action_pose_reference,
                )

        self._last_final_absolute_action = np.asarray(
            unnormalized_action.get(ROBOTWIN_EEF_ACTION_KEY),
            dtype=np.float32,
        ).copy()

        batch.act = unnormalized_action
        return batch

    def forward(self, batch, state=None, **kwargs):
        raise NotImplementedError(
            "GrootSimPolicy.forward is not used by RLinf training or the current "
            "DreamZero RoboTwin websocket eval path. RoboTwin eval uses "
            "lazy_joint_forward_causal(); RLinf training uses RLinf "
            "DreamZeroPolicy.sft_forward()."
        )

        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs
        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.get_action(normalized_input)
        normalized_action = model_pred["action_pred"].float()

        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch

    def joint_forward(self, batch, video=None, state=None, **kwargs):
        raise NotImplementedError(
            "joint_forward is not used by RLinf training or DreamZero eval; "
            "DreamZero eval uses lazy_joint_forward_causal."
        )
        # 0. Ensure model is on GPU

        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        if video is not None:
            for key in normalized_input:
                if 'images' in key:
                    print("key", key, normalized_input[key].shape)
                    normalized_input[key] = video

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.joint_video_action(normalized_input)
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch, video_pred

    def lazy_joint_forward_causal(self, batch, video=None, latent_video=None, state=None, video_only=False, **kwargs):

        transform_start_time = time.perf_counter()
        raw_external_past_actions = batch.obs.pop("external_past_action.eef_position", None)
        external_past_actions = self._normalize_external_past_actions(raw_external_past_actions)

        robotwin_eef_s0_anchor = None
        robotwin_eef_action_anchor = None
        if self._uses_robotwin_eef_s0_relative_action():
            robotwin_eef_s0_anchor = self._prepare_robotwin_eef_s0_window_anchor(
                batch.obs
            )
            robotwin_eef_action_anchor = robotwin_eef_s0_anchor.copy()
        elif self._uses_robotwin_eef_chunk_relative_action():
            robotwin_eef_s0_anchor = self._extract_s0_anchor_from_obs(batch.obs)
            robotwin_eef_action_anchor = robotwin_eef_s0_anchor.copy()

        # Save original observation before any modification (for relative action conversion)
        original_obs_for_relative = {k: v.copy() if isinstance(v, np.ndarray) else v.clone() if torch.is_tensor(v) else v
                                     for k, v in batch.obs.items()}
        episode_first_raw_obs = dict(batch.obs)

        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)
            # Also unsqueeze the saved original obs
            original_obs_for_relative = unsqueeze_dict_values(original_obs_for_relative)
            if robotwin_eef_s0_anchor is not None:
                robotwin_eef_s0_anchor = np.expand_dims(
                    robotwin_eef_s0_anchor,
                    axis=0,
                )
            if robotwin_eef_action_anchor is not None:
                robotwin_eef_action_anchor = np.expand_dims(
                    robotwin_eef_action_anchor,
                    axis=0,
                )
        elif robotwin_eef_s0_anchor is not None:
            raise ValueError(
                "RoboTwin causal relative eval only supports unbatched raw "
                "observations that become batch size 1 inside GrootSimPolicy."
            )

        # 2. Apply transforms/normalization
        batch = self.apply(
            batch,
            robotwin_eef_s0_anchor=robotwin_eef_s0_anchor,
        )
        normalized_input = batch.normalized_obs

        transform_time = time.perf_counter() - transform_start_time

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        if self._uses_episode_first_frame_token() and (
            video is not None or latent_video is not None
        ):
            raise ValueError(
                "use_episode_first_frame_token=True is not compatible with "
                "video/latent_video override in lazy_joint_forward_causal(); "
                "RoboTwin eval should use observation images as the single video source."
            )

        self._attach_episode_first_images(episode_first_raw_obs, normalized_input)

        if video is not None:
            for key in normalized_input:
                if 'images' in key:
                    print("key", key, normalized_input[key].shape)
                    normalized_input[key] = video

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        model_start_time = time.perf_counter()

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.lazy_joint_video_action_causal(
                normalized_input,
                latent_video=latent_video,
                video_only=video_only,
                external_past_actions=external_past_actions,
            )
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        model_time = time.perf_counter() - model_start_time

        untransform_start_time = time.perf_counter()

        # 4. Unnormalize actions (pass obs for relative action conversion)
        if not video_only:
            batch = self.unapply(
                Batch(normalized_action=normalized_action),
                obs=original_obs_for_relative,
                robotwin_eef_s0_anchor=robotwin_eef_s0_anchor,
                robotwin_eef_action_anchor=robotwin_eef_action_anchor,
            )
            self._write_robotwin_action_trace(
                raw_obs=original_obs_for_relative,
                normalized_input=normalized_input,
                normalized_action=normalized_action,
                robotwin_eef_s0_anchor=robotwin_eef_s0_anchor,
            )
        else:
            batch = Batch(normalized_action=normalized_action)

        # 5. Remove batch dimension if we added it
        if video_only:
            if not is_batched:
                batch.normalized_action = batch.normalized_action.squeeze(0)
        elif not is_batched:
            batch.act = squeeze_dict_values(batch.act)

        untransform_time = time.perf_counter() - untransform_start_time
        total_time = transform_time + model_time + untransform_time

        if self.rank == 0:
            print(f"Inference Time: Total {total_time:.3f} seconds, "
                  f"Transform: {transform_time:.3f} seconds, "
                  f"Model: {model_time:.3f} seconds, "
                  f"Untransform: {untransform_time:.3f} seconds")

        return batch, video_pred

    def lazy_joint_forward_causal_gt_cond(self, batch, video=None, latent_video=None, state=None, **kwargs):
        raise NotImplementedError(
            "lazy_joint_forward_causal_gt_cond is not used by RLinf training or "
            "the current DreamZero RoboTwin websocket eval path. RoboTwin eval "
            "uses lazy_joint_forward_causal()."
        )

        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        if video is not None:
            for key in normalized_input:
                if 'images' in key:
                    print("key", key, normalized_input[key].shape)
                    normalized_input[key] = video

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.lazy_joint_video_action_causal_gt_cond(normalized_input, latent_video=latent_video)
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch, video_pred

    def lazy_joint_forward(self, batch, video=None, state=None, **kwargs):
        raise NotImplementedError(
            "lazy_joint_forward is not used by RLinf training or DreamZero eval; "
            "DreamZero eval uses lazy_joint_forward_causal."
        )

        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        if video is not None:
            for key in normalized_input:
                if 'images' in key:
                    print("key", key, normalized_input[key].shape)
                    normalized_input[key] = video

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.lazy_joint_video_action(normalized_input)
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch, video_pred

    def lazy_joint_forward_efficient(self, batch, video=None, state=None, prompt_embs=None, prompt_emb_nega=None, **kwargs):
        raise NotImplementedError(
            "lazy_joint_forward_efficient is not used by RLinf training or the "
            "current DreamZero RoboTwin websocket eval path. RoboTwin eval uses "
            "lazy_joint_forward_causal()."
        )

        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        if video is not None:
            for key in normalized_input:
                if 'images' in key:
                    print("key", key, normalized_input[key].shape)
                    normalized_input[key] = video

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.lazy_joint_video_action_efficient(normalized_input, prompt_embs=prompt_embs, prompt_emb_nega=prompt_emb_nega)
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch, video_pred

    def language_forward(self, batch, state=None, **kwargs):

        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.get_language(normalized_input)
        # normalized_action = model_pred["action_pred"].float()
        output = model_pred["output"]
        # 4. Unnormalize actions
        # batch = self.unapply(Batch(normalized_action=normalized_action))

        # # 5. Remove batch dimension if we added it
        # if not is_batched:
        #     batch.act = squeeze_dict_values(batch.act)
        return output
        # return batch


    def gt_video_action_pred(self, batch, video=None, state=None, **kwargs):

        # 1. Check if input is batched and add batch dimension if needed
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        # 2. Apply transforms/normalization
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs

        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()

        if video is not None:
            for key in normalized_input:
                if 'images' in key:
                    print("key", key, normalized_input[key].shape)
                    normalized_input[key] = video

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        # 3. Model inference
        with torch.inference_mode():
            # with maybe_autocast:
            model_pred = self.trained_model.gt_video_action_pred(normalized_input)
        normalized_action = model_pred["action_pred"].float()
        video_pred = model_pred["video_pred"]

        # 4. Unnormalize actions (pass obs for relative action conversion)
        original_obs = batch.obs
        batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)

        # 5. Remove batch dimension if we added it
        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch, video_pred

    def _check_state_is_batched(self, obs: dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    @property
    def raw_data_image_transform(self) -> Callable:
        """
        Get the raw data image transform for the policy
        """
        return lambda x: x

    @property
    def state_delta_indices(self) -> np.ndarray:
        return self._state_delta_indices

    @property
    def video_delta_indices(self) -> np.ndarray:
        return self._video_delta_indices


def unsqueeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.array(v)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            unsqueezed_data[k] = np.array([v])
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data


def tree_get_leading_dim(data: dict[str, Any]) -> int:
    """
    Get the batch size (leading dimension) of all tensors in the structure.

    Args:
        data: Nested structure containing tensors
        strict: If True, raises error if batch sizes don't match.
                If False, returns None if inconsistent.

    Returns:
        The common batch size, or None if inconsistent (when strict=False)

    Raises:
        ValueError: If batch sizes are inconsistent (when strict=True)
        ValueError: If no tensors found
    """
    batch_sizes = []
    tensor_paths = []

    def collect_batch_sizes(path, x):
        if hasattr(x, "shape") and len(x.shape) > 0:
            batch_sizes.append(x.shape[0])
            tensor_paths.append(".".join(map(str, path)))

    # Collect all batch sizes with their paths
    tree.map_structure_with_path(collect_batch_sizes, data)

    if not batch_sizes:
        raise ValueError("No tensors found in the data structure")

    # Check if all batch sizes are the same
    first_batch_size = batch_sizes[0]

    if not all(bs == first_batch_size for bs in batch_sizes):
        inconsistent_info = [f"{path}: {bs}" for path, bs in zip(tensor_paths, batch_sizes)]
        error_msg = "Inconsistent batch sizes found:\n" + "\n".join(inconsistent_info)

        raise ValueError(error_msg)

    return first_batch_size


class GrootSimRLPolicy(BaseGrootSimPolicy):
    """
    A class for Offline RL policies,
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        device: int | str,
        q_fn_path: str | None = None,
        n_action_samples: int = 1,
        n_denoising_steps: int = 4,
        gaussian_std: float = 0.0,
        inference_batch_size: int = 1,
        model_config_overrides: list[str] | None = [],
    ):
        """
        Implicit Policy Extraction (IPE) GrootSimPolicy.
        """
        super().__init__(embodiment_tag=embodiment_tag, model_path=model_path, device=device)

        self.base_policy = GrootSimPolicy(
            embodiment_tag=embodiment_tag,
            model_path=str(model_path),
            device=device,
        )
        self.base_policy.trained_model.action_head.num_inference_timesteps = n_denoising_steps

        if q_fn_path is None:
            if n_action_samples > 1:
                raise ValueError("If n_action_samples > 1, q_fn_path should be provided.")
            self.qfn = None
        else:
            assert n_action_samples > 1, "If q_fn_path is provided, n_action_samples should be > 1."
            self.qfn = GrootSimPolicy(
                embodiment_tag=EmbodimentTag.GR1_UNIFIED_OFFLINE_RL,
                model_path=q_fn_path,
                device=device,
                skip_assert_delta_indices=True,
            )

        self.n_action_samples = n_action_samples
        self.gaussian_std = gaussian_std
        self.inference_batch_size = inference_batch_size

    def batch_obs(self, obs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """
        Check if input is batched and add batch dimension if needed
        """
        is_batched = self.base_policy._check_state_is_batched(obs)
        if not is_batched:
            obs = unsqueeze_dict_values(obs)

        return obs, is_batched

    def add_action_noise(self, action: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0.0:
            return action
        noise = torch.randn_like(action) * std
        noisy_action = action + noise
        return noisy_action

    def sample_actions_from_base_policy(
        self,
        batch: Batch,
        n_action_samples: int,  # number of actions to sample PER item in the batch
        inference_batch_size: int = 1,
    ) -> torch.Tensor:
        # a) Apply transforms/normalization
        batch = self.base_policy.apply(batch)
        normalized_input = batch.normalized_obs
        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.base_policy.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        with torch.inference_mode():
            model_pred = self.base_policy.trained_model.get_action(
                normalized_input,
                num_action_samples=n_action_samples,
                inference_batch_size=inference_batch_size,
                validate=False,
            )
            normalized_action = model_pred["action_pred"]
            normalized_action = self.add_action_noise(normalized_action, self.gaussian_std)
        return normalized_action  # [num_envs, n_action_samples, horizon, action_dim]

    def get_action_values(
        self, batch: Batch, actions: torch.Tensor, inference_batch_size: int = 1
    ) -> torch.Tensor:
        assert self.qfn is not None, "Q-function must be provided to evaluate actions."

        # a) Apply transforms/normalization
        batch = self.qfn.apply(batch)
        normalized_input = batch.normalized_obs
        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.qfn.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        with torch.inference_mode():
            # [num_envs, ...]
            backbone_inputs, action_inputs = self.qfn.trained_model.prepare_input(normalized_input)
            backbone_outputs = self.qfn.trained_model.backbone(backbone_inputs)
            q_values = self.qfn.trained_model.action_head.get_q_pred(
                backbone_outputs,
                action_inputs,
                actions,
                inference_batch_size=inference_batch_size,
            )

        return q_values

    def forward(self, batch, state=None, **kwargs):
        orig_obs = batch.obs.copy()
        orig_obs, is_batched = self.batch_obs(orig_obs)

        # Get number of environments
        num_envs = 1 if not is_batched else tree_get_leading_dim(orig_obs)

        # 1/ Sample normalized actions from the base policy
        actions = self.sample_actions_from_base_policy(
            Batch(obs=orig_obs),
            n_action_samples=self.n_action_samples,
            inference_batch_size=self.inference_batch_size,
        )  # [num_envs, n_action_samples, horizon, action_dim]

        # 2/ Evaluate the sampled actions with the Q-function
        if self.qfn is not None:
            q_values = self.get_action_values(
                Batch(obs=orig_obs),
                actions=actions,
                inference_batch_size=self.inference_batch_size,
            )

            best_indices = torch.argmax(q_values, dim=1)  # [num_envs]
            normalized_action = actions[
                torch.arange(num_envs), best_indices
            ].float()  # [num_envs, horizon, action_dim]
        else:
            normalized_action = actions.float()

        batch = self.base_policy.unapply(Batch(normalized_action=normalized_action), obs=orig_obs)

        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)
        return batch

    @property
    def video_delta_indices(self) -> np.ndarray:
        return self.base_policy.video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray:
        return self.base_policy.state_delta_indices

    @property
    def raw_data_image_transform(self) -> Callable:
        return lambda x: x
