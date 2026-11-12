from dataclasses import dataclass, field
import logging
import math
import time
from typing import TypeAlias, cast
import os
import torch.nn.functional as F

from accelerate import load_checkpoint_and_dispatch

from einops import rearrange
from hydra.utils import instantiate
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from safetensors.torch import load_file
import json
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)

WAN_HF_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN22_HF_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"

def hf_download(filename: str, repo_id: str = WAN_HF_REPO_ID) -> str:
    """Download a file from the specified HuggingFace repo to HF cache."""
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return path


def ensure_file(path: str | None, hf_filename: str, repo_id: str = WAN_HF_REPO_ID) -> str:
    """Return a valid local path: use `path` if it exists, otherwise download from HuggingFace."""
    if path is not None and os.path.exists(path):
        return path
    return hf_download(hf_filename, repo_id)

from torch.distributions import Beta
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torchvision.transforms import v2
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.n1_5.action_head.base_action_head import ActionHead
from groot.vla.model.dreamzero.modules.flow_match_scheduler import FlowMatchScheduler
from groot.vla.model.dreamzero.modules.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from groot.vla.model.dreamzero.modules.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler


KVCacheType: TypeAlias = torch.Tensor
DEFAULT_SIGMA_SHIFT = 5.0


def resolve_action_sigma_shifts(
    action_train_sigma_shift: float | None,
    action_inference_sigma_shift: float | None,
    *,
    video_sigma_shift: float = DEFAULT_SIGMA_SHIFT,
) -> tuple[float, float]:
    """Resolve action train/inference shifts while preserving legacy defaults."""
    train_shift = (
        video_sigma_shift
        if action_train_sigma_shift is None
        else float(action_train_sigma_shift)
    )
    inference_shift = (
        train_shift
        if action_inference_sigma_shift is None
        else float(action_inference_sigma_shift)
    )
    for name, value in (
        ("video_sigma_shift", video_sigma_shift),
        ("action_train_sigma_shift", train_shift),
        ("action_inference_sigma_shift", inference_shift),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number, got {value!r}.")
    return train_shift, inference_shift


def _check_and_clean(
    tensor: torch.Tensor,
    name: str,
    rank: int = 0,
    step: int = 0,
) -> torch.Tensor:
    """检查 tensor 是否含 NaN/Inf，若有则打印日志并用 nan_to_num 清零后返回。

    不跳过 batch，只做原地修复，确保后续计算不会因异常值直接崩溃。
    """
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    if has_nan or has_inf:
        safe = tensor.float()
        logger.warning(
            f"[NaN/Inf] rank={rank} step={step} tensor='{name}' "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"nan={has_nan.item()} inf={has_inf.item()} "
            f"max={safe[torch.isfinite(safe)].max().item() if torch.isfinite(safe).any() else 'all-bad'} "
            f"min={safe[torch.isfinite(safe)].min().item() if torch.isfinite(safe).any() else 'all-bad'}"
        )
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


def _check_loss(
    loss: torch.Tensor,
    name: str,
    rank: int = 0,
    step: int = 0,
) -> torch.Tensor:
    """检查标量 loss 是否为 NaN/Inf，若有则打印日志并置零，避免反向传播崩溃。"""
    if not torch.isfinite(loss):
        logger.warning(
            f"[NaN/Inf LOSS] rank={rank} step={step} loss='{name}' value={loss.item()}"
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss


@dataclass
class WANPolicyHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )
    tiled: bool = field(default=True, metadata={"help": "Whether to use tiled input."})
    tile_size_height: int = field(default=34, metadata={"help": "Tile size height."})
    tile_size_width: int = field(default=34, metadata={"help": "Tile size width."})
    tile_stride_height: int = field(default=18, metadata={"help": "Tile stride height."})
    tile_stride_width: int = field(default=16, metadata={"help": "Tile stride width."})
    num_frame_per_block: int = field(default=1, metadata={"help": "Number of frames per block."})
    # Target video (H, W) for Wan22 resize. When set, videos are resized to this before VAE so latent
    # spatial size matches. Use height/width divisible by 32 for WanVideoVAE38 (16x) so latent H,W are even.
    target_video_height: int | None = field(default=None, metadata={"help": "Target video height for resize (e.g. 160 for even latent with VAE38)."})
    target_video_width: int | None = field(default=None, metadata={"help": "Target video width for resize (e.g. 320)."})

    lora_rank: int = field(default=4, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=4, metadata={"help": "LoRA alpha."})
    lora_target_modules: str = field(default="q,k,v,o,ffn.0,ffn.2")
    init_lora_weights: str = field(default="kaiming", metadata={"help": "LoRA initialization method."})
    train_architecture: str= field(default="lora", metadata={"help": "Train architecture."})
    skip_component_loading: bool = field(default=False, metadata={"help": "Skip loading individual component weights (used when loading from full pretrained model)."})

    use_gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing."})
    qformer_cfg: dict = field(default=None, metadata={"help": "Qformer configuration."})
    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    action_train_sigma_shift: float | None = field(
        default=None,
        metadata={"help": "Action training sigma shift. None inherits the video shift."},
    )
    action_inference_sigma_shift: float | None = field(
        default=None,
        metadata={"help": "Action inference sigma shift. None inherits the action training shift."},
    )
    # High noise emphasis for BASE (coupled) training - applies Beta distribution to BOTH video and action together
    use_high_noise_emphasis: bool = field(
        default=False, metadata={"help": "Use Beta distribution for noise sampling (biases BOTH video and action towards high noise levels together)."}
    )
    high_noise_beta_alpha: float = field(
        default=3.0, metadata={"help": "Beta alpha for high noise emphasis. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher = more high noise bias."}
    )
    # Decoupled noise sampling config for training-inference alignment
    # When enabled: video uses Beta(alpha,beta) biased towards high noise, action uses independent uniform
    decouple_video_action_noise: bool = field(
        default=False, metadata={"help": "Decouple video/action noise: video uses Beta distribution (high noise bias), action uses independent uniform."}
    )
    video_noise_beta_alpha: float = field(
        default=3.0, metadata={"help": "Beta alpha for video noise. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher alpha = more bias to high noise."}
    )
    video_noise_beta_beta: float = field(
        default=1.0, metadata={"help": "Beta beta for video noise. Keep at 1.0."}
    )
    # Decoupled inference config - allows video to stay noisy while action fully denoises
    decouple_inference_noise: bool = field(
        default=False, metadata={"help": "Use decoupled noise schedules during inference (video stays noisy, action fully denoises)."}
    )
    video_inference_final_noise: float = field(
        default=0.8, metadata={"help": "Final noise level for video during decoupled inference (0.0-1.0). E.g., 0.8 means video ends at 80% noise."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)
    defer_lora_injection: bool = field(default=False, metadata={"help": "Defer LoRA injection until after loading pretrained weights."})
    use_clip_image_encoder: bool = field(
        default=False,
        metadata={"help": "Legacy checkpoint field; image encoder conditioning is unsupported and must remain disabled."},
    )
    image_context_type: str = field(
        default="clip",
        metadata={"help": "Image context source when use_clip_image_encoder is enabled: clip, vae, siglip2, dinov2, or qwen3_vl."},
    )
    image_context_token_scope: str = field(
        default="all",
        metadata={"help": "Which token types can directly attend to CLIP image context: all, video, action, non_video, non_action."},
    )
    use_past_action_context: bool = field(
        default=False,
        metadata={"help": "Use final past action chunks as clean action context during policy inference/training."},
    )
    past_action_context_dropout_train: bool = field(
        default=False,
        metadata={"help": "Apply train-only random fixed-size token subsets over past action context."},
    )
    past_action_context_drop_ratio: float = field(
        default=0.0,
        metadata={"help": "Fixed ratio used to derive how many past action context tokens to drop during training."},
    )
    use_episode_first_frame_token: bool = field(
        default=False,
        metadata={"help": "Legacy checkpoint field; episode first-frame tokens are unsupported and must remain disabled."},
    )
    episode_first_frame_rope_index: int = field(
        default=1000,
        metadata={"help": "Reserved video RoPE time index for episode first-frame tokens. Current table length is 1024."},
    )
    video_condition_on_action_state: bool = field(
        default=True,
        metadata={"help": "Whether noisy video queries attend to current action/state register tokens."},
    )

    vl_self_attention_cfg: dict = field(default=None)
    text_encoder_cfg: dict = field(default=None)
    image_encoder_cfg: dict = field(default=None)
    vae_cfg: dict = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class WANPolicyHead(ActionHead):
    config_class = WANPolicyHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: WANPolicyHeadConfig,
    ):
        super().__init__()
        if config.use_clip_image_encoder:
            raise ValueError(
                "Image encoder conditioning is no longer supported; "
                "set use_clip_image_encoder=False."
            )
        if config.use_episode_first_frame_token:
            raise ValueError(
                "Episode first-frame tokens are no longer supported; "
                "set use_episode_first_frame_token=False."
            )
        self.tiled = config.tiled
        self.tile_size_height = config.tile_size_height
        self.tile_size_width = config.tile_size_width
        self.tile_stride_height = config.tile_stride_height
        self.tile_stride_width = config.tile_stride_width
        self.num_frame_per_block = config.num_frame_per_block   # 2
        self.hidden_size = config.hidden_size   # 64
        self.num_frames = config.num_frames # 33
        self.text_encoder = instantiate(config.text_encoder_cfg)    # WanTextEncoder
        self.image_encoder = instantiate(config.image_encoder_cfg)  # WanImageEncoder
        self.vae = instantiate(config.vae_cfg)
        self.sigma_shift = DEFAULT_SIGMA_SHIFT
        (
            self.action_train_sigma_shift,
            self.action_inference_sigma_shift,
        ) = resolve_action_sigma_shifts(
            getattr(config, "action_train_sigma_shift", None),
            getattr(config, "action_inference_sigma_shift", None),
            video_sigma_shift=self.sigma_shift,
        )
        self.scheduler = FlowMatchScheduler(
            shift=self.sigma_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=self.action_train_sigma_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.model_names = ['text_encoder']

        # Runtime override shared by policy and video-only inference.
        self.num_inference_steps = int(os.getenv("NUM_INFERENCE_STEPS", "16"))
        if self.num_inference_steps <= 0:
            raise ValueError("NUM_INFERENCE_STEPS must be a positive integer.")
        self.seed = 1140
        self.cfg_scale = 5.0
        self.denoising_strength = 1.0
        self.kv_cache1: KVCacheType | None = None
        self.kv_cache_neg: KVCacheType | None = None
        self.crossattn_cache: KVCacheType | None = None
        self.crossattn_cache_neg: KVCacheType | None = None

        self.global_step = 0
        self.max_steps = 0
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_target_modules = config.lora_target_modules
        self.init_lora_weights = config.init_lora_weights
        self.train_architecture = config.train_architecture
        self.clip_feas = None
        self.ys = None
        self.current_start_frame = 0
        self.language = None
        self.past_action_history: torch.Tensor | None = None
        self._past_action_context_train_logged = False
        self._past_action_context_infer_logged = False
        self._inference_scheduler_logged = False

        self.ip_rank = 0
        self.ip_size = 1
        self.ip_group = None

        self._device = "cuda"
        self.dynamic_cache_schedule = os.getenv("DYNAMIC_CACHE_SCHEDULE", "False").lower() == "true"


        num_dit_steps = 8   # 怎么又有个8？
        if os.getenv("NUM_DIT_STEPS") is not None:
            num_dit_steps = int(os.getenv("NUM_DIT_STEPS"))
        if num_dit_steps == 5:
            self.dit_step_mask = [True, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False]
        elif num_dit_steps == 6:
            self.dit_step_mask = [True, True, False, False, False, True, False, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 7:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 8:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True]
        else:
            self.dit_step_mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        assert self.dit_step_mask[0] == True, "first step must be True"

        self.normalize_video = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        if self.training:
            self.scheduler.set_timesteps(1000, training=True)
            self.action_scheduler.set_timesteps(1000, training=True)


        self.input_embedding_dim = config.input_embedding_dim

        self.cpu_offload = False

        self.model = instantiate(config.diffusion_model_cfg)
        self.video_condition_on_action_state = bool(config.video_condition_on_action_state)
        self.model.video_condition_on_action_state = self.video_condition_on_action_state
        for block in getattr(self.model, "blocks", []):
            block.video_condition_on_action_state = self.video_condition_on_action_state
            if hasattr(block, "self_attn"):
                block.self_attn.video_condition_on_action_state = self.video_condition_on_action_state
        self.use_clip_image_encoder = config.use_clip_image_encoder
        self.image_context_type = config.image_context_type
        self.image_context_token_scope = config.image_context_token_scope
        self.use_episode_first_frame_token = bool(config.use_episode_first_frame_token)
        self.episode_first_frame_rope_index = int(config.episode_first_frame_rope_index)
        if self.use_episode_first_frame_token:
            if config.use_past_action_context:
                raise ValueError(
                    "use_episode_first_frame_token=True currently requires "
                    "use_past_action_context=False."
                )
            if self.image_context_token_scope != "all":
                raise ValueError(
                    "use_episode_first_frame_token=True currently requires "
                    "image_context_token_scope='all'."
                )
            if not (0 <= self.episode_first_frame_rope_index < 1024):
                raise ValueError(
                    "episode_first_frame_rope_index must be in [0, 1023], "
                    f"got {self.episode_first_frame_rope_index}."
                )
        valid_image_context_types = {"clip", "vae", "siglip2", "dinov2", "qwen3_vl"}
        if self.use_clip_image_encoder and self.image_context_type not in valid_image_context_types:
            raise ValueError(
                "image_context_type must be one of "
                f"{sorted(valid_image_context_types)} when use_clip_image_encoder is enabled, got {self.image_context_type!r}"
            )
        valid_image_context_token_scopes = {"all", "video", "action", "non_video", "non_action"}
        if self.image_context_token_scope not in valid_image_context_token_scopes:
            raise ValueError(
                "image_context_token_scope must be one of "
                f"{sorted(valid_image_context_token_scopes)}, got {self.image_context_token_scope!r}"
            )
        if self.image_context_type in {"siglip2", "dinov2", "qwen3_vl"}:
            assert self.image_context_token_scope == "all", (
                "new image encoders currently require image_context_token_scope='all'; "
                f"got {self.image_context_token_scope!r} for {self.image_context_type!r}"
            )
        if self.image_context_token_scope != "all":
            assert self.use_clip_image_encoder and self.image_context_type == "clip", "only allow clip."
            assert not config.use_past_action_context, (
                "image_context_token_scope != 'all' does not support past action context yet."
            )
        self.model.image_context_type = self.image_context_type
        self.model.image_context_token_scope = self.image_context_token_scope
        self.action_dim = config.action_dim # 32
        self.action_horizon = config.action_horizon # 24
        # 仅为兼容已有配置和 checkpoint 而保留；当前推理实际读取
        # self.num_inference_steps（默认 16，可通过 NUM_INFERENCE_STEPS 覆盖）。
        self.num_inference_timesteps = config.num_inference_timesteps

        if not config.skip_component_loading:
            text_enc_path = ensure_file(
                self.text_encoder.text_encoder_pretrained_path,
                "models_t5_umt5-xxl-enc-bf16.pth",
            )
            self.text_encoder.load_state_dict(torch.load(text_enc_path, map_location='cpu'))

            if self.use_clip_image_encoder and self.image_context_type == "clip":
                img_enc_path = ensure_file(
                    self.image_encoder.image_encoder_pretrained_path,
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                )
                self.image_encoder.model.load_state_dict(torch.load(img_enc_path, map_location='cpu'), strict=False)

        # Wan2.2 (WanVideoVAE38, z_dim=48) uses Wan2.2_VAE.pth; Wan2.1 uses Wan2.1_VAE.pth
        if not config.skip_component_loading:
            vae_hf_filename = "Wan2.2_VAE.pth" if getattr(self.vae, "z_dim", 16) == 48 else "Wan2.1_VAE.pth"
            vae_repo_id = WAN22_HF_REPO_ID if getattr(self.vae, "z_dim", 16) == 48 else WAN_HF_REPO_ID
            vae_path = ensure_file(
                self.vae.vae_pretrained_path,
                vae_hf_filename,
                repo_id=vae_repo_id,
            )
            self.vae.model.load_state_dict(torch.load(vae_path, map_location='cpu'))

        if not config.skip_component_loading:
            dit_dir = self.model.diffusion_model_pretrained_path
            # Wan2.2 (in_dim=48) uses Wan2.2-TI2V-5B repo; Wan2.1 uses Wan2.1-I2V-14B-480P
            dit_repo_id = WAN22_HF_REPO_ID if getattr(self.model, "in_dim", 16) == 48 else WAN_HF_REPO_ID
            if dit_dir is None or not os.path.isdir(dit_dir):
                index_path = hf_hub_download(repo_id=dit_repo_id, filename="diffusion_pytorch_model.safetensors.index.json")
                dit_dir = os.path.dirname(index_path)
                with open(index_path, 'r') as f:
                    index = json.load(f)
                for shard_file in set(index["weight_map"].values()):
                    hf_hub_download(repo_id=dit_repo_id, filename=shard_file)

            if dit_dir is not None:
                safetensors_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors")
                safetensors_index_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors.index.json")
                state_dict = {}

                if os.path.exists(safetensors_index_path):
                    # Handle sharded safetensors
                    print(f"Loading sharded safetensors using index: {safetensors_index_path}")

                    with open(safetensors_index_path, 'r') as f:
                        index = json.load(f)

                    # Load each shard
                    for shard_file in set(index["weight_map"].values()):
                        shard_path = os.path.join(dit_dir, shard_file)
                        print(f"Loading shard: {shard_path}")
                        shard_state_dict = load_file(shard_path)
                        state_dict.update(shard_state_dict)

                elif os.path.exists(safetensors_path):
                    # Handle single safetensors file
                    print(f"Loading weights from safetensors: {safetensors_path}")
                    state_dict = load_file(safetensors_path)

                else:
                    raise ValueError(f"No safetensors file found at {safetensors_path} or {safetensors_index_path}")

                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

                if getattr(self.model, "action_architecture", "base") == "split":
                    missing_key_set = set(missing_keys)
                    control_keys = {
                        key
                        for key in self.model.state_dict()
                        if ".self_attn.control_" in key
                    }
                    # 没有control模块，报错
                    if not control_keys:
                        raise RuntimeError(
                            "Split attention is enabled but the model has no control "
                            "attention parameters."
                        )

                    # 有额外的control模块，报错
                    loaded_control_keys = sorted(control_keys - missing_key_set)
                    if loaded_control_keys:
                        raise RuntimeError(
                            "Wan component weights unexpectedly contain split control "
                            f"attention parameters: {loaded_control_keys}"
                        )

                    missing_video_source_keys = sorted(
                        control_key.replace(
                            ".self_attn.control_", ".self_attn.", 1
                        )
                        for control_key in control_keys
                        if control_key.replace(
                            ".self_attn.control_", ".self_attn.", 1
                        )
                        in missing_key_set
                    )
                    # video attn缺失，报错
                    if missing_video_source_keys:
                        raise RuntimeError(
                            "Cannot initialize split control attention because Wan "
                            "component weights are missing video attention parameters: "
                            f"{missing_video_source_keys}"
                        )

                    self.model.initialize_control_attention_from_video()
                    missing_keys = [
                        key for key in missing_keys if key not in control_keys
                    ]
                    print(
                        "Initialized split control Q/K/V/O and QK-Norm from "
                        "the loaded Wan video attention weights"
                    )

                if missing_keys:    # 存在state、action和image相关参数
                    print(f"Missing keys when loading pretrained weights: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")

                print("Successfully loaded pretrained weights")
        else:
            print("Skipping individual component loading (loading from full pretrained model)")
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        # Video noise Beta distribution (biased towards high noise levels when enabled)
        self.video_beta_dist = Beta(config.video_noise_beta_alpha, config.video_noise_beta_beta)
        # High noise emphasis Beta distribution for coupled training (applies to both video and action)
        self.high_noise_beta_dist = Beta(config.high_noise_beta_alpha, 1.0)
        # self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self._noise_logged = False
        self.defer_lora_injection = config.defer_lora_injection
        print("defer_lora_injection@@", self.defer_lora_injection)
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def _reset_past_action_history(self):
        self.past_action_history = None

    def _append_past_action_history(self, action_chunk: torch.Tensor):
        # Expected shape: [batch, action_horizon, action_dim].
        if action_chunk is None:
            return
        if action_chunk.shape[-1] != self.action_dim:
            raise ValueError(
                f"past action dim mismatch: got {action_chunk.shape[-1]}, expected {self.action_dim}"
            )
        if action_chunk.shape[1] != self.action_horizon:
            raise ValueError(
                f"past action chunk length mismatch: got {action_chunk.shape[1]}, expected {self.action_horizon}"
            )
        if self.action_horizon != self.model.num_action_per_block:
            raise ValueError(
                "use_past_action_context currently expects action_horizon == "
                f"num_action_per_block, got {self.action_horizon} and {self.model.num_action_per_block}"
            )

        action_chunk = action_chunk.detach()
        if self.past_action_history is None:
            self.past_action_history = action_chunk
        else:
            self.past_action_history = torch.cat([self.past_action_history, action_chunk], dim=1)

        if self.model.local_attn_size != -1:
            max_chunks = max(1, (self.model.local_attn_size - 1) // self.num_frame_per_block)
            max_tokens = max_chunks * self.model.num_action_per_block
            self.past_action_history = self.past_action_history[:, -max_tokens:].detach()
        else:
            raise ValueError(
                "use_past_action_context requires finite local_attn_size; got local_attn_size=-1."
            )

    def _prepare_external_past_action_history(
        self,
        past_actions: torch.Tensor | None,
        *,
        current_start_frame: int | None = None,
    ):
        """Align external env-side action history to the active KV-cache window."""
        if past_actions is None:
            return None
        if past_actions.ndim == 2:
            past_actions = past_actions.unsqueeze(0)
        if past_actions.ndim != 3:
            raise ValueError(f"external past action history must be 3D, got {past_actions.shape}")
        if past_actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"external past action dim mismatch: got {past_actions.shape[-1]}, "
                f"expected {self.action_dim}"
            )
        if past_actions.shape[1] == 0:
            return None

        block = self.model.num_action_per_block
        usable_tokens = (past_actions.shape[1] // block) * block
        if usable_tokens == 0:
            return None
        if usable_tokens != past_actions.shape[1]:
            raise ValueError(
                "external past action history length must be divisible by "
                f"num_action_per_block={block}, got length={past_actions.shape[1]}."
            )

        if self.model.local_attn_size == -1:
            raise ValueError(
                "use_past_action_context requires finite local_attn_size; got local_attn_size=-1."
            )

        max_chunks = max(1, (self.model.local_attn_size - 1) // self.num_frame_per_block)
        if current_start_frame is None:
            chunks_since_anchor = max_chunks
        else:
            chunks_since_anchor = max(0, (current_start_frame - 1) // self.num_frame_per_block)
            chunks_since_anchor = min(chunks_since_anchor, max_chunks)
        if chunks_since_anchor == 0:
            return None

        max_tokens = chunks_since_anchor * block
        past_actions = past_actions[:, -max_tokens:]
        return past_actions.detach()

    def _log_past_action_context_once(self, message: str, *, infer: bool = False):
        flag_name = "_past_action_context_infer_logged" if infer else "_past_action_context_train_logged"
        if self.ip_rank == 0 and not getattr(self, flag_name):
            print(message)
            setattr(self, flag_name, True)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

        if self.train_architecture == "lora" and not self.defer_lora_injection:
            print("Adding LoRA to model")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
        elif self.train_architecture == "lora" and self.defer_lora_injection:
            print("Deferring LoRA injection until after pretrained weights are loaded")
        else:
            self.print_trainable_params()

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        if not self.defer_lora_injection:
            self.print_trainable_params()

    def print_trainable_params(self):
        """Print trainable parameters of the diffusion model."""
        trainable_params = []
        total_params = 0
        trainable_total = 0

        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params.append(name)
                trainable_total += param.numel()

        print(f"Total parameters in diffusion model: {total_params:,}")
        print(f"Trainable parameters in diffusion model: {trainable_total:,}")
        # print(trainable_params)


    def inject_lora_after_loading(self):
        """
        Inject LoRA adapters after pretrained weights have been loaded.
        This should be called when defer_lora_injection=True.
        """
        if self.train_architecture == "lora":
            print("Injecting LoRA after loading pretrained weights")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
            # self.model.registers.requires_grad_(True)
            # self.model.time_modality_projection.requires_grad_(True)

            self.text_encoder.requires_grad_(False)
            self.image_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.print_trainable_params()
        else:
            print("LoRA injection not needed (train_architecture != 'lora')")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_diffusion_model:
                self.model.eval()
            self.text_encoder.eval()
            self.image_encoder.eval()
            self.vae.eval()

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.dtype,
                computation_device='cuda',
            ),
        )

        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        print("offloadd")
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                # print("offload", module)
                                module.offload()
                    else:
                        print("tocpu")
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    print("onload")
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            # print("onload", module)
                            module.onload()
                else:
                    print("togpu")
                    model.to(self._device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

    def _create_kv_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_seqlen: int,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU KV cache for the Wan model.
        Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
        """
        num_heads = self.model.num_heads  # 40 for Wan2.1, 24 for Wan2.2
        head_dim = self.model.dim // self.model.num_heads  # 128 for both
        kv_cache1: KVCacheType = []
        kv_cache_neg: KVCacheType = []
        for _ in range(self.model.num_layers):
            kv_cache1.append(
                torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device),
            )
            kv_cache_neg.append(
                torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device),
            )

        return kv_cache1, kv_cache_neg

    def _create_crossattn_caches(
        self, batch_size: int, dtype: torch.dtype, device: torch.device,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
        """
        num_heads = self.model.num_heads  # 40 for Wan2.1, 24 for Wan2.2
        head_dim = self.model.dim // self.model.num_heads  # 128 for both
        crossattn_cache: KVCacheType = []
        crossattn_cache_neg: KVCacheType = []

        for _ in range(self.model.num_layers):
            crossattn_cache.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            )
            crossattn_cache_neg.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            )

        return crossattn_cache, crossattn_cache_neg

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def preprocess_image(self, image):
        image = (image * (2 / 255) - 1).permute(0, 1, 4, 2, 3)
        return image

    def encode_prompt(self, input_ids, attention_mask):
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(input_ids, attention_mask)
        prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb

    def _ensure_vae_on_device(self, ref_tensor):
        """Lazily move the VAE to the correct device/dtype on first use."""
        if not getattr(self, '_vae_device_ready', False):
            self.vae.to(device=ref_tensor.device, dtype=torch.bfloat16)
            self.vae.eval()
            self._vae_device_ready = True

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        self._ensure_vae_on_device(input_video)
        with torch.no_grad():
            latents = self.vae.encode(input_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def _encode_episode_first_frame_token(self, data: BatchFeature, *, target_h, target_w):
        """输入 episode_first_images: uint8 [B,T,H,W,C]，输出 ff_x: VAE latent [B,C,T,H,W]。
        ff_x 只作为 episode 首帧 context 进入 DiT attention，不参与 loss。"""
        if "episode_first_images" not in data:
            raise KeyError(
                "use_episode_first_frame_token=True requires batch key "
                "'episode_first_images'."
            )
        episode_first_images = data["episode_first_images"]
        if episode_first_images.ndim != 5:
            raise ValueError(
                "episode_first_images must have shape [B,T,H,W,C], "
                f"got {tuple(episode_first_images.shape)}."
            )
        if episode_first_images.dtype != torch.uint8:
            raise TypeError(
                "episode_first_images must be uint8 before video normalization, "
                f"got {episode_first_images.dtype}."
            )
        ff_videos = rearrange(episode_first_images, "b t h w c -> b c t h w")
        ff_videos = ff_videos.float() / 255.0
        b, c, t, h, w = ff_videos.shape
        ff_videos = ff_videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        ff_videos = self.normalize_video(ff_videos)
        ff_videos = ff_videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        ff_videos = ff_videos.to(dtype=self.dtype)
        if target_h is not None and target_w is not None:
            b, c, t, h, w = ff_videos.shape
            if (h, w) != (target_h, target_w):
                ff_videos = torch.nn.functional.interpolate(
                    ff_videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)
        return self.encode_video(
            ff_videos.to(dtype=self.dtype),
            self.tiled,
            (self.tile_size_height, self.tile_size_width),
            (self.tile_stride_height, self.tile_stride_width),
        )

    def encode_image(self, image, num_frames, height, width, encode_clip=True):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            batch_size = image.shape[0]
            image_context = None
            if encode_clip and self.image_context_type in {"clip", "siglip2", "dinov2", "qwen3_vl"}:
                image_context = self.image_encoder.encode_image(image)
            image_input = image.transpose(1, 2)
            image_zeros = torch.zeros(batch_size, 3, num_frames-1, height, width, dtype=torch.bfloat16, device=self._device)
            self._ensure_vae_on_device(image_input)
            with torch.no_grad():
                y = self.vae.encode(torch.concat([image_input, image_zeros], dim=2))
            # Build mask to match VAE output shape (VAE may use different spatial downsampling, e.g. WanVideoVAE38 uses patch_size=2 -> height/16)
            # y shape is B * 16 * (1+(T-1)/4) * H_latent * W_latent
            num_t = y.shape[2]
            h_latent, w_latent = y.shape[3], y.shape[4]
            msk = torch.zeros(batch_size, 4, num_t, h_latent, w_latent, dtype=y.dtype, device=self._device)
            msk[:, :, 0:1, :, :] = 1
            new_image = y[:, :, 0:1]
            if encode_clip and self.image_context_type == "vae":
                image_context = rearrange(new_image, "b c t h w -> b (t h w) c")
            # concat: B * (4+16) * (1+(T-1)/4) * H_latent * W_latent
            y = torch.concat([msk, y], dim=1)
        return image_context, y, new_image

    def _encode_inference_image(self, image, num_frames, height, width):
        if (
            self.use_clip_image_encoder
            or self.model.concat_first_frame_latent
            or self.model.model_type == "i2v"
        ):
            return self.encode_image(
                image, num_frames, height, width, encode_clip=self.use_clip_image_encoder
            )

        # The causal VAE's first latent does not depend on the unused zero tail.
        image_input = image.transpose(1, 2)
        self._ensure_vae_on_device(image_input)
        with torch.no_grad(), torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self._device).type
        ):
            first_frame_latent = self.vae.encode(image_input)
        return None, None, first_frame_latent

    def prepare_extra_input(self, latents=None):
        return {}

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming") -> nn.Module:
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = get_peft_model(model, lora_config)
        for param in model.parameters():
            param.data = param.to(torch.float32)
        return model

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        data = action_input
        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id
        # print("embodiment_id", embodiment_id)
        has_real_action = action_input.has_real_action
        action_mask = action_input.action_mask
        action_valid_mask = getattr(action_input, "action_valid_mask", None)
        image_valid_mask = getattr(action_input, "image_valid_mask", None)

        state_features = action_input.state

        actions = action_input.action

        _rank = self.rank if hasattr(self, "rank") else 0
        _step = self.global_step

        if actions.numel() > 0:
            assert action_valid_mask is not None, "action_valid_mask is required for padded action loss."
            action_valid_mask = action_valid_mask.to(device=actions.device, dtype=torch.bool)
        elif self.config.use_past_action_context:
            raise ValueError("use_past_action_context requires action training batches.")

        # ── 关键输入 NaN/Inf 检查（异常值置零，不跳过 batch）──────────────────
        actions = _check_and_clean(actions, "actions", _rank, _step)
        state_features = _check_and_clean(state_features, "state_features", _rank, _step)

        # q99 state/action normalization permits the configured [-2.0, 2.0] range.
        if actions.numel() > 0:
            assert actions.min() >= -2.0 and actions.max() <= 2.0, "actions must be in [-2.0,2.0] range"
        videos = data["images"]
        if not hasattr(self, "_shape_logged"):
            print(f"[SHAPE] images={tuple(videos.shape)}")
            self._shape_logged = True

        videos = rearrange(videos, "b t h w c -> b c t h w")
        # print("videos", videos.shape)       # videos torch.Size([1, 3, 33, 352, 640])

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)   # 0_1 -> -1_1
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)

        # shape of B * max_length * dim
        prompt_embs = self.encode_prompt(data["text"], data["text_attention_mask"])
        prompt_embs = _check_and_clean(prompt_embs, "prompt_embs", _rank, _step)

        # Resize video to the configured target resolution before VAE/image encoding.
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)

        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        ff_x = None
        if self.use_episode_first_frame_token:
            ff_x = self._encode_episode_first_frame_token(
                data,
                target_h=target_h,
                target_w=target_w,
            )
            if torch.rand((), device=ff_x.device).item() < 0.001:
                _dist_rank = dist.get_rank() if dist.is_initialized() else 0
                print(
                    "[FF_TOKEN] "
                    f"rank={_dist_rank} phase=train "
                    f"episode_first_images={tuple(data['episode_first_images'].shape)} "
                    f"ff_x={tuple(ff_x.shape)} "
                    f"rope_index={self.episode_first_frame_rope_index}",
                    flush=True,
                )

        latents = self.encode_video(videos, self.tiled, (self.tile_size_height, self.tile_size_width), (self.tile_stride_height, self.tile_stride_width))
        # VAE 编码后检查 latents（bfloat16 下溢/上溢是常见 NaN 来源）
        latents = _check_and_clean(latents, "video_latents", _rank, _step)

        # image_valid_mask: [B, T_image]；VAE 时间压缩后 latent_video_valid_mask: [B, T_latent]。
        # 当前 causal VAE 的时间关系为 T_latent = 1 + (T_image - 1) // 4（如 33 -> 9）。
        assert image_valid_mask is not None, "image_valid_mask is required for padded video dynamics loss."
        image_valid_mask = image_valid_mask.to(device=latents.device, dtype=torch.bool)
        valid_image_frames = image_valid_mask.sum(dim=1)
        valid_latent_frames = 1 + (valid_image_frames - 1) // 4
        latent_frame_indices = torch.arange(latents.shape[2], device=latents.device)[None, :]
        latent_video_valid_mask = latent_frame_indices < valid_latent_frames[:, None]
        expected_latent_frames = 1 + (image_valid_mask.shape[1] - 1) // 4
        assert expected_latent_frames == latents.shape[2], (
            f"image/latent temporal mismatch: image_valid_mask={image_valid_mask.shape}, "
            f"latents={latents.shape}"
        )

        # Training only needs image encoding when the model consumes its outputs.
        clip_feas, ys = None, None
        if (
            self.use_clip_image_encoder
            or self.model.concat_first_frame_latent
            or self.model.model_type == "i2v"
        ):
            _, _, num_frames, height, width = videos.shape
            image = videos[:, :, :1].transpose(1, 2)
            clip_feas, ys, _ = self.encode_image(
                image, num_frames, height, width, encode_clip=self.use_clip_image_encoder
            )

        latents = latents.to(self._device)
        if clip_feas is not None:
            clip_feas = clip_feas.to(device=self._device, dtype=latents.dtype)
        if ys is not None:
            ys = ys.to(self._device)
        prompt_embs = prompt_embs.to(self._device)
        if clip_feas is not None:
            clip_feas = _check_and_clean(clip_feas, "clip_feas", _rank, _step)
        if ys is not None:
            ys = ys.to(self._device)
            ys = _check_and_clean(ys, "image_condition_ys", _rank, _step)

        # Loss
        noise = torch.randn_like(latents)   # 1,48,9,10,20

        # specific to autoregressive
        noise = noise.transpose(1, 2)
        latents = latents.transpose(1, 2)

        # ============ VIDEO TIMESTEP SAMPLING ============
        if self.config.decouple_video_action_noise:
            # Decoupled mode follows the official DreamZero schedule: Beta(3, 1)
            # samples near 1, and scheduler index 0 corresponds to the largest
            # sigma, so (1 - ratio) biases video timesteps toward high noise.
            video_noise_ratio = self.video_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - video_noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "DECOUPLED"
        elif self.config.use_high_noise_emphasis:
            # High noise emphasis mode (coupled): BOTH video and action use Beta distribution
            noise_ratio = self.high_noise_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "HIGH_NOISE_EMPHASIS"
        else:
            # Original: uniform sampling over full range
            timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (noise.shape[0], noise.shape[1]))
            noise_mode = "STANDARD"

        timestep_id_block = timestep_id[:, 1:].reshape(
                    timestep_id.shape[0], -1, self.num_frame_per_block)
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]

        if actions.numel() > 0:
            noise_action = torch.randn_like(actions)
            assert actions.shape[1] / (noise.shape[1]-1) == (self.model.num_action_per_block // self.num_frame_per_block), f"actions.shape, {actions.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
            assert (noise.shape[1]-1) / state_features.shape[1] == (self.num_frame_per_block // self.model.num_state_per_block), f"state_features.shape, {state_features.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"

            # ============ ACTION TIMESTEP SAMPLING ============
            if self.config.decouple_video_action_noise:
                # Decoupled: sample action timestep independently with full range
                timestep_action_id = torch.randint(
                    0,
                    self.action_scheduler.num_train_timesteps,
                    (actions.shape[0], actions.shape[1])
                )
                action_mode = "INDEPENDENT"
            else:
                # Original coupled: action timestep derived from video timestep
                timestep_action_id = timestep_id_block.repeat(1, 1, actions.shape[1]//(noise.shape[1]-1))
                timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)
                action_mode = "COUPLED"

            # Log noise mode once
            if not self._noise_logged:
                video_mean = timestep_id.float().mean().item()
                action_mean = timestep_action_id.float().mean().item()
                shift_log = (
                    f"video_sigma_shift={self.sigma_shift:g} "
                    f"action_train_sigma_shift={self.action_train_sigma_shift:g} "
                    f"action_inference_sigma_shift={self.action_inference_sigma_shift:g}"
                )
                if noise_mode == "DECOUPLED":
                    print(
                        f"[NOISE] Mode={noise_mode} | "
                        f"Video: Beta({self.config.video_noise_beta_alpha},"
                        f"{self.config.video_noise_beta_beta}) high-noise bias "
                        f"mean_t={video_mean:.0f} | Action: {action_mode} Uniform "
                        f"mean_t={action_mean:.0f} | {shift_log}"
                    )
                elif noise_mode == "HIGH_NOISE_EMPHASIS":
                    print(
                        f"[NOISE] Mode={noise_mode} | "
                        f"Video+Action: Beta({self.config.high_noise_beta_alpha},1) "
                        f"mean_t={video_mean:.0f} | Action: {action_mode} | {shift_log}"
                    )
                else:
                    print(
                        f"[NOISE] Mode={noise_mode} | Video+Action: Uniform "
                        f"mean_t={video_mean:.0f} | Action: {action_mode} | {shift_log}"
                    )
                self._noise_logged = True
        else:
            noise_action = None
            timestep_action_id = None

        timestep_id_block = timestep_id_block.reshape(timestep_id_block.shape[0], -1)
        timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block], dim=1)
        _, num_frames, num_channels, height, width = noise.shape
        # DiT patch_embedding uses stride (1,2,2), so sequence length is num_frames * (H//2) * (W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        seq_len = num_frames * tokens_per_frame

        timestep = self.scheduler.timesteps[timestep_id].to(self._device)
        noisy_latents = self.scheduler.add_noise(latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = self.scheduler.training_target(latents, noise, timestep).transpose(1, 2)

        if actions.numel() > 0:
            timestep_action = self.action_scheduler.timesteps[timestep_action_id].to(self._device)
            noisy_actions = self.action_scheduler.add_noise(
                actions.flatten(0, 1),
                noise_action.flatten(0, 1),
                timestep_action.flatten(0, 1),
            ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
            training_target_action = self.action_scheduler.training_target(
                actions,
                noise_action,
                timestep_action,
            )
        else:
            timestep_action = None
            noisy_actions = None
            training_target_action = None

        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            if actions.numel() > 0:
                if torch.rand((), device=actions.device).item() < 0.001:
                    _dist_rank = dist.get_rank() if dist.is_initialized() else 0
                    print(
                        "[ATTENTION_MODE] "
                        f"rank={_dist_rank} phase=train "
                        f"mode={'V1' if self.video_condition_on_action_state else 'V0'}_"
                        f"{'A1' if self.config.use_past_action_context else 'A0'} "
                        f"video_condition_on_action_state={self.video_condition_on_action_state} "
                        f"use_past_action_context={self.config.use_past_action_context}",
                        flush=True,
                    )
                past_action_keep_tokens = None
                if self.config.use_past_action_context:
                    drop_ratio = float(getattr(self.config, "past_action_context_drop_ratio", 0.0))
                    if drop_ratio < 0.0 or drop_ratio > 1.0:
                        raise ValueError(
                            "past_action_context_drop_ratio must be in [0, 1], "
                            f"got {drop_ratio}."
                        )
                    if getattr(self.config, "past_action_context_dropout_train", False) and drop_ratio > 0.0:
                        total_tokens = actions.shape[1]
                        drop_tokens = int(total_tokens * drop_ratio)
                        past_action_keep_tokens = total_tokens - drop_tokens
                    self._log_past_action_context_once(
                        "[PAST_ACTION_CONTEXT] enabled train: "
                        f"clean_video={tuple(latents.transpose(1, 2).shape)}, "
                        f"noisy_video={tuple(noisy_latents.transpose(1, 2).shape)}, "
                        f"noisy_action={tuple(noisy_actions.shape)}, "
                        f"state={tuple(state_features.shape)}, "
                        f"past_action={tuple(actions.shape)}, "
                        f"past_action_keep_tokens={past_action_keep_tokens}"
                    )
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    action=noisy_actions, timestep_action=timestep_action,
                    action_feature_mask=action_mask,
                    clean_x=latents.transpose(1, 2),
                    past_actions=actions if self.config.use_past_action_context else None,
                    past_action_keep_tokens=past_action_keep_tokens,
                    ff_x=ff_x,
                    ff_rope_frame_index=self.episode_first_frame_rope_index,
                )
            else:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, timestep_action=timestep_action,
                    clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    clean_x=latents.transpose(1, 2),
                    ff_x=ff_x,
                    ff_rope_frame_index=self.episode_first_frame_rope_index,
                )

            # Per-sample dynamics loss
            # DiT patch_embedding uses stride (1,2,2), so output spatial size can be smaller than
            # latent when H or W is odd (e.g. latent 11x20 -> model output 10x20). Crop target to match.
            if training_target.shape != video_noise_pred.shape:
                training_target = training_target[
                    ..., : video_noise_pred.shape[3], : video_noise_pred.shape[4]
                ]
            dynamics_loss_per_sample = torch.nn.functional.mse_loss(
                video_noise_pred.float(), training_target.float(), reduction='none'
            ).mean(dim=(1,3,4))  # shape: [B, num_frames]

            weight_dynamics = dynamics_loss_per_sample * self.scheduler.training_weight(timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1])).to(self._device)  # [B, T_latent] = [B, T_latent] * [B, T_latent]
            # weight_dynamics / latent_video_valid_mask: [B, T_latent]；仅有效视频 latent 帧进入 loss。
            dynamics_valid_mask = latent_video_valid_mask.to(device=weight_dynamics.device, dtype=weight_dynamics.dtype)  # [B, T_latent]
            weighted_dynamics_loss = (weight_dynamics * dynamics_valid_mask).sum() / dynamics_valid_mask.sum().clamp_min(1.0)  # scalar = ([B, T_latent] * [B, T_latent]).sum() / scalar
            # ── dynamics loss NaN/Inf 检查 ──────────────────────────────────
            weighted_dynamics_loss = _check_loss(weighted_dynamics_loss, "dynamics_loss", _rank, _step)

            if actions.numel() > 0:
                action_loss_per_sample = torch.nn.functional.mse_loss(
                    action_noise_pred.float(), training_target_action.float(), reduction='none'
                )  # [B, T_action, D_action]
                # action feat (overall) = action mask (feat padding & action qulaity) * valid mask (temporal padding)
                action_feature_mask = action_mask.float() * action_valid_mask[:, :, None].float()  # [B, T_action, D_action] = [B, T_action, D_action] * [B, T_action, 1]
                action_loss_per_sample = action_loss_per_sample * action_feature_mask  # [B, T_action, D_action] = [B, T_action, D_action] * [B, T_action, D_action]
                action_loss_per_step = action_loss_per_sample.sum(dim=2) / action_feature_mask.sum(dim=2).clamp_min(1.0)  # [B, T_action] = [B, T_action] / [B, T_action]

                action_step_mask = (
                    has_real_action[:, None].float()        # for video only masking
                    * action_valid_mask.float()             # for temporal padding
                    * action_mask.any(dim=2).float()        # for feature/quality masking
                )  # [B, T_action]
                weight_action = action_loss_per_step * self.action_scheduler.training_weight(
                    timestep_action.flatten(0, 1),
                ).unflatten(0, (noise_action.shape[0], noise_action.shape[1])).to(self._device)  # [B, T_action] = [B, T_action] * [B, T_action]
                weighted_action_loss = (weight_action * action_step_mask).sum() / action_step_mask.sum().clamp_min(1.0)  # scalar = ([B, T_action] * [B, T_action]).sum() / scalar
                # ── action loss NaN/Inf 检查 ────────────────────────────────
                weighted_action_loss = _check_loss(weighted_action_loss, "action_loss", _rank, _step)
                loss = weighted_dynamics_loss + weighted_action_loss
            else:
                weighted_action_loss = torch.tensor(0.0, device=self._device)
                loss = weighted_dynamics_loss

            # ── 最终 total loss NaN/Inf 检查（backward 前最后一道防线）───────
            loss = _check_loss(loss, "total_loss", _rank, _step)

        # Record log
        output_dict = {
            "loss": loss,
            "dynamics_loss": weighted_dynamics_loss,
            "action_loss": weighted_action_loss,
        }

        return BatchFeature(data=output_dict)

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise

    def _get_caches(
        self, kv_caches_input: list[KVCacheType],
    ) -> list[KVCacheType]:
        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            assert len(kv_caches_input) == 2
            if self.ip_rank == 0:
                kv_caches = [kv_caches_input[0]]
            else:
                kv_caches = [kv_caches_input[1]]
        else:
            assert len(kv_caches_input) <= 2
            kv_caches = [kv_caches_input[0]]
            if self.cfg_scale != 1.0:
                kv_caches.append(kv_caches_input[1])
        return kv_caches

    def _prepare_text_inputs(self, data: BatchFeature) -> list[tuple[torch.Tensor, torch.Tensor]]:

        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            if self.ip_rank == 0:
                text_inputs = [(data["text"], data["text_attention_mask"])]
            else:
                text_inputs = [(data["text_negative"], data["text_attention_mask_negative"])]
        else:
            text_inputs = [(data["text"], data["text_attention_mask"])]
            if self.cfg_scale != 1.0:
                text_inputs.append((data["text_negative"], data["text_attention_mask_negative"]))
        return text_inputs


    def _run_diffusion_steps(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        kv_caches: list[KVCacheType],
        crossattn_caches: list[KVCacheType],
        kv_cache_metadata: dict[str, bool | int],
        past_actions: torch.Tensor | None = None,
        ff_x: torch.Tensor | None = None,
        ff_rope_frame_index: int | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        predictions = []
        for prompt_index, prompt_emb in enumerate(context):
            kv_cache = kv_caches[prompt_index]
            crossattn_cache = crossattn_caches[prompt_index]
            use_trt = (
                not kv_cache_metadata["update_kv_cache"]
                and self.trt_engine is not None
                and past_actions is None
                and ff_x is None
            )
            if use_trt:
                obs_noise_pred, action_noise_pred = self.trt_engine(
                    noisy_input,
                    timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    context=prompt_emb,
                    y=y,
                    clip_feature=clip_feature,
                    kv_cache=kv_cache,
                )
            else:
                with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
                    obs_noise_pred, action_noise_pred, updated_kv_caches = self.model(
                        noisy_input,
                        timestep,
                        action=action,
                        timestep_action=timestep_action,
                        state=state,
                        embodiment_id=embodiment_id,
                        context=prompt_emb,
                        seq_len=seq_len,
                        y=y,
                        clip_feature=clip_feature,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start_frame=kv_cache_metadata["start_frame"],
                        past_actions=past_actions,
                        ff_x=ff_x,
                        ff_rope_frame_index=ff_rope_frame_index,
                    )
                if kv_cache_metadata["update_kv_cache"]:
                    for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                        kv_cache[block_index] = updated_kv_cache.clone()
            obs_noise_pred = obs_noise_pred.clone()
            if action_noise_pred is not None:
                action_noise_pred = action_noise_pred.clone()
            else:
                action_noise_pred = torch.tensor(0.0, device=obs_noise_pred.device) # dummy action noise prediction
            predictions.append((obs_noise_pred, action_noise_pred))
        return self._exchange_predictions(predictions)

    def _exchange_predictions(
        self,
        predictions: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.ip_size == 1:
            return predictions

        assert len(predictions) == 1
        my_predictions = list(predictions[0])

        other_predictions = [torch.empty_like(pred) for pred in my_predictions]

        send_ops = [
            dist.P2POp(op=dist.isend, tensor=pred, group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group)
            for pred in my_predictions
        ]
        recv_ops = [
            dist.P2POp(op=dist.irecv, tensor=other_pred, group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group)
            for other_pred in other_predictions
        ]
        ops = send_ops + recv_ops

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        output_predictions: list[tuple[torch.Tensor, torch.Tensor] | None] = [None for _ in range(self.ip_size)]
        output_predictions[self.ip_rank] = tuple(my_predictions)
        output_predictions[(self.ip_rank + 1) % self.ip_size] = tuple(other_predictions)
        assert all(isinstance(pred, tuple) for pred in output_predictions)
        return cast(list[tuple[torch.Tensor, torch.Tensor]], output_predictions)

    def should_run_model(self, index, current_timestep, prev_predictions):

        # The legacy skip schedules were tuned for 16 denoising steps only.
        # Other step counts always recompute predictions, including when
        # video-only inference overrides num_inference_steps after loading.
        if self.num_inference_steps != 16:
            return True

        if not self.dynamic_cache_schedule:
            return self.dit_step_mask[index]

        # Always run first 2 steps to establish history
        if len(prev_predictions) < 2:
            return True

        if self.skip_countdown > 1:
            self.skip_countdown -= 1
            return False
        elif self.skip_countdown == 1:
            self.skip_countdown = 0
            return True

        v_last = prev_predictions[-1][1].flatten(1).float()
        v_prev = prev_predictions[-2][1].flatten(1).float()
        sim = torch.nn.functional.cosine_similarity(v_last, v_prev, dim=1).mean()

        thresholds = [0.95, 0.93]
        countdowns = [4, 2]

        for threshold, countdown in zip(thresholds, countdowns):
            if sim > threshold:
                self.skip_countdown = countdown
                return False

        return True

    def lazy_joint_video_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        latent_video: torch.Tensor | None = None,
        video_only: bool = False,
        external_past_actions: torch.Tensor | None = None,
    ) -> BatchFeature:
        start_time = time.perf_counter()
        if self.config.use_past_action_context and video_only:
            raise ValueError("use_past_action_context is incompatible with video_only=True.")
        if self.config.use_past_action_context and self.action_horizon != self.model.num_action_per_block:
            raise ValueError(
                "use_past_action_context currently expects action_horizon == num_action_per_block, "
                f"got {self.action_horizon} and {self.model.num_action_per_block}."
            )
        if not video_only:
            if torch.rand(()).item() < 0.001:
                _dist_rank = dist.get_rank() if dist.is_initialized() else 0
                print(
                    "[ATTENTION_MODE] "
                    f"rank={_dist_rank} phase=infer "
                    f"mode={'V1' if self.video_condition_on_action_state else 'V0'}_"
                    f"{'A1' if self.config.use_past_action_context else 'A0'} "
                    f"video_condition_on_action_state={self.video_condition_on_action_state} "
                    f"use_past_action_context={self.config.use_past_action_context}",
                    flush=True,
                )
        if external_past_actions is not None and not self.config.use_past_action_context:
            self._log_past_action_context_once(
                "[OBS_ACTION_HISTORY] received external past actions, but "
                "use_past_action_context=False; ignoring.",
                infer=True,
            )

        # Tracking time taken on GPU for various operations.
        start_text_encoder_event = torch.cuda.Event(enable_timing=True)
        end_text_encoder_event = torch.cuda.Event(enable_timing=True)
        start_image_encoder_event = torch.cuda.Event(enable_timing=True)
        end_image_encoder_event = torch.cuda.Event(enable_timing=True)
        start_vae_event = torch.cuda.Event(enable_timing=True)
        end_vae_event = torch.cuda.Event(enable_timing=True)
        start_kv_event = torch.cuda.Event(enable_timing=True)
        end_kv_event = torch.cuda.Event(enable_timing=True)
        start_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_inference_steps)]
        end_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_inference_steps)]

        self.set_frozen_modules_to_eval_mode()
        data = action_input

        videos = data["images"]

        embodiment_id = action_input.embodiment_id
        state_features = action_input.state

        videos = rearrange(videos, "b t h w c -> b c t h w")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            videos = videos.to(dtype=self.dtype)
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)

        state_features = state_features.to(dtype=torch.bfloat16)
        videos = videos.to(dtype=torch.bfloat16)

        # Wan 5B: resize to target resolution so latent matches DiT
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        ff_x = None
        if self.use_episode_first_frame_token:
            ff_x = self._encode_episode_first_frame_token(
                data,
                target_h=target_h,
                target_w=target_w,
            )
            if torch.rand((), device=ff_x.device).item() < 0.001:
                _dist_rank = dist.get_rank() if dist.is_initialized() else 0
                print(
                    "[FF_TOKEN] "
                    f"rank={_dist_rank} phase=infer "
                    f"episode_first_images={tuple(data['episode_first_images'].shape)} "
                    f"ff_x={tuple(ff_x.shape)} "
                    f"rope_index={self.episode_first_frame_rope_index}",
                    flush=True,
                )

        # Plan A: when the KV cache is full and we have to reset mid-episode,
        # we must re-anchor on the LATEST observation frame (current env step),
        # not the oldest one in the received window. Otherwise first_image and
        # state would fall on different env timesteps (e.g., first_image@env150
        # vs state@env192), which is out-of-distribution w.r.t. training where
        # they always share the same anchor A.
        cache_full_reset = False
        if self.language is None:
            print("language is None, reset current_start_frame to 0")
            self.language = data["text"]
            self.current_start_frame = 0
            self._reset_past_action_history()
        elif not torch.equal(self.language, data["text"]):
            print("language changed, reset current_start_frame to 0")
            self.current_start_frame = 0
            self.language = data["text"]
            self._reset_past_action_history()
        elif videos.shape[2] == 1:
            print("videos.shape[2] == 1, reset current_start_frame to 0")
            self.current_start_frame = 0
            self._reset_past_action_history()
        elif self.current_start_frame >= self.model.local_attn_size:
            print(
                "current_start_frame >= local_attn_size, reset current_start_frame to 0 "
                "(cache-full reset; anchoring on latest frame)"
            )
            self.current_start_frame = 0
            cache_full_reset = True
            self._reset_past_action_history()

        external_past_action_history = None
        if self.config.use_past_action_context and external_past_actions is not None:
            external_past_action_history = self._prepare_external_past_action_history(
                external_past_actions,
                current_start_frame=self.current_start_frame,
            )
            if external_past_action_history is not None:
                self._log_past_action_context_once(
                    "[OBS_ACTION_HISTORY] enabled infer: "
                    f"current_start_frame={self.current_start_frame}, "
                    f"external_past_action_history={tuple(external_past_action_history.shape)}",
                    infer=True,
                )
        past_action_context = (
            external_past_action_history
            if external_past_actions is not None
            else self.past_action_history
        )

        if self.ip_rank == 0:
            print("videos shape", videos.shape, self.num_frames)

        start_text_encoder_event.record()

        text_inputs = self._prepare_text_inputs(data)
        prompt_embs = [self.encode_prompt(text, attention_mask) for text, attention_mask in text_inputs]

        end_text_encoder_event.record()

        start_image_encoder_event.record()

        _, _, num_frames, height, width = videos.shape
        if videos.shape[2] == 4 or videos.shape[2] == 9:
            # special case for real-world eval where language is updated
            image = videos[:, :, -1:].transpose(1, 2)
        elif cache_full_reset:
            # Plan A: pick the latest (current env-step) frame so that
            # first_image, state_features, and the next predicted chunk all
            # share the same temporal anchor. The older 7 frames in the
            # received window are intentionally discarded here.
            image = videos[:, :, -1:].transpose(1, 2)
        else:
            image = videos[:, :, :1].transpose(1, 2)

        if self.current_start_frame == 0:
            clip_feas, ys, image = self._encode_inference_image(
                image, self.num_frames, height, width
            )
            self.clip_feas = clip_feas.to(dtype=image.dtype) if clip_feas is not None else None
            self.ys = ys.to(dtype=image.dtype) if ys is not None else None

        if self.model.concat_first_frame_latent or self.model.model_type == "i2v":
            assert self.ys is not None, "ys must be set when image latent conditioning is enabled"
        if self.use_clip_image_encoder:
            assert self.clip_feas is not None, "clip_feas must be set when use_clip_image_encoder is enabled"

        end_image_encoder_event.record()

        start_vae_event.record()

        if latent_video is not None and self.current_start_frame != 0:
            image = latent_video
            if self.ip_rank == 0:
                print("image shape@@", image.shape)
        elif self.current_start_frame != 0:
            # this is for real world execution
            if (videos.shape[2] - 1) // 4 == self.num_frame_per_block:
                print("no further action")
            elif videos.shape[2] // 4 != self.num_frame_per_block:
                # Repeating videos along dim 2.
                repeat_factor = self.num_frame_per_block // (videos.shape[2] // 4)
                videos = torch.repeat_interleave(videos, repeat_factor, dim=2)

                first_frame = videos[:, :, 0:1]  # Extract first frame
                videos = torch.cat([first_frame, videos], dim=2)
            else:
                first_frame = videos[:, :, 0:1]  # Extract first frame
                videos = torch.cat([first_frame, videos], dim=2)

            image = self.vae.encode(
                videos,
                tiled=self.tiled,
                tile_size=(self.tile_size_height, self.tile_size_width),
                tile_stride=(self.tile_stride_height, self.tile_stride_width),
            )

        end_vae_event.record()

        noise_obs = self.generate_noise(
            (image.shape[0], image.shape[1], self.num_frame_per_block, image.shape[3], image.shape[4]),
            seed=self.seed,
            device='cuda',
            dtype=torch.bfloat16,
        )
        noise_action = self.generate_noise((image.shape[0], self.action_horizon, self.model.action_dim), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        batch_size, num_channels, num_frames, height, width = noise_obs.shape

        ######### Generate video #########
        # DiT patch_embedding uses stride (1,2,2), so tokens per frame = (H//2)*(W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        frame_seqlen = tokens_per_frame
        seq_len = num_frames * frame_seqlen

        image = image.transpose(1, 2)
        noise_obs = noise_obs.transpose(1, 2)

        if self.current_start_frame == 0:
            # Reinitialize KV cache and crossattn cache for each new sequence.
            self.kv_cache1, self.kv_cache_neg = self._create_kv_caches(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
                frame_seqlen=frame_seqlen,
            )
            self.crossattn_cache, self.crossattn_cache_neg = self._create_crossattn_caches(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
            )

        assert self.kv_cache1 is not None
        assert self.kv_cache_neg is not None
        assert self.crossattn_cache is not None
        assert self.crossattn_cache_neg is not None
        kv_caches = self._get_caches(
            [self.kv_cache1, self.kv_cache_neg],
        )
        crossattn_caches = self._get_caches(
            [self.crossattn_cache, self.crossattn_cache_neg],
        )

        start_kv_event.record()

        if self.current_start_frame == 0:
            timestep = torch.ones([batch_size, 1], device=noise_obs.device, dtype=torch.int64) * 0
            self._run_diffusion_steps(
                noisy_input=image.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=frame_seqlen,
                y=self.ys[:, :, 0:1] if self.ys is not None else None,
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=0,
                    update_kv_cache=True,
                ),
                ff_x=ff_x,
                ff_rope_frame_index=self.episode_first_frame_rope_index,
            )
            self.current_start_frame += 1

        timestep = torch.ones([batch_size, self.num_frame_per_block], device=noise_obs.device, dtype=torch.int64) * 0

        if self.current_start_frame != 1:
            current_ref_latents = image[:, -self.num_frame_per_block:]
            if self.ys is not None:
                if self.current_start_frame <= self.ys.shape[2]:
                    y = self.ys[:, :, self.current_start_frame - self.num_frame_per_block : self.current_start_frame]
                else:
                    y = self.ys[:, :, -self.num_frame_per_block:]
            else:
                y = None
            self._run_diffusion_steps(
                noisy_input=current_ref_latents.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=seq_len,
                y=y,
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=self.current_start_frame - self.num_frame_per_block,
                    update_kv_cache=True,
                ),
                ff_x=ff_x,
                ff_rope_frame_index=self.episode_first_frame_rope_index,
            )

        end_kv_event.record()

        noisy_input = noise_obs
        noisy_input_action = noise_action

        # Step 3.1: Spatial denoising loop

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.action_scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)
        sample_scheduler_action.set_timesteps(
            self.num_inference_steps,
            device=noise_obs.device,
            shift=self.action_inference_sigma_shift,
        )
        # Decoupled inference: video sigmas end at video_final_noise instead of 0
        # This rescales the schedule so video still takes all denoising steps,
        # but ends at a higher noise level (e.g., 1.0 → 0.9 → 0.8 instead of 1.0 → 0.5 → 0.0)
        if self.config.decouple_inference_noise:
            video_final_noise = self.config.video_inference_final_noise
            # Rescale video sigmas: map [sigma_max, 0] -> [sigma_max, video_final_noise]
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max + video_final_noise
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64)
            if self.ip_rank == 0:
                print(f"Decoupled inference: video sigmas {sigma_max:.3f} -> {sample_scheduler.sigmas[-1].item():.3f}")

        if not self._inference_scheduler_logged:
            if not dist.is_initialized() or dist.get_rank() == 0:
                middle_index = self.num_inference_steps // 2
                print(
                    "[INFERENCE_SCHEDULER] "
                    f"video_shift={self.sigma_shift:g} "
                    f"action_train_shift={self.action_train_sigma_shift:g} "
                    f"action_inference_shift={self.action_inference_sigma_shift:g} "
                    f"steps={self.num_inference_steps} "
                    f"middle_index={middle_index} "
                    f"video_sigma={sample_scheduler.sigmas[middle_index].item():.6f} "
                    f"action_sigma={sample_scheduler_action.sigmas[middle_index].item():.6f} "
                    f"video_timestep={sample_scheduler.timesteps[middle_index].item()} "
                    f"action_timestep={sample_scheduler_action.timesteps[middle_index].item()}",
                    flush=True,
                )
            self._inference_scheduler_logged = True

        start_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps]
        end_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps]
        prev_predictions = []
        self.skip_countdown = 0
        dit_compute_steps = 0
        for index, current_timestep in enumerate(sample_scheduler.timesteps):
            start_diffusion_events[index].record()

            # Get timesteps from respective schedulers
            action_timestep = sample_scheduler_action.timesteps[index]
            video_timestep = sample_scheduler.timesteps[index]  # Already rescaled if decoupled

            # set current timestep
            timestep = torch.ones(
                [batch_size, self.num_frame_per_block],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * video_timestep
            timestep_action = torch.ones(
                [batch_size, self.action_horizon],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * action_timestep

            # check if we need to run the DIT step
            should_run_model = self.should_run_model(index, current_timestep, prev_predictions)
            if should_run_model:
                dit_compute_steps += 1
                if self.ys is not None:
                    if self.current_start_frame + self.num_frame_per_block <= self.ys.shape[2]:
                        y = self.ys[:, :, self.current_start_frame : self.current_start_frame + self.num_frame_per_block]
                    else:
                        y = self.ys[:, :, -self.num_frame_per_block:]
                else:
                    y = None
                predictions = self._run_diffusion_steps(
                    noisy_input=noisy_input.transpose(1, 2),
                    timestep=timestep,
                    action=None if video_only else noisy_input_action,
                    timestep_action=None if video_only else timestep_action,
                    state=None if video_only else state_features,
                    embodiment_id=None if video_only else embodiment_id,
                    context=prompt_embs,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=self.clip_feas,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    kv_cache_metadata=dict(
                        start_frame=self.current_start_frame,
                        update_kv_cache=False,
                    ),
                    past_actions=past_action_context if self.config.use_past_action_context else None,
                    ff_x=ff_x,
                    ff_rope_frame_index=self.episode_first_frame_rope_index,
                )
                flow_pred_cond, flow_pred_cond_action = predictions[0]
                flow_pred_uncond, flow_pred_uncond_action = predictions[1]

                flow_pred = flow_pred_uncond + self.cfg_scale * (flow_pred_cond - flow_pred_uncond)
                prev_predictions.append((current_timestep, flow_pred, flow_pred_cond_action))
                max_cache_size = 2
                if len(prev_predictions) > max_cache_size:
                    prev_predictions.pop(0)

            else:
                assert len(prev_predictions) > 0, "prev_predictions must be set when skipping"
                _, flow_pred, flow_pred_cond_action = prev_predictions[-1]

            end_diffusion_events[index].record()

            # Video: denoising step (uses rescaled schedule if decoupled)
            noisy_input = sample_scheduler.step(
                model_output=flow_pred.transpose(1, 2),
                timestep=video_timestep,
                sample=noisy_input,
                step_index=index,
                return_dict=False,
            )[0]

            # Action: fully denoise only for labeled policy inference.
            # In video-only mode DiT receives action=None and returns a scalar
            # placeholder, so stepping the action scheduler would mismatch shape.
            if not video_only:
                noisy_input_action = sample_scheduler_action.step(
                    model_output=flow_pred_cond_action,
                    timestep=action_timestep,
                    sample=noisy_input_action,
                    step_index=index,
                    return_dict=False,
                )[0]

        latents = noisy_input
        latents_action = noisy_input_action
        output = latents

        if self.config.use_past_action_context:
            assert latents_action is not None, "past action context requires action prediction."
            if external_past_actions is None:
                self._append_past_action_history(latents_action)
                self._log_past_action_context_once(
                    "[PAST_ACTION_CONTEXT] enabled infer: "
                    f"past_action_history={tuple(self.past_action_history.shape)}",
                    infer=True,
                )
            elif external_past_action_history is None:
                if self.ip_rank == 0 and not getattr(self, "_obs_action_history_too_short_logged", False):
                    print(
                        "[OBS_ACTION_HISTORY] external history present but too short for one action block; "
                        "skipping predicted-history append."
                    )
                    self._obs_action_history_too_short_logged = True

        if self.current_start_frame == 1:
            output = torch.cat([image, output], dim=1)
        self.current_start_frame += self.num_frame_per_block

        # Do torch.cuda.synchronize() to ensure all operations are completed before timing.
        # This isn't expected to affect inference performance since it's at the end of an inference step.
        torch.cuda.synchronize()

        total_time = time.perf_counter() - start_time
        text_encoder_time = start_text_encoder_event.elapsed_time(end_text_encoder_event) / 1000
        image_encoder_time = start_image_encoder_event.elapsed_time(end_image_encoder_event) / 1000
        vae_time = start_vae_event.elapsed_time(end_vae_event) / 1000
        kv_creation_time = start_kv_event.elapsed_time(end_kv_event) / 1000
        diffusion_times = [s.elapsed_time(e) for s, e in zip(start_diffusion_events, end_diffusion_events)]
        diffusion_time = sum(diffusion_times) / 1000
        scheduler_time = total_time - kv_creation_time - diffusion_time - text_encoder_time - image_encoder_time - vae_time

        if self.ip_rank == 0:
            print(f"Time taken: Total {total_time:.2f} seconds, "
                  f"Text Encoder {text_encoder_time:.2f} seconds, "
                  f"Image Encoder {image_encoder_time:.2f} seconds, "
                  f"VAE {vae_time:.2f} seconds, "
                  f"KV Cache Creation {kv_creation_time:.2f} seconds, "
                  f"Diffusion {diffusion_time:.2f} seconds, "
                  f"DIT Compute Steps {dit_compute_steps} steps, "
                  f"Scheduler {scheduler_time:.2f} seconds")

        result = {"action_pred": latents_action, "video_pred": output.transpose(1, 2)}

        return BatchFeature(data=result)

    def cache_predict_order1(self, current_timestep, timestep_1, f1, timestep_2, f2):
        h_curr = current_timestep - timestep_1
        h_past = timestep_1 - timestep_2

        v_prime = (f1 - f2) / h_past

        # Prediction
        damping_factor = 0.25
        flow_pred = f1 + (v_prime * h_curr) * damping_factor
        return flow_pred

    def post_initialize(self):
        # Move models to the cuda device and set the dtype to bfloat16.
        print("Moving models to the cuda device and setting the dtype to bfloat16.")
        self.model.to(device=self._device, dtype=torch.bfloat16)
        self.text_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.image_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.vae.to(device=self._device, dtype=torch.bfloat16)
        import os
        ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"
        LOAD_TRT_ENGINE = os.getenv("LOAD_TRT_ENGINE", None)

        # Compile the encoders and VAE; keep _forward_blocks eager for variable token counts.
        if not ENABLE_TENSORRT:
            print("Torch compiling the TextEncoder, ImageEncoder, and VAE modules (Wan _forward_blocks not compiled).")

            self.text_encoder.forward = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.text_encoder.forward)

            if self.image_context_type == "clip":
                self.image_encoder.model.visual.forward = torch.compile(
                    mode="reduce-overhead", fullgraph=True, dynamic=False,
                )(self.image_encoder.model.visual.forward)

            self.vae.model.encode = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.vae.model.encode)

        self.trt_engine = None
        if LOAD_TRT_ENGINE is not None:
            print(f"Loading TRT engine from {LOAD_TRT_ENGINE}")
            import groot.control.tensorrt_utils as trt_utils
            model_path = LOAD_TRT_ENGINE
            self.trt_engine = trt_utils.load_tensorrt_engine(model_path, model_type="ar_14B")

    def parallelize(self, device_mesh: DeviceMesh) -> None:
        ip_mesh = device_mesh["ip"]
        self.ip_rank = ip_mesh.get_local_rank()
        self.ip_size = ip_mesh.size()
        self.ip_group = ip_mesh.get_group()

        assert self.ip_size == 1 or self.ip_size == 2, "ip_size must be 1 or 2"
        assert self.ip_rank >= 0 and self.ip_rank < self.ip_size, "ip_rank must be in [0, ip_size)"

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
