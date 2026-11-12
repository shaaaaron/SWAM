from __future__ import annotations

import os
import random

import torch
import torch.nn as nn


def _model_param(module: nn.Module) -> nn.Parameter:
    return next(module.parameters())


class _FrozenImageEncoder(nn.Module):
    def __init__(self, image_encoder_pretrained_path: str | None = None):
        super().__init__()
        self.image_encoder_pretrained_path = image_encoder_pretrained_path

    def _freeze(self) -> None:
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

class WanSigLIP2ImageEncoder(_FrozenImageEncoder):
    def __init__(self, image_encoder_pretrained_path: str | None = None):
        super().__init__(image_encoder_pretrained_path)
        from transformers import AutoConfig, AutoProcessor, SiglipVisionModel

        if image_encoder_pretrained_path is None:
            raise ValueError("image_encoder_pretrained_path is required for SigLIP2")

        config = AutoConfig.from_pretrained(image_encoder_pretrained_path, local_files_only=True)
        # The local siglip2-so400m-patch14-224 checkpoint is published with
        # the HF SigLIP config/model classes even though the model family is SigLIP2.
        if config.model_type != "siglip":
            raise ValueError(
                "This SigLIP2 wrapper expects the local checkpoint layout used by "
                f"siglip2-so400m-patch14-224, got model_type={config.model_type!r}"
            )
        self.model = SiglipVisionModel.from_pretrained(image_encoder_pretrained_path, local_files_only=True)
        self.image_processor = AutoProcessor.from_pretrained(
            image_encoder_pretrained_path,
            local_files_only=True,
        ).image_processor
        self._freeze()

    def encode_image(self, videos) -> torch.Tensor:
        """
        Input:  CLIP-style iterable of [1, 3, H, W], floating dtype, scale [-1, 1].
        Output: image tokens [B, 256, 1152], model dtype.
        """
        # Match the original CLIP interface: videos yields [1,3,H,W] items with shared H/W.
        input_shape = tuple(videos.shape) if isinstance(videos, torch.Tensor) else None
        videos = list(videos)
        input_shape = input_shape or f"list[{len(videos)}]"
        images = torch.cat(videos, dim=0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"SigLIP2 image encoder expects [B,3,H,W] after cat, got {tuple(images.shape)}")
        images = images.detach().float().mul(0.5).add(0.5).clamp_(0.0, 1.0).cpu()
        with torch.no_grad():
            processed = self.image_processor(images=images, return_tensors="pt", do_rescale=False)
            model_param = _model_param(self.model)
            pixel_values = processed["pixel_values"].to(device=model_param.device, dtype=model_param.dtype)
            output = self.model(pixel_values=pixel_values).last_hidden_state.clone()
            if random.random() < 0.001:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(
                    f"[IMAGE_ENCODER_SHAPE] rank={rank} encoder=siglip2 "
                    f"input={input_shape} after_cat={tuple(images.shape)} "
                    f"output={tuple(output.shape)} output_dtype={output.dtype} output_device={output.device}",
                    flush=True,
                )
            return output


class WanDINOv2ImageEncoder(_FrozenImageEncoder):
    def __init__(self, image_encoder_pretrained_path: str | None = None):
        super().__init__(image_encoder_pretrained_path)
        from transformers import AutoImageProcessor, Dinov2Model

        if image_encoder_pretrained_path is None:
            raise ValueError("image_encoder_pretrained_path is required for DINOv2")

        self.model = Dinov2Model.from_pretrained(image_encoder_pretrained_path, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(
            image_encoder_pretrained_path,
            local_files_only=True,
        )
        self._freeze()

    def encode_image(self, videos) -> torch.Tensor:
        """
        Input:  CLIP-style iterable of [1, 3, H, W], floating dtype, scale [-1, 1].
        Output: image tokens [B, 257, 1024], model dtype.
        """
        # Match the original CLIP interface: videos yields [1,3,H,W] items with shared H/W.
        input_shape = tuple(videos.shape) if isinstance(videos, torch.Tensor) else None
        videos = list(videos)
        input_shape = input_shape or f"list[{len(videos)}]"
        images = torch.cat(videos, dim=0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"DINOv2 image encoder expects [B,3,H,W] after cat, got {tuple(images.shape)}")
        images = images.detach().float().mul(0.5).add(0.5).clamp_(0.0, 1.0).cpu()
        with torch.no_grad():
            processed = self.image_processor(images=images, return_tensors="pt", do_rescale=False)
            model_param = _model_param(self.model)
            pixel_values = processed["pixel_values"].to(device=model_param.device, dtype=model_param.dtype)
            output = self.model(pixel_values=pixel_values).last_hidden_state.clone()
            if random.random() < 0.001:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(
                    f"[IMAGE_ENCODER_SHAPE] rank={rank} encoder=dinov2 "
                    f"input={input_shape} after_cat={tuple(images.shape)} "
                    f"output={tuple(output.shape)} output_dtype={output.dtype} output_device={output.device}",
                    flush=True,
                )
            return output


class WanQwen3VLImageEncoder(_FrozenImageEncoder):
    def __init__(self, image_encoder_pretrained_path: str | None = None):
        super().__init__(image_encoder_pretrained_path)
        from safetensors import safe_open
        from transformers import AutoConfig, AutoProcessor, Qwen3VLVisionModel

        if image_encoder_pretrained_path is None:
            raise ValueError("image_encoder_pretrained_path is required for Qwen3-VL")

        config = AutoConfig.from_pretrained(
            image_encoder_pretrained_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = Qwen3VLVisionModel(config.vision_config)
        state_path = os.path.join(image_encoder_pretrained_path, "model.safetensors")
        state_dict = {}
        with safe_open(state_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("model.visual."):
                    state_dict[key[len("model.visual."):]] = handle.get_tensor(key)
        self.model.load_state_dict(state_dict, strict=True)

        self.merge_size = int(config.vision_config.spatial_merge_size)
        self.image_processor = AutoProcessor.from_pretrained(
            image_encoder_pretrained_path,
            local_files_only=True,
            trust_remote_code=True,
        ).image_processor
        self._freeze()

    def encode_image(self, videos) -> torch.Tensor:
        """
        Input:  CLIP-style iterable of [1, 3, H, W], floating dtype, scale [-1, 1].
        Output: image tokens [B, L, 2048], model dtype, where L follows the HF processor grid.
        Qwen3-VL image-only processing uses grid_t=1, temporal_patch_size=2,
        and spatial_merge_size=2, so L = 1 * (grid_h // 2) * (grid_w // 2).
        """
        # Match the original CLIP interface: videos yields [1,3,H,W] items with shared H/W.
        input_shape = tuple(videos.shape) if isinstance(videos, torch.Tensor) else None
        videos = list(videos)
        input_shape = input_shape or f"list[{len(videos)}]"
        images = torch.cat(videos, dim=0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Qwen3-VL image encoder expects [B,3,H,W] after cat, got {tuple(images.shape)}")
        images = images.detach().float().mul(0.5).add(0.5).clamp_(0.0, 1.0).cpu()
        with torch.no_grad():
            processed = self.image_processor(images=images, return_tensors="pt", do_rescale=False)
            model_param = _model_param(self.model)
            pixel_values = processed["pixel_values"].to(device=model_param.device, dtype=model_param.dtype)
            grid_thw = processed["image_grid_thw"].to(device=model_param.device)
            output, _deepstack_features = self.model(pixel_values, grid_thw=grid_thw)
            token_count = (grid_thw[:, 1] // self.merge_size) * (grid_thw[:, 2] // self.merge_size) * grid_thw[:, 0]
            if not torch.all(token_count == token_count[0]):
                raise ValueError(f"Qwen3-VL wrapper expects fixed token count, got {token_count.tolist()}")
            output = output.reshape(images.shape[0], int(token_count[0].item()), -1).clone()
            if random.random() < 0.001:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(
                    f"[IMAGE_ENCODER_SHAPE] rank={rank} encoder=qwen3_vl "
                    f"input={input_shape} after_cat={tuple(images.shape)} grid_thw={grid_thw.tolist()} "
                    f"output={tuple(output.shape)} output_dtype={output.dtype} output_device={output.device}",
                    flush=True,
                )
            return output
