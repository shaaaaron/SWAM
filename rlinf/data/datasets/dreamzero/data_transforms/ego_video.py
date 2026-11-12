# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video-only DreamZero transforms for human activity and general videos."""

from typing import Any

import numpy as np
from einops import rearrange
from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.transform.base import ComposedModalityTransform
from groot.vla.data.transform.concat import ConcatTransform
from groot.vla.data.transform.video import (
    VideoColorJitter,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
    DreamTransform as DreamTransformBase,
)

from rlinf.data.datasets.dreamzero.data_transforms.base import RolloutObsLayout

_VIDEO_KEYS = ["video.cam_high"]
_VIDEO_BACKEND = "torchvision"
_EGO_VIEW_LAYOUTS = {"single", "repeat_three", "top_gray"}
_TRAINING_PROMPT_PREFIX = (
    "A first-person video shows a person performing the following activity: "
)
_TRAINING_PROMPT_SUFFIX = " The video is recorded from the person's point of view."
# OpenVid 与 Panda70M 共用通用场景前缀，后接 metadata 中的原生 caption。
_OPENVID_TRAINING_PROMPT_PREFIX = "A video depicts the following scene or activity, describing the main objects, actions, and interactions visible in the footage: "


def ego_resize_resolution(
    layout: str, target_height: int, target_width: int
) -> tuple[int, int]:
    """返回布局拼接前单路视频应使用的 resize 高宽。"""
    if layout not in _EGO_VIEW_LAYOUTS:
        raise ValueError(
            f"Unsupported ego_view_layout {layout!r}; "
            f"expected one of {sorted(_EGO_VIEW_LAYOUTS)}."
        )
    if layout == "single":
        return target_height, target_width
    if target_height % 2 != 0 or target_width % 2 != 0:
        raise ValueError(
            "repeat_three/top_gray require even image resolution, "
            f"got {target_width}x{target_height}."
        )
    return target_height // 2, target_width // 2


def concat_ego_video_views(images: np.ndarray, layout: str) -> np.ndarray:
    """将单路 Ego 视频转换为单路或 RobotWin 风格三槽画布。

    images: np.ndarray [V=1, T, C, H, W], uint8, [0, 255].
    """
    if layout not in _EGO_VIEW_LAYOUTS:
        raise ValueError(
            f"Unsupported ego_view_layout {layout!r}; "
            f"expected one of {sorted(_EGO_VIEW_LAYOUTS)}."
        )
    if images.ndim != 5 or images.shape[0] != 1:
        raise ValueError(
            "ego video expects one view with shape [1,T,C,H,W], "
            f"got {images.shape}."
        )
    if layout == "single":
        return images

    main = images[0]
    _, _, height, width = main.shape
    fill_value = 127 if layout == "top_gray" else 0
    canvas = np.full(
        (1, main.shape[0], main.shape[1], 2 * height, 2 * width),
        fill_value,
        dtype=images.dtype,
    )
    canvas[0, :, :, :height, :] = np.repeat(main, 2, axis=-1)
    if layout == "repeat_three":
        canvas[0, :, :, height:, :width] = main
        canvas[0, :, :, height:, width:] = main
    return canvas


class EgoVideoDreamTransform(DreamTransformBase):
    """Create structural state/action registers without action supervision."""

    has_real_action: bool = True
    video_frames_per_chunk: int = 8
    max_chunk_size: int = 4
    ego_view_layout: str = "single"

    def _prepare_video(self, data: dict) -> np.ndarray:
        images = rearrange(data["video"], "t v h w c -> v t c h w")
        return concat_ego_video_views(images, self.ego_view_layout)

    def _num_video_chunks(self, data: dict[str, Any]) -> int:
        video = np.asarray(data["video"])
        num_frames = int(video.shape[0])
        remainder = (num_frames - 1) % self.video_frames_per_chunk
        # An episode-first frame may be prepended, yielding T = 8 * num_chunks + 2.
        if num_frames <= 1 or remainder not in (0, 1):
            raise ValueError(
                "ego_video expects T = 8 * num_chunks + 1, or +2 with an "
                "episode-first frame, "
                f"got T={num_frames}."
            )
        num_chunks = (num_frames - 1) // self.video_frames_per_chunk
        if not 1 <= num_chunks <= self.max_chunk_size:
            raise ValueError(
                f"ego_video num_chunks must be in [1, {self.max_chunk_size}], "
                f"got {num_chunks}."
            )
        return num_chunks

    def _prepare_state(self, data: dict):
        num_tokens = self._num_video_chunks(data) * self.state_horizon
        state = np.zeros((num_tokens, self.max_state_dim), dtype=np.float32)
        return state, np.zeros_like(state, dtype=bool), num_tokens

    def _prepare_action(self, data: dict):
        num_tokens = self._num_video_chunks(data) * self.action_horizon
        action = np.zeros((num_tokens, self.max_action_dim), dtype=np.float32)
        return action, np.zeros_like(action, dtype=bool), num_tokens

    def apply_single(self, data: dict) -> dict:
        transformed = super().apply_single(data)
        # LAPA placeholders are internal to the Groot transform and are not
        # consumed by RLinf SFT. Keep mixed video/robotics sample keys aligned.
        transformed.pop("lapa_action", None)
        transformed.pop("lapa_action_mask", None)
        if self.training:
            transformed["has_real_action"] = np.asarray(
                self.has_real_action, dtype=bool
            )
        return transformed


class EgoVideoDataTransform:
    """Shared single-view transform for video-only datasets."""

    TAG = ""
    DEFAULT_TAG_MAPPING: dict[str, int] = {}
    DEFAULT_ACTION_HORIZON = 48
    ROLLOUT_OBS_LAYOUT = RolloutObsLayout(
        video_fields=(("main_images", "video.cam_high"),),
        state_fields=(),
        language_env_key="task_descriptions",
    )

    @staticmethod
    def format_training_prompt(instruction: str) -> str:
        """Build the first-person activity prompt for Ego video."""
        return _TRAINING_PROMPT_PREFIX + instruction + _TRAINING_PROMPT_SUFFIX

    @staticmethod
    def concat_multiview_video(images: np.ndarray) -> np.ndarray:
        """继承统一 embodiment 接口；Ego 训练路径不应调用此方法。"""
        raise RuntimeError(
            "Ego video layout is handled by "
            "EgoVideoDreamTransform._prepare_video(); "
            "concat_multiview_video() must not be called."
        )

    @staticmethod
    def get_modality_config() -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(
                delta_indices=list(range(25)),
                eval_delta_indices=[0],
                modality_keys=list(_VIDEO_KEYS),
            ),
            "state": ModalityConfig(delta_indices=[], modality_keys=[]),
            "action": ModalityConfig(delta_indices=[], modality_keys=[]),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.language.action_text"],
            ),
        }

    @staticmethod
    def get_transform(
        *,
        tokenizer_path: str,
        cfg: Any,
        embodiment_tag_mapping: dict[str, int],
    ) -> ComposedModalityTransform:
        if "has_real_action" not in cfg or bool(cfg.has_real_action):
            raise ValueError(
                f"{cfg.embodiment_tag} datasets must explicitly set "
                "has_real_action: false."
            )
        ego_view_layout = str(cfg.get("ego_view_layout", "single"))
        if ego_view_layout not in _EGO_VIEW_LAYOUTS:
            raise ValueError(
                f"Unsupported ego_view_layout {ego_view_layout!r}; "
                f"expected one of {sorted(_EGO_VIEW_LAYOUTS)}."
            )
        target_height = int(cfg.get("image_resolution_height", 256))
        target_width = int(cfg.get("image_resolution_width", 320))
        resize_height, resize_width = ego_resize_resolution(
            ego_view_layout,
            target_height,
            target_width,
        )
        video_keys = list(_VIDEO_KEYS)
        max_chunk_size = int(
            cfg.action_head_cfg.config.diffusion_model_cfg.max_chunk_size
        )
        color_jitter = cfg.get("color_jitter") or {}
        transforms: list[Any] = [
            VideoToTensor(apply_to=video_keys, backend=_VIDEO_BACKEND),
            VideoResize(
                apply_to=video_keys,
                backend=_VIDEO_BACKEND,
                height=resize_height,
                width=resize_width,
                interpolation="linear",
            ),
            VideoColorJitter(
                apply_to=video_keys,
                backend=_VIDEO_BACKEND,
                brightness=color_jitter.get("brightness", 0.3),
                contrast=color_jitter.get("contrast", 0.4),
                saturation=color_jitter.get("saturation", 0.5),
                hue=color_jitter.get("hue", 0.08),
            ),
            VideoToNumpy(apply_to=video_keys, backend=_VIDEO_BACKEND),
            ConcatTransform(
                apply_to=[],
                video_concat_order=video_keys,
                state_concat_order=[],
                action_concat_order=[],
            ),
            EgoVideoDreamTransform(
                default_instruction=str(
                    cfg.get("default_instruction", "Perform the described activity.")
                ),
                language_dropout_prob=float(cfg.get("language_dropout_prob", 0.0)),
                always_use_default_instruction=bool(
                    cfg.get("always_use_default_instruction", False)
                ),
                max_state_dim=int(cfg.get("max_state_dim", 16)),
                max_action_dim=int(cfg.get("max_action_dim", 16)),
                max_length=int(cfg.get("max_seq_len", 512)),
                state_horizon=int(cfg.get("state_horizon", 1)),
                action_horizon=int(cfg.get("action_horizon", 48)),
                tokenizer_path=tokenizer_path,
                embodiment_tag_mapping=embodiment_tag_mapping,
                has_real_action=bool(cfg.has_real_action),
                max_chunk_size=max_chunk_size,
                ego_view_layout=ego_view_layout,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class Ego4DDataTransform(EgoVideoDataTransform):
    """Video-only Ego4D dataset transform."""

    TAG = "ego4d"
    DEFAULT_TAG_MAPPING = {"ego4d": 35}


class EgoverseDataTransform(EgoVideoDataTransform):
    """Video-only Egoverse dataset transform."""

    TAG = "egoverse"
    DEFAULT_TAG_MAPPING = {"egoverse": 36}


class OpenVidDataTransform(EgoVideoDataTransform):
    """Video-only OpenVid transform with a non-egocentric prompt."""

    TAG = "openvid"
    DEFAULT_TAG_MAPPING = {"openvid": 38}

    @staticmethod
    def format_training_prompt(instruction: str) -> str:
        """为 OpenVid / Panda70M 的原生 caption 添加通用场景前缀。"""
        return _OPENVID_TRAINING_PROMPT_PREFIX + instruction
