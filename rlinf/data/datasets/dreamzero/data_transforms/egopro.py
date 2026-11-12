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

"""Video-only EgoPro training with a head camera and two wrist cameras."""

from typing import Any

import numpy as np
from einops import rearrange
from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.transform.base import ComposedModalityTransform
from groot.vla.data.transform.concat import ConcatTransform
from groot.vla.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)

from rlinf.data.datasets.dreamzero.data_transforms.base import RolloutObsLayout
from rlinf.data.datasets.dreamzero.data_transforms.ego_video import (
    EgoVideoDreamTransform,
)

_VIDEO_KEYS = ["video.head_left", "video.left_wrist", "video.right_wrist"]
_VIDEO_BACKEND = "torchvision"


class EgoProDreamTransform(EgoVideoDreamTransform):
    """Use the Ego video-only registers with a real three-camera canvas."""

    def _prepare_video(self, data: dict) -> np.ndarray:
        images = rearrange(data["video"], "t v h w c -> v t c h w")
        return EgoProDataTransform.concat_multiview_video(images)


class EgoProDataTransform:
    """Provide the fixed head/left-wrist/right-wrist training layout."""

    TAG = "egopro"
    DEFAULT_TAG_MAPPING = {"egopro": 40}
    DEFAULT_ACTION_HORIZON = 48
    ROLLOUT_OBS_LAYOUT = RolloutObsLayout(
        video_fields=(
            ("main_images", "video.head_left"),
            ("wrist_images", "video.left_wrist"),
            ("extra_view_images", "video.right_wrist"),
        ),
        state_fields=(),
        language_env_key="task_descriptions",
    )

    @staticmethod
    def format_training_prompt(instruction: str) -> str:
        return (
            "A multi-view video shows a person performing the following activity: "
            + instruction
            + " The video is split into three views: The top view shows the "
            "head-mounted camera, the bottom-left view shows the left wrist "
            "camera, and the bottom-right view shows the right wrist camera."
        )

    @staticmethod
    def concat_multiview_video(images: np.ndarray) -> np.ndarray:
        """2x2 grid: head spans top row; left/right wrist on bottom row."""
        if images.ndim != 5:
            raise ValueError(
                "egopro expects video views with shape [3,T,C,H,W], "
                f"got {images.shape}."
            )
        v, t, c, h, w = images.shape
        if v != 3:
            raise ValueError(
                f"egopro expects exactly 3 views, got v={v} with shape {images.shape}"
            )
        cam_high = images[0]
        cam_left_wrist = images[1]
        cam_right_wrist = images[2]
        concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)
        concat_images[0, :, :, :h, :] = np.repeat(cam_high, 2, axis=-1)
        concat_images[0, :, :, h:, :w] = cam_left_wrist
        concat_images[0, :, :, h:, w:] = cam_right_wrist
        return concat_images

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
                "egopro datasets must explicitly set has_real_action: false."
            )
        image_resolution_height = int(cfg.get("image_resolution_height", 128))
        image_resolution_width = int(cfg.get("image_resolution_width", 160))
        if image_resolution_height <= 0 or image_resolution_width <= 0:
            raise ValueError(
                "egopro requires positive per-view image resolution, "
                f"got {image_resolution_width}x{image_resolution_height}."
            )
        video_keys = list(_VIDEO_KEYS)
        color_jitter = cfg.get("color_jitter") or {}
        return ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_keys, backend=_VIDEO_BACKEND),
                VideoCrop(
                    apply_to=video_keys,
                    backend=_VIDEO_BACKEND,
                    scale=0.95,
                    mode="random",
                ),
                VideoResize(
                    apply_to=video_keys,
                    backend=_VIDEO_BACKEND,
                    height=image_resolution_height,
                    width=image_resolution_width,
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
                EgoProDreamTransform(
                    default_instruction=str(
                        cfg.get(
                            "default_instruction", "Perform the described activity."
                        )
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
                    has_real_action=False,
                    max_chunk_size=int(
                        cfg.action_head_cfg.config.diffusion_model_cfg.max_chunk_size
                    ),
                ),
            ]
        )
