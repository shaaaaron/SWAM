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

"""NOE/UMI DreamZero adapter with a 16D dual-arm EEF action/state contract."""

from typing import Any

import numpy as np
from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.transform.base import ComposedModalityTransform, ModalityTransform
from groot.vla.data.transform.concat import ConcatTransform
from groot.vla.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from groot.vla.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)

from rlinf.data.datasets.dreamzero.data_transforms.base import RolloutObsLayout
from rlinf.data.datasets.dreamzero.data_transforms.dream_transform import DreamTransform

_VIDEO_KEYS = [
    "video.image",
    "video.wrist_image_1",
    "video.wrist_image_2",
]
_STATE_KEYS = ["state.eef_position"]
_ACTION_KEYS = ["action.eef_position"]
_LEFT_WRIST_KEY = "video.wrist_image_1"
_RIGHT_WRIST_KEY = "video.wrist_image_2"

_VIDEO_BACKEND = "torchvision"

_TRAINING_PROMPT_PREFIX = "A multi-view video shows that a dual-arm ego-centric human "
_MULTIVIEW_LAYOUT = (
    " The video is split into three views: The top view shows the overhead "
    "camera, the bottom-left view shows the left wrist camera, "
    "and the bottom-right view shows the right wrist camera."
)


class NoeUmiWristRotation(ModalityTransform):
    """Orient NOE/UMI wrist videos before common video augmentation."""

    apply_to: list[str] = [_LEFT_WRIST_KEY, _RIGHT_WRIST_KEY]

    def set_metadata(self, dataset_metadata) -> None:
        super().set_metadata(dataset_metadata)
        self._swap_video_resolution(dataset_metadata, _LEFT_WRIST_KEY)
        self._swap_video_resolution(dataset_metadata, _RIGHT_WRIST_KEY)

    @staticmethod
    def _swap_video_resolution(dataset_metadata, key: str) -> None:
        _, sub_key = key.split(".", 1)
        if sub_key not in dataset_metadata.modalities.video:
            raise ValueError(
                f"Video key {sub_key!r} not found in dataset metadata. "
                f"Available keys: {dataset_metadata.modalities.video.keys()}"
            )
        video_metadata = dataset_metadata.modalities.video[sub_key]
        width, height = video_metadata.resolution
        video_metadata.resolution = (height, width)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        data[_LEFT_WRIST_KEY] = np.ascontiguousarray(
            np.rot90(data[_LEFT_WRIST_KEY], k=-1, axes=(-3, -2))
        )
        data[_RIGHT_WRIST_KEY] = np.ascontiguousarray(
            np.rot90(data[_RIGHT_WRIST_KEY], k=1, axes=(-3, -2))
        )
        return data


class NoeUmi16DDataTransform:
    """Provides modality config and composed transform for NOE/UMI 16D data."""

    TAG = "noe_umi_16d"
    DEFAULT_TAG_MAPPING = {"noe_umi_16d": 34}
    DEFAULT_ACTION_HORIZON = 48
    ROLLOUT_OBS_LAYOUT = RolloutObsLayout(
        video_fields=(
            ("main_images", "video.image"),
            ("wrist_images", "video.wrist_image_1"),
            ("extra_view_images", "video.wrist_image_2"),
        ),
        state_fields=(("states", "state.eef_position"),),
        language_env_key="task_descriptions",
        binarize_gripper=False,
        fill_missing_video_keys=True,
    )

    @staticmethod
    def format_training_prompt(instruction: str) -> str:
        """Build multi-view layout prompt for NOE/UMI human demonstrations."""
        return _TRAINING_PROMPT_PREFIX + instruction + _MULTIVIEW_LAYOUT

    @staticmethod
    def concat_multiview_video(images: np.ndarray) -> np.ndarray:
        """2x2 grid: overhead spans top row; left/right wrist on bottom row."""
        v, t, c, h, w = images.shape
        if v < 3:
            raise ValueError(
                f"noe_umi_16d expects at least 3 video views, got v={v} with shape {images.shape}"
            )
        image = images[0]
        wrist_image_1 = images[1]
        wrist_image_2 = images[2]
        concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)
        concat_images[0, :, :, :h, :] = np.repeat(image, 2, axis=-1)
        concat_images[0, :, :, h:, :w] = wrist_image_1
        concat_images[0, :, :, h:, w:] = wrist_image_2
        return concat_images

    @staticmethod
    def get_modality_config() -> dict[str, ModalityConfig]:
        """Return modality config dict for NOE/UMI 16D data."""
        return {
            "video": ModalityConfig(
                delta_indices=list(range(25)),
                eval_delta_indices=[0],
                modality_keys=list(_VIDEO_KEYS),
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(_STATE_KEYS),
            ),
            "action": ModalityConfig(
                delta_indices=[],
                modality_keys=[],
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.language.action_text"],
            ),
            "lapa_action": ModalityConfig(
                delta_indices=[0],
                modality_keys=["lapa_action"],
            ),
        }

    @staticmethod
    def get_transform(
        *,
        tokenizer_path: str,
        cfg: Any,
        embodiment_tag_mapping: dict[str, int],
    ) -> ComposedModalityTransform:
        """Build the full ``ComposedModalityTransform`` chain for NOE/UMI."""
        return NoeUmi16DDataTransform._build_composed_transform(
            tokenizer_path=tokenizer_path,
            state_horizon=int(cfg.get("state_horizon", 1)),
            action_horizon=int(
                cfg.get("action_horizon", NoeUmi16DDataTransform.DEFAULT_ACTION_HORIZON)
            ),
            max_state_dim=int(cfg.get("max_state_dim", 16)),
            max_action_dim=int(cfg.get("max_action_dim", 16)),
            max_length=int(cfg.get("max_seq_len", 512)),
            default_instruction=str(
                cfg.get("default_instruction", "Perform the default behavior.")
            ),
            language_dropout_prob=float(cfg.get("language_dropout_prob", 0.0)),
            always_use_default_instruction=bool(
                cfg.get("always_use_default_instruction", False)
            ),
            image_resolution_width=int(cfg.get("image_resolution_width", 160)),
            image_resolution_height=int(cfg.get("image_resolution_height", 128)),
            concat_method=str(cfg.get("concat_method", "canvas")),
            has_real_action=bool(cfg.get("has_real_action", True)),
            embodiment_tag_mapping=dict(embodiment_tag_mapping),
            color_jitter=cfg.get("color_jitter"),
        )

    @staticmethod
    def _build_composed_transform(
        tokenizer_path: str,
        state_horizon: int,
        action_horizon: int,
        max_state_dim: int,
        max_action_dim: int,
        max_length: int,
        default_instruction: str,
        language_dropout_prob: float,
        always_use_default_instruction: bool,
        image_resolution_width: int,
        image_resolution_height: int,
        concat_method: str,
        has_real_action: bool,
        embodiment_tag_mapping: dict[str, int],
        color_jitter: dict[str, Any] | None = None,
    ) -> ComposedModalityTransform:
        color_jitter = color_jitter or {}
        vk = list(_VIDEO_KEYS)
        state_k = list(_STATE_KEYS)
        action_k = list(_ACTION_KEYS)

        transforms: list[Any] = [
            NoeUmiWristRotation(),
            VideoToTensor(apply_to=vk, backend=_VIDEO_BACKEND),
            VideoCrop(apply_to=vk, backend=_VIDEO_BACKEND, scale=0.95, mode="random"),
            VideoResize(
                apply_to=vk,
                backend=_VIDEO_BACKEND,
                height=image_resolution_height,
                width=image_resolution_width,
                interpolation="linear",
            ),
            VideoColorJitter(
                apply_to=vk,
                backend=_VIDEO_BACKEND,
                brightness=color_jitter.get("brightness", 0.3),
                contrast=color_jitter.get("contrast", 0.4),
                saturation=color_jitter.get("saturation", 0.5),
                hue=color_jitter.get("hue", 0.08),
            ),
            VideoToNumpy(apply_to=vk, backend=_VIDEO_BACKEND),
            StateActionToTensor(apply_to=state_k),
            StateActionTransform(
                apply_to=state_k,
                normalization_modes={"state.eef_position": "q99"},
            ),
        ]
        transforms.extend(
            [
                StateActionToTensor(apply_to=action_k),
                StateActionTransform(
                    apply_to=action_k,
                    normalization_modes={"action.eef_position": "q99"},
                ),
            ]
        )
        transforms.extend(
            [
                ConcatTransform(
                    apply_to=[],
                    video_concat_order=vk,
                    state_concat_order=state_k,
                    action_concat_order=action_k,
                ),
                DreamTransform(
                    default_instruction=default_instruction,
                    language_dropout_prob=language_dropout_prob,
                    always_use_default_instruction=always_use_default_instruction,
                    max_state_dim=max_state_dim,
                    max_action_dim=max_action_dim,
                    max_length=max_length,
                    state_horizon=state_horizon,
                    action_horizon=action_horizon,
                    tokenizer_path=tokenizer_path,
                    embodiment_tag_mapping=embodiment_tag_mapping,
                    video_concat_layout=concat_method,
                    has_real_action=has_real_action,
                ),
            ]
        )

        return ComposedModalityTransform(transforms=transforms)
