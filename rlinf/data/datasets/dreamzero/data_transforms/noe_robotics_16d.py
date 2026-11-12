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

"""DreamZero transform for canonical 16D NOE robotics datasets."""

from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TVF
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
from rlinf.data.datasets.dreamzero.noe_robotics_contract import (
    NOE_ROBOTICS_EMBODIMENT_ID,
    NOE_ROBOTICS_TAG,
)

_VIDEO_KEYS = [
    "video.image",
    "video.wrist_image_1",
    "video.wrist_image_2",
]
_STATE_KEYS = ["state.eef_position"]
_ACTION_KEYS = ["action.eef_position"]
_VIDEO_BACKEND = "torchvision"

_TRAINING_PROMPT_PREFIX = "A multi-view video shows that a dual-arm robot "
_MULTIVIEW_LAYOUT = (
    " The video is split into three views: The top view shows the main "
    "camera, the bottom-left view shows the left wrist camera, "
    "and the bottom-right view shows the right wrist camera."
)


class NoeRoboticsVideoToTensor(VideoToTensor):
    """Convert each view independently when source resolutions vary by episode."""

    def check_input(self, data: dict[str, Any]) -> None:
        for key in self.apply_to:
            assert key in data, (
                f"Key {key} not found in data. Available keys: {data.keys()}"
            )
            assert isinstance(data[key], np.ndarray), (
                f"Video {key} is not a numpy array"
            )
            assert data[key].ndim == 4, (
                f"Video {key} must have 4 dimensions, got {data[key].ndim}"
            )
            assert data[key].dtype == np.uint8, (
                f"Video {key} must have dtype uint8, got {data[key].dtype}"
            )
            assert data[key].shape[-1] == 3, (
                f"Video {key} must have 3 channels, got shape {data[key].shape}"
            )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        try:
            self.check_input(data)
        except AssertionError as exc:
            raise ValueError(
                "Input data does not match the expected format for "
                f"{self.__class__.__name__}: {exc}"
            ) from exc

        for key in self.apply_to:
            data[key] = VideoToTensor.to_tensor(
                data[key],
                output_on_cuda=self.output_on_cuda,
            )
        return data


class NoeRoboticsViewResize(ModalityTransform):
    """Resize each source view independently before shared video transforms."""

    height: int
    width: int

    def _resize_one(self, frames: torch.Tensor) -> torch.Tensor:
        return TVF.resize(
            frames,
            [self.height, self.width],
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("NoeRoboticsViewResize target size must be positive")
        for key in self.apply_to:
            frames = data[key]
            if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
                raise ValueError(
                    f"Expected {key} as [T,C,H,W], got "
                    f"{type(frames)} with shape={getattr(frames, 'shape', None)}"
                )
            data[key] = self._resize_one(frames)
        return data

    def set_metadata(self, dataset_metadata) -> None:
        super().set_metadata(dataset_metadata)
        for key in self.apply_to:
            subkey = key.split(".", 1)[1]
            if subkey not in dataset_metadata.modalities.video:
                raise ValueError(f"Metadata is missing video.{subkey}")
            dataset_metadata.modalities.video[subkey].resolution = (
                self.width,
                self.height,
            )


class NoeRobotics16DDataTransform:
    """Provide modality configuration and transforms for NOE robotics data."""

    TAG = NOE_ROBOTICS_TAG
    DEFAULT_TAG_MAPPING = {NOE_ROBOTICS_TAG: NOE_ROBOTICS_EMBODIMENT_ID}
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
        """Build the robotics three-view training prompt."""
        return _TRAINING_PROMPT_PREFIX + instruction + _MULTIVIEW_LAYOUT

    @staticmethod
    def concat_multiview_video(images: np.ndarray) -> np.ndarray:
        """Place the main view above the two wrist views on a 2x2 canvas."""
        v, t, c, h, w = images.shape
        if v < 3:
            raise ValueError(
                "noe_robotics_16d expects at least 3 video views, "
                f"got v={v} with shape {images.shape}"
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
        """Return the canonical NOE robotics modality configuration."""
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
        """Build the composed transform chain for NOE robotics data."""
        return NoeRobotics16DDataTransform._build_composed_transform(
            tokenizer_path=tokenizer_path,
            state_horizon=int(cfg.get("state_horizon", 1)),
            action_horizon=int(
                cfg.get(
                    "action_horizon",
                    NoeRobotics16DDataTransform.DEFAULT_ACTION_HORIZON,
                )
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
        video_keys = list(_VIDEO_KEYS)
        state_keys = list(_STATE_KEYS)
        action_keys = list(_ACTION_KEYS)
        transforms: list[Any] = [
            NoeRoboticsVideoToTensor(apply_to=video_keys, backend=_VIDEO_BACKEND),
            NoeRoboticsViewResize(
                apply_to=video_keys,
                height=image_resolution_height,
                width=image_resolution_width,
            ),
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
            StateActionToTensor(apply_to=state_keys),
            StateActionTransform(
                apply_to=state_keys,
                normalization_modes={"state.eef_position": "q99"},
            ),
            StateActionToTensor(apply_to=action_keys),
            StateActionTransform(
                apply_to=action_keys,
                normalization_modes={"action.eef_position": "q99"},
            ),
            ConcatTransform(
                apply_to=[],
                video_concat_order=video_keys,
                state_concat_order=state_keys,
                action_concat_order=action_keys,
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
        return ComposedModalityTransform(transforms=transforms)
