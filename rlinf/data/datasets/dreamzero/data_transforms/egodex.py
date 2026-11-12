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

"""EgoDex 单视角双手 absolute state/action 训练转换。"""

from typing import Any

import numpy as np
from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.transform.base import ComposedModalityTransform
from groot.vla.data.transform.concat import ConcatTransform
from groot.vla.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from groot.vla.data.transform.video import (
    VideoColorJitter,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)

from rlinf.data.datasets.dreamzero.data_transforms.base import RolloutObsLayout
from rlinf.data.datasets.dreamzero.data_transforms.dream_transform import DreamTransform

VIDEO_KEYS = ["video.egocentric"]
STATE_KEYS = ["state.left_hand_position", "state.right_hand_position"]
ACTION_KEYS = ["action.left_hand_position", "action.right_hand_position"]
LANGUAGE_KEYS = ["annotation.task"]

_VIDEO_BACKEND = "torchvision"


class EgoDexDataTransform:
    """将 EgoDex 样本转换为 DreamZero 需要的单视角 absolute action 格式。"""

    TAG = "egodex"
    DEFAULT_TAG_MAPPING = {"egodex": 37}
    DEFAULT_ACTION_HORIZON = 48
    ROLLOUT_OBS_LAYOUT = RolloutObsLayout(
        video_fields=(("main_images", "video.egocentric"),),
        state_fields=(),
    )

    @staticmethod
    def _action_horizon_from_cfg(cfg: Any | None) -> int:
        """从配置读取 EgoDex action horizon，并校验其为正整数。"""
        action_horizon = int(
            EgoDexDataTransform.DEFAULT_ACTION_HORIZON
            if cfg is None
            else cfg.get(
                "action_horizon",
                EgoDexDataTransform.DEFAULT_ACTION_HORIZON,
            )
        )
        if action_horizon <= 0:
            raise ValueError(
                f"EgoDex action_horizon must be positive, got {action_horizon}."
            )
        return action_horizon

    @staticmethod
    def format_training_prompt(instruction: str) -> str:
        """输入任务文本，输出第一人称双手训练 prompt。"""
        if not instruction.strip():
            return ""
        return f"A first-person video shows human hands performing the task: {instruction}"

    @staticmethod
    def concat_multiview_video(images: np.ndarray) -> np.ndarray:
        """输入 [1,T,C,H,W]，原样返回单路 EgoDex 视频。"""
        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError(
                f"EgoDex expects one video view [1,T,C,H,W], got {images.shape}."
            )
        return images

    @staticmethod
    def get_modality_config(cfg: Any | None = None) -> dict[str, ModalityConfig]:
        """返回 EgoDex 训练使用的视频、状态、动作和语言键。"""
        action_horizon = EgoDexDataTransform._action_horizon_from_cfg(cfg)
        return {
            "video": ModalityConfig(
                delta_indices=list(range(25)),
                eval_delta_indices=[0],
                modality_keys=list(VIDEO_KEYS),
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(STATE_KEYS),
            ),
            "action": ModalityConfig(
                delta_indices=list(range(action_horizon)),
                modality_keys=list(ACTION_KEYS),
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(LANGUAGE_KEYS),
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
        """按配置构建直接 resize 的 EgoDex DreamZero transform。"""
        video_keys = list(VIDEO_KEYS)
        state_keys = list(STATE_KEYS)
        action_keys = list(ACTION_KEYS)
        action_horizon = EgoDexDataTransform._action_horizon_from_cfg(cfg)
        color_jitter = cfg.get("color_jitter") or {}

        transforms: list[Any] = [
            VideoToTensor(apply_to=video_keys, backend=_VIDEO_BACKEND),
            VideoResize(
                apply_to=video_keys,
                backend=_VIDEO_BACKEND,
                height=int(cfg.get("image_resolution_height", 256)),
                width=int(cfg.get("image_resolution_width", 320)),
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
                normalization_modes={key: "q99" for key in state_keys},
            ),
            StateActionToTensor(apply_to=action_keys),
            StateActionTransform(
                apply_to=action_keys,
                normalization_modes={key: "q99" for key in action_keys},
            ),
            ConcatTransform(
                apply_to=[],
                video_concat_order=video_keys,
                state_concat_order=state_keys,
                action_concat_order=action_keys,
            ),
            DreamTransform(
                default_instruction=str(
                    cfg.get("default_instruction", "Perform the described task.")
                ),
                language_dropout_prob=float(cfg.get("language_dropout_prob", 0.0)),
                always_use_default_instruction=bool(
                    cfg.get("always_use_default_instruction", False)
                ),
                max_state_dim=int(cfg.get("max_state_dim", 16)),
                max_action_dim=int(cfg.get("max_action_dim", 16)),
                max_length=int(cfg.get("max_seq_len", 512)),
                state_horizon=int(cfg.get("state_horizon", 1)),
                action_horizon=action_horizon,
                tokenizer_path=tokenizer_path,
                embodiment_tag_mapping=dict(embodiment_tag_mapping),
                has_real_action=bool(cfg.get("has_real_action", True)),
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
