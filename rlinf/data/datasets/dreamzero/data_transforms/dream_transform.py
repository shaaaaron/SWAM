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

"""RLinf :class:`DreamTransform` subclass with embodiment-specific multi-view concat."""

from __future__ import annotations

from typing import Any

import numpy as np
from einops import rearrange
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
    DreamTransform as DreamTransformBase,
)


def resolve_registry_tag(embodiment_tag: Any) -> str:
    """Map Groot ``EmbodimentTag`` or registry tag string to ``_EMBODIMENT_REGISTRY`` key."""
    if hasattr(embodiment_tag, "value"):
        return str(embodiment_tag.value)
    return str(embodiment_tag)


def concat_generic_grid_views(images: np.ndarray) -> np.ndarray:
    """Fallback 2x2 grid for unknown embodiments."""
    v, t, c, h, w = images.shape
    concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)
    if v > 0:
        concat_images[0, :, :, :h, :w] = images[0]
    if v > 1:
        concat_images[0, :, :, h:, :w] = images[1]
    if v > 2:
        concat_images[0, :, :, :h, w:] = images[2]
    return concat_images


def concat_multiview_video(embodiment_tag: Any, images: Any) -> np.ndarray:
    """Concat multi-view frames using the embodiment registered in ``_EMBODIMENT_REGISTRY``."""
    from rlinf.data.datasets.dreamzero.data_transforms import (
        _EMBODIMENT_REGISTRY,
        _require_embodiment,
    )

    arr = np.asarray(images)
    tag = resolve_registry_tag(embodiment_tag)
    if tag in _EMBODIMENT_REGISTRY:
        return _require_embodiment(tag).concat_multiview_video(arr)
    return concat_generic_grid_views(arr)


class DreamTransform(DreamTransformBase):
    """DreamTransform that delegates multi-view layout to ``data_transforms`` registry.

    .. note::

        The ``codebase`` branch of DreamZero-dev has dropped ``video_concat_layout``
        from the base :class:`DreamTransform`, while the ``codebase-dev-xxy`` branch
        still carries it.  To stay compatible with both, this subclass manages
        ``video_concat_layout`` as its own instance attribute instead of forwarding
        it to ``super().__init__()``.
    """

    has_real_action: bool = True

    def __init__(self, **kwargs: Any) -> None:
        # Pop video_concat_layout before passing to base class; the codebase
        # branch no longer defines this field, and passing it would raise a
        # Pydantic "Extra inputs are not permitted" error.
        # We set it after super().__init__() via object.__setattr__ so that
        # (a) Pydantic's internal state is ready and (b) we bypass field
        # validation on branches where the base class lacks this field.
        video_concat_layout = kwargs.pop("video_concat_layout", "stretch_main")
        super().__init__(**kwargs)
        object.__setattr__(self, "video_concat_layout", video_concat_layout)

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        """Collate with RLinf prompt wrapping (supports all registered embodiments)."""
        import tree

        from rlinf.data.datasets.dreamzero.dreamzero import DreamZeroCollator

        data.pop("lapa_action", None)
        data.pop("dream_actions", None)
        data_split = [
            tree.map_structure(lambda x: x[i], data) for i in range(batch_size)
        ]
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return DreamZeroCollator.collate_batch(
            data_split_processed,
            self.tokenizer,
            self.embodiment_tag_mapping,
        )

    def apply_single(self, data: dict) -> dict:
        # 消费 UMI quality metadata 并且合并进入 existing state/action masks.
        from rlinf.data.datasets.dreamzero.dreamzero import (
            _NOE_UMI_ACTION_QUALITY_KEY,
            _NOE_UMI_STATE_QUALITY_KEY,
        )

        state_quality = data.pop(_NOE_UMI_STATE_QUALITY_KEY, None)
        action_quality = data.pop(_NOE_UMI_ACTION_QUALITY_KEY, None)
        transformed = super().apply_single(data)
        if self.training and state_quality is not None and action_quality is not None:
            state_quality = np.asarray(state_quality, dtype=bool).reshape(-1)
            action_quality = np.asarray(action_quality, dtype=bool).reshape(-1)
            action_horizon = int(self.action_horizon)
            if action_horizon <= 0 or action_quality.size % action_horizon != 0:
                raise ValueError(
                    "UMI action quality mask must contain complete action chunks, "
                    f"got size={action_quality.size}, action_horizon={action_horizon}."
                )
            if state_quality.size != action_quality.size // action_horizon:
                raise ValueError(
                    "UMI state/action quality masks disagree on chunk count: "
                    f"state={state_quality.size}, action={action_quality.size}, "
                    f"action_horizon={action_horizon}."
                )

            # 如果state不合格，屏蔽整个action
            action_quality = action_quality.copy()
            for chunk_idx, state_ok in enumerate(state_quality.tolist()):
                if not state_ok:
                    start = chunk_idx * action_horizon
                    action_quality[start : start + action_horizon] = False

            state_mask = transformed["state_mask"]
            if state_mask.shape[0] != state_quality.size:
                raise ValueError(
                    "UMI state quality mask length does not match transformed state: "
                    f"quality={state_quality.size}, state={state_mask.shape}."
                )
            # feat dim & quality mask = feat dim mask & quality mask
            transformed["state_mask"] = state_mask & state_quality[:, None]

            action_mask = transformed["action_mask"]
            if action_mask.shape[0] != action_quality.size:
                raise ValueError(
                    "UMI action quality mask length does not match transformed action: "
                    f"quality={action_quality.size}, action={action_mask.shape}."
                )
            # feat dim & quality mask = feat dim mask & quality mask
            transformed["action_mask"] = action_mask & action_quality[:, None]

        # LAPA 样本已在基类中统一映射到 action/action_mask；不向 collator 传递中间字段。
        transformed.pop("lapa_action", None)
        transformed.pop("lapa_action_mask", None)

        if self.training:
            transformed["has_real_action"] = np.asarray(
                self.has_real_action, dtype=bool
            )
        return transformed

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video'].

        When ``video_concat_layout == "latent"`` keeps views as ``[V, T, C, H, W]``
        so that per-view VAE encode + DiT token concat happen downstream.
        """
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        # --- latent mode: keep views separate, no canvas stitching ---
        if self.video_concat_layout == "latent":
            return images

        if images.shape[0] > 1:
            return concat_multiview_video(self.embodiment_tag, images)
        return images
