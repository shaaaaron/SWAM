# Copyright 2026 The RLinf Authors.

import numpy as np
import pytest
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
    DreamTransform as DreamTransformBase,
)
from omegaconf import OmegaConf

from rlinf.data.datasets.dreamzero.data_transforms import egodex
from rlinf.data.datasets.dreamzero.data_transforms.dream_transform import (
    DreamTransform,
)
from rlinf.data.datasets.dreamzero.data_transforms.noe_umi_16d import (
    NoeUmi16DDataTransform,
)
from rlinf.data.datasets.dreamzero.data_transforms.oxe_droid import (
    OxeDroidDataTransform,
)
from rlinf.data.datasets.dreamzero.data_transforms.robotwin import (
    RobotwinDataTransform,
)


@pytest.mark.parametrize(
    "transform_cls",
    [
        RobotwinDataTransform,
        NoeUmi16DDataTransform,
        OxeDroidDataTransform,
    ],
)
@pytest.mark.parametrize("configured_value", [None, False])
def test_robotics_transform_passes_action_supervision_flag(
    transform_cls,
    configured_value,
    monkeypatch,
):
    captured = {}
    sentinel = object()

    def fake_build(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        transform_cls,
        "_build_composed_transform",
        staticmethod(fake_build),
    )
    cfg = {} if configured_value is None else {"has_real_action": configured_value}

    result = transform_cls.get_transform(
        tokenizer_path="unused",
        cfg=OmegaConf.create(cfg),
        embodiment_tag_mapping={transform_cls.TAG: 0},
    )

    assert result is sentinel
    assert captured["has_real_action"] is (
        True if configured_value is None else configured_value
    )


@pytest.mark.parametrize("configured_value", [None, False])
def test_egodex_transform_passes_action_supervision_flag(
    configured_value,
    monkeypatch,
):
    captured = {}

    def fake_dream_transform(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(egodex, "DreamTransform", fake_dream_transform)
    monkeypatch.setattr(
        egodex,
        "ComposedModalityTransform",
        lambda *, transforms: transforms,
    )
    cfg = {} if configured_value is None else {"has_real_action": configured_value}

    egodex.EgoDexDataTransform.get_transform(
        tokenizer_path="unused",
        cfg=OmegaConf.create(cfg),
        embodiment_tag_mapping={"egodex": 37},
    )

    assert captured["has_real_action"] is (
        True if configured_value is None else configured_value
    )


def test_egodex_modality_config_uses_configured_action_horizon():
    modality = egodex.EgoDexDataTransform.get_modality_config(
        OmegaConf.create({"action_horizon": 12})
    )

    assert modality["action"].delta_indices == list(range(12))


def test_egodex_transform_uses_direct_320x256_resize(monkeypatch):
    captured = {}

    class FakeResize:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(egodex, "VideoResize", FakeResize)
    monkeypatch.setattr(egodex, "DreamTransform", lambda **kwargs: object())
    monkeypatch.setattr(
        egodex,
        "ComposedModalityTransform",
        lambda *, transforms: transforms,
    )

    egodex.EgoDexDataTransform.get_transform(
        tokenizer_path="unused",
        cfg=OmegaConf.create(
            {"image_resolution_height": 256, "image_resolution_width": 320}
        ),
        embodiment_tag_mapping={"egodex": 37},
    )

    assert captured["height"] == 256
    assert captured["width"] == 320


def test_egodex_empty_instruction_stays_empty():
    assert egodex.EgoDexDataTransform.format_training_prompt("") == ""
    assert egodex.EgoDexDataTransform.format_training_prompt("   ") == ""


def test_disabling_action_supervision_keeps_robotics_state_and_action(monkeypatch):
    state = np.arange(16, dtype=np.float32).reshape(1, 16)
    action = np.arange(32, dtype=np.float32).reshape(2, 16)

    def fake_apply_single(_self, _data):
        return {
            "state": state.copy(),
            "action": action.copy(),
            "has_real_action": np.asarray(True, dtype=bool),
        }

    monkeypatch.setattr(DreamTransformBase, "apply_single", fake_apply_single)
    transform = DreamTransform.model_construct(
        training=True,
        has_real_action=False,
    )

    transformed = transform.apply_single({})

    np.testing.assert_array_equal(transformed["state"], state)
    np.testing.assert_array_equal(transformed["action"], action)
    assert transformed["has_real_action"].shape == ()
    assert not bool(transformed["has_real_action"])
