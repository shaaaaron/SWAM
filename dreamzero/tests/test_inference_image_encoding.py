from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
    WANPolicyHead,
    WANPolicyHeadConfig,
)


def test_image_encoder_conditioning_is_disabled_by_default():
    assert WANPolicyHeadConfig().use_clip_image_encoder is False


@pytest.mark.parametrize("architecture", ["split", "full"])
@pytest.mark.parametrize("source", ["clip", "vae", "siglip2", "dinov2", "qwen3_vl"])
def test_image_encoder_conditioning_fails_before_component_loading(
    monkeypatch, architecture, source
):
    instantiate = Mock()
    monkeypatch.setattr(f"{WANPolicyHead.__module__}.instantiate", instantiate)
    config = WANPolicyHeadConfig(
        use_clip_image_encoder=True,
        image_context_type=source,
        diffusion_model_cfg={"action_architecture": architecture},
    )

    with pytest.raises(ValueError, match="set use_clip_image_encoder=False"):
        WANPolicyHead(config)

    instantiate.assert_not_called()


def test_episode_first_frame_tokens_are_disabled_by_default():
    assert WANPolicyHeadConfig().use_episode_first_frame_token is False


@pytest.mark.parametrize("architecture", ["split", "full"])
@pytest.mark.parametrize("skip_component_loading", [False, True])
def test_episode_first_frame_tokens_fail_before_component_loading(
    monkeypatch, architecture, skip_component_loading
):
    instantiate = Mock()
    monkeypatch.setattr(f"{WANPolicyHead.__module__}.instantiate", instantiate)
    config = WANPolicyHeadConfig(
        use_clip_image_encoder=False,
        use_episode_first_frame_token=True,
        skip_component_loading=skip_component_loading,
        diffusion_model_cfg={"action_architecture": architecture},
    )

    with pytest.raises(ValueError, match="set use_episode_first_frame_token=False"):
        WANPolicyHead(config)

    instantiate.assert_not_called()


@pytest.mark.parametrize("architecture", ["split", "full"])
@pytest.mark.parametrize(
    "legacy_fields",
    [{}, {"use_clip_image_encoder": False, "use_episode_first_frame_token": False}],
)
def test_disabled_legacy_conditioning_keeps_normal_component_initialization(
    monkeypatch, architecture, legacy_fields
):
    instantiate = Mock(side_effect=RuntimeError("component construction reached"))
    monkeypatch.setattr(f"{WANPolicyHead.__module__}.instantiate", instantiate)
    config = WANPolicyHeadConfig(
        num_frames=33,
        diffusion_model_cfg={"action_architecture": architecture},
        **legacy_fields,
    )

    with pytest.raises(RuntimeError, match="component construction reached"):
        WANPolicyHead(config)

    instantiate.assert_called_once_with(config.text_encoder_cfg)


def test_unconditioned_inference_encodes_only_the_anchor_frame():
    image = torch.randn(2, 1, 3, 16, 32, requires_grad=True)
    latent = torch.randn(2, 48, 1, 1, 2)

    def encode(video):
        assert video.shape == (2, 3, 1, 16, 32)
        torch.testing.assert_close(video, image.transpose(1, 2))
        assert not torch.is_grad_enabled()
        assert torch.is_autocast_enabled("cpu")
        return latent

    head = SimpleNamespace(
        use_clip_image_encoder=False,
        model=SimpleNamespace(concat_first_frame_latent=False, model_type="ti2v"),
        vae=SimpleNamespace(encode=Mock(side_effect=encode)),
        _device="cpu",
        _ensure_vae_on_device=Mock(),
        encode_image=Mock(side_effect=AssertionError("unexpected padded encoding")),
    )
    context, conditions, result = WANPolicyHead._encode_inference_image(
        head, image, 33, 16, 32
    )

    assert context is None
    assert conditions is None
    assert result is latent
    head.vae.encode.assert_called_once()
    head._ensure_vae_on_device.assert_called_once()
    head.encode_image.assert_not_called()


@pytest.mark.parametrize(
    "concat,i2v", [case for case in product((False, True), repeat=2) if any(case)]
)
def test_vae_conditioning_keeps_the_original_encoding(concat, i2v):
    image = torch.randn(1, 1, 3, 16, 32)
    expected = (object(), object(), object())
    head = SimpleNamespace(
        use_clip_image_encoder=False,
        model=SimpleNamespace(
            concat_first_frame_latent=concat, model_type="i2v" if i2v else "ti2v"
        ),
        encode_image=Mock(return_value=expected),
    )

    result = WANPolicyHead._encode_inference_image(head, image, 33, 16, 32)

    assert result is expected
    head.encode_image.assert_called_once_with(image, 33, 16, 32, encode_clip=False)
