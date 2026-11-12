# Copyright 2026 The RLinf Authors.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from groot.vla.data.schema import EmbodimentTag
from groot.vla.data.transform.base import IdentityModalityTransform
from omegaconf import OmegaConf

import rlinf.data.datasets.dreamzero.data_transforms as data_transform_registry
from rlinf.data.datasets.dreamzero.data_transforms import (
    embodiment_tag_mapping_for_embodiment,
    noe_robotics_16d,
)
from rlinf.data.datasets.dreamzero.data_transforms.dream_transform import (
    concat_multiview_video,
)
from rlinf.data.datasets.dreamzero.data_transforms.noe_robotics_16d import (
    NoeRobotics16DDataTransform,
    NoeRoboticsViewResize,
)

VIDEO_KEYS = [
    "video.image",
    "video.wrist_image_1",
    "video.wrist_image_2",
]


def test_noe_robotics_video_to_tensor_accepts_per_view_source_resolutions():
    transform = noe_robotics_16d.NoeRoboticsVideoToTensor(
        apply_to=VIDEO_KEYS,
        backend="torchvision",
    )
    data = {
        "video.image": np.zeros((2, 12, 16, 3), dtype=np.uint8),
        "video.wrist_image_1": np.zeros((2, 12, 20, 3), dtype=np.uint8),
        "video.wrist_image_2": np.zeros((2, 10, 16, 3), dtype=np.uint8),
    }

    output = transform.apply(data)

    assert output["video.image"].shape == (2, 3, 12, 16)
    assert output["video.wrist_image_1"].shape == (2, 3, 12, 20)
    assert output["video.wrist_image_2"].shape == (2, 3, 10, 16)
    assert all(value.dtype == torch.float32 for value in output.values())


def test_noe_robotics_pipeline_normalizes_views_before_shared_transform(monkeypatch):
    monkeypatch.setattr(
        noe_robotics_16d,
        "DreamTransform",
        lambda **kwargs: IdentityModalityTransform(),
    )
    pipeline = NoeRobotics16DDataTransform._build_composed_transform(
        tokenizer_path="unused",
        state_horizon=4,
        action_horizon=48,
        max_state_dim=16,
        max_action_dim=16,
        max_length=512,
        default_instruction="unused",
        language_dropout_prob=0.0,
        always_use_default_instruction=False,
        image_resolution_width=8,
        image_resolution_height=6,
        concat_method="canvas",
        has_real_action=True,
        embodiment_tag_mapping={"noe_robotics_16d": 39},
    )
    video_metadata = {
        key.split(".", 1)[1]: SimpleNamespace(resolution=(16, 12)) for key in VIDEO_KEYS
    }
    metadata = SimpleNamespace(modalities=SimpleNamespace(video=video_metadata))
    data = {
        "video.image": np.zeros((2, 12, 16, 3), dtype=np.uint8),
        "video.wrist_image_1": np.zeros((2, 12, 20, 3), dtype=np.uint8),
        "video.wrist_image_2": np.zeros((2, 10, 16, 3), dtype=np.uint8),
    }

    for transform in pipeline.transforms[:2]:
        transform.set_metadata(metadata)
        data = transform.apply(data)

    assert isinstance(pipeline.transforms[0], noe_robotics_16d.NoeRoboticsVideoToTensor)
    assert isinstance(pipeline.transforms[1], NoeRoboticsViewResize)
    assert {tuple(data[key].shape) for key in VIDEO_KEYS} == {(2, 3, 6, 8)}

    crop = pipeline.transforms[2]
    assert crop.scale == 0.95
    crop.set_metadata(metadata)
    data = crop.apply(data)
    assert len({tuple(data[key].shape) for key in VIDEO_KEYS}) == 1


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (np.zeros((2, 12, 16, 3), dtype=np.float32), "dtype uint8"),
        (np.zeros((2, 12, 16), dtype=np.uint8), "4 dimensions"),
        (np.zeros((2, 12, 16, 4), dtype=np.uint8), "3 channels"),
    ],
)
def test_noe_robotics_video_to_tensor_still_rejects_malformed_video(frames, message):
    transform = noe_robotics_16d.NoeRoboticsVideoToTensor(
        apply_to=["video.image"],
        backend="torchvision",
    )

    with pytest.raises(ValueError, match=message):
        transform.apply({"video.image": frames})


def test_noe_robotics_transform_identity_and_keys():
    cfg = NoeRobotics16DDataTransform.get_modality_config()

    assert NoeRobotics16DDataTransform.TAG == "noe_robotics_16d"
    assert NoeRobotics16DDataTransform.DEFAULT_TAG_MAPPING == {"noe_robotics_16d": 39}
    assert cfg["video"].modality_keys == VIDEO_KEYS
    assert cfg["state"].modality_keys == ["state.eef_position"]
    assert cfg["language"].modality_keys == ["annotation.language.action_text"]


def test_noe_robotics_transform_is_registered_with_id_39():
    assert embodiment_tag_mapping_for_embodiment("noe_robotics_16d") == {
        "noe_robotics_16d": 39
    }


def test_noe_robotics_tag_is_registered_in_dreamzero_schema():
    values = {tag.value for tag in EmbodimentTag}
    if "noe_robotics_16d" not in values:
        pytest.skip("installed Groot does not yet expose the NOE enum value")
    assert EmbodimentTag("noe_robotics_16d").value == "noe_robotics_16d"


def test_noe_robotics_prompt_describes_robot_and_three_views():
    prompt = NoeRobotics16DDataTransform.format_training_prompt("pick up the cup.")

    assert "dual-arm robot" in prompt
    assert "wrist_image_1" in prompt
    assert "wrist_image_2" in prompt
    assert "human" not in prompt


def test_per_view_resize_has_fixed_stretch_contract():
    transform = NoeRoboticsViewResize(
        apply_to=VIDEO_KEYS,
        height=128,
        width=160,
    )
    data = {
        "video.image": torch.zeros(2, 3, 720, 1280, dtype=torch.uint8),
        "video.wrist_image_1": torch.zeros(2, 3, 480, 640, dtype=torch.uint8),
        "video.wrist_image_2": torch.zeros(2, 3, 368, 640, dtype=torch.uint8),
    }

    output = transform.apply(data)

    assert {tuple(output[key].shape) for key in data} == {(2, 3, 128, 160)}
    assert "mode" not in NoeRoboticsViewResize.model_fields
    assert "pad_value" not in NoeRoboticsViewResize.model_fields


def test_per_view_resize_updates_each_metadata_resolution():
    video_metadata = {
        key.split(".", 1)[1]: SimpleNamespace(resolution=(640, 480))
        for key in VIDEO_KEYS
    }
    metadata = SimpleNamespace(modalities=SimpleNamespace(video=video_metadata))
    transform = NoeRoboticsViewResize(
        apply_to=VIDEO_KEYS,
        height=128,
        width=160,
    )

    transform.set_metadata(metadata)

    assert {entry.resolution for entry in video_metadata.values()} == {(160, 128)}


def test_noe_robotics_canvas_places_main_and_wrist_views():
    images = np.zeros((3, 1, 1, 2, 3), dtype=np.uint8)
    images[0] = 10
    images[1] = 20
    images[2] = 30

    output = NoeRobotics16DDataTransform.concat_multiview_video(images)

    assert output.shape == (1, 1, 1, 4, 6)
    assert np.all(output[..., :2, :] == 10)
    assert np.all(output[..., 2:, :3] == 20)
    assert np.all(output[..., 2:, 3:] == 30)


def test_registered_concat_uses_the_common_single_argument_interface(monkeypatch):
    images = np.zeros((3, 1, 1, 2, 3), dtype=np.uint8)
    captured = []

    class FakeTransform:
        @staticmethod
        def concat_multiview_video(received):
            captured.append(received)
            return received

    monkeypatch.setattr(
        data_transform_registry,
        "_require_embodiment",
        lambda _tag: FakeTransform,
    )

    output = concat_multiview_video("noe_robotics_16d", images)

    assert output is images
    assert captured == [images]


def test_noe_robotics_transform_does_not_forward_legacy_view_options(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_build(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        NoeRobotics16DDataTransform,
        "_build_composed_transform",
        staticmethod(fake_build),
    )
    result = NoeRobotics16DDataTransform.get_transform(
        tokenizer_path="unused",
        cfg=OmegaConf.create(
            {
                "view_resize_mode": "contain",
                "view_pad_value": 23,
                "video_crop_scale": 0.9,
            }
        ),
        embodiment_tag_mapping={"noe_robotics_16d": 39},
    )

    assert result is sentinel
    assert "view_resize_mode" not in captured
    assert "view_pad_value" not in captured
    assert "video_crop_scale" not in captured
