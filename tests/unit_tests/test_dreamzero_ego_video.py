# Copyright 2026 The RLinf Authors.

import numpy as np
import pytest
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
    DreamTransform as DreamTransformBase,
)

from rlinf.data.datasets.dreamzero.data_transforms.dream_transform import (
    DreamTransform,
)
from rlinf.data.datasets.dreamzero.data_transforms.ego_video import (
    Ego4DDataTransform,
    EgoverseDataTransform,
    EgoVideoDataTransform,
    EgoVideoDreamTransform,
    OpenVidDataTransform,
    concat_ego_video_views,
    ego_resize_resolution,
)
from rlinf.data.datasets.dreamzero.data_transforms.noe_umi_16d import (
    NoeUmi16DDataTransform,
)
from rlinf.data.datasets.dreamzero.dreamzero import DreamZeroCollator


def _bare_transform() -> EgoVideoDreamTransform:
    return EgoVideoDreamTransform.model_construct(
        state_horizon=1,
        action_horizon=48,
        max_state_dim=16,
        max_action_dim=16,
        video_frames_per_chunk=8,
        max_chunk_size=4,
    )


@pytest.mark.parametrize("has_real_action", [True, False])
def test_dream_transform_drops_lapa_intermediate_fields(
    monkeypatch, has_real_action
):
    action = np.ones((48, 16), dtype=np.float32)
    action_mask = np.ones_like(action, dtype=bool)

    monkeypatch.setattr(
        DreamTransformBase,
        "apply_single",
        lambda _self, _data: {
            "action": action,
            "action_mask": action_mask,
            "lapa_action": np.zeros_like(action),
            "lapa_action_mask": np.zeros_like(action_mask),
        },
    )
    transform = DreamTransform.model_construct(
        training=True,
        has_real_action=has_real_action,
    )

    transformed = transform.apply_single({})

    assert transformed["action"] is action
    assert transformed["action_mask"] is action_mask
    assert "lapa_action" not in transformed
    assert "lapa_action_mask" not in transformed
    assert transformed["has_real_action"].item() is has_real_action


def test_ego_video_transform_drops_lapa_intermediate_fields(monkeypatch):
    action = np.zeros((48, 16), dtype=np.float32)
    action_mask = np.zeros_like(action, dtype=bool)

    monkeypatch.setattr(
        DreamTransformBase,
        "apply_single",
        lambda _self, _data: {
            "action": action,
            "action_mask": action_mask,
            "lapa_action": np.zeros_like(action),
            "lapa_action_mask": np.zeros_like(action_mask),
        },
    )
    transform = EgoVideoDreamTransform.model_construct(
        training=True,
        has_real_action=False,
    )

    transformed = transform.apply_single({})

    assert transformed["action"] is action
    assert transformed["action_mask"] is action_mask
    assert "lapa_action" not in transformed
    assert "lapa_action_mask" not in transformed
    assert transformed["has_real_action"].item() is False


@pytest.mark.parametrize(
    ("num_frames", "num_chunks"),
    [(9, 1), (17, 2), (25, 3), (33, 4), (34, 4)],
)
def test_ego_placeholder_lengths_follow_video_chunks(num_frames, num_chunks):
    transform = _bare_transform()
    data = {"video": np.zeros((num_frames, 1, 8, 8, 3), dtype=np.uint8)}

    state, state_mask, state_tokens = transform._prepare_state(data)
    action, action_mask, action_tokens = transform._prepare_action(data)

    assert state.shape == (num_chunks, 16)
    assert action.shape == (num_chunks * 48, 16)
    assert state_tokens == num_chunks
    assert action_tokens == num_chunks * 48
    assert not state_mask.any()
    assert not action_mask.any()


def test_ego_placeholder_rejects_invalid_video_length():
    transform = _bare_transform()
    with pytest.raises(ValueError, match=r"8 \* num_chunks \+ 1"):
        transform._prepare_action(
            {"video": np.zeros((16, 1, 8, 8, 3), dtype=np.uint8)}
        )


def test_ego_prompt_receives_normalized_terminal_punctuation():
    prompt = EgoVideoDataTransform.format_training_prompt("prepare food.")
    assert "activity: prepare food. The video" in prompt
    assert "food.." not in prompt


def test_openvid_prompt_adds_prefix_without_claiming_first_person_view():
    prompt = OpenVidDataTransform.format_training_prompt(
        "a pink minivan drives through a parking lot."
    )
    assert prompt == (
        "A video depicts the following scene or activity, describing the main "
        "objects, actions, and interactions visible in the footage: "
        "a pink minivan drives through a parking lot."
    )
    assert "pink minivan" in prompt
    assert "first-person" not in prompt
    assert "person's point of view" not in prompt


def test_collator_keeps_empty_instruction_unwrapped():
    class RecordingTokenizer:
        def __init__(self):
            self.texts = None

        def __call__(self, texts, **_kwargs):
            self.texts = texts
            batch_size = len(texts)
            return np.zeros((batch_size, 4), dtype=np.int64), np.ones(
                (batch_size, 4), dtype=np.int64
            )

    tokenizer = RecordingTokenizer()
    DreamZeroCollator.collate_batch(
        [
            {"text": "   ", "embodiment_id": np.asarray(34, dtype=np.int64)},
            {
                "text": "Pick up the cup",
                "embodiment_id": np.asarray(34, dtype=np.int64),
            },
        ],
        tokenizer,
        {"noe_umi_16d": 34},
    )

    assert tokenizer.texts == [
        "",
        NoeUmi16DDataTransform.format_training_prompt("pick up the cup."),
    ]


def test_ego_sources_use_distinct_embodiment_ids():
    assert Ego4DDataTransform.DEFAULT_TAG_MAPPING == {"ego4d": 35}
    assert EgoverseDataTransform.DEFAULT_TAG_MAPPING == {"egoverse": 36}
    assert OpenVidDataTransform.DEFAULT_TAG_MAPPING == {"openvid": 38}


def test_ego_single_view_layout_keeps_the_original_view():
    images = np.arange(12, dtype=np.uint8).reshape(1, 1, 1, 3, 4)

    output = concat_ego_video_views(images, "single")

    assert output.shape == images.shape
    assert np.array_equal(output, images)


def test_ego_repeat_three_layout_uses_robotwin_canvas_slots():
    main = np.arange(6, dtype=np.uint8).reshape(1, 1, 1, 2, 3)

    output = concat_ego_video_views(main, "repeat_three")

    assert output.shape == (1, 1, 1, 4, 6)
    assert np.array_equal(
        output[0, 0, 0, :2], np.repeat(main[0, 0, 0], 2, axis=-1)
    )
    assert np.array_equal(output[0, 0, 0, 2:, :3], main[0, 0, 0])
    assert np.array_equal(output[0, 0, 0, 2:, 3:], main[0, 0, 0])


def test_ego_top_gray_layout_fills_bottom_slots_with_127():
    main = np.arange(6, dtype=np.uint8).reshape(1, 1, 1, 2, 3)

    output = concat_ego_video_views(main, "top_gray")

    assert output.shape == (1, 1, 1, 4, 6)
    assert np.array_equal(
        output[0, 0, 0, :2], np.repeat(main[0, 0, 0], 2, axis=-1)
    )
    assert np.all(output[0, 0, 0, 2:] == 127)


def test_ego_view_layout_rejects_unknown_mode():
    with pytest.raises(ValueError, match="ego_view_layout"):
        concat_ego_video_views(np.zeros((1, 1, 1, 2, 2), dtype=np.uint8), "bad")


def test_ego_generic_multiview_interface_is_not_used():
    with pytest.raises(RuntimeError, match="EgoVideoDreamTransform._prepare_video"):
        EgoVideoDataTransform.concat_multiview_video(
            np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)
        )


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ("single", (256, 320)),
        ("repeat_three", (128, 160)),
        ("top_gray", (128, 160)),
    ],
)
def test_ego_layout_uses_expected_pre_canvas_resize(layout, expected):
    assert ego_resize_resolution(layout, 256, 320) == expected


def test_ego_three_slot_layout_requires_even_target_resolution():
    with pytest.raises(ValueError, match="even image resolution"):
        ego_resize_resolution("repeat_three", 255, 320)


def test_ego_collator_pads_temporal_registers_and_keeps_action_unsupervised():
    class DummyTokenizer:
        def __call__(self, texts, **_kwargs):
            batch_size = len(texts)
            return np.zeros((batch_size, 4), dtype=np.int64), np.ones(
                (batch_size, 4), dtype=np.int64
            )

    def sample(num_chunks: int, embodiment_id: int) -> dict:
        num_frames = num_chunks * 8 + 1
        num_actions = num_chunks * 48
        return {
            "images": np.full(
                (num_frames, 2, 2, 3), num_chunks, dtype=np.uint8
            ),
            "state": np.zeros((num_chunks, 16), dtype=np.float32),
            "state_mask": np.zeros((num_chunks, 16), dtype=bool),
            "action": np.zeros((num_actions, 16), dtype=np.float32),
            "action_mask": np.zeros((num_actions, 16), dtype=bool),
            "has_real_action": np.asarray(False, dtype=bool),
            "embodiment_id": np.asarray(embodiment_id, dtype=np.int64),
            "text": "perform the activity.",
        }

    batch = DreamZeroCollator.collate_batch(
        [sample(1, 35), sample(2, 36)],
        DummyTokenizer(),
        {"ego4d": 35, "egoverse": 36},
    )

    assert tuple(batch["images"].shape) == (2, 17, 2, 2, 3)
    assert tuple(batch["state"].shape) == (2, 2, 16)
    assert tuple(batch["action"].shape) == (2, 96, 16)
    assert batch["image_valid_mask"].sum(dim=1).tolist() == [9, 17]
    assert batch["state_valid_mask"].sum(dim=1).tolist() == [1, 2]
    assert batch["action_valid_mask"].sum(dim=1).tolist() == [48, 96]
    assert batch["has_real_action"].tolist() == [False, False]
    assert batch["action_mask"].sum().item() == 0
    assert np.all(batch["images"][0, 9:].numpy() == 1)


def test_collator_pads_multi_anchor_samples_to_configured_temporal_capacity():
    class DummyTokenizer:
        def __call__(self, texts, **_kwargs):
            batch_size = len(texts)
            return np.zeros((batch_size, 4), dtype=np.int64), np.ones(
                (batch_size, 4), dtype=np.int64
            )

    def sample(num_chunks: int) -> dict:
        return {
            "images": np.full(
                (num_chunks * 8 + 1, 2, 2, 3), num_chunks, dtype=np.uint8
            ),
            "state": np.zeros((num_chunks, 16), dtype=np.float32),
            "state_mask": np.ones((num_chunks, 16), dtype=bool),
            "action": np.zeros((num_chunks * 48, 16), dtype=np.float32),
            "action_mask": np.ones((num_chunks * 48, 16), dtype=bool),
            "has_real_action": np.asarray(True, dtype=bool),
            "embodiment_id": np.asarray(33, dtype=np.int64),
            "text": "perform the activity.",
        }

    batch = DreamZeroCollator.collate_batch(
        [sample(1), sample(2)],
        DummyTokenizer(),
        {"robotwin": 33},
        temporal_pad_lengths={
            "images": 33,
            "state": 4,
            "action": 192,
        },
    )

    assert tuple(batch["images"].shape) == (2, 33, 2, 2, 3)
    assert tuple(batch["state"].shape) == (2, 4, 16)
    assert tuple(batch["action"].shape) == (2, 192, 16)
    assert batch["image_valid_mask"].sum(dim=1).tolist() == [9, 17]
    assert batch["state_valid_mask"].sum(dim=1).tolist() == [1, 2]
    assert batch["action_valid_mask"].sum(dim=1).tolist() == [48, 96]
    assert np.all(batch["images"][0, 9:].numpy() == 1)
    assert np.all(batch["images"][1, 17:].numpy() == 2)
    assert batch["state"][:, 2:].sum().item() == 0
    assert batch["action"][:, 96:].sum().item() == 0


def test_collator_rejects_temporal_samples_larger_than_configured_capacity():
    class DummyTokenizer:
        def __call__(self, texts, **_kwargs):
            batch_size = len(texts)
            return np.zeros((batch_size, 4), dtype=np.int64), np.ones(
                (batch_size, 4), dtype=np.int64
            )

    sample = {
        "images": np.zeros((17, 2, 2, 3), dtype=np.uint8),
        "embodiment_id": np.asarray(33, dtype=np.int64),
        "text": "perform the activity.",
    }

    with pytest.raises(ValueError, match="observed=17, target=9"):
        DreamZeroCollator.collate_batch(
            [sample],
            DummyTokenizer(),
            {"robotwin": 33},
            temporal_pad_lengths={"images": 9},
        )
