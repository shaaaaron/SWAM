# Copyright 2026 The RLinf Authors.

import json

import pytest

from rlinf.data.datasets.dreamzero.ego_instructions import (
    EgoInstructionIndex,
    append_scene_instruction,
    compose_dense_instruction,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _write_subtasks(root):
    _write_jsonl(
        root / "meta" / "subtask" / "subtasks_v1.jsonl",
        [
            {
                "episode_index": 3,
                "subtasks": [
                    {
                        "start_frame": 0,
                        "end_frame": 100,
                        "caption": "  action   A ",
                        "tag": "valid",
                    },
                    {
                        "start_frame": 100,
                        "end_frame": 200,
                        "caption": "wait",
                        "tag": "no_action",
                    },
                    {
                        "start_frame": 200,
                        "end_frame": 300,
                        "caption": "action B",
                        "tag": "valid",
                    },
                    {
                        "start_frame": 250,
                        "end_frame": 350,
                        "caption": "action B",
                        "tag": "valid",
                    },
                    {
                        "start_frame": 320,
                        "end_frame": 400,
                        "caption": "action C",
                        "tag": "valid",
                    },
                ],
            }
        ],
    )


def _write_scenes(root):
    _write_jsonl(
        root / "meta" / "episodes_extra.jsonl",
        [
            {
                "episode_index": 3,
                "metadata": {
                    "scenario_caption": "  a   kitchen counter ",
                    "scenarios": ["Kitchen", "Food preparation"],
                    "task_type": "Food Processing",
                    "caption": "prepare food",
                },
            },
            {
                "episode_index": 4,
                "scenario_caption": "an outdoor garden",
                "task_type": "Gardening",
            },
        ],
    )


def _write_dense_captions(root):
    _write_jsonl(
        root / "meta" / "dense_caption" / "dense_captions_v1.jsonl",
        [
            {
                "episode_index": 3,
                "source_id": "source-3",
                "dense_captions": [
                    {
                        "start_frame": 0,
                        "end_frame": 100,
                        "action_caption": "  pick   up the cup ",
                        "object_caption": "the cup is lifted",
                        "tag": "valid",
                    },
                    {
                        "start_frame": 100,
                        "end_frame": 220,
                        "action_caption": "place the cup on the shelf.",
                        "object_caption": "the cup rests on the shelf.",
                        "tag": "valid",
                    },
                    {
                        "start_frame": 180,
                        "end_frame": 220,
                        "action_caption": "place the cup on the shelf.",
                        "object_caption": "the cup rests on the shelf.",
                        "tag": "valid",
                    },
                    {
                        "start_frame": 220,
                        "end_frame": 300,
                        "action_caption": "walk away",
                        "object_caption": "the cup remains on the shelf",
                        "tag": "no_action",
                    },
                ],
            }
        ],
    )


def test_ego_subtask_merges_valid_intersections_in_time_order(tmp_path):
    _write_subtasks(tmp_path)
    index = EgoInstructionIndex.load(
        tmp_path,
        use_subtask_instruction=True,
        use_scene_instruction=False,
    )
    assert index.ego_caption_segment_mode == "then"

    prompt = index.get_ego_block_prompt(
        3,
        90,
        330,
        episode_instruction="episode fallback",
    )

    assert prompt == "action A then action B then action C"


def test_ego_subtask_falls_back_once_when_window_has_no_valid_segment(tmp_path):
    _write_subtasks(tmp_path)
    index = EgoInstructionIndex.load(
        tmp_path,
        use_subtask_instruction=True,
        use_scene_instruction=False,
    )

    assert (
        index.get_ego_block_prompt(
            3,
            100,
            200,
            episode_instruction=" episode   fallback ",
        )
        == "episode fallback"
    )


def test_ego_subtask_can_select_maximum_overlap(tmp_path):
    _write_subtasks(tmp_path)
    index = EgoInstructionIndex.load(
        tmp_path,
        use_subtask_instruction=True,
        use_scene_instruction=False,
        ego_caption_segment_mode="max",
    )

    assert (
        index.get_ego_block_prompt(
            3,
            90,
            330,
            episode_instruction="episode fallback",
        )
        == "action B"
    )


def test_disabled_subtask_keeps_default_instruction_without_sidecar(tmp_path):
    index = EgoInstructionIndex.load(
        tmp_path,
        use_subtask_instruction=False,
        use_scene_instruction=False,
    )

    assert (
        index.get_ego_block_prompt(
            99,
            0,
            192,
            episode_instruction="default caption",
        )
        == "default caption"
    )


def test_ego_dense_caption_uses_maximum_valid_overlap(tmp_path):
    _write_dense_captions(tmp_path)
    index = EgoInstructionIndex.load(
        tmp_path,
        use_subtask_instruction=False,
        use_scene_instruction=False,
        use_dense_caption_instruction=True,
    )
    assert index.ego_caption_segment_mode == "max"

    assert (
        index.get_ego_block_prompt(
            3,
            80,
            220,
            episode_instruction="episode fallback",
        )
        == "place the cup on the shelf. the cup rests on the shelf."
    )


def test_ego_dense_caption_can_exclude_object_caption(tmp_path):
    _write_dense_captions(tmp_path)
    index = EgoInstructionIndex.load(
        tmp_path,
        use_subtask_instruction=False,
        use_scene_instruction=False,
        use_dense_caption_instruction=True,
        dense_caption_include_object=False,
    )

    assert (
        index.get_ego_block_prompt(
            3,
            80,
            180,
            episode_instruction="episode fallback",
        )
        == "place the cup on the shelf."
    )


@pytest.mark.parametrize(
    ("include_object", "expected"),
    [
        (
            True,
            "pick up the cup. the cup is lifted. then "
            "place the cup on the shelf. the cup rests on the shelf.",
        ),
        (
            False,
            "pick up the cup. then place the cup on the shelf.",
        ),
    ],
)
def test_ego_dense_caption_then_mode_is_independent_of_object_caption(
    tmp_path,
    include_object,
    expected,
):
    _write_dense_captions(tmp_path)
    index = EgoInstructionIndex.load(
        tmp_path,
        use_subtask_instruction=False,
        use_scene_instruction=False,
        use_dense_caption_instruction=True,
        dense_caption_include_object=include_object,
        ego_caption_segment_mode="then",
    )

    assert (
        index.get_ego_block_prompt(
            3,
            80,
            180,
            episode_instruction="episode fallback",
        )
        == expected
    )


def test_ego_dense_caption_max_mode_keeps_first_segment_on_overlap_tie(tmp_path):
    _write_dense_captions(tmp_path)
    index = EgoInstructionIndex.load(
        tmp_path,
        use_subtask_instruction=False,
        use_scene_instruction=False,
        use_dense_caption_instruction=True,
        dense_caption_include_object=False,
        ego_caption_segment_mode="max",
    )

    assert (
        index.get_ego_block_prompt(
            3,
            50,
            150,
            episode_instruction="episode fallback",
        )
        == "pick up the cup."
    )


def test_ego_dense_caption_falls_back_when_no_valid_segment_overlaps(tmp_path):
    _write_dense_captions(tmp_path)
    index = EgoInstructionIndex.load(
        tmp_path,
        use_subtask_instruction=False,
        use_scene_instruction=False,
        use_dense_caption_instruction=True,
    )

    assert (
        index.get_ego_block_prompt(
            3,
            230,
            290,
            episode_instruction=" episode   fallback ",
        )
        == "episode fallback"
    )


def test_compose_dense_instruction_normalizes_text_and_punctuation():
    assert (
        compose_dense_instruction("  pick   up the cup ", "the cup is lifted")
        == "pick up the cup. the cup is lifted."
    )


def test_ego_subtask_and_dense_caption_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        EgoInstructionIndex.load(
            tmp_path,
            use_subtask_instruction=True,
            use_scene_instruction=False,
            use_dense_caption_instruction=True,
        )


def test_ego_caption_segment_mode_rejects_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="must be 'then' or 'max'"):
        EgoInstructionIndex.load(
            tmp_path,
            use_subtask_instruction=False,
            use_scene_instruction=False,
            ego_caption_segment_mode="all",
        )


def test_ego_scene_supports_nested_and_flat_episode_metadata(tmp_path):
    _write_scenes(tmp_path)
    index = EgoInstructionIndex.load(
        tmp_path,
        use_subtask_instruction=False,
        use_scene_instruction=True,
    )

    nested = index.get_ego_scenario(3)
    flat = index.get_ego_scenario(4)

    assert nested == {
        "scenario_tags": ["Kitchen", "Food preparation"],
        "scenario_caption": "a kitchen counter",
        "task_type": "Food Processing",
        "episode_caption": "prepare food",
    }
    assert flat["scenario_caption"] == "an outdoor garden"
    assert flat["task_type"] == "Gardening"


@pytest.mark.parametrize(
    ("instruction", "scene", "expected"),
    [
        ("pick up the cup", "a kitchen", "pick up the cup. Scene: a kitchen"),
        ("pick up the cup.", "a kitchen", "pick up the cup. Scene: a kitchen"),
        ("pick up the cup?", "a kitchen", "pick up the cup? Scene: a kitchen"),
        ("pick up the cup", "", "pick up the cup"),
        ("", "a kitchen", "Scene: a kitchen"),
    ],
)
def test_append_scene_instruction(instruction, scene, expected):
    assert append_scene_instruction(instruction, scene) == expected


@pytest.mark.parametrize(
    ("use_subtask", "use_dense", "use_scene", "missing_name"),
    [
        (True, False, False, "subtasks_v1.jsonl"),
        (False, True, False, "dense_captions_v1.jsonl"),
        (False, False, True, "episodes_extra.jsonl"),
    ],
)
def test_enabled_ego_sidecar_must_exist(
    tmp_path,
    use_subtask,
    use_dense,
    use_scene,
    missing_name,
):
    with pytest.raises(FileNotFoundError, match=missing_name):
        EgoInstructionIndex.load(
            tmp_path,
            use_subtask_instruction=use_subtask,
            use_dense_caption_instruction=use_dense,
            use_scene_instruction=use_scene,
        )
