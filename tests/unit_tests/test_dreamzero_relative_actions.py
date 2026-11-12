import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from rlinf.data.datasets.dreamzero.dreamzero import DreamZeroLeRobotDataset
from rlinf.data.datasets.dreamzero.relative_actions import (
    RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND,
    ACTION_QUATERNION_MODE_PRESERVE,
    ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS,
    ROBOTWIN_EEF_ACTION_KEY,
    ROBOTWIN_EEF_STATE_KEY,
    ensure_relative_action_cache_compatible,
    ensure_relative_action_metadata_compatible,
    is_dual_arm_eef_se3_relative_action,
    is_dual_arm_eef_se3_relative_state,
    is_robotwin_eef_se3_chunk_relative,
    is_robotwin_eef_se3_relative_action,
    is_robotwin_eef_se3_s0_relative_action,
    is_robotwin_eef_se3_s0_relative_state,
    relative_action_mode_embodiment_tag,
    robotwin_eef_abs_to_rel_wxyz,
    robotwin_eef_rel_to_abs_wxyz,
    temporal_continuous_robotwin_eef_action_wxyz,
)
from rlinf.models.embodiment.dreamzero import _validate_checkpoint_weight_keys
from rlinf.models.embodiment.dreamzero.dreamzero_config import (
    validate_dreamzero_sft_model_cfg,
)


def _quat_from_z(theta):
    half = theta / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _pose16(left_pos, left_quat, left_grip, right_pos, right_quat, right_grip):
    return np.asarray(
        [
            *left_pos,
            *left_quat,
            left_grip,
            *right_pos,
            *right_quat,
            right_grip,
        ],
        dtype=np.float32,
    )


def test_temporal_continuous_action_aligns_each_chunk_to_state():
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pose = _pose16([0, 0, 0], identity, 0.1, [0, 0, 0], identity, 0.2)
    neg_pose = pose.copy()
    neg_pose[[3, 4, 5, 6, 11, 12, 13, 14]] *= -1
    actions = np.stack([pose, neg_pose, neg_pose, pose, neg_pose, neg_pose])
    anchors = np.stack([pose, pose, pose, neg_pose, neg_pose, neg_pose])

    output = temporal_continuous_robotwin_eef_action_wxyz(
        actions,
        anchors,
        action_horizon=3,
    )

    np.testing.assert_allclose(output[:, :3], actions[:, :3])
    np.testing.assert_allclose(output[:, 7], actions[:, 7])
    np.testing.assert_allclose(output[:, 8:11], actions[:, 8:11])
    np.testing.assert_allclose(output[:, 15], actions[:, 15])
    for start in (0, 3):
        for quat_sl in (slice(3, 7), slice(11, 15)):
            assert np.dot(output[start, quat_sl], anchors[start, quat_sl]) >= 0
            assert np.all(
                np.sum(output[start : start + 3, quat_sl][1:] * output[start : start + 3, quat_sl][:-1], axis=-1)
                >= 0
            )


def test_temporal_continuous_preserves_zero_quaternion_without_special_fallback():
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pose = _pose16([0, 0, 0], identity, 0.1, [0, 0, 0], identity, 0.2)
    pose[3:7] = 0.0
    pose[11:15] = 0.0
    output = temporal_continuous_robotwin_eef_action_wxyz(
        pose[None], pose[None], action_horizon=48
    )
    np.testing.assert_array_equal(output, pose[None])


def _quaternion_continuity_sample(num_actions, action_horizon=48):
    actions = np.stack(
        [
            _pose16(
                [index / 100.0, 0.2, 0.8],
                _quat_from_z(index / 500.0),
                0.1 + index / 1000.0,
                [-0.2, index / 100.0, 0.9],
                _quat_from_z(-index / 500.0),
                0.9 - index / 1000.0,
            )
            for index in range(num_actions)
        ]
    )
    states = actions[::action_horizon].copy()
    states[1::2, 3:7] *= -1
    states[::2, 11:15] *= -1
    actions[::3, 3:7] *= -1
    actions[::2, 11:15] *= -1
    return states, actions


@pytest.mark.parametrize(
    "mode",
    [ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS, ACTION_QUATERNION_MODE_PRESERVE],
)
def test_modality_dict_uses_each_chunk_state_without_changing_inputs(mode):
    dataset = object.__new__(DreamZeroLeRobotDataset)
    dataset.embodiment_tag = "robotwin"
    dataset.relative_action = False
    dataset.action_quaternion_mode = mode
    dataset.action_horizon = 48
    dataset._source_video_key = {}
    dataset.return_episode_first_images = False
    dataset._state_components = {
        ROBOTWIN_EEF_STATE_KEY: ("observation.state", slice(0, 16))
    }
    dataset._action_components = {ROBOTWIN_EEF_ACTION_KEY: ("action", slice(0, 16))}
    dataset._ego_instruction_index = None
    dataset.episode_instruction_key = None
    dataset.language_keys = ["annotation.human.action.task_description"]
    dataset._language_sources = {}
    dataset._tasks = {}
    states, actions = _quaternion_continuity_sample(4 * dataset.action_horizon)
    original_states, original_actions = states.copy(), actions.copy()

    output = dataset._build_modality_dict(
        {"observation.state": states, "action": actions, "task": "move the object"}
    )

    np.testing.assert_array_equal(output[ROBOTWIN_EEF_STATE_KEY], original_states)
    np.testing.assert_array_equal(states, original_states)
    np.testing.assert_array_equal(actions, original_actions)
    output_actions = output[ROBOTWIN_EEF_ACTION_KEY]
    non_rotation = [0, 1, 2, 7, 8, 9, 10, 15]
    np.testing.assert_array_equal(
        output_actions[:, non_rotation], original_actions[:, non_rotation]
    )
    if mode == ACTION_QUATERNION_MODE_PRESERVE:
        np.testing.assert_array_equal(output_actions, original_actions)
        return

    for chunk_index, state in enumerate(original_states):
        start = chunk_index * dataset.action_horizon
        chunk = output_actions[start : start + dataset.action_horizon]
        for quat_slice in (slice(3, 7), slice(11, 15)):
            assert np.dot(chunk[0, quat_slice], state[quat_slice]) > 0.99
            assert np.all(
                np.sum(chunk[1:, quat_slice] * chunk[:-1, quat_slice], axis=-1) > 0
            )
    for quat_slice in (slice(3, 7), slice(11, 15)):
        equivalent_rotation = np.abs(
            np.sum(output_actions[:, quat_slice] * original_actions[:, quat_slice], axis=-1)
        )
        np.testing.assert_allclose(equivalent_rotation, 1.0, atol=1e-6)


@pytest.mark.parametrize("num_actions", [48, 96, 109])
def test_temporal_continuous_chunk_anchors_match_per_action_anchors(num_actions):
    horizon = 48
    states, actions = _quaternion_continuity_sample(num_actions, horizon)
    per_action_states = np.repeat(states, horizon, axis=0)[:num_actions]
    original_states, original_actions = states.copy(), actions.copy()
    original_per_action_states = per_action_states.copy()

    chunk_output = temporal_continuous_robotwin_eef_action_wxyz(
        actions, states, action_horizon=horizon
    )
    per_action_output = temporal_continuous_robotwin_eef_action_wxyz(
        actions, per_action_states, action_horizon=horizon
    )

    np.testing.assert_array_equal(chunk_output, per_action_output)
    np.testing.assert_array_equal(states, original_states)
    np.testing.assert_array_equal(per_action_states, original_per_action_states)
    np.testing.assert_array_equal(actions, original_actions)


def test_left_hand_pose_reference_uses_left_anchor_for_both_arms():
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    anchor = _pose16([1, 2, 3], identity, 0.2, [10, 20, 30], identity, 0.8)[None]
    action = _pose16([2, 4, 6], identity, 0.3, [4, 8, 12], identity, 0.7)[None]

    per_arm = robotwin_eef_abs_to_rel_wxyz(action, anchor)
    left_hand = robotwin_eef_abs_to_rel_wxyz(
        action,
        anchor,
        RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND,
    )

    np.testing.assert_allclose(per_arm[0, :3], [1, 2, 3])
    np.testing.assert_allclose(per_arm[0, 8:11], [-6, -12, -18])
    np.testing.assert_allclose(left_hand[0, :3], [1, 2, 3])
    np.testing.assert_allclose(left_hand[0, 8:11], [3, 6, 9])
    np.testing.assert_allclose(left_hand[0, [7, 15]], action[0, [7, 15]])
    np.testing.assert_allclose(
        robotwin_eef_rel_to_abs_wxyz(
            left_hand,
            anchor,
            RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND,
        ),
        action,
        atol=1e-6,
    )


def test_relative_metadata_rejects_pose_reference_mismatch():
    with pytest.raises(ValueError, match="pose-reference mismatch"):
        ensure_relative_action_metadata_compatible(
            embodiment_tag="robotwin",
            relative_action=True,
            relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
            relative_action_pose_reference=RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND,
            metadata_relative_action_mode=(
                RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ
            ),
            metadata_relative_action_pose_reference="per_arm",
        )


def test_action_quaternion_metadata_mode_must_match_config():
    with pytest.raises(ValueError, match="Action quaternion mode mismatch"):
        ensure_relative_action_metadata_compatible(
            embodiment_tag="robotwin",
            relative_action=False,
            relative_action_mode=None,
            action_quaternion_mode=ACTION_QUATERNION_MODE_TEMPORAL_CONTINUOUS,
            metadata_action_quaternion_mode=ACTION_QUATERNION_MODE_PRESERVE,
        )


def test_robotwin_eef_se3_relative_roundtrip_and_gripper_passthrough():
    state = _pose16(
        [1.0, 2.0, 3.0],
        _quat_from_z(np.pi / 2),
        0.25,
        [-1.0, 0.5, 1.5],
        _quat_from_z(-np.pi / 4),
        0.75,
    )
    action = np.stack(
        [
            _pose16(
                [1.0, 3.0, 3.0],
                _quat_from_z(np.pi),
                0.1,
                [-0.5, 0.5, 2.0],
                _quat_from_z(np.pi / 4),
                0.9,
            ),
            _pose16(
                [2.0, 2.0, 4.0],
                -_quat_from_z(np.pi / 2),
                -0.2,
                [-1.0, 1.0, 1.5],
                -_quat_from_z(-np.pi / 2),
                0.3,
            ),
        ],
        axis=0,
    )

    state_anchor = state[None, :]
    relative = robotwin_eef_abs_to_rel_wxyz(action, state_anchor)
    recovered = robotwin_eef_rel_to_abs_wxyz(relative, state_anchor)

    np.testing.assert_allclose(relative[:, [7, 15]], action[:, [7, 15]], atol=1e-6)
    np.testing.assert_allclose(recovered[:, [0, 1, 2, 8, 9, 10]], action[:, [0, 1, 2, 8, 9, 10]], atol=1e-5)
    np.testing.assert_allclose(np.abs(recovered[:, [3, 4, 5, 6, 11, 12, 13, 14]]), np.abs(action[:, [3, 4, 5, 6, 11, 12, 13, 14]]), atol=1e-5)
    assert np.all(relative[:, 3] >= 0.0)
    assert np.all(relative[:, 11] >= 0.0)


def test_robotwin_action_only_mode_keeps_state_absolute_and_reuses_relative_action():
    dataset = object.__new__(DreamZeroLeRobotDataset)
    dataset.relative_action = True
    dataset.relative_action_pose_reference = "per_arm"
    dataset.relative_action_keys = ["eef_position"]
    dataset.state_keys = [ROBOTWIN_EEF_STATE_KEY]
    dataset.sampling_mode = "test"
    dataset.embodiment_tag = "robotwin"

    state = np.stack(
        [
            _pose16(
                [1, 2, 3],
                _quat_from_z(0.25),
                0.1,
                [-1, 0, 2],
                _quat_from_z(-0.5),
                0.9,
            ),
            _pose16(
                [2, 2, 4],
                _quat_from_z(0.75),
                0.2,
                [0, 1, 2],
                _quat_from_z(0.5),
                0.8,
            ),
        ]
    )
    action = state[::-1].copy()

    dataset.relative_action_mode = (
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ
    )
    action_only_out = {ROBOTWIN_EEF_STATE_KEY: state.copy()}
    action_only_anchor = dataset._prepare_dual_arm_eef_relative_state(action_only_out)
    action_only_action = dataset._convert_action_abs_to_relative(
        action, ROBOTWIN_EEF_ACTION_KEY, action_only_anchor
    )
    np.testing.assert_array_equal(action_only_out[ROBOTWIN_EEF_STATE_KEY], state)

    dataset.relative_action_mode = RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ
    full_relative_out = {ROBOTWIN_EEF_STATE_KEY: state.copy()}
    full_relative_anchor = dataset._prepare_dual_arm_eef_relative_state(full_relative_out)
    full_relative_action = dataset._convert_action_abs_to_relative(
        action, ROBOTWIN_EEF_ACTION_KEY, full_relative_anchor
    )
    np.testing.assert_allclose(action_only_action, full_relative_action, atol=1e-6)
    np.testing.assert_allclose(
        full_relative_out[ROBOTWIN_EEF_STATE_KEY],
        robotwin_eef_abs_to_rel_wxyz(state, state[0:1]),
        atol=1e-6,
    )


def test_robotwin_relative_mode_capabilities():
    assert is_robotwin_eef_se3_s0_relative_action(
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ
    )
    assert not is_robotwin_eef_se3_s0_relative_state(
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ
    )
    assert is_robotwin_eef_se3_s0_relative_state(
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ
    )
    assert is_robotwin_eef_se3_relative_action(
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ
    )
    assert is_robotwin_eef_se3_chunk_relative(
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ
    )
    assert not is_robotwin_eef_se3_s0_relative_action(
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ
    )


def test_robotwin_chunk_relative_uses_each_state_for_state_and_action():
    dataset = object.__new__(DreamZeroLeRobotDataset)
    dataset.relative_action = True
    dataset.relative_action_pose_reference = "per_arm"
    dataset.relative_action_keys = ["eef_position"]
    dataset.relative_action_mode = RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ
    dataset.state_keys = [ROBOTWIN_EEF_STATE_KEY]
    dataset.sampling_mode = "multi_anchor"
    dataset.embodiment_tag = "robotwin"
    dataset.action_horizon = 2

    state = np.stack(
        [
            _pose16([1, 2, 3], _quat_from_z(0.25), 0.1, [-1, 0, 2], _quat_from_z(-0.5), 0.9),
            _pose16([4, 1, 2], _quat_from_z(1.0), 0.2, [2, 3, 1], _quat_from_z(0.75), 0.8),
        ]
    )
    action = np.stack(
        [
            _pose16([2, 2, 3], _quat_from_z(0.5), 0.3, [0, 0, 2], _quat_from_z(-0.25), 0.7),
            _pose16([1, 3, 3], _quat_from_z(0.75), 0.4, [-1, 1, 2], _quat_from_z(0.0), 0.6),
            _pose16([5, 1, 2], _quat_from_z(1.25), 0.5, [3, 3, 1], _quat_from_z(1.0), 0.5),
            _pose16([4, 2, 2], _quat_from_z(1.5), 0.6, [2, 4, 1], _quat_from_z(1.25), 0.4),
        ]
    )

    out = {ROBOTWIN_EEF_STATE_KEY: state.copy()}
    absolute_state = dataset._prepare_dual_arm_eef_relative_state(out)
    relative_action = dataset._convert_action_abs_to_relative(
        action, ROBOTWIN_EEF_ACTION_KEY, absolute_state
    )

    np.testing.assert_allclose(
        out[ROBOTWIN_EEF_STATE_KEY],
        robotwin_eef_abs_to_rel_wxyz(state, state),
        atol=1e-6,
    )
    expected_action = robotwin_eef_abs_to_rel_wxyz(
        action.reshape(2, 2, 16), state[:, None, :]
    ).reshape(4, 16)
    np.testing.assert_allclose(relative_action, expected_action, atol=1e-6)
    np.testing.assert_allclose(
        robotwin_eef_rel_to_abs_wxyz(
            relative_action.reshape(2, 2, 16), state[:, None, :]
        ).reshape(4, 16),
        action,
        atol=1e-5,
    )

    with pytest.raises(ValueError, match="complete action chunks"):
        dataset._convert_action_abs_to_relative(
            action[:3], ROBOTWIN_EEF_ACTION_KEY, absolute_state
        )


@pytest.mark.parametrize(
    ("tag", "mode"),
    [
        ("noe_umi_16d", RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ),
        ("noe_robotics_16d", RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ),
    ],
)
def test_noe_full_relative_mode_reuses_robotwin_se3_conversion(tag, mode):
    dataset = object.__new__(DreamZeroLeRobotDataset)
    dataset.relative_action = True
    dataset.relative_action_pose_reference = "per_arm"
    dataset.relative_action_keys = ["eef_position"]
    dataset.state_keys = [ROBOTWIN_EEF_STATE_KEY]
    dataset.sampling_mode = "test"

    state = np.stack(
        [
            _pose16(
                [0.1, -0.2, 0.3],
                _quat_from_z(0.25),
                0.1,
                [-0.4, 0.5, 0.6],
                _quat_from_z(-0.5),
                0.9,
            ),
            _pose16(
                [0.2, -0.1, 0.4],
                _quat_from_z(0.75),
                0.2,
                [-0.3, 0.4, 0.7],
                _quat_from_z(0.5),
                0.8,
            ),
        ]
    )
    action = state[::-1].copy()

    outputs = {}
    for current_tag, current_mode in (
        ("robotwin", RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ),
        (tag, mode),
    ):
        dataset.embodiment_tag = current_tag
        dataset.relative_action_mode = current_mode
        out = {ROBOTWIN_EEF_STATE_KEY: state.copy()}
        anchor = dataset._prepare_dual_arm_eef_relative_state(out)
        relative_action = dataset._convert_action_abs_to_relative(
            action,
            ROBOTWIN_EEF_ACTION_KEY,
            anchor,
        )
        outputs[current_mode] = (out[ROBOTWIN_EEF_STATE_KEY], relative_action)

    robotwin_state, robotwin_action = outputs[
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ
    ]
    relative_state, relative_action = outputs[mode]
    np.testing.assert_allclose(relative_state, robotwin_state, atol=1e-6)
    np.testing.assert_allclose(relative_action, robotwin_action, atol=1e-6)
    assert is_dual_arm_eef_se3_relative_action(mode)
    assert is_dual_arm_eef_se3_relative_state(mode)
    assert not is_robotwin_eef_se3_s0_relative_action(mode)
    assert relative_action_mode_embodiment_tag(mode) == tag


@pytest.mark.parametrize(
    ("tag", "mode"),
    [
        ("noe_umi_16d", RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ),
        ("noe_robotics_16d", RELATIVE_ACTION_MODE_NOE_ROBOTICS_EEF_SE3_S0_WXYZ),
    ],
)
def test_noe_relative_model_config_is_accepted(tag, mode):
    model_cfg = OmegaConf.create(
        {
            "embodiment_tag": tag,
            "tokenizer_path": "dummy-tokenizer",
            "diffusion_model_pretrained_path": "dummy-diffusion",
            "image_encoder_pretrained_path": "dummy-image-encoder",
            "text_encoder_pretrained_path": "dummy-text-encoder",
            "vae_pretrained_path": "dummy-vae",
            "action_horizon": 48,
            "relative_action": True,
            "relative_action_mode": mode,
            "relative_action_keys": ["eef_position"],
            "action_head_cfg": {
                "config": {
                    "num_frames": 33,
                    "use_past_action_context": False,
                    "diffusion_model_cfg": {"max_chunk_size": 4},
                }
            },
        }
    )

    validated = validate_dreamzero_sft_model_cfg(model_cfg)
    assert validated.relative_action_mode == mode

    invalid_cfg = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
    invalid_cfg.relative_action_mode = RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ
    with pytest.raises(ValueError, match="requires embodiment_tag='robotwin'"):
        validate_dreamzero_sft_model_cfg(invalid_cfg)


def _valid_dreamzero_model_cfg(**overrides):
    values = {
        "model_type": "dreamzero",
        "model_path": None,
        "tokenizer_path": "dummy-tokenizer",
        "diffusion_model_pretrained_path": "dummy-diffusion",
        "image_encoder_pretrained_path": "dummy-image-encoder",
        "text_encoder_pretrained_path": "dummy-text-encoder",
        "vae_pretrained_path": "dummy-vae",
        "embodiment_tag": "robotwin",
        "embodiment_tag_mapping": {"robotwin": 33},
        "action_horizon": 48,
        "relative_action": False,
        "relative_action_mode": None,
        "relative_action_keys": [],
        "action_head_cfg": {
            "config": {
                "num_frames": 33,
                "use_past_action_context": False,
                "diffusion_model_cfg": {"max_chunk_size": 4},
            }
        },
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _write_checkpoint_config(checkpoint_dir, model_cfg, **overrides):
    checkpoint_dir.mkdir()
    values = OmegaConf.to_container(model_cfg, resolve=True)
    values.update(overrides)
    (checkpoint_dir / "config.json").write_text(
        json.dumps(values),
        encoding="utf-8",
    )


def test_checkpoint_config_mode_defaults_to_merge(tmp_path):
    base_cfg = _valid_dreamzero_model_cfg()
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint_config(
        checkpoint_dir,
        base_cfg,
        checkpoint_only_field="inherited",
    )
    model_cfg = _valid_dreamzero_model_cfg(model_path=str(checkpoint_dir))

    validated = validate_dreamzero_sft_model_cfg(model_cfg)

    assert validated.checkpoint_only_field == "inherited"


def test_robotwin_chunk_relative_model_config_is_accepted():
    model_cfg = _valid_dreamzero_model_cfg(
        relative_action=True,
        relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
        relative_action_keys=["eef_position"],
    )

    validated = validate_dreamzero_sft_model_cfg(model_cfg)

    assert (
        validated.relative_action_mode
        == RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ
    )


def test_checkpoint_config_mode_yaml_only_ignores_checkpoint_config(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    model_cfg = _valid_dreamzero_model_cfg(
        model_path=str(checkpoint_dir),
        checkpoint_config_mode="yaml_only",
    )

    validated = validate_dreamzero_sft_model_cfg(model_cfg)

    assert validated.embodiment_tag == "robotwin"
    assert OmegaConf.to_container(validated.embodiment_tag_mapping) == {
        "robotwin": 33
    }
    assert validated.model_path == str(checkpoint_dir)


def test_checkpoint_config_mode_rejects_unknown_value():
    model_cfg = _valid_dreamzero_model_cfg(
        checkpoint_config_mode="unsupported",
    )

    with pytest.raises(ValueError, match="checkpoint_config_mode"):
        validate_dreamzero_sft_model_cfg(model_cfg)


def test_yaml_only_rejects_incompatible_checkpoint_weight_keys():
    with pytest.raises(RuntimeError, match="2 missing keys and 1 unexpected keys"):
        _validate_checkpoint_weight_keys(
            "yaml_only",
            ["missing.weight", "missing.bias"],
            ["unexpected.weight"],
        )


def test_checkpoint_weight_key_check_preserves_legacy_merge_behavior():
    _validate_checkpoint_weight_keys(
        "merge",
        ["missing.weight"],
        ["unexpected.weight"],
    )
    _validate_checkpoint_weight_keys("yaml_only", [], [])


def test_robotwin_eef_se3_rejects_action_cache():
    for mode in (
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ,
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
        RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ,
    ):
        with pytest.raises(ValueError, match="incompatible with action cache"):
            ensure_relative_action_cache_compatible(
                relative_action=True,
                relative_action_mode=mode,
                use_past_action_context=True,
            )

    with pytest.raises(ValueError, match="incompatible with action cache"):
        ensure_relative_action_cache_compatible(
            relative_action=True,
            relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
            use_obs_action_history=True,
        )


def test_robotwin_eef_se3_rejects_missing_relative_metadata_mode():
    with pytest.raises(ValueError, match="requires metadata relative_action_mode"):
        ensure_relative_action_metadata_compatible(
            embodiment_tag="robotwin",
            relative_action=True,
            relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
        )

    with pytest.raises(ValueError, match="Unsupported metadata relative_action_mode"):
        ensure_relative_action_metadata_compatible(
            embodiment_tag="robotwin",
            relative_action=True,
            relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
            metadata_relative_action_mode="old_relative_mode",
        )

    with pytest.raises(ValueError, match="relative_action=False"):
        ensure_relative_action_metadata_compatible(
            embodiment_tag="robotwin",
            relative_action=False,
            relative_action_mode=None,
            metadata_relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
        )

    ensure_relative_action_metadata_compatible(
        embodiment_tag="robotwin",
        relative_action=True,
        relative_action_mode=(
            RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ
        ),
        metadata_relative_action_mode=(
            RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ
        ),
    )

    ensure_relative_action_metadata_compatible(
        embodiment_tag="robotwin",
        relative_action=True,
        relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
        metadata_relative_action_mode=(
            RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ
        ),
    )

    ensure_relative_action_metadata_compatible(
        embodiment_tag="noe_umi_16d",
        relative_action=True,
        relative_action_mode=RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ,
        metadata_relative_action_mode=(
            RELATIVE_ACTION_MODE_NOE_UMI_EEF_SE3_S0_WXYZ
        ),
    )

    with pytest.raises(ValueError, match="requires embodiment_tag"):
        ensure_relative_action_metadata_compatible(
            embodiment_tag="noe_umi_16d",
            relative_action=True,
            relative_action_mode=RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
            metadata_relative_action_mode=(
                RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ
            ),
        )

    with pytest.raises(ValueError, match="requires metadata relative_action_mode"):
        ensure_relative_action_metadata_compatible(
            embodiment_tag="robotwin",
            relative_action=True,
            relative_action_mode=(
                RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ
            ),
            metadata_relative_action_mode=(
                RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ
            ),
        )


def test_robotwin_eef_se3_rejects_non_16d_components():
    state = np.zeros((1, 15), dtype=np.float32)
    action = np.zeros((2, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="state anchor must be 16D"):
        robotwin_eef_abs_to_rel_wxyz(action, state)

    with pytest.raises(ValueError, match="action must be 16D"):
        robotwin_eef_rel_to_abs_wxyz(action[:, :15], np.zeros((1, 16), dtype=np.float32))


def test_robotwin_eef_se3_rejects_ndim_mismatch():
    action = np.zeros((2, 16), dtype=np.float32)
    state = np.zeros(16, dtype=np.float32)

    with pytest.raises(ValueError, match="ndim mismatch"):
        robotwin_eef_abs_to_rel_wxyz(action, state)


def test_robotwin_eef_se3_only_accepts_2d_or_3d_components():
    action = np.zeros((1, 2, 3, 16), dtype=np.float32)
    state = np.zeros((1, 1, 1, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="only support 2D or 3D"):
        robotwin_eef_abs_to_rel_wxyz(action, state)
