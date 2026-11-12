from types import SimpleNamespace

import numpy as np
import pytest

from groot.vla.data.relative_actions import (
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ,
    RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ,
    RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND,
    ROBOTWIN_EEF_ACTION_KEY,
    ROBOTWIN_EEF_STATE_KEY,
    ensure_relative_action_metadata_compatible,
    is_robotwin_eef_se3_s0_relative_action,
    is_robotwin_eef_se3_relative_action,
    is_robotwin_eef_se3_s0_relative_state,
    robotwin_eef_abs_to_rel_wxyz,
    robotwin_eef_rel_to_abs_wxyz,
)
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


def _state() -> np.ndarray:
    state = np.zeros((2, 16), dtype=np.float32)
    state[:, [3, 11]] = 1.0
    state[0, [0, 1, 2, 8, 9, 10]] = [1, 2, 3, -1, 0, 2]
    state[1, [0, 1, 2, 8, 9, 10]] = [2, 2, 4, 0, 1, 2]
    state[:, [7, 15]] = [[0.1, 0.9], [0.2, 0.8]]
    return state


def test_left_hand_pose_reference_round_trip():
    state = _state()[0:1]
    action = _state()[1:2]
    relative = robotwin_eef_abs_to_rel_wxyz(
        action,
        state,
        RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND,
    )

    np.testing.assert_allclose(relative[0, :3], [1, 0, 1])
    np.testing.assert_allclose(relative[0, 8:11], [-1, -1, -1])
    np.testing.assert_allclose(
        robotwin_eef_rel_to_abs_wxyz(
            relative,
            state,
            RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND,
        ),
        action,
        atol=1e-6,
    )


def _policy(mode: str) -> GrootSimPolicy:
    policy = object.__new__(GrootSimPolicy)
    policy.train_cfg = {"relative_action": True}
    policy.relative_action_mode = mode
    policy.relative_action_pose_reference = "per_arm"
    return policy


class _NormalizedAction:
    def cpu(self):
        return self


class _EvalTransform:
    def __init__(self, action):
        self.action = action

    def unapply(self, batch):
        assert "action" in batch
        return {ROBOTWIN_EEF_ACTION_KEY: self.action.copy()}


def test_action_only_mode_keeps_eval_state_absolute():
    state = _state()
    obs = {ROBOTWIN_EEF_STATE_KEY: state.copy()}
    prepared = _policy(
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ
    )._convert_obs_state_abs_to_s0_relative(
        obs,
        s0_anchor=state[0:1],
    )

    assert prepared is obs
    np.testing.assert_array_equal(prepared[ROBOTWIN_EEF_STATE_KEY], state)


def test_full_relative_mode_still_converts_eval_state():
    state = _state()
    obs = {ROBOTWIN_EEF_STATE_KEY: state.copy()}
    prepared = _policy(
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ
    )._convert_obs_state_abs_to_s0_relative(
        obs,
        s0_anchor=state[0:1],
    )

    assert prepared is not obs
    np.testing.assert_allclose(
        prepared[ROBOTWIN_EEF_STATE_KEY],
        robotwin_eef_abs_to_rel_wxyz(state, state[0:1]),
        atol=1e-6,
    )


def test_action_only_mode_capabilities_and_metadata_match():
    mode = RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ
    assert is_robotwin_eef_se3_s0_relative_action(mode)
    assert not is_robotwin_eef_se3_s0_relative_state(mode)

    ensure_relative_action_metadata_compatible(
        embodiment_tag="robotwin",
        relative_action=True,
        relative_action_mode=mode,
        metadata_relative_action_mode=mode,
    )
    with pytest.raises(ValueError, match="requires metadata relative_action_mode"):
        ensure_relative_action_metadata_compatible(
            embodiment_tag="robotwin",
            relative_action=True,
            relative_action_mode=mode,
            metadata_relative_action_mode=(
                RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_WXYZ
            ),
        )


def test_action_only_mode_restores_relative_action_with_window_s0():
    state = _state()
    s0_anchor = state[0:1]
    absolute_action = state[::-1].copy()
    relative_action = robotwin_eef_abs_to_rel_wxyz(absolute_action, s0_anchor)

    policy = _policy(RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ)
    policy.relative_action_keys = ["eef_position"]
    policy.eval_transform = _EvalTransform(relative_action)
    batch = SimpleNamespace(normalized_action=_NormalizedAction())

    restored = policy.unapply(
        batch,
        robotwin_eef_s0_anchor=s0_anchor,
    ).act[ROBOTWIN_EEF_ACTION_KEY]
    np.testing.assert_allclose(restored, absolute_action, atol=1e-6)


def test_action_only_mode_restores_left_hand_relative_action():
    state = _state()
    s0_anchor = state[0:1]
    absolute_action = state[::-1].copy()
    relative_action = robotwin_eef_abs_to_rel_wxyz(
        absolute_action,
        s0_anchor,
        RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND,
    )

    policy = _policy(RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_S0_ACTION_ONLY_WXYZ)
    policy.relative_action_pose_reference = RELATIVE_ACTION_POSE_REFERENCE_LEFT_HAND
    policy.relative_action_keys = ["eef_position"]
    policy.eval_transform = _EvalTransform(relative_action)
    batch = SimpleNamespace(normalized_action=_NormalizedAction())

    restored = policy.unapply(
        batch,
        robotwin_eef_s0_anchor=s0_anchor,
    ).act[ROBOTWIN_EEF_ACTION_KEY]
    np.testing.assert_allclose(restored, absolute_action, atol=1e-6)


def test_chunk_relative_mode_uses_current_state_for_state_and_action():
    state = _state()[:1]
    absolute_action = np.concatenate([_state()[:1], _state()[1:2]], axis=0)
    relative_action = robotwin_eef_abs_to_rel_wxyz(
        absolute_action,
        state,
    )

    policy = _policy(RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ)
    obs = {ROBOTWIN_EEF_STATE_KEY: state.copy()}
    prepared = policy._convert_obs_state_abs_to_s0_relative(
        obs,
        s0_anchor=state,
    )
    np.testing.assert_allclose(
        prepared[ROBOTWIN_EEF_STATE_KEY],
        robotwin_eef_abs_to_rel_wxyz(state, state),
        atol=1e-6,
    )

    policy.relative_action_keys = ["eef_position"]
    policy.eval_transform = _EvalTransform(relative_action)
    batch = SimpleNamespace(normalized_action=_NormalizedAction())
    restored = policy.unapply(batch, obs=obs).act[ROBOTWIN_EEF_ACTION_KEY]
    np.testing.assert_allclose(restored, absolute_action, atol=1e-6)
    assert is_robotwin_eef_se3_relative_action(
        RELATIVE_ACTION_MODE_ROBOTWIN_EEF_SE3_CHUNK_WXYZ
    )
