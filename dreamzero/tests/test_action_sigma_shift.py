import math

import pytest

from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
    WANPolicyHeadConfig,
    resolve_action_sigma_shifts,
)
from groot.vla.model.dreamzero.modules.flow_match_scheduler import FlowMatchScheduler
from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import (
    FlowUniPCMultistepScheduler,
)


@pytest.mark.parametrize(
    ("train", "inference", "expected"),
    [
        (None, None, (5.0, 5.0)),
        (1.0, None, (1.0, 1.0)),
        (5.0, 1.0, (5.0, 1.0)),
    ],
)
def test_resolve_action_sigma_shifts(train, inference, expected):
    assert resolve_action_sigma_shifts(train, inference) == expected


def test_policy_head_config_keeps_action_shift_overrides_nullable():
    config = WANPolicyHeadConfig()

    assert config.action_train_sigma_shift is None
    assert config.action_inference_sigma_shift is None


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_resolve_action_sigma_shifts_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite positive"):
        resolve_action_sigma_shifts(value, None)

    with pytest.raises(ValueError, match="finite positive"):
        resolve_action_sigma_shifts(None, value)


def test_action_shift_changes_sigma_mapping_and_preserves_legacy_default():
    video_scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    legacy_action_scheduler = FlowMatchScheduler(
        shift=resolve_action_sigma_shifts(None, None)[0],
        sigma_min=0.0,
        extra_one_step=True,
    )
    unshifted_action_scheduler = FlowMatchScheduler(
        shift=1.0,
        sigma_min=0.0,
        extra_one_step=True,
    )

    for scheduler in (
        video_scheduler,
        legacy_action_scheduler,
        unshifted_action_scheduler,
    ):
        scheduler.set_timesteps(1000, training=True)

    assert legacy_action_scheduler.sigmas.equal(video_scheduler.sigmas)
    assert legacy_action_scheduler.timesteps.equal(video_scheduler.timesteps)
    assert video_scheduler.sigmas[500] > unshifted_action_scheduler.sigmas[500]


def test_action_inference_shift_changes_unipc_schedule():
    legacy_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    unshifted_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )

    legacy_scheduler.set_timesteps(16, device="cpu", shift=5.0)
    unshifted_scheduler.set_timesteps(16, device="cpu", shift=1.0)

    assert legacy_scheduler.sigmas[8] > unshifted_scheduler.sigmas[8]
    assert legacy_scheduler.timesteps[8] > unshifted_scheduler.timesteps[8]
