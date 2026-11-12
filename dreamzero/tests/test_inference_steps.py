from types import SimpleNamespace

import pytest

from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import WANPolicyHead


@pytest.mark.parametrize('dynamic', [False, True])
def test_50_steps_recompute_every_prediction_after_runtime_override(dynamic):
    head = SimpleNamespace(
        num_inference_steps=16,
        dynamic_cache_schedule=dynamic,
        dit_step_mask=[True] + [False] * 15,
    )
    # The video-only CLI overrides this attribute after the model is loaded.
    head.num_inference_steps = 50
    assert all(WANPolicyHead.should_run_model(head, i, None, []) for i in range(50))


def test_16_steps_keep_the_existing_static_skip_schedule():
    mask = [True, True, True, False, False, False, True, False,
            False, False, True, False, False, True, True, True]
    head = SimpleNamespace(num_inference_steps=16, dynamic_cache_schedule=False, dit_step_mask=mask)
    assert [WANPolicyHead.should_run_model(head, i, None, []) for i in range(16)] == mask


def test_16_steps_keep_dynamic_skip_countdown():
    head = SimpleNamespace(num_inference_steps=16, dynamic_cache_schedule=True, skip_countdown=2)
    assert WANPolicyHead.should_run_model(head, 0, None, []) is True
    assert WANPolicyHead.should_run_model(head, 4, None, [None, None]) is False
    assert head.skip_countdown == 1
    assert WANPolicyHead.should_run_model(head, 5, None, [None, None]) is True
    assert head.skip_countdown == 0
