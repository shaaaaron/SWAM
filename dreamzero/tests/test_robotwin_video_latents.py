from types import SimpleNamespace

import torch

from robotwin_eval.dreamzero_robotwin_server import RoboTwinPolicy


def _policy(*, is_first_call: bool) -> RoboTwinPolicy:
    policy = object.__new__(RoboTwinPolicy)
    policy._is_first_call = is_first_call
    policy._policy = SimpleNamespace(
        trained_model=SimpleNamespace(
            action_head=SimpleNamespace(num_frame_per_block=2),
        ),
    )
    return policy


def _latents(num_latents: int) -> torch.Tensor:
    return torch.arange(num_latents).reshape(1, 1, num_latents, 1, 1)


def test_initial_reference_latent_is_preserved():
    video_pred = _latents(3)

    result = _policy(is_first_call=True)._video_latents_for_episode(video_pred)

    assert result is video_pred


def test_normal_prediction_block_is_preserved():
    video_pred = _latents(2)

    result = _policy(is_first_call=False)._video_latents_for_episode(video_pred)

    assert result is video_pred


def test_mid_episode_cache_reset_reference_latent_is_removed():
    video_pred = _latents(3)

    result = _policy(is_first_call=False)._video_latents_for_episode(video_pred)

    torch.testing.assert_close(result, video_pred[:, :, 1:])
    assert result.shape[2] == 2
    assert result.is_contiguous()


def test_unexpected_latent_count_is_preserved(caplog):
    video_pred = _latents(4)

    result = _policy(is_first_call=False)._video_latents_for_episode(video_pred)

    assert result is video_pred
    assert "keeping all latents" in caplog.text
