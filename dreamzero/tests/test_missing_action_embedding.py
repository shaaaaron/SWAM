import torch

from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    MultiEmbodimentActionEncoder,
)


def _copy_normal_weights(source, target):
    target.W1.load_state_dict(source.W1.state_dict())
    target.W2.load_state_dict(source.W2.state_dict())
    target.W3.load_state_dict(source.W3.state_dict())


def test_disabled_path_matches_legacy_and_keeps_missing_param_in_graph():
    torch.manual_seed(0)
    actions = torch.randn(2, 4, 3)
    timesteps = torch.rand(2, 4)
    cat_ids = torch.zeros(2, dtype=torch.long)

    reference = MultiEmbodimentActionEncoder(3, 6, 1)
    disabled = MultiEmbodimentActionEncoder(
        3,
        6,
        1,
        enable_missing_action_embedding=False,
    )
    _copy_normal_weights(reference, disabled)

    expected = reference(actions, timesteps, cat_ids)
    actual = disabled(actions, timesteps, cat_ids)
    assert torch.equal(expected, actual)

    actual.square().mean().backward()
    assert disabled.W_missing.grad is not None
    assert torch.count_nonzero(disabled.W_missing.grad) == 0


def test_invalid_channel_uses_missing_projection_not_raw_value():
    torch.manual_seed(1)
    actions = torch.randn(2, 4, 3)
    timesteps = torch.rand(2, 4)
    cat_ids = torch.zeros(2, dtype=torch.long)
    encoder = MultiEmbodimentActionEncoder(
        3,
        6,
        1,
        enable_missing_action_embedding=True,
    )
    encoder.W_missing.data.normal_(0.0, 0.02)

    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[:, :, 1] = False
    output = encoder(actions, timesteps, cat_ids, action_feature_mask=mask)

    changed = actions.clone()
    changed[:, :, 1] += 1000.0
    changed_output = encoder(
        changed,
        timesteps,
        cat_ids,
        action_feature_mask=mask,
    )
    assert torch.allclose(output, changed_output)


def test_enabled_path_rejects_misaligned_feature_mask():
    encoder = MultiEmbodimentActionEncoder(
        3,
        6,
        1,
        enable_missing_action_embedding=True,
    )
    actions = torch.zeros(2, 4, 3)
    timesteps = torch.zeros(2, 4)
    cat_ids = torch.zeros(2, dtype=torch.long)

    try:
        encoder(
            actions,
            timesteps,
            cat_ids,
            action_feature_mask=torch.ones(2, 4, 2, dtype=torch.bool),
        )
    except ValueError as exc:
        assert "action_feature_mask" in str(exc)
    else:
        raise AssertionError("misaligned action_feature_mask must fail loudly")


def test_missing_projection_is_shared_across_embodiments():
    encoder = MultiEmbodimentActionEncoder(
        3,
        6,
        2,
        enable_missing_action_embedding=True,
    )
    encoder.W1.W.data.zero_()
    encoder.W1.b.data.zero_()
    encoder.W_missing.data.normal_(0.0, 0.02)

    actions = torch.randn(2, 4, 3)
    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[:, :, 1] = False
    cat_ids = torch.tensor([0, 1], dtype=torch.long)

    first_layer = encoder._embed_first_layer(
        actions,
        cat_ids,
        action_feature_mask=mask,
    )
    assert torch.allclose(first_layer[0], first_layer[1])
