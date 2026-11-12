import pytest
import torch

from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CausalWanModel,
    CausalWanSelfAttention,
)


def _set_scaled_identity(linear: torch.nn.Linear, scale: float) -> None:
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.in_features) * scale)
        linear.bias.zero_()


def _make_attention(*, split: bool = True) -> CausalWanSelfAttention:
    return CausalWanSelfAttention(
        dim=4,
        num_heads=1,
        frame_seqlen=1,
        local_attn_size=3,
        num_frame_per_block=1,
        qk_norm=False,
        num_action_per_block=2,
        num_state_per_block=1,
        split_video_control_attention=split,
    )


def test_split_attention_routes_ff_tokens_through_video_projection() -> None:
    attention = _make_attention()
    _set_scaled_identity(attention.q, 1.0)
    _set_scaled_identity(attention.k, 2.0)
    _set_scaled_identity(attention.v, 3.0)
    _set_scaled_identity(attention.control_q, 4.0)
    _set_scaled_identity(attention.control_k, 5.0)
    _set_scaled_identity(attention.control_v, 6.0)
    _set_scaled_identity(attention.o, 7.0)
    _set_scaled_identity(attention.control_o, 8.0)

    x = torch.arange(24, dtype=torch.float32).view(1, 6, 4)
    control_span = attention._control_token_span(
        seq_len=x.shape[1],
        action_register_length=2,
        past_action_context_length=1,
        ff_context_length=1,
    )
    assert control_span == (2, 5)

    q, k, v = attention._project_qkv(x, control_span)
    expected_q = torch.cat([x[:, :2], x[:, 2:5] * 4, x[:, 5:]], dim=1)
    expected_k = torch.cat([x[:, :2] * 2, x[:, 2:5] * 5, x[:, 5:] * 2], dim=1)
    expected_v = torch.cat([x[:, :2] * 3, x[:, 2:5] * 6, x[:, 5:] * 3], dim=1)
    torch.testing.assert_close(q.flatten(2), expected_q)
    torch.testing.assert_close(k.flatten(2), expected_k)
    torch.testing.assert_close(v.flatten(2), expected_v)

    attention_output = x.view(1, 6, 1, 4)
    projected = attention._project_attention_output(attention_output, control_span)
    expected_output = torch.cat([x[:, :2] * 7, x[:, 2:5] * 8, x[:, 5:] * 7], dim=1)
    torch.testing.assert_close(projected, expected_output)


def test_control_attention_initialization_is_exact_and_single_use() -> None:
    attention = _make_attention()
    video_modules = (
        attention.q,
        attention.k,
        attention.v,
        attention.o,
    )
    control_modules = (
        attention.control_q,
        attention.control_k,
        attention.control_v,
        attention.control_o,
    )
    for index, module in enumerate(video_modules, start=1):
        _set_scaled_identity(module, float(index))

    attention.initialize_control_attention_from_video()

    for video_module, control_module in zip(video_modules, control_modules, strict=True):
        for key, value in video_module.state_dict().items():
            torch.testing.assert_close(control_module.state_dict()[key], value)
    with pytest.raises(RuntimeError, match="already been copied from video attention"):
        attention.initialize_control_attention_from_video()


def test_split_projections_preserve_all_gradients() -> None:
    attention = _make_attention()
    x = torch.randn(2, 6, 4, requires_grad=True)
    control_span = (2, 5)

    q, k, v = attention._project_qkv(x, control_span)
    projected = attention._project_attention_output(
        torch.randn(2, 6, 1, 4, requires_grad=True), control_span
    )
    loss = q.square().mean() + k.square().mean() + v.square().mean()
    loss = loss + projected.square().mean()
    loss.backward()

    for name in (
        "q",
        "k",
        "v",
        "o",
        "control_q",
        "control_k",
        "control_v",
        "control_o",
    ):
        grad = getattr(attention, name).weight.grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


def test_split_projections_preserve_autocast_dtype() -> None:
    attention = _make_attention()
    x = torch.randn(2, 6, 4)
    attention_output = torch.randn(2, 6, 1, 4)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        q, k, v = attention._project_qkv(x, (2, 5))
        projected = attention._project_attention_output(
            attention_output, (2, 5)
        )

    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert projected.dtype == torch.bfloat16


def test_split_attention_routes_video_only_tokens_through_video_projection() -> None:
    attention = _make_attention()
    _set_scaled_identity(attention.q, 1.0)
    _set_scaled_identity(attention.k, 2.0)
    _set_scaled_identity(attention.v, 3.0)
    _set_scaled_identity(attention.o, 4.0)

    x = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
    control_span = attention._control_token_span(
        seq_len=x.shape[1],
        action_register_length=None,
        past_action_context_length=0,
        ff_context_length=0,
    )
    assert control_span is None

    q, k, v = attention._project_qkv(x, control_span)
    torch.testing.assert_close(q.flatten(2), x)
    torch.testing.assert_close(k.flatten(2), x * 2)
    torch.testing.assert_close(v.flatten(2), x * 3)

    projected = attention._project_attention_output(
        x.view(1, 4, 1, 4), control_span
    )
    torch.testing.assert_close(projected, x * 4)


def test_split_attention_rejects_invalid_control_layout() -> None:
    attention = _make_attention()

    for action_register_length in (0, -1):
        with pytest.raises(
            ValueError,
            match="action_register_length must be positive",
        ):
            attention._control_token_span(
                seq_len=4,
                action_register_length=action_register_length,
                past_action_context_length=0,
                ff_context_length=0,
            )

    with pytest.raises(
        ValueError,
        match="Past action context requires current action/state registers",
    ):
        attention._control_token_span(
            seq_len=4,
            action_register_length=None,
            past_action_context_length=1,
            ff_context_length=0,
        )

    with pytest.raises(ValueError, match="at least one leading video token"):
        attention._control_token_span(
            seq_len=4,
            action_register_length=3,
            past_action_context_length=1,
            ff_context_length=0,
        )


def test_model_initializes_every_control_attention_block() -> None:
    model = CausalWanModel(
        model_type="t2v",
        patch_size=(1, 1, 1),
        frame_seqlen=1,
        text_len=4,
        in_dim=2,
        dim=4,
        ffn_dim=8,
        freq_dim=4,
        text_dim=4,
        out_dim=2,
        num_heads=1,
        num_layers=2,
        max_chunk_size=1,
        num_frame_per_block=1,
        action_dim=2,
        max_state_dim=2,
        hidden_size=4,
        num_action_per_block=2,
        num_state_per_block=1,
        action_architecture="split",
    )

    model.initialize_control_attention_from_video()

    assert model._control_attention_copied_from_video
    assert model.blocks[0].self_attn._emit_split_routing_log
    assert not model.blocks[1].self_attn._emit_split_routing_log
    assert all(
        block.self_attn._control_attention_copied_from_video for block in model.blocks
    )
    with pytest.raises(RuntimeError, match="already been copied from video attention"):
        model.initialize_control_attention_from_video()
