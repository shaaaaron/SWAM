from typing import Any, TypeAlias

from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule
from groot.vla.model.n1_5.modules.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
    WanRMSNorm,
    rope_action_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from torch.nn.attention.flex_attention import BlockMask
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.distributed as dist
import os
import imageio
import numpy as np

ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(
        self,
        action_dim,
        hidden_size,
        num_embodiments,
        enable_missing_action_embedding=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        self.action_dim = action_dim
        self.enable_missing_action_embedding = bool(
            enable_missing_action_embedding
        )

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W_missing = nn.Parameter(torch.zeros(action_dim, hidden_size))
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def _embed_first_layer(self, actions, cat_ids, action_feature_mask=None):
        """
        进行missing channel的是用
        """
        if not self.enable_missing_action_embedding:
            normal = self.W1(actions, cat_ids)
            # 仍然经过w missing，将模块保持在计算图中
            return normal + (0.0 * self.W_missing.float().sum()).to(
                dtype=normal.dtype,
                device=normal.device,
            )

        if action_feature_mask is None:
            validity = torch.ones_like(actions)
            mask_path = "implicit_all_valid"
            mask_dtype = None
        else:
            if action_feature_mask.dtype != torch.bool:
                raise TypeError(
                    f"Action feature mask dtype must be torch.bool, "
                    f"got {action_feature_mask.dtype}."
                )
            validity = torch.as_tensor(
                action_feature_mask,
                dtype=actions.dtype,
                device=actions.device,
            )
            if validity.shape != actions.shape:
                raise ValueError(
                    "action_feature_mask must match actions shape [B, T, D]; "
                    f"got mask={tuple(validity.shape)}, actions={tuple(actions.shape)}"
                )
            mask_path = "explicit_mask"
            mask_dtype = action_feature_mask.dtype

        if torch.rand((), device=actions.device).item() < 0.001:
            rank = dist.get_rank() if dist.is_initialized() else 0
            valid_ratio = validity.float().mean().item()
            print(
                "[MISSING_ACTION_EMBEDDING] "
                f"rank={rank} path={mask_path} shape={tuple(validity.shape)} "
                f"mask_dtype={mask_dtype} compute_dtype={validity.dtype} "
                f"valid_ratio={valid_ratio:.6f}",
                flush=True,
            )

        normal = self.W1(actions * validity, cat_ids)  # [B,T,D] .* [B,T,D] -> [B,T,H]
        missing = (1.0 - validity) @ self.W_missing.to(dtype=normal.dtype)  # [B,T,D] @ [D,H] -> [B,T,H]
        return normal + missing

    def forward(self, actions, timesteps, cat_ids, action_feature_mask=None):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # Standard action MLP step for shape => (B, T, w)
        a_emb = self._embed_first_layer(
            actions,
            cat_ids,
            action_feature_mask=action_feature_mask,
        )

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


def causal_rope_action_apply(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index):
    if ENABLE_TENSORRT:
        return causal_rope_action_apply_no_polar(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index)
    else:
        return causal_rope_action_apply_polar(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index)


def build_image_context_query_mask(
    *,  # 强制关键字传参
    seq_len: int,   # 总长度 [video][action][state][extra]
    video_len: int,
    action_len: int,
    state_len: int,
    token_scope: str,
    device: torch.device,
    extra_len: int = 0,
) -> torch.Tensor | None:
    """输入各段 token 长度，输出 image-context cross-attn query mask；extra_len 覆盖 past_action/ff 等尾部 context。"""
    if token_scope == "all":
        return None

    expected_len = video_len + action_len + state_len + extra_len
    assert expected_len == seq_len, (
        "image_context_token_scope assumes query layout [video][action][state], "
        f"but got seq_len={seq_len}, video_len={video_len}, "
        f"action_len={action_len}, state_len={state_len}, extra_len={extra_len}."
    )

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    video_start = 0
    video_end = video_len
    action_start = video_end
    action_end = action_start + action_len
    state_start = action_end
    state_end = state_start + state_len
    selected_starts: list[int] = []

    if token_scope == "video":
        mask[video_start:video_end] = True
        selected_starts.append(video_start)
    elif token_scope == "action":
        mask[action_start:action_end] = True
        selected_starts.append(action_start)
    elif token_scope == "non_video":
        mask[action_start:action_end] = True
        mask[state_start:state_end] = True
        selected_starts.extend([action_start, state_start])
    elif token_scope == "non_action":
        mask[video_start:video_end] = True
        mask[state_start:state_end] = True
        selected_starts.extend([video_start, state_start])
    else:
        raise ValueError(f"Unsupported image_context_token_scope: {token_scope!r}")

    if torch.rand((), device=device).item() < 0.001:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(
            f"[IMAGE_CONTEXT_TOKEN_SCOPE] rank={rank} scope={token_scope} "
            f"starts={selected_starts} "
            f"seq_len={seq_len}",
            flush=True,
        )
    return mask


def rope_action_only_apply(
    x: torch.Tensor,
    freqs_action: torch.Tensor,
    start_index: int = 0,
):
    if ENABLE_TENSORRT:
        return rope_action_only_apply_no_polar(x, freqs_action, start_index)
    return rope_action_only_apply_polar(x, freqs_action, start_index)


def rope_action_only_apply_no_polar(
    x: torch.Tensor,
    freqs_action: torch.Tensor,
    start_index: int = 0,
):
    B, seq_len, n, d = x.shape
    freqs_action = freqs_action[start_index:start_index + seq_len]
    x = x.reshape(B, seq_len, n, -1, 2)
    x_real = x[..., 0]
    x_imag = x[..., 1]
    freqs_action = freqs_action.view(seq_len, 1, -1, 2)
    freqs_cos = freqs_action[..., 0].unsqueeze(0)
    freqs_sin = freqs_action[..., 1].unsqueeze(0)
    x_real_rotated = x_real * freqs_cos - x_imag * freqs_sin
    x_imag_rotated = x_real * freqs_sin + x_imag * freqs_cos
    return torch.stack((x_real_rotated, x_imag_rotated), dim=-1).flatten(3)


def rope_action_only_apply_polar(
    x: torch.Tensor,
    freqs_action: torch.Tensor,
    start_index: int = 0,
):
    B, seq_len, n, _ = x.shape
    freqs_action = freqs_action[start_index:start_index + seq_len]
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )
    freqs_action = freqs_action.view(seq_len, 1, -1).unsqueeze(0)
    return torch.view_as_real(x * freqs_action).flatten(3)


def causal_rope_action_apply_no_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    B, seq_len, n, d = x.shape

    # (B, seq_len, n, d) -> (B, seq_len, n, d/2, 2)
    x = x.reshape(B, seq_len, n, -1, 2)
    x_real = x[..., 0]
    x_imag = x[..., 1]

    # Split freqs into cos and sin components
    freqs = freqs.unsqueeze(0).view(1, freqs.shape[0], 1, -1, 2)
    freqs_cos = freqs[..., 0] # Shape: (1, seq_len', 1, d/2)
    freqs_sin = freqs[..., 1] # Shape: (1, seq_len', 1, d/2)

    #  Handle the Action/State Register Frequencies
    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)

        freqs_action_slice = freqs_action[
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block
        ]
        freqs_state_slice = freqs_state[
            action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block
        ]

        # Combine the action/state tokens for this frame
        freqs_1d = torch.cat([freqs_action_slice, freqs_state_slice], dim=0).view(
            action_register_length, 1, -1, 2
        )

        # Split the new action/state frequencies
        freqs_cos_1d = freqs_1d[..., 0]
        freqs_sin_1d = freqs_1d[..., 1]

        # Append the action/state register sin/cos to the main sequence sin/cos
        freqs_cos = torch.cat([freqs_cos[0], freqs_cos_1d], dim=0).unsqueeze(0)
        freqs_sin = torch.cat([freqs_sin[0], freqs_sin_1d], dim=0).unsqueeze(0)

    x_real_rotated = x_real * freqs_cos - x_imag * freqs_sin
    x_imag_rotated = x_real * freqs_sin + x_imag * freqs_cos

    x_rotated = torch.stack((x_real_rotated, x_imag_rotated), dim=-1)

    return x_rotated.flatten(3)

def causal_rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        freqs_action = freqs_action[
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block
        ]
        freqs_state = freqs_state[
            action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action, freqs_state], dim=0).view(action_register_length, 1, -1)
        freqs = torch.cat([freqs, freqs_1d], dim=0)

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)

    return x


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=1,
                 video_condition_on_action_state=True,
                 split_video_control_attention=False):
        assert dim % num_heads == 0
        super().__init__()
        if not isinstance(split_video_control_attention, bool):
            raise TypeError(
                "split_video_control_attention must be a bool, "
                f"got {type(split_video_control_attention).__name__}."
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 21 * frame_seqlen if local_attn_size == -1 else local_attn_size * frame_seqlen
        # print(f"CausalWanSelfAttention: max_attention_size={self.max_attention_size}")
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.video_condition_on_action_state = video_condition_on_action_state
        self.split_video_control_attention = split_video_control_attention
        self._control_attention_copied_from_video = False
        self._emit_split_routing_log = False    # 控制第一个 block 打印
        self._split_routing_logged = False      # 打印 video/control token 路由
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        if self.split_video_control_attention:
            self.control_q = nn.Linear(dim, dim)
            self.control_k = nn.Linear(dim, dim)
            self.control_v = nn.Linear(dim, dim)
            self.control_o = nn.Linear(dim, dim)
            self.control_norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
            self.control_norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim)
        self.causal_attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim, causal=True)

    @torch.no_grad()
    def initialize_control_attention_from_video(self) -> None:
        if not self.split_video_control_attention:
            raise RuntimeError(
                "Cannot initialize control attention when "
                "split_video_control_attention=False."
            )
        if self._control_attention_copied_from_video:
            raise RuntimeError("Control attention has already been copied from video attention.")

        module_pairs = (
            (self.control_q, self.q),
            (self.control_k, self.k),
            (self.control_v, self.v),
            (self.control_o, self.o),
            (self.control_norm_q, self.norm_q),
            (self.control_norm_k, self.norm_k),
        )
        for control_module, video_module in module_pairs:
            control_module.load_state_dict(video_module.state_dict(), strict=True)
        self._control_attention_copied_from_video = True

    def _control_token_span(
        self,
        *,
        seq_len: int,
        action_register_length: int | None,
        past_action_context_length: int,
        ff_context_length: int,
    ) -> tuple[int, int] | None:
        if past_action_context_length < 0:
            raise ValueError(
                "past_action_context_length must be non-negative, "
                f"got {past_action_context_length}."
            )
        if ff_context_length < 0:
            raise ValueError(
                f"ff_context_length must be non-negative, got {ff_context_length}."
            )
        if action_register_length is None:
            if past_action_context_length != 0:
                raise ValueError(
                    "Past action context requires current action/state registers."
                )
            return None
        if action_register_length <= 0:
            raise ValueError(
                "action_register_length must be positive, "
                f"got {action_register_length}."
            )

        control_length = action_register_length + past_action_context_length
        control_end = seq_len - ff_context_length
        control_start = control_end - control_length
        if control_start <= 0:
            raise ValueError(
                "Split attention requires at least one leading video token; "
                f"got seq_len={seq_len}, action_register_length={action_register_length}, "
                f"past_action_context_length={past_action_context_length}, "
                f"ff_context_length={ff_context_length}."
            )
        return control_start, control_end

    def _project_qkv(
        self,
        x: torch.Tensor,
        control_span: tuple[int, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        按 token 类型分别计算 Q/K/V。
        - 当 control_span 为 None 时，所有 token 使用原始 video Q/K/V；
        - 否则，control_span 指定的 action/state 及历史动作 token 使用 control Q/K/V，
        """

        b, s = x.shape[:2]
        n, d = self.num_heads, self.head_dim

        if control_span is None:
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        control_start, control_end = control_span
        if not (0 < control_start <= control_end <= s):
            raise ValueError(
                f"Invalid control token span {control_span} for sequence length {s}."
            )

        video_x = torch.cat([x[:, :control_start], x[:, control_end:]], dim=1)
        control_x = x[:, control_start:control_end]

        video_q = self.norm_q(self.q(video_x)).view(b, video_x.shape[1], n, d)
        video_k = self.norm_k(self.k(video_x)).view(b, video_x.shape[1], n, d)
        video_v = self.v(video_x).view(b, video_x.shape[1], n, d)
        if control_x.shape[1] == 0:
            return video_q, video_k, video_v

        control_q = self.control_norm_q(self.control_q(control_x)).view(
            b, control_x.shape[1], n, d
        )
        control_k = self.control_norm_k(self.control_k(control_x)).view(
            b, control_x.shape[1], n, d
        )
        control_v = self.control_v(control_x).view(b, control_x.shape[1], n, d)

        video_prefix_length = control_start
        q = torch.cat(
            [video_q[:, :video_prefix_length], control_q, video_q[:, video_prefix_length:]],
            dim=1,
        )
        k = torch.cat(
            [video_k[:, :video_prefix_length], control_k, video_k[:, video_prefix_length:]],
            dim=1,
        )
        v = torch.cat(
            [video_v[:, :video_prefix_length], control_v, video_v[:, video_prefix_length:]],
            dim=1,
        )
        return q, k, v

    def _project_attention_output(
        self,
        x: torch.Tensor,
        control_span: tuple[int, int] | None,
    ) -> torch.Tensor:
        """
        按 token 类型分别执行 attention 输出投影。
        """
        x = x.flatten(2)
        if control_span is None:
            return self.o(x)

        control_start, control_end = control_span
        video_prefix = self.o(x[:, :control_start])
        control = self.control_o(x[:, control_start:control_end])
        video_suffix = self.o(x[:, control_end:])
        return torch.cat([video_prefix, control, video_suffix], dim=1)

    def _visualize_attention_mask(self, total_len, first_image_len, image_blocks_len,
                                   action_len, state_len, num_image_blocks,
                                   num_action_blocks, num_state_blocks,
                                   num_frame_per_block, frame_seqlen,
                                   num_action_per_block, num_state_per_block):
        """
        Create and print a visualization of the attention mask pattern.
        Returns a binary mask [total_len, total_len] where 1 = can attend, 0 = cannot attend.

        Legacy utility for the old monolithic blockwise mask. The active
        labeled training path now uses the split V0/V1/A0/A1 helpers below.
        """
        # Token ranges
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len

        # Create mask tensor
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)

        # First image: self-attention only
        mask[first_image_start:first_image_end, first_image_start:first_image_end] = True

        # Image blocks
        for block_idx in range(num_image_blocks):
            block_start = image_blocks_start + block_idx * num_frame_per_block * frame_seqlen
            block_end = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen

            # Attend to first image
            mask[block_start:block_end, first_image_start:first_image_end] = True

            # Attend to previous and current image blocks
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            mask[block_start:block_end, image_kv_start:block_end] = True

            # Attend to current action block
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_start + (block_idx + 1) * num_action_per_block
            mask[block_start:block_end, action_block_start:action_block_end] = True

            # Attend to current state block
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[block_start:block_end, state_block_start:state_block_end] = True

        # Action blocks
        for block_idx in range(num_action_blocks):
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_start + (block_idx + 1) * num_action_per_block

            # Attend to first image
            mask[action_block_start:action_block_end, first_image_start:first_image_end] = True

            # Attend to previous and current image blocks
            image_block_end = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, image_block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            mask[action_block_start:action_block_end, image_kv_start:image_block_end] = True

            # Self-attention
            mask[action_block_start:action_block_end, action_block_start:action_block_end] = True

            # Attend to current state block
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[action_block_start:action_block_end, state_block_start:state_block_end] = True

        # State blocks: self-attention only
        for block_idx in range(num_state_blocks):
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[state_block_start:state_block_end, state_block_start:state_block_end] = True

        return mask

    def _blockwise_causal_flash_attn(self, q, k, v, frame_seqlen, num_frame_per_block=1,
                                       action_horizon=None, state_horizon=None,
                                       num_action_per_block=None, num_state_per_block=None,
                                       visualize_mask=False):
        """
        Implement blockwise causal attention using flash_attention.
        Matches the pattern from _prepare_blockwise_causal_attn_mask:

        Legacy path for non-teacher-forcing forward without KV cache. Current
        labeled training uses the split teacher-forcing helpers, and inference
        uses the KV-cache branch.

        Structure:
        - First image: conditioning only, cannot attend to anything
        - Image blocks: can attend to first image + previous image blocks + current action block + current state block
        - Action blocks: can attend to previous image blocks + current image block + current state block + first image
        - State blocks: conditioning only, cannot attend to anything

        Args:
            q, k, v: Query, key, value tensors [B, L, num_heads, head_dim]
            frame_seqlen: Number of tokens per frame
            num_frame_per_block: Number of frames per attention block
            action_horizon: Total number of action tokens (if None, no action/state tokens)
            state_horizon: Total number of state tokens (if None, no action/state tokens)
            num_action_per_block: Number of action tokens per block
            num_state_per_block: Number of state tokens per block
            visualize_mask: If True, print the attention mask pattern

        Returns:
            Attention output [B, L, num_heads, head_dim]
        """
        b, total_len, n, d = q.shape

        # Check if we have action/state tokens
        has_action_state = (action_horizon is not None and state_horizon is not None)

        if not has_action_state:
            # OPTIMIZED: Simple blockwise causal attention (without action/state tokens)
            num_frames = total_len // frame_seqlen
            block_size = frame_seqlen * num_frame_per_block
            assert (num_frames - 1) % num_frame_per_block == 0
            num_blocks = (num_frames - 1) // num_frame_per_block

            # Handle edge case when sequence is too short (no blocks to process)
            if num_blocks <= 0:
                # Process entire sequence as a single block
                return self.attn(q, k, v)

            # OPTIMIZATION: For global attention, process all blocks in one call with causal masking
            if self.local_attn_size == -1:
                # Single flash_attention call with causal=True for all blocks at once
                # This is much faster than looping!
                return self.causal_attn(q, k, v)

            # With local attention, still need loop but optimize it
            # Pre-allocate output tensor
            output = torch.empty_like(q)

            # First frame: conditioning, self-attention only.
            # Mirrors _process_clean_image_only() at L580-584. Without this,
            # output[:, :frame_seqlen] stays as torch.empty_like garbage and
            # downstream blocks see NaN/Inf — which cascades into the whole
            # forward and silently corrupts training (see
            # docs/cotrain_video_only_nan_bug.md).
            output[:, :frame_seqlen] = self.attn(
                q[:, :frame_seqlen],
                k[:, :frame_seqlen],
                v[:, :frame_seqlen],
            )

            # Pre-compute block boundaries
            block_starts = [frame_seqlen + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]
            kv_starts = [max(0, end - self.local_attn_size * frame_seqlen) for end in block_ends]

            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]
                kv_start = kv_starts[block_idx]

                output[:, block_start:block_end] = self.attn(
                    q[:, block_start:block_end],
                    k[:, kv_start:block_end],
                    v[:, kv_start:block_end]
                )

            return output

        assert action_horizon is not None and state_horizon is not None
        assert num_action_per_block is not None and num_state_per_block is not None

        # Sequence layout: [first_image] [image_blocks] [action_blocks] [state_blocks]
        first_image_len = frame_seqlen
        action_len = action_horizon
        state_len = state_horizon
        image_blocks_len = total_len - first_image_len - action_len - state_len

        num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seqlen)
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block
        assert num_image_blocks == num_action_blocks == num_state_blocks

        # ----- Token ranges -----
        first_image_start = 0
        first_image_end   = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end   = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end   = action_start + action_len
        state_start  = action_end
        state_end    = state_start + state_len

        # Visualize attention mask if requested
        if visualize_mask:
            print("\n" + "="*80)
            print("ATTENTION MASK VISUALIZATION")
            print(f"Total length: {total_len}")
            print(f"First image:  [{first_image_start}:{first_image_end}]")
            print(f"Image blocks: [{image_blocks_start}:{image_blocks_end}] num_blocks={num_image_blocks}")
            print(f"Action:       [{action_start}:{action_end}] num_blocks={num_action_blocks}")
            print(f"State:        [{state_start}:{state_end}] num_blocks={num_state_blocks}")
            print("="*80 + "\n")

        output = torch.empty_like(q)

        # First image: self-attention only
        output[:, first_image_start:first_image_end] = self.attn(
            q[:, first_image_start:first_image_end],
            k[:, first_image_start:first_image_end],
            v[:, first_image_start:first_image_end]
        )

        # Pre-compute block index arrays
        image_block_starts  = [image_blocks_start + i * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
        image_block_ends    = [image_blocks_start + (i + 1) * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
        if self.local_attn_size != -1:
            image_kv_starts = [max(image_blocks_start, e - self.local_attn_size * frame_seqlen) for e in image_block_ends]
        else:
            image_kv_starts = [image_blocks_start] * num_image_blocks

        action_block_starts = [action_start + i * num_action_per_block for i in range(num_action_blocks)]
        action_block_ends   = [action_start + (i + 1) * num_action_per_block for i in range(num_action_blocks)]
        state_block_starts  = [state_start + i * num_state_per_block for i in range(num_state_blocks)]
        state_block_ends    = [state_start + (i + 1) * num_state_per_block for i in range(num_state_blocks)]

        # ----- Image blocks -----
        for block_idx in range(num_image_blocks):
            image_block_start = image_block_starts[block_idx]
            image_block_end   = image_block_ends[block_idx]
            image_kv_start    = image_kv_starts[block_idx]
            action_block_start, action_block_end = action_block_starts[block_idx], action_block_ends[block_idx]
            state_block_start, state_block_end   = state_block_starts[block_idx], state_block_ends[block_idx]
            parts_k = [k[:, first_image_start:first_image_end], k[:, image_kv_start:image_block_end],
                       k[:, action_block_start:action_block_end], k[:, state_block_start:state_block_end]]
            parts_v = [v[:, first_image_start:first_image_end], v[:, image_kv_start:image_block_end],
                       v[:, action_block_start:action_block_end], v[:, state_block_start:state_block_end]]
            output[:, image_block_start:image_block_end] = self.attn(
                q[:, image_block_start:image_block_end], torch.cat(parts_k, dim=1), torch.cat(parts_v, dim=1)
            )

        # ----- Action blocks -----
        for block_idx in range(num_action_blocks):
            action_block_start, action_block_end = action_block_starts[block_idx], action_block_ends[block_idx]
            image_block_end = image_block_ends[block_idx]
            state_block_start, state_block_end = state_block_starts[block_idx], state_block_ends[block_idx]
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, image_block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            parts_k = [k[:, first_image_start:first_image_end], k[:, image_kv_start:image_block_end],
                       k[:, action_block_start:action_block_end], k[:, state_block_start:state_block_end]]
            parts_v = [v[:, first_image_start:first_image_end], v[:, image_kv_start:image_block_end],
                       v[:, action_block_start:action_block_end], v[:, state_block_start:state_block_end]]
            output[:, action_block_start:action_block_end] = self.attn(
                q[:, action_block_start:action_block_end], torch.cat(parts_k, dim=1), torch.cat(parts_v, dim=1)
            )

        # ----- State blocks: self-attention only -----
        for block_idx in range(num_state_blocks):
            state_block_start, state_block_end = state_block_starts[block_idx], state_block_ends[block_idx]
            output[:, state_block_start:state_block_end] = self.attn(
                q[:, state_block_start:state_block_end],
                k[:, state_block_start:state_block_end],
                v[:, state_block_start:state_block_end]
            )

        return output

    def _process_clean_image_only(
        self,
        clean_image_q,
        clean_image_k,
        clean_image_v,
        clean_frames,
        ff_k=None,
        ff_v=None,
    ):
        """Process clean image blocks with causal attention pattern - OPTIMIZED

        First frame: conditioning, cannot attend to anything (self-attention only)
        Block i: attends to first frame + previous blocks (0 to i-1) + current block

        OPTIMIZATION: Instead of looping through blocks, we batch process them together
        by using a single flash_attention call with properly structured KV cache.
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        assert (clean_frames - 1) % self.num_frame_per_block == 0
        num_blocks = (clean_frames - 1) // self.num_frame_per_block

        if num_blocks == 0:
            # Only first frame - single attention call
            parts_k = [clean_image_k[:, :self.frame_seqlen]]
            parts_v = [clean_image_v[:, :self.frame_seqlen]]
            if ff_k is not None:
                parts_k.append(ff_k)
                parts_v.append(ff_v)
            return self.attn(
                clean_image_q[:, :self.frame_seqlen],
                torch.cat(parts_k, dim=1),
                torch.cat(parts_v, dim=1),
            )

        # Pre-allocate output tensor (avoids list append + cat overhead)
        b, total_len, n, d = clean_image_q.shape
        output = torch.empty_like(clean_image_q)

        # First frame: conditioning, self-attention only
        parts_k = [clean_image_k[:, :self.frame_seqlen]]
        parts_v = [clean_image_v[:, :self.frame_seqlen]]
        if ff_k is not None:
            parts_k.append(ff_k)
            parts_v.append(ff_v)
        output[:, :self.frame_seqlen] = self.attn(
            clean_image_q[:, :self.frame_seqlen],
            torch.cat(parts_k, dim=1),
            torch.cat(parts_v, dim=1),
        )

        # OPTIMIZATION: Process all blocks together with causal masking
        # For global attention (no local_attn_size), we can process all blocks in one call
        if self.local_attn_size == -1:
            # 当前 RLinf/DreamZero 5B 配置 max_chunk_size=4、num_frame_per_block=2，
            # local_attn_size=9，不会走 global attention 分支；ff token 只支持 local 分支。
            raise NotImplementedError(
                "global clean-image attention is not used by the current DreamZero 5B "
                "configuration; use finite local_attn_size."
            )
        else:
            # With local attention, we still need to loop but with optimizations
            # Pre-compute all block boundaries to reduce overhead
            block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]

            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]

                q_block = clean_image_q[:, block_start:block_end]

                # Context: first frame + recent blocks within local_attn_size
                image_kv_start = max(self.frame_seqlen, block_end - self.local_attn_size * self.frame_seqlen)
                parts_k = [
                    clean_image_k[:, :self.frame_seqlen],  # First frame
                    clean_image_k[:, image_kv_start:block_end],  # Recent blocks + current
                ]
                parts_v = [
                    clean_image_v[:, :self.frame_seqlen],
                    clean_image_v[:, image_kv_start:block_end],
                ]
                if ff_k is not None:
                    parts_k.append(ff_k)
                    parts_v.append(ff_v)
                k_context = torch.cat(parts_k, dim=1)
                v_context = torch.cat(parts_v, dim=1)

                output[:, block_start:block_end] = self.attn(q_block, k_context, v_context)

        return output

    def _process_state_blocks(self, state_q, state_k, state_v, state_horizon):
        """Process state blocks: self-attention only - OPTIMIZED

        OPTIMIZATION: State blocks only do self-attention within each block.
        Instead of looping, we can process all blocks in a single call with block-diagonal masking,
        or even simpler: just one attention call since they're independent.
        """
        num_blocks = state_horizon // self.num_state_per_block

        if num_blocks == 1:
            # Single block - one attention call
            return self.attn(state_q, state_k, state_v)

        # OPTIMIZATION: Since each state block only attends to itself (no cross-block attention),
        # we can process all blocks in a single batched call. Flash attention will handle this
        # efficiently. The blocks are independent, so this is safe.
        # Alternative: reshape and process as separate batch items

        # Pre-allocate output
        output = torch.empty_like(state_q)

        # Process all blocks (keeping loop for now due to block-diagonal pattern)
        # This could be further optimized with custom masking
        for block_idx in range(num_blocks):
            state_block_start = block_idx * self.num_state_per_block
            state_block_end = state_block_start + self.num_state_per_block

            output[:, state_block_start:state_block_end] = self.attn(
                state_q[:, state_block_start:state_block_end],
                state_k[:, state_block_start:state_block_end],
                state_v[:, state_block_start:state_block_end]
            )

        return output


    def _process_noisy_image_only(self, noisy_image_q, noisy_image_k, noisy_image_v,
                                   clean_image_k, clean_image_v, half_frames):
        """Process noisy image blocks with teacher forcing pattern (VIDEO-ONLY).

        Video-only counterpart of _process_noisy_image_blocks (L673-737),
        with the action/state register tokens stripped out.

        ⭐ CRITICAL — DO NOT change attention pattern:
        Train-time attention pattern MUST match the KV-cache inference path
        (see CausalWanSelfAttention.forward L1060-1099). The previous
        implementation `self.attn(noisy_q, full_k, full_v)` with full_k =
        cat(clean_k, noisy_k) was non-causal — every noisy_q saw ALL noisy
        frames including the future ones, which let the model trivially
        reconstruct (loss → 0.034) but corrupted the causal generation
        ability inherited from the Wan2.2 baseline. KV-cache inference
        only sees PAST frames, so train and inference were optimizing
        different attention patterns.

        Per-block attention (mirrors labeled L719-735 minus action/state):
            First noisy frame: conditioning, self-attention only
            Block i: attends to first_clean_frame + clean_blocks[0:i]
                                + current_noisy_block

        This makes the per-block context monotonically grow with block
        index — exactly what KV-cache inference will replay later.
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (half_frames - 1) // self.num_frame_per_block

        output = torch.empty_like(noisy_image_q)

        # First noisy frame: conditioning, self-attention only.
        output[:, :self.frame_seqlen] = self.attn(
            noisy_image_q[:, :self.frame_seqlen],
            noisy_image_k[:, :self.frame_seqlen],
            noisy_image_v[:, :self.frame_seqlen],
        )

        if num_blocks == 0:
            return output

        noisy_block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        noisy_block_ends = [
            min(start + block_size, noisy_image_q.shape[1]) for start in noisy_block_starts
        ]
        clean_context_ends = [self.frame_seqlen + i * block_size for i in range(num_blocks)]

        for block_idx in range(num_blocks):
            noisy_start = noisy_block_starts[block_idx]
            noisy_end = noisy_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]

            q_block = noisy_image_q[:, noisy_start:noisy_end]

            # Mirrors labeled _process_noisy_image_blocks L721-733 minus
            # action/state context. Per-block context: first_clean_frame +
            # clean_blocks[0:i] + current_noisy_block.
            k_context = torch.cat([
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_start:noisy_end],
            ], dim=1)
            v_context = torch.cat([
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_start:noisy_end],
            ], dim=1)

            output[:, noisy_start:noisy_end] = self.attn(q_block, k_context, v_context)

        return output

    def _process_noisy_image_blocks(self, noisy_image_q, noisy_image_k, noisy_image_v,
                                     clean_image_k, clean_image_v,
                                     noisy_action_k, noisy_action_v, noisy_state_k, noisy_state_v,
                                     half_frames,
                                     ff_k=None,
                                     ff_v=None):
        """Process noisy image blocks with teacher forcing visibility.

        Main visibility contract, block i:
          V0_noisy_video[i] =
            clean_video[<i] + noisy_video[i]

          V1_noisy_video[i] =
            clean_video[<i] + noisy_video[i] + noisy_action[i] + state[i]
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (half_frames - 1) // self.num_frame_per_block

        output = torch.empty_like(noisy_image_q)

        # Note: 首个 noisy frame 沿用原实现，只做 self-attention，不接 action/state。
        parts_k = [noisy_image_k[:, :self.frame_seqlen]]
        parts_v = [noisy_image_v[:, :self.frame_seqlen]]
        if ff_k is not None:
            parts_k.append(ff_k)
            parts_v.append(ff_v)
        output[:, :self.frame_seqlen] = self.attn(
            noisy_image_q[:, :self.frame_seqlen],
            torch.cat(parts_k, dim=1),
            torch.cat(parts_v, dim=1),
        )

        if num_blocks == 0:
            return output

        noisy_block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        noisy_block_ends = [min(start + block_size, noisy_image_q.shape[1]) for start in noisy_block_starts]
        clean_context_ends = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
        action_block_ends = [start + self.num_action_per_block for start in action_block_starts]
        state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
        state_block_ends = [start + self.num_state_per_block for start in state_block_starts]

        for block_idx in range(num_blocks):
            noisy_start = noisy_block_starts[block_idx]
            noisy_end = noisy_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]
            action_start = action_block_starts[block_idx]
            action_end = action_block_ends[block_idx]
            state_start = state_block_starts[block_idx]
            state_end = state_block_ends[block_idx]

            q_block = noisy_image_q[:, noisy_start:noisy_end]

            parts_k = [
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_start:noisy_end],
            ]
            parts_v = [
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_start:noisy_end],
            ]
            if self.video_condition_on_action_state:
                parts_k.extend([
                    noisy_action_k[:, action_start:action_end],
                    noisy_state_k[:, state_start:state_end],
                ])
                parts_v.extend([
                    noisy_action_v[:, action_start:action_end],
                    noisy_state_v[:, state_start:state_end],
                ])
            if ff_k is not None:
                parts_k.append(ff_k)
                parts_v.append(ff_v)

            k_context = torch.cat(parts_k, dim=1)
            v_context = torch.cat(parts_v, dim=1)
            output[:, noisy_start:noisy_end] = self.attn(q_block, k_context, v_context)

        return output

    def _process_noisy_action_blocks(self, noisy_action_q, noisy_action_k, noisy_action_v,
                                      clean_image_k, clean_image_v,
                                      noisy_image_k, noisy_image_v,
                                      noisy_state_k, noisy_state_v,
                                      half_frames,
                                      clean_past_action_k=None, clean_past_action_v=None,
                                      past_action_keep_tokens=None,
                                      ff_k=None,
                                      ff_v=None):
        """Process noisy action blocks with teacher-forcing visibility.

        Main visibility contract, block i:
          A0_noisy_action[i] =
            clean_video[<i] + noisy_video[i] + noisy_action[i] + state[i]

          A1_noisy_action[i] =
            clean_video[<i] + noisy_video[i] + noisy_action[i] + state[i]
            + clean_past_action[<i]
        """
        num_blocks = (half_frames - 1) // self.num_frame_per_block

        if num_blocks == 0:
            return torch.empty_like(noisy_action_q)

        output = torch.empty_like(noisy_action_q)

        action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
        action_block_ends   = [start + self.num_action_per_block for start in action_block_starts]
        clean_context_ends = [self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block for i in range(num_blocks)]
        noisy_image_starts = [self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block for i in range(num_blocks)]
        noisy_image_ends   = [start + self.frame_seqlen * self.num_frame_per_block for start in noisy_image_starts]
        state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
        state_block_ends   = [start + self.num_state_per_block for start in state_block_starts]

        for block_idx in range(num_blocks):
            action_start = action_block_starts[block_idx]
            action_end   = action_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]
            image_start = noisy_image_starts[block_idx]
            image_end   = noisy_image_ends[block_idx]
            state_start = state_block_starts[block_idx]
            state_end   = state_block_ends[block_idx]

            q_block = noisy_action_q[:, action_start:action_end]

            parts_k = [
                clean_image_k[:, :clean_end],
                noisy_image_k[:, image_start:image_end],
                noisy_action_k[:, action_start:action_end],
                noisy_state_k[:, state_start:state_end],
            ]
            parts_v = [
                clean_image_v[:, :clean_end],
                noisy_image_v[:, image_start:image_end],
                noisy_action_v[:, action_start:action_end],
                noisy_state_v[:, state_start:state_end],
            ]
            if ff_k is not None:
                parts_k.append(ff_k)
                parts_v.append(ff_v)
            past_action_end = block_idx * self.num_action_per_block
            if clean_past_action_k is not None:
                assert clean_past_action_v is not None
                selected_past_action_idx = None
                if past_action_keep_tokens is None:
                    past_action_start = 0
                    past_action_end_idx = past_action_end
                else:
                    if past_action_keep_tokens < 0:
                        raise ValueError(
                            "past_action_keep_tokens must be non-negative, "
                            f"got {past_action_keep_tokens}."
                        )
                    if past_action_end <= past_action_keep_tokens:
                        past_action_start = 0
                        past_action_end_idx = past_action_end
                    else:
                        selected_past_action_idx = torch.randperm(
                            past_action_end,
                            device=clean_past_action_k.device,
                        )[:past_action_keep_tokens]
                        selected_past_action_idx = selected_past_action_idx.sort().values
                        if torch.rand((), device=clean_past_action_k.device).item() < 0.001:
                            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                            if rank == 0:
                                print(
                                    "[PAST_ACTION_DROPOUT] "
                                    f"block_idx={block_idx} "
                                    f"past_action_end={past_action_end} "
                                    f"past_action_keep_tokens={past_action_keep_tokens} "
                                    f"selected_tokens={selected_past_action_idx.numel()}",
                                    flush=True,
                                )
                        past_action_start = 0
                        past_action_end_idx = 0
                if block_idx == 0:
                    assert past_action_end == 0
                if selected_past_action_idx is not None:
                    if selected_past_action_idx.numel() > 0:
                        parts_k.insert(1, clean_past_action_k.index_select(1, selected_past_action_idx))
                        parts_v.insert(1, clean_past_action_v.index_select(1, selected_past_action_idx))
                elif past_action_end_idx > past_action_start:
                    parts_k.insert(1, clean_past_action_k[:, past_action_start:past_action_end_idx])
                    parts_v.insert(1, clean_past_action_v[:, past_action_start:past_action_end_idx])

            k_context = torch.cat(parts_k, dim=1)
            v_context = torch.cat(parts_v, dim=1)

            output[:, action_start:action_end] = self.attn(q_block, k_context, v_context)

        return output

    def _process_clean_past_action_blocks(self, past_action_q, past_action_k, past_action_v):
        """Process clean-past-action blocks as pure action-history memory.

        This helper is shared by teacher forcing and inference:
          clean_past_action[i] = clean_past_action[<=i]

        Past-action tokens stay outside the rolling video KV cache and are
        small enough to recompute explicitly.
        """
        if past_action_q is None or past_action_q.shape[1] == 0:
            return None

        if past_action_q.shape[1] % self.num_action_per_block != 0:
            raise ValueError(
                "past action context length must be divisible by num_action_per_block, "
                f"got {past_action_q.shape[1]} and {self.num_action_per_block}."
            )

        output = torch.empty_like(past_action_q)
        num_blocks = past_action_q.shape[1] // self.num_action_per_block

        for block_idx in range(num_blocks):
            action_start = block_idx * self.num_action_per_block
            action_end = action_start + self.num_action_per_block
            # Keep clean-past-action as pure action-history memory. The main
            # action query already attends to clean_video[<i] directly.
            q_block = past_action_q[:, action_start:action_end]
            output[:, action_start:action_end] = self.attn(
                q_block,
                past_action_k[:, :action_end],
                past_action_v[:, :action_end],
            )

        return output

    def _process_ff_image_context(self, ff_q, ff_k, ff_v):
        """输入 append 在序列尾部的 ff q/k/v，输出 ff 自身 attention 结果；其它 token 只把 ff 当只读 context。"""
        return self.attn(ff_q, ff_k, ff_v)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
        past_action_context_length: int = 0,
        past_action_keep_tokens: int | None = None,
        ff_context_length: int = 0,
        ff_freqs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            ff_context_length: 尾部 ff token 数；ff_freqs 是它们独立的 video RoPE。
        """
        s = x.shape[1]

        control_span = None
        if self.split_video_control_attention:
            control_span = self._control_token_span(
                seq_len=s,
                action_register_length=action_register_length,
                past_action_context_length=past_action_context_length,
                ff_context_length=ff_context_length,
            )
        q, k, v = self._project_qkv(x, control_span)

        updated_kv_cache: torch.Tensor | None = None

        if kv_cache is None:
            if is_tf:
                # Teacher forcing training.
                if ff_context_length < 0:
                    raise ValueError(f"ff_context_length must be non-negative, got {ff_context_length}.")
                if ff_context_length and ff_freqs is None:
                    raise ValueError("ff_context_length requires ff_freqs.")
                base_seq_len = s - past_action_context_length - ff_context_length
                if base_seq_len <= 0:
                    raise ValueError(
                        "invalid context lengths for sequence length "
                        f"{s}: past_action_context_length={past_action_context_length}, "
                        f"ff_context_length={ff_context_length}."
                    )
                if action_register_length is not None:
                    half_seq_len = (base_seq_len - action_register_length) // 2
                    q_context = q[:, :half_seq_len]
                    k_context = k[:, :half_seq_len]
                    q_noisy = q[:, half_seq_len:base_seq_len]
                    k_noisy = k[:, half_seq_len:base_seq_len]
                else:
                    if past_action_context_length:
                        raise ValueError("past action context requires action_register_length.")
                    half_seq_len = base_seq_len // 2
                    q_context = q[:, :half_seq_len]
                    k_context = k[:, :half_seq_len]
                    q_noisy = q[:, half_seq_len:base_seq_len]
                    k_noisy = k[:, half_seq_len:base_seq_len]
                past_action_end = base_seq_len + past_action_context_length
                q_past_action = q[:, base_seq_len:past_action_end] if past_action_context_length else None
                k_past_action = k[:, base_seq_len:past_action_end] if past_action_context_length else None
                q_ff = q[:, past_action_end:] if ff_context_length else None
                k_ff = k[:, past_action_end:] if ff_context_length else None
                roped_query = []
                roped_key = []

                # rope should be same for clean and noisy parts
                rq_context = rope_action_apply(
                    x=q_context,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=None,
                ).type_as(v)
                rk_context = rope_action_apply(
                    x=k_context,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=None,
                ).type_as(v)

                rq_noisy = rope_action_apply(
                    x=q_noisy,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)
                rk_noisy = rope_action_apply(
                    x=k_noisy,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)

                roped_query.append(rq_context)
                roped_key.append(rk_context)
                roped_query.append(rq_noisy)
                roped_key.append(rk_noisy)
                if past_action_context_length:
                    assert q_past_action is not None and k_past_action is not None
                    rq_past_action = rope_action_only_apply(
                        q_past_action,
                        freqs_action=freqs_action,
                    ).type_as(v)
                    rk_past_action = rope_action_only_apply(
                        k_past_action,
                        freqs_action=freqs_action,
                    ).type_as(v)
                    roped_query.append(rq_past_action)
                    roped_key.append(rk_past_action)
                if ff_context_length:
                    assert q_ff is not None and k_ff is not None and ff_freqs is not None
                    rq_ff = rope_action_apply(
                        x=q_ff,
                        freqs=ff_freqs,
                        freqs_action=freqs_action,
                        freqs_state=freqs_state,
                        action_register_length=None,
                    ).type_as(v)
                    rk_ff = rope_action_apply(
                        x=k_ff,
                        freqs=ff_freqs,
                        freqs_action=freqs_action,
                        freqs_state=freqs_state,
                        action_register_length=None,
                    ).type_as(v)
                    roped_query.append(rq_ff)
                    roped_key.append(rk_ff)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                if action_register_length is not None:
                    # Teacher forcing structure:
                    # Clean half: [image tokens only]
                    # Noisy half: [image tokens][action][state]
                    # Causality only applies to image blocks!

                    # Clean half contains ONLY image tokens
                    clean_image_seq_len = half_seq_len
                    clean_frames = clean_image_seq_len // self.frame_seqlen

                    # Noisy half contains image + action + state tokens
                    noisy_image_seq_len = half_seq_len
                    noisy_frames = noisy_image_seq_len // self.frame_seqlen
                    num_image_blocks = (noisy_frames - 1) // self.num_frame_per_block
                    action_horizon = num_image_blocks * self.num_action_per_block
                    state_horizon = num_image_blocks * self.num_state_per_block

                    # Block layout must match actual register length.
                    if roped_query.shape[1] != clean_image_seq_len + noisy_image_seq_len + action_horizon + state_horizon + past_action_context_length + ff_context_length:
                        raise ValueError(
                            "Sequence length does not match block layout. "
                            f"Got noisy_frames={noisy_frames}, num_image_blocks={num_image_blocks}, "
                            f"action_register_length={action_register_length}. "
                            "Ensure (noisy_frames - 1) // num_frame_per_block >= 1 and register length equals "
                            "num_blocks * (num_action_per_block + num_state_per_block)."
                        )

                    # Split clean and noisy parts
                    # Clean: [image tokens only]
                    clean_image_q = roped_query[:, :clean_image_seq_len]
                    clean_image_k = roped_key[:, :clean_image_seq_len]
                    clean_image_v = v[:, :clean_image_seq_len]
                    # Noisy: [image][action][state]
                    img_end = clean_image_seq_len + noisy_image_seq_len
                    act_end = img_end + action_horizon
                    state_end = act_end + state_horizon

                    noisy_image_q = roped_query[:, clean_image_seq_len:img_end]
                    noisy_action_q = roped_query[:, img_end:act_end]
                    noisy_state_q  = roped_query[:, act_end:state_end]
                    ff_start = state_end + past_action_context_length
                    clean_past_action_q = roped_query[:, state_end:ff_start] if past_action_context_length else None
                    ff_q = roped_query[:, ff_start:] if ff_context_length else None

                    noisy_image_k = roped_key[:, clean_image_seq_len:img_end]
                    noisy_action_k = roped_key[:, img_end:act_end]
                    noisy_state_k  = roped_key[:, act_end:state_end]
                    clean_past_action_k = roped_key[:, state_end:ff_start] if past_action_context_length else None
                    ff_k = roped_key[:, ff_start:] if ff_context_length else None

                    noisy_image_v = v[:, clean_image_seq_len:img_end]
                    noisy_action_v = v[:, img_end:act_end]
                    noisy_state_v  = v[:, act_end:state_end]
                    clean_past_action_v = v[:, state_end:ff_start] if past_action_context_length else None
                    ff_v = v[:, ff_start:] if ff_context_length else None

                    # Self-attention visibility contract, block i:
                    #   clean_video[i] = clean_video[<=i]
                    #   clean_past_action[i] = clean_past_action[<=i]
                    #   V0_noisy_video[i] = clean_video[<i] + noisy_video[i]
                    #   V1_noisy_video[i] = clean_video[<i] + noisy_video[i]
                    #                       + noisy_action[i] + state[i]
                    #   A0_noisy_action[i] = clean_video[<i] + noisy_video[i]
                    #                       + noisy_action[i] + state[i]
                    #   A1_noisy_action[i] = clean_video[<i] + noisy_video[i]
                    #                       + noisy_action[i] + state[i]
                    #                       + clean_past_action[<i]
                    #   state[i] = state[i]
                    # ========== Process CLEAN (context) image tokens ==========
                    clean_image_outputs = self._process_clean_image_only(
                        clean_image_q, clean_image_k, clean_image_v, clean_frames,
                        ff_k=ff_k, ff_v=ff_v)

                    # ========== Process NOISY tokens ==========
                    # Image attention is V0/V1 gated inside _process_noisy_image_blocks.
                    noisy_image_outputs = self._process_noisy_image_blocks(
                        noisy_image_q, noisy_image_k, noisy_image_v,
                        clean_image_k, clean_image_v,
                        noisy_action_k, noisy_action_v, noisy_state_k, noisy_state_v,
                        noisy_frames,
                        ff_k=ff_k,
                        ff_v=ff_v,
                    )

                    # Action sees clean/current video, current action/state,
                    # and optionally clean_past_action[<i].
                    noisy_action_outputs = self._process_noisy_action_blocks(
                        noisy_action_q, noisy_action_k, noisy_action_v,
                        clean_image_k, clean_image_v,
                        noisy_image_k, noisy_image_v,
                        noisy_state_k, noisy_state_v,
                        noisy_frames,
                        clean_past_action_k=clean_past_action_k,
                        clean_past_action_v=clean_past_action_v,
                        past_action_keep_tokens=past_action_keep_tokens,
                        ff_k=ff_k,
                        ff_v=ff_v,
                    )

                    # Noisy state blocks: self-attention only
                    noisy_state_outputs = self._process_state_blocks(
                        noisy_state_q, noisy_state_k, noisy_state_v, state_horizon)

                    outputs = [clean_image_outputs, noisy_image_outputs, noisy_action_outputs, noisy_state_outputs]
                    if past_action_context_length:
                        clean_past_action_outputs = self._process_clean_past_action_blocks(
                            clean_past_action_q,
                            clean_past_action_k,
                            clean_past_action_v,
                        )
                        assert clean_past_action_outputs is not None
                        outputs.append(clean_past_action_outputs)
                    if ff_context_length:
                        ff_outputs = self._process_ff_image_context(ff_q, ff_k, ff_v)
                        outputs.append(ff_outputs)

                    # Concatenate outputs in layout:
                    # [clean_img][noisy_img][noisy_act][noisy_state][clean_past_action][ff_img]
                    x = torch.cat(outputs, dim=1)
                else:
                    raise NotImplementedError(
                        "Video-only teacher-forcing path is currently unsupported. "
                        "The legacy implementation is kept below for reference."
                    )
                    # ⭐ Video-only teacher forcing path — MUST mirror the
                    # labeled path (L900-953) exactly except for stripping
                    # the action/state register tokens. Earlier this branch
                    # called self.attn(noisy_q, full_k=cat(clean_k, noisy_k),
                    # full_v=cat(clean_v, noisy_v)) which was NON-CAUSAL on
                    # noisy frames (every noisy_q saw the full noisy K/V
                    # including future frames). That made train-time
                    # reconstruction trivially easy but broke the causal
                    # generation ability inherited from Wan2.2 — at
                    # KV-cache inference time the model only sees past
                    # frames, mismatching what it learned during training.
                    #
                    # Symptom: dynamics_loss → 0.034 in 200 steps, but
                    # autoregressive video generation produced output
                    # uncorrelated with GT. See _process_noisy_image_only
                    # docstring for the exact attention pattern.
                    half_frames = half_seq_len // self.frame_seqlen
                    clean_q = roped_query[:, :half_seq_len]
                    clean_k = roped_key[:, :half_seq_len]
                    clean_v = v[:, :half_seq_len]
                    noisy_q = roped_query[:, half_seq_len:]
                    noisy_k = roped_key[:, half_seq_len:]
                    noisy_v = v[:, half_seq_len:]

                    # Clean (context) image: causal chunked attention,
                    # mirrors labeled L926-927 (same _process_clean_image_only).
                    clean_image_outputs = self._process_clean_image_only(
                        clean_q, clean_k, clean_v, half_frames)

                    # Noisy image: causal chunked teacher forcing. Mirrors
                    # labeled _process_noisy_image_blocks (L673-737) with
                    # action/state register stripped. See dedicated method
                    # docstring for attention pattern proof.
                    noisy_image_outputs = self._process_noisy_image_only(
                        noisy_q, noisy_k, noisy_v,
                        clean_k, clean_v, half_frames)

                    x = torch.cat([clean_image_outputs, noisy_image_outputs], dim=1)

            else:
                raise RuntimeError(
                    "Unexpected non-teacher-forcing forward without kv_cache. "
                    "Current DreamZero training should pass clean_x; inference should pass kv_cache."
                )
                roped_query = rope_action_apply(
                    x=q,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)
                roped_key = rope_action_apply(
                    x=k,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)

                if action_register_length is not None:
                    _denom = self.num_action_per_block + self.num_state_per_block
                    _num_blocks = (action_register_length // _denom) if _denom > 0 else 0
                    action_horizon = _num_blocks * self.num_action_per_block
                    state_horizon = _num_blocks * self.num_state_per_block
                else:
                    action_horizon = None
                    state_horizon = None

                # Use blockwise causal flash attention without massive padding
                visualize = False
                x = self._blockwise_causal_flash_attn(
                    roped_query, roped_key, v, self.frame_seqlen, self.num_frame_per_block,
                    action_horizon=action_horizon,
                    state_horizon=state_horizon,
                    num_action_per_block=self.num_action_per_block if action_register_length else None,
                    num_state_per_block=self.num_state_per_block if action_register_length else None,
                    visualize_mask=visualize,
                )

        else:
            action_state_index = (current_start_frame - 1) // self.num_frame_per_block

            ff_query: torch.Tensor | None = None
            ff_key: torch.Tensor | None = None
            ff_v: torch.Tensor | None = None
            if ff_context_length:
                if ff_context_length < 0:
                    raise ValueError(f"ff_context_length must be non-negative, got {ff_context_length}.")
                if ff_freqs is None:
                    raise ValueError("ff_context_length requires ff_freqs.")
                ff_query = q[:, -ff_context_length:]
                ff_key = k[:, -ff_context_length:]
                ff_v = v[:, -ff_context_length:]
                q = q[:, :-ff_context_length]
                k = k[:, :-ff_context_length]
                v = v[:, :-ff_context_length]

            past_action_query: torch.Tensor | None = None
            past_action_key: torch.Tensor | None = None
            past_action_v: torch.Tensor | None = None
            if past_action_context_length:
                if action_register_length is None:
                    raise ValueError("past action context requires current action/state registers.")
                past_action_query = q[:, -past_action_context_length:]
                past_action_key = k[:, -past_action_context_length:]
                past_action_v = v[:, -past_action_context_length:]
                q = q[:, :-past_action_context_length]
                k = k[:, :-past_action_context_length]
                v = v[:, :-past_action_context_length]

            roped_query = causal_rope_action_apply(
                x=q,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)
            roped_key = causal_rope_action_apply(
                x=k,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)

            # split roped_query and roped_action_query (the last action_register_length tokens)
            roped_action_query: torch.Tensor | None = None
            roped_action_key: torch.Tensor | None = None
            action_v: torch.Tensor | None = None

            if action_register_length is not None:
                roped_action_query = roped_query[:, -action_register_length:]
                roped_query = roped_query[:, :-action_register_length]
                roped_action_key = roped_key[:, -action_register_length:]
                roped_key = roped_key[:, :-action_register_length]
                action_v = v[:, -action_register_length:]
                v = v[:, :-action_register_length]
                assert roped_action_query is not None
                assert roped_action_key is not None
                assert action_v is not None

            roped_past_action_query: torch.Tensor | None = None
            roped_past_action_key: torch.Tensor | None = None
            if past_action_context_length:
                assert past_action_query is not None
                assert past_action_key is not None
                assert past_action_v is not None
                if past_action_context_length % self.num_action_per_block != 0:
                    raise ValueError(
                        "past action context length must be divisible by num_action_per_block, "
                        f"got {past_action_context_length} and {self.num_action_per_block}."
                    )
                roped_past_action_query = rope_action_only_apply(
                    past_action_query,
                    freqs_action=freqs_action,
                ).type_as(v)
                roped_past_action_key = rope_action_only_apply(
                    past_action_key,
                    freqs_action=freqs_action,
                ).type_as(v)

            roped_ff_query: torch.Tensor | None = None
            roped_ff_key: torch.Tensor | None = None
            if ff_context_length:
                assert ff_query is not None and ff_key is not None and ff_v is not None and ff_freqs is not None
                roped_ff_query = rope_action_apply(
                    x=ff_query,
                    freqs=ff_freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=None,
                ).type_as(v)
                roped_ff_key = rope_action_apply(
                    x=ff_key,
                    freqs=ff_freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=None,
                ).type_as(v)

            num_new_tokens = roped_query.shape[1]
            assert roped_key.shape[1] == num_new_tokens
            assert v.shape[1] == num_new_tokens

            # If we are using local attention and the current KV cache size is larger
            # than the local attention size, we need to truncate the KV cache

            updated_kv_cache = kv_cache
            updated_k = updated_kv_cache[0]
            updated_v = updated_kv_cache[1]
            # Assign new keys/values directly up to current_end
            new_k = torch.cat([updated_k, roped_key], dim=1)
            new_v = torch.cat([updated_v, v], dim=1)

            # We may need to truncate the KV cache if it's size is larger than the max attention size.
            new_k = new_k[:, -self.max_attention_size:]
            new_v = new_v[:, -self.max_attention_size:]
            # print(f"new_k.shape={new_k.shape}, new_v.shape={new_v.shape}")

            # updated_k / updated_v = 历史 video cache
            # roped_key / v = 当前 video token 的 K/V
            # new_k / new_v = 历史 video cache + 当前 video K/V

            if action_register_length is not None:
                if action_register_length != (
                    self.num_action_per_block + self.num_state_per_block
                ):
                    raise ValueError(
                        "Inference expects one current action/state block, got "
                        f"action_register_length={action_register_length}, "
                        f"num_action_per_block={self.num_action_per_block}, "
                        f"num_state_per_block={self.num_state_per_block}."
                    )

                action_end = self.num_action_per_block
                current_action_q = roped_action_query[:, :action_end]
                current_state_q = roped_action_query[:, action_end:]
                current_action_k = roped_action_key[:, :action_end]
                current_state_k = roped_action_key[:, action_end:]
                current_action_v = action_v[:, :action_end]
                current_state_v = action_v[:, action_end:]

                # Inference mirrors the training visibility contract:
                #   V0 video  -> history_video + current_video
                #   V1 video  -> history_video + current_video + action + state
                #   A0 action -> history_video + current_video + action + state
                #   A1 action -> history_video + current_video + action + state
                #                + clean_past_action
                #   state     -> state only
                #
                # Past-action tokens are kept outside the rolling video KV cache.
                video_k_parts = [new_k]
                video_v_parts = [new_v]
                if self.video_condition_on_action_state:
                    video_k_parts.extend([current_action_k, current_state_k])
                    video_v_parts.extend([current_action_v, current_state_v])
                if ff_context_length:
                    assert roped_ff_key is not None and ff_v is not None
                    video_k_parts.append(roped_ff_key)
                    video_v_parts.append(ff_v)
                video_out = self.attn(
                    roped_query,
                    torch.cat(video_k_parts, dim=1),
                    torch.cat(video_v_parts, dim=1),
                )

                action_k_parts = [new_k, current_action_k, current_state_k]
                action_v_parts = [new_v, current_action_v, current_state_v]
                if past_action_context_length:
                    assert roped_past_action_key is not None
                    assert past_action_v is not None
                    action_k_parts.append(roped_past_action_key)
                    action_v_parts.append(past_action_v)
                if ff_context_length:
                    assert roped_ff_key is not None and ff_v is not None
                    action_k_parts.append(roped_ff_key)
                    action_v_parts.append(ff_v)
                action_out = self.attn(
                    current_action_q,
                    torch.cat(action_k_parts, dim=1),
                    torch.cat(action_v_parts, dim=1),
                )

                if current_state_q.shape[1] == 0:
                    raise ValueError("Inference expects at least one state token in the current action/state block.")
                state_out = self.attn(
                    current_state_q,
                    current_state_k,
                    current_state_v,
                )

                x_current = torch.cat([video_out, action_out, state_out], dim=1)
                if past_action_context_length:
                    assert roped_past_action_query is not None
                    assert roped_past_action_key is not None
                    assert past_action_v is not None
                    x_past = self._process_clean_past_action_blocks(
                        roped_past_action_query,
                        roped_past_action_key,
                        past_action_v,
                    )
                    assert x_past is not None
                    x = torch.cat([x_current, x_past], dim=1)
                else:
                    x = x_current
                if ff_context_length:
                    assert roped_ff_query is not None and roped_ff_key is not None and ff_v is not None
                    x_ff = self._process_ff_image_context(roped_ff_query, roped_ff_key, ff_v)
                    x = torch.cat([x, x_ff], dim=1)
            else:
                attn_k_parts = [new_k]
                attn_v_parts = [new_v]
                if ff_context_length:
                    assert roped_ff_key is not None and ff_v is not None
                    attn_k_parts.append(roped_ff_key)
                    attn_v_parts.append(ff_v)
                video_out = self.attn(
                    roped_query,
                    torch.cat(attn_k_parts, dim=1),
                    torch.cat(attn_v_parts, dim=1),
                )
                if ff_context_length:
                    assert roped_ff_query is not None and roped_ff_key is not None and ff_v is not None
                    ff_out = self._process_ff_image_context(roped_ff_query, roped_ff_key, ff_v)
                    x = torch.cat([video_out, ff_out], dim=1)
                else:
                    x = video_out
            updated_kv_cache = torch.stack([new_k, new_v], dim=0)


        # output
        x = self._project_attention_output(x, control_span)
        if (
            self._emit_split_routing_log
            and not self._split_routing_logged
            and control_span is not None
        ):
            control_start, control_end = control_span
            control_tokens = control_end - control_start
            print(
                "[SplitAttention] layer=0 Q/K/V/O route verified: "
                f"mode={'train' if kv_cache is None else 'inference'}, "
                f"sequence_tokens={s}, leading_video_tokens={control_start}, "
                f"control_tokens={control_tokens}, "
                f"action_state_tokens={action_register_length}, "
                f"past_action_tokens={past_action_context_length}, "
                f"ff_video_tokens={ff_context_length}, dtype={x.dtype}, "
                f"device={x.device}"
            )
            self._split_routing_logged = True
        return x, updated_kv_cache


def _split_token_modulation(e: torch.Tensor, length: int) -> tuple[torch.Tensor, ...]:
    """Validate the token count before splitting modulation components."""
    if e.shape[1] != length:
        raise ValueError(
            "Modulation token length must match the feature token length: "
            f"got {e.shape[1]}, expected {length}."
        )
    return e.split(1, dim=2)


class FullActionSelfAttention(CausalWanSelfAttention):
    """Mixed video/action attention with a 3072-dim video and 1024-dim action stream.

    Q/K/V live in the shared video attention space (24 x 128). The action
    stream has independent projections and an independent output projection,
    while the attention visibility contract is shared with DreamZero's
    existing blockwise causal attention helpers.
    """

    def __init__(self, *, action_dim: int, **kwargs):
        super().__init__(split_video_control_attention=False, **kwargs)
        self.action_dim = action_dim
        self.action_q = nn.Linear(action_dim, self.dim)
        self.action_k = nn.Linear(action_dim, self.dim)
        self.action_v = nn.Linear(action_dim, self.dim)
        self.action_o = nn.Linear(self.dim, action_dim)
        self.action_norm_q = WanRMSNorm(self.dim, eps=self.eps) if self.qk_norm else nn.Identity()
        self.action_norm_k = WanRMSNorm(self.dim, eps=self.eps) if self.qk_norm else nn.Identity()

    def _project_action_qkv(self, x: torch.Tensor):
        b, s = x.shape[:2]
        q = self.action_norm_q(self.action_q(x)).view(b, s, self.num_heads, self.head_dim)
        k = self.action_norm_k(self.action_k(x)).view(b, s, self.num_heads, self.head_dim)
        v = self.action_v(x).view(b, s, self.num_heads, self.head_dim)
        return q, k, v

    def _video_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.o(x.flatten(2))

    def _action_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_o(x.flatten(2))

    def forward_full(
        self,
        video_x: torch.Tensor,
        action_x: torch.Tensor,
        state_x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
        clean_video_length: int = 0,
        past_action_x: torch.Tensor | None = None,
        past_action_keep_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        action_length = action_x.shape[1]
        state_length = state_x.shape[1]
        action_register_length = action_length + state_length
        past_length = 0 if past_action_x is None else past_action_x.shape[1]

        video_q, video_k, video_v = self._project_qkv(video_x, None)
        action_q, action_k, action_v = self._project_action_qkv(action_x)
        state_q, state_k, state_v = self._project_action_qkv(state_x)
        past_q = past_k = past_v = None
        if past_action_x is not None:
            past_q, past_k, past_v = self._project_action_qkv(past_action_x)

        if kv_cache is None:
            if not is_tf:
                raise RuntimeError("Full action training without teacher forcing is unsupported.")
            if clean_video_length <= 0 or clean_video_length >= video_x.shape[1]:
                raise ValueError(
                    "Full action teacher forcing requires both clean and noisy video tokens."
                )
            noisy_video_length = video_x.shape[1] - clean_video_length
            if clean_video_length % self.frame_seqlen or noisy_video_length % self.frame_seqlen:
                raise ValueError("Video token lengths must be divisible by frame_seqlen.")

            clean_q = video_q[:, :clean_video_length]
            clean_k = video_k[:, :clean_video_length]
            clean_v = video_v[:, :clean_video_length]
            noisy_q = torch.cat([video_q[:, clean_video_length:], action_q, state_q], dim=1)
            noisy_k = torch.cat([video_k[:, clean_video_length:], action_k, state_k], dim=1)
            noisy_v = torch.cat([video_v[:, clean_video_length:], action_v, state_v], dim=1)

            roped_clean_q = rope_action_apply(
                clean_q, freqs, freqs_action, freqs_state, action_register_length=None
            ).type_as(video_v)
            roped_clean_k = rope_action_apply(
                clean_k, freqs, freqs_action, freqs_state, action_register_length=None
            ).type_as(video_v)
            roped_noisy_q = rope_action_apply(
                noisy_q,
                freqs,
                freqs_action,
                freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
            ).type_as(video_v)
            roped_noisy_k = rope_action_apply(
                noisy_k,
                freqs,
                freqs_action,
                freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
            ).type_as(video_v)

            roped_past_q = roped_past_k = None
            if past_length:
                assert past_q is not None and past_k is not None
                roped_past_q = rope_action_only_apply(past_q, freqs_action).type_as(video_v)
                roped_past_k = rope_action_only_apply(past_k, freqs_action).type_as(video_v)
            noisy_video_q = roped_noisy_q[:, :noisy_video_length]
            noisy_action_q = roped_noisy_q[:, noisy_video_length:noisy_video_length + action_length]
            noisy_state_q = roped_noisy_q[:, noisy_video_length + action_length:]
            noisy_video_k = roped_noisy_k[:, :noisy_video_length]
            noisy_action_k = roped_noisy_k[:, noisy_video_length:noisy_video_length + action_length]
            noisy_state_k = roped_noisy_k[:, noisy_video_length + action_length:]
            noisy_video_v = noisy_v[:, :noisy_video_length]
            noisy_action_v = noisy_v[:, noisy_video_length:noisy_video_length + action_length]
            noisy_state_v = noisy_v[:, noisy_video_length + action_length:]

            clean_frames = clean_video_length // self.frame_seqlen
            noisy_frames = noisy_video_length // self.frame_seqlen
            num_image_blocks = (noisy_frames - 1) // self.num_frame_per_block
            state_horizon = num_image_blocks * self.num_state_per_block
            expected_action = num_image_blocks * self.num_action_per_block
            if action_length != expected_action or state_length != state_horizon:
                raise ValueError(
                    "Full action register length does not match the video block layout: "
                    f"action={action_length}/{expected_action}, state={state_length}/{state_horizon}."
                )

            clean_out = self._process_clean_image_only(
                roped_clean_q, roped_clean_k, clean_v, clean_frames,
            )
            noisy_out = self._process_noisy_image_blocks(
                noisy_video_q, noisy_video_k, noisy_video_v,
                roped_clean_k, clean_v,
                noisy_action_k, noisy_action_v,
                noisy_state_k, noisy_state_v,
                noisy_frames,
            )
            action_out = self._process_noisy_action_blocks(
                noisy_action_q, noisy_action_k, noisy_action_v,
                roped_clean_k, clean_v,
                noisy_video_k, noisy_video_v,
                noisy_state_k, noisy_state_v,
                noisy_frames,
                clean_past_action_k=roped_past_k,
                clean_past_action_v=past_v,
                past_action_keep_tokens=past_action_keep_tokens,
            )
            state_out = self._process_state_blocks(
                noisy_state_q, noisy_state_k, noisy_state_v, state_length
            )
            video_parts = [self._video_output(clean_out), self._video_output(noisy_out)]
            action_parts = [self._action_output(action_out), self._action_output(state_out)]
            if past_length:
                assert roped_past_q is not None and roped_past_k is not None and past_v is not None
                past_out = self._process_clean_past_action_blocks(roped_past_q, roped_past_k, past_v)
                assert past_out is not None
                action_parts.append(self._action_output(past_out))
            return torch.cat(video_parts, dim=1), torch.cat(action_parts, dim=1), None

        if is_tf:
            raise RuntimeError("Full action inference received a KV cache with teacher forcing enabled.")
        if action_length != self.num_action_per_block or state_length != self.num_state_per_block:
            raise ValueError("Full action inference expects exactly one current action/state block.")

        current_q = torch.cat([video_q, action_q, state_q], dim=1)
        current_k = torch.cat([video_k, action_k, state_k], dim=1)
        action_state_index = (current_start_frame - 1) // self.num_frame_per_block
        roped_current_q = causal_rope_action_apply(
            current_q, freqs, freqs_action, freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=self.num_state_per_block,
            action_state_index=action_state_index,
        ).type_as(video_v)
        roped_current_k = causal_rope_action_apply(
            current_k, freqs, freqs_action, freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=self.num_state_per_block,
            action_state_index=action_state_index,
        ).type_as(video_v)
        roped_video_q = roped_current_q[:, :video_x.shape[1]]
        roped_action_q = roped_current_q[:, video_x.shape[1]:video_x.shape[1] + action_length]
        roped_state_q = roped_current_q[:, -state_length:]
        roped_video_k = roped_current_k[:, :video_x.shape[1]]
        roped_action_k = roped_current_k[:, video_x.shape[1]:video_x.shape[1] + action_length]
        roped_state_k = roped_current_k[:, -state_length:]

        roped_past_q = roped_past_k = None
        if past_length:
            assert past_q is not None and past_k is not None
            roped_past_q = rope_action_only_apply(past_q, freqs_action).type_as(video_v)
            roped_past_k = rope_action_only_apply(past_k, freqs_action).type_as(video_v)
        cached_k, cached_v = kv_cache[0], kv_cache[1]
        new_k = torch.cat([cached_k, roped_video_k], dim=1)
        new_v = torch.cat([cached_v, video_v], dim=1)
        new_k = new_k[:, -self.max_attention_size:]
        new_v = new_v[:, -self.max_attention_size:]

        video_k_parts = [new_k]
        video_v_parts = [new_v]
        if self.video_condition_on_action_state:
            video_k_parts.extend([roped_action_k, roped_state_k])
            video_v_parts.extend([action_v, state_v])
        video_out = self.attn(roped_video_q, torch.cat(video_k_parts, dim=1), torch.cat(video_v_parts, dim=1))

        action_k_parts = [new_k, roped_action_k, roped_state_k]
        action_v_parts = [new_v, action_v, state_v]
        if past_length:
            assert roped_past_k is not None and past_v is not None
            action_k_parts.append(roped_past_k)
            action_v_parts.append(past_v)
        action_out = self.attn(roped_action_q, torch.cat(action_k_parts, dim=1), torch.cat(action_v_parts, dim=1))
        state_out = self.attn(roped_state_q, roped_state_k, state_v)

        action_parts = [self._action_output(action_out), self._action_output(state_out)]
        if past_length:
            assert roped_past_q is not None and roped_past_k is not None and past_v is not None
            past_out = self._process_clean_past_action_blocks(roped_past_q, roped_past_k, past_v)
            assert past_out is not None
            action_parts.append(self._action_output(past_out))
        return self._video_output(video_out), torch.cat(action_parts, dim=1), torch.stack([new_k, new_v], dim=0)


class FullActionCrossAttention(nn.Module):
    """Text-only cross attention with independent action-side projections."""

    def __init__(self, action_dim, video_dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        self.action_dim = action_dim
        self.video_dim = video_dim
        self.num_heads = num_heads
        self.head_dim = video_dim // num_heads
        self.q = nn.Linear(action_dim, video_dim)
        self.k = nn.Linear(video_dim, video_dim)
        self.v = nn.Linear(video_dim, video_dim)
        self.o = nn.Linear(video_dim, action_dim)
        self.norm_q = WanRMSNorm(video_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(video_dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = AttentionModule(num_heads=num_heads, head_dim=self.head_dim)

    def forward(self, x, context):
        b = x.shape[0]
        q = self.norm_q(self.q(x)).view(b, -1, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(context)).view(b, -1, self.num_heads, self.head_dim)
        v = self.v(context).view(b, -1, self.num_heads, self.head_dim)
        out = self.attn(q, k, v).flatten(2)
        return self.o(out)


class CausalWanFullActionAttentionBlock(nn.Module):
    """Blockwise causal Wan block with separate video and action residual streams."""

    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, frame_seqlen,
                 local_attn_size=-1, sink_size=0, num_frame_per_block=1,
                 qk_norm=True, cross_attn_norm=False, eps=1e-6,
                 num_action_per_block=32, num_state_per_block=1,
                 video_condition_on_action_state=True, action_dim=1024,
                 action_ffn_dim=4096, action_condition_on_text=True):
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.action_condition_on_text = bool(action_condition_on_text)
        self.video_condition_on_action_state = video_condition_on_action_state
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = FullActionSelfAttention(
            action_dim=action_dim,
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
            video_condition_on_action_state=video_condition_on_action_state,
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.action_norm1 = WanLayerNorm(action_dim, eps)
        self.action_norm3 = WanLayerNorm(action_dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.action_cross_attn = FullActionCrossAttention(action_dim, dim, num_heads, qk_norm, eps)
        self.action_norm2 = WanLayerNorm(action_dim, eps)
        self.action_ffn = nn.Sequential(
            nn.Linear(action_dim, action_ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(action_ffn_dim, action_dim),
        )
        self.action_modulation = nn.Parameter(torch.randn(1, 6, action_dim) / action_dim**0.5)

    def forward(self, video_x=None, action_x=None, state_x=None,
                e_video=None, e_action=None, **kwargs):
        if kwargs.get('image_context_len', 0) != 0 or any(
            kwargs.get(name) is not None for name in (
                'image_context_query_mask', 'video_image_context_query_mask',
                'action_image_context_query_mask',
            )
        ):
            raise ValueError(
                "Full action architecture does not support image encoder conditioning."
            )
        if (kwargs.get('ff_x') is not None or kwargs.get('ff_freqs') is not None
                or kwargs.get('ff_context_length', 0) != 0):
            raise ValueError(
                "Full action architecture does not support episode first-frame tokens."
            )
        kwargs.pop('ff_x', None)
        if video_x is None:
            # Observed video primes the KV cache before action denoising starts.
            return CausalWanAttentionBlock.forward(self, **kwargs)
        past_action_x = kwargs.get('past_action_x')
        action_stream = torch.cat([action_x, state_x] + ([] if past_action_x is None else [past_action_x]), dim=1)
        video_stream = video_x
        video_e = _split_token_modulation(self.modulation.unsqueeze(1) + e_video, video_stream.shape[1])
        action_e = _split_token_modulation(self.action_modulation.unsqueeze(1) + e_action, action_stream.shape[1])
        video_norm = self.norm1(video_stream).float() * (1 + video_e[1].squeeze(2)) + video_e[0].squeeze(2)
        action_norm = self.action_norm1(action_stream).float() * (1 + action_e[1].squeeze(2)) + action_e[0].squeeze(2)
        action_current_norm = action_norm[:, :action_x.shape[1]]
        state_start = action_x.shape[1]
        action_state_norm = action_norm[:, state_start:state_start + state_x.shape[1]]
        past_norm = action_norm[:, state_start + state_x.shape[1]:] if past_action_x is not None else None
        video_y, action_y, updated_cache = self.self_attn.forward_full(
            video_norm, action_current_norm, action_state_norm,
            past_action_x=past_norm,
            past_action_keep_tokens=kwargs.get('past_action_keep_tokens'),
            freqs=kwargs['freqs'],
            freqs_action=kwargs['freqs_action'],
            freqs_state=kwargs['freqs_state'],
            kv_cache=kwargs.get('kv_cache'),
            current_start_frame=kwargs.get('current_start_frame', 0),
            is_tf=kwargs.get('is_tf', True),
            clean_video_length=kwargs.get('clean_video_length', 0),
        )
        video_stream = video_stream + video_y * video_e[2].squeeze(2)
        action_stream = action_stream + action_y * action_e[2].squeeze(2)
        video_cross = self.cross_attn(self.norm3(video_stream), kwargs['context'])
        video_stream = video_stream + video_cross
        video_ffn = self.ffn(
            self.norm2(video_stream).float() * (1 + video_e[4].squeeze(2)) + video_e[3].squeeze(2)
        )
        video_stream = video_stream + video_ffn * video_e[5].squeeze(2)
        if self.action_condition_on_text:
            action_cross = self.action_cross_attn(self.action_norm3(action_stream), kwargs['context'])
            action_stream = action_stream + action_cross
        action_ffn = self.action_ffn(
            self.action_norm2(action_stream).float() * (1 + action_e[4].squeeze(2)) + action_e[3].squeeze(2)
        )
        action_stream = action_stream + action_ffn * action_e[5].squeeze(2)
        return video_stream, action_stream, updated_cache


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=1,
                 video_condition_on_action_state=True,
                 split_video_control_attention=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.video_condition_on_action_state = video_condition_on_action_state
        self.split_video_control_attention = split_video_control_attention

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
            video_condition_on_action_state=video_condition_on_action_state,
            split_video_control_attention=split_video_control_attention,
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        context: torch.Tensor,
        image_context_len: int = 0,
        kv_cache: torch.Tensor | None = None,
        crossattn_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
        past_action_context_length: int = 0,
        image_context_query_mask: torch.Tensor | None = None,
        past_action_keep_tokens: int | None = None,
        ff_context_length: int = 0,
        ff_freqs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            ff_context_length/ff_freqs: 透传给 self-attn 的 episode 首帧 token 长度和 RoPE。
        """
        e = _split_token_modulation(self.modulation.unsqueeze(1) + e, x.shape[1])

        # self-attention:
        y, updated_kv_cache = self.self_attn(
            x=(self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)),
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            is_tf=is_tf,
            current_start_frame=current_start_frame,
            past_action_context_length=past_action_context_length,
            past_action_keep_tokens=past_action_keep_tokens,
            ff_context_length=ff_context_length,
            ff_freqs=ff_freqs,
        )
        x = x + (y * e[2].squeeze(2))

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, e):
            cross = self.cross_attn(
                self.norm3(x),
                context,
                image_context_len=image_context_len,
                image_context_query_mask=image_context_query_mask,
            )
            x = x + cross
            y = self.ffn(
                (self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            )
            x = x + (y * e[5].squeeze(2))
            return x

        x = cross_attn_ffn(x, context, e)
        return x, updated_kv_cache


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        e = _split_token_modulation(self.modulation.unsqueeze(1) + e, x.shape[1])

        # Match Wan's FP32 video output projection, including the matmul.
        with torch.amp.autocast('cuda', dtype=torch.float32):
            head_input = self.norm(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
            x = self.head(head_input)
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 frame_seqlen=220,
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 max_chunk_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 num_frame_per_block=1,
                 action_dim=32,
                 num_registers=8,
                 max_state_dim=64,
                 max_num_embodiments=32,
                 hidden_size=1024,
                 diffusion_model_pretrained_path=None,
                 num_action_per_block=32,
                 num_state_per_block=1,
                 concat_first_frame_latent=True,
                 video_condition_on_action_state=True,
                 action_architecture='base',
                 action_hidden_size=1024,
                 action_ffn_dim=4096,
                 action_condition_on_text=True,
                 image_context_dim=1280,
                 enable_missing_action_embedding=False):
        r"""
        Initialize the diffusion model backbone.

        Args:
            concat_first_frame_latent (`bool`, *optional*, defaults to True):
                If True, concat [x; y] before patch_embedding (14B I2V style). If False, latent only (5B pretrained style; first-frame via CLIP).
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.frame_seqlen = frame_seqlen
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block
        self.diffusion_model_pretrained_path = diffusion_model_pretrained_path
        self.action_dim = action_dim
        self.num_registers = num_registers
        self.max_state_dim = max_state_dim
        self.max_num_embodiments = max_num_embodiments
        self.hidden_size = hidden_size
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.concat_first_frame_latent = concat_first_frame_latent
        self.video_condition_on_action_state = video_condition_on_action_state
        self.enable_missing_action_embedding = bool(
            enable_missing_action_embedding
        )
        if action_architecture not in {'base', 'split', 'full'}:
            raise ValueError(
                "action_architecture must be 'base', 'split', or 'full', "
                f"got {action_architecture!r}."
            )
        self.action_architecture = action_architecture
        self.action_hidden_size = action_hidden_size if action_architecture == 'full' else dim
        self.action_ffn_dim = action_ffn_dim if action_architecture == 'full' else ffn_dim
        self.action_condition_on_text = bool(action_condition_on_text)
        self._control_attention_copied_from_video = False
        self.image_context_dim = image_context_dim
        self.image_context_type = 'clip'
        self.image_context_token_scope = 'all'

        # 当前 action/state head 只实例化 category 0；恢复多 embodiment 时需同时保留真实 embodiment_id。
        max_num_embodiments = 1

        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_hidden_size,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=self.action_hidden_size,
            num_embodiments=max_num_embodiments,
            enable_missing_action_embedding=self.enable_missing_action_embedding,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=self.action_hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=action_dim,
        )

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        if action_architecture == 'full':
            self.action_time_embedding = nn.Sequential(
                nn.Linear(freq_dim, self.action_hidden_size),
                nn.SiLU(),
                nn.Linear(self.action_hidden_size, self.action_hidden_size),
            )
            self.action_time_projection = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.action_hidden_size, self.action_hidden_size * 6),
            )

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        if action_architecture == 'full':
            self.blocks = nn.ModuleList([
                CausalWanFullActionAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads, frame_seqlen,
                    self.local_attn_size, sink_size, num_frame_per_block, qk_norm,
                    cross_attn_norm, eps, num_action_per_block, num_state_per_block,
                    video_condition_on_action_state=video_condition_on_action_state,
                    action_dim=self.action_hidden_size,
                    action_ffn_dim=self.action_ffn_dim,
                    action_condition_on_text=self.action_condition_on_text,
                )
                for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                CausalWanAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads, frame_seqlen,
                    self.local_attn_size, sink_size, num_frame_per_block, qk_norm,
                    cross_attn_norm, eps, num_action_per_block, num_state_per_block,
                    video_condition_on_action_state=video_condition_on_action_state,
                    split_video_control_attention=action_architecture == 'split',
                )
                for _ in range(num_layers)
            ])
        if self.action_architecture == 'split':
            if not self.blocks:
                raise ValueError(
                    "action_architecture='split' requires at least one transformer block."
                )
            self.blocks[0].self_attn._emit_split_routing_log = True

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads

        self.freqs_action = rope_params(1024*10, d)
        self.freqs_state = rope_params(1024, d)
        self.freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]
        if model_type in ('i2v', 'ti2v'):
            self.img_emb = MLPProj(image_context_dim, dim)
            self.vae_img_emb = MLPProj(48, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = True
        self.independent_first_frame = False if self.num_frame_per_block == 1 else True

    @torch.no_grad()
    def initialize_control_attention_from_video(self) -> None:
        if self.action_architecture != 'split':
            raise RuntimeError(
                "Control attention initialization requires action_architecture='split'."
            )
        if self._control_attention_copied_from_video:
            raise RuntimeError("Control attention has already been copied from video attention.")
        if len(self.blocks) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} transformer blocks, got {len(self.blocks)}."
            )

        for block_index, block in enumerate(self.blocks):
            if not block.self_attn.split_video_control_attention:
                raise RuntimeError(
                    f"Block {block_index} does not have split video/control attention enabled."
                )
            block.self_attn.initialize_control_attention_from_video()
        self._control_attention_copied_from_video = True


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1, action_horizon=1, state_horizon=1, num_action_per_block=30, num_state_per_block=1
    ) -> BlockMask:
        """
        We will divide the token sequence into the following format:
        [first image (conditioning)] [image blocks] [action blocks] [state blocks]

        Legacy mask builder for the old monolithic blockwise path. It does not
        model the V0/V1 video_condition_on_action_state switch or past-action
        context.

        Structure:
        - First image: conditioning only, cannot attend to anything
        - Image blocks: can attend to first image + previous image block + current action block + current state block
        - Action blocks: can attend to previous image block + current image block + current state block
        - State blocks: conditioning only, cannot attend to anything

        Block alignment:
        - num_image_blocks = (num_frames - 1) // num_frame_per_block
        - num_action_blocks = action_horizon // num_action_per_block
        - num_state_blocks = state_horizon // num_state_per_block
        - num_image_blocks = num_action_blocks + 1 = num_state_blocks + 1
        """
        # Calculate block structure
        num_image_blocks = (num_frames - 1) // num_frame_per_block
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block

        # Verify the relationship: num_image_blocks = num_action_blocks + 1 = num_state_blocks + 1
        assert num_image_blocks == num_action_blocks, \
            f"image_blocks mismatch: {num_image_blocks} != {num_action_blocks}"
        assert num_image_blocks == num_state_blocks, \
            f"image_blocks mismatch: {num_image_blocks} != {num_state_blocks}"

        # Token ranges
        first_image_len = frame_seqlen  # First image (conditioning)
        image_blocks_len = num_image_blocks * num_frame_per_block * frame_seqlen
        action_len = action_horizon
        state_len = state_horizon
        total_length = first_image_len + image_blocks_len + action_len + state_len

        # print("total_length", total_length, first_image_len, image_blocks_len, action_len, state_len)
        # Padding to multiple of 128
        # padded_length = math.ceil(total_length / 128) * 128 - total_length
        padded_length = math.ceil((local_attn_size * frame_seqlen + (local_attn_size - 1) + 32 * (local_attn_size - 1))/128) * 128 - total_length
        total_padded_length = total_length + padded_length
        # print("total_padded_length", total_padded_length, total_length, padded_length)

        # Define token ranges for each modality
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len

        # Precompute block indices for each token
        block_indices = torch.zeros(total_padded_length, device=device, dtype=torch.long)

        # First image gets special block index -1 (conditioning, cannot attend to anything)
        block_indices[first_image_start:first_image_end] = -1

        # Assign block indices for image blocks (0 to num_image_blocks-1)
        for block_idx in range(num_image_blocks):
            start_idx = image_blocks_start + block_idx * num_frame_per_block * frame_seqlen
            end_idx = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            block_indices[start_idx:end_idx] = block_idx

        # Assign block indices for action tokens (0 to num_action_blocks-1)
        for block_idx in range(num_action_blocks):
            start_idx = action_start + block_idx * num_action_per_block
            end_idx = action_start + (block_idx + 1) * num_action_per_block
            block_indices[start_idx:end_idx] = block_idx

        # Assign block indices for state tokens (0 to num_state_blocks-1)
        for block_idx in range(num_state_blocks):
            start_idx = state_start + block_idx * num_state_per_block
            end_idx = state_start + (block_idx + 1) * num_state_per_block
            block_indices[start_idx:end_idx] = block_idx

        # Padding tokens get block index of last block + 1 (won't attend to anything)
        block_indices[total_length:] = num_image_blocks

        def attention_mask(b, h, q_idx, kv_idx):
            # Self-attention
            self_attn = (q_idx == kv_idx)

            # Determine which modality q and kv belong to
            q_is_first_image = (q_idx >= first_image_start) & (q_idx < first_image_end)
            q_is_image_block = (q_idx >= image_blocks_start) & (q_idx < image_blocks_end)
            q_is_action = (q_idx >= action_start) & (q_idx < action_end)
            q_is_state = (q_idx >= state_start) & (q_idx < state_end)

            kv_is_first_image = (kv_idx >= first_image_start) & (kv_idx < first_image_end)
            kv_is_image_block = (kv_idx >= image_blocks_start) & (kv_idx < image_blocks_end)
            kv_is_action = (kv_idx >= action_start) & (kv_idx < action_end)
            kv_is_state = (kv_idx >= state_start) & (kv_idx < state_end)

            q_block = block_indices[q_idx]
            kv_block = block_indices[kv_idx]

            # First image query (conditioning) - cannot attend to anything
            first_image_mask = q_is_first_image & False

            # Image block query
            image_to_first = q_is_image_block & kv_is_first_image  # Image block to first image: always allowed
            image_to_image = q_is_image_block & kv_is_image_block & (kv_block <= q_block)  # Image block to image block: can attend to current and previous image blocks
            image_to_action = q_is_image_block & kv_is_action & (kv_block == q_block)  # Image block to action: can attend to current action block
            image_to_state = q_is_image_block & kv_is_state & (kv_block == q_block)  # Image block to state: can attend to current state block

            image_block_mask = image_to_first | image_to_image | image_to_action | image_to_state

            # Action query
            action_to_image = q_is_action & kv_is_image_block & (kv_block <= q_block)  # Action to image block: can attend to current and all previous image blocks
            action_to_action = q_is_action & kv_is_action & (kv_block == q_block)  # Action to action: only same block
            action_to_state = q_is_action & kv_is_state & (kv_block == q_block)  # Action to state: only same block
            action_to_first = q_is_action & kv_is_first_image  # Action to first image: always allowed

            action_mask = action_to_image | action_to_action | action_to_state | action_to_first

            # State query (conditioning) - cannot attend to anything
            state_mask = q_is_state & False

            # Combine all masks
            return self_attn | first_image_mask | image_block_mask | action_mask | state_mask

        block_mask = create_block_mask(
            attention_mask, B=None, H=None,
            Q_LEN=total_padded_length,
            KV_LEN=total_padded_length,
            _compile=False, device=device
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Created blockwise causal attention mask:")
            print(f"  first_image_tokens={first_image_len} (conditioning)")
            print(f"  num_image_blocks={num_image_blocks} (blocks of {num_frame_per_block * frame_seqlen})")
            print(f"  num_action_blocks={num_action_blocks} (blocks of {num_action_per_block})")
            print(f"  num_state_blocks={num_state_blocks} (blocks of {num_state_per_block})")
            print(f"  total_length={total_length}, padded_length={padded_length}")
            print(block_mask)

            # Debug: materialize a small slice of the mask into 0/1 strings
            try:
                dense_mask = create_mask(
                    attention_mask,
                    B=None,
                    H=None,
                    Q_LEN=total_padded_length,
                    KV_LEN=total_padded_length,
                    device=device,
                )[0, 0]  # [Q, K]
                preview_q = min(979, dense_mask.shape[0])
                preview_k = min(979, dense_mask.shape[1])
                print("Block mask (preview):")
                for qi in range(preview_q):
                    row = dense_mask[qi, :preview_k].to(torch.int8).tolist()
                    print(" ".join(str(int(v)) for v in row))
            except Exception as err:
                print("[warn] Failed to materialize block mask preview:", err)

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(self.local_attn_size * frame_seqlen/128) * 128 - total_length
        # padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)


        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(local_attn_size * frame_seqlen/128) * 128 - total_length
        # padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_blocks_full(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        embodiment_id: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        past_actions: torch.Tensor | None,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """
        一次full arch的dit forward
        """
        video_x = x.flatten(start_dim=2).transpose(1, 2)
        batch_size = video_x.shape[0]
        if video_x.shape[1] != seq_len:
            raise ValueError(f"Expected {seq_len} video tokens, got {video_x.shape[1]}.")
        if action is None or timestep_action is None or state is None:
            raise ValueError("Full action architecture requires action, timestep_action, and state.")

        embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=video_x.device)
        action_x = self.action_encoder(action, timestep_action, embodiment_id)
        state_x = self.state_encoder(state, embodiment_id)
        past_x = None
        past_timestep = None
        if past_actions is not None:
            if past_actions.shape[-1] != self.action_dim:
                raise ValueError("past_actions action dimension does not match the model.")
            past_timestep = torch.zeros(
                past_actions.shape[:2], device=past_actions.device, dtype=timestep_action.dtype
            )
            past_x = self.action_encoder(past_actions, past_timestep, embodiment_id)

        frame_count = timestep.shape[1]
        video_timestep = timestep.unsqueeze(-1).expand(batch_size, frame_count, seq_len // frame_count).reshape(batch_size, -1)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            video_embedding = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, video_timestep.flatten()).float()
            ).unflatten(0, (batch_size, -1))
            video_e = self.time_projection(video_embedding)
        video_e = video_e.unflatten(2, (6, self.dim))

        stride = timestep_action.shape[1] // state_x.shape[1]
        state_timestep = timestep_action[:, ::stride]
        action_timestep = torch.cat([timestep_action, state_timestep], dim=1)
        if past_timestep is not None:
            action_timestep = torch.cat([action_timestep, past_timestep], dim=1)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_embedding = self.action_time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, action_timestep.flatten()).float()
            )
            action_e = self.action_time_projection(action_embedding)
        action_e = action_e.unflatten(0, (batch_size, -1)).unflatten(2, (6, self.action_hidden_size))

        context = self.text_embedding(context)
        updated_kv_caches = []
        for block_index, block in enumerate(self.blocks):
            if not isinstance(block, CausalWanFullActionAttentionBlock):
                raise RuntimeError("Full action architecture was paired with a split transformer block.")
            video_result, action_result, updated_cache = block(
                video_x=video_x,
                action_x=action_x,
                state_x=state_x,
                e_video=video_e,
                e_action=action_e,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                kv_cache=kv_cache[block_index],
                current_start_frame=current_start_frame,
                is_tf=False,
                clean_video_length=0,
                past_action_x=past_x,
            )
            video_x = video_result
            action_x = action_result[:, :action_x.shape[1]]
            state_start = action_x.shape[1]
            state_x = action_result[:, state_start:state_start + state_x.shape[1]]
            if past_x is not None:
                past_x = action_result[:, state_start + state_x.shape[1]:]
            updated_kv_caches.append(updated_cache)

        action_noise_pred = self.action_decoder(action_x, embodiment_id)
        x_video = self.head(video_x, video_embedding.unsqueeze(2))
        return x_video, action_noise_pred, updated_kv_caches


    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        past_actions: torch.Tensor | None,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
        ff_tokens: torch.Tensor | None = None,
        ff_freqs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        r"""
        输入 patch 后 video latent；输出 video/action 预测和更新后的 video KV cache。
        """
        if self.action_architecture == 'full' and any(
            value is not None for value in (action, timestep_action, state)
        ):  # 确认模型结构，并且排除video only调用
            return self._forward_blocks_full(
                x=x,
                seq_len=seq_len,
                freqs=freqs,
                timestep=timestep,
                context=context,
                embodiment_id=embodiment_id,
                action=action,
                timestep_action=timestep_action,
                state=state,
                past_actions=past_actions,
                kv_cache=kv_cache,
                current_start_frame=current_start_frame,
            )
        x = x.flatten(start_dim=2).transpose(1, 2)

        B = x.shape[0]
        F = timestep.shape[1]

        control_inputs = (action, timestep_action, state)
        has_any_control_input = any(value is not None for value in control_inputs)
        has_all_control_inputs = all(value is not None for value in control_inputs)
        if has_any_control_input and not has_all_control_inputs:
            raise ValueError(
                "action, timestep_action, and state must either all be provided "
                "or all be None."
            )

        if action is not None:
            # action/state head 当前只有 category 0；此处置零用于匹配单 embodiment 参数布局。
            embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            state_features = self.state_encoder(state, embodiment_id)

            action_length = action_features.shape[1]
            extra_register = torch.cat([action_features, state_features], dim=1)
            action_register_length = extra_register.shape[1]
            x = torch.cat([x, extra_register], dim=1)
        else:
            state_features = None
            action_length = 0
            action_register_length = None

        past_action_context_length = 0
        past_action_timestep = None
        if past_actions is not None:
            if action is None:
                raise ValueError("past_actions requires current action/state inputs.")
            if past_actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"past_actions dim mismatch: got {past_actions.shape[-1]}, expected {self.action_dim}"
                )
            if past_actions.shape[1] % self.num_action_per_block != 0:
                raise ValueError(
                    "past_actions length must be divisible by num_action_per_block, "
                    f"got {past_actions.shape[1]} and {self.num_action_per_block}."
                )
            past_action_timestep = torch.zeros(
                past_actions.shape[:2],
                device=past_actions.device,
                dtype=timestep_action.dtype,
            )
            past_action_features = self.action_encoder(
                past_actions,
                past_action_timestep,
                embodiment_id,
            )
            if torch.rand((), device=past_actions.device).item() < 0.001:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(
                    f"[PAST_ACTION_SAMPLE] rank={rank} past_actions={tuple(past_actions.shape)} "
                    f"past_action_features={tuple(past_action_features.shape)}",
                    flush=True,
                )
            past_action_context_length = past_action_features.shape[1]
            x = torch.cat([x, past_action_features], dim=1)

        ff_context_length = 0
        if ff_tokens is not None:
            if ff_freqs is None:
                raise ValueError("ff_tokens requires ff_freqs.")
            ff_context_length = ff_tokens.shape[1]
            x = torch.cat([x, ff_tokens.to(device=x.device, dtype=x.dtype)], dim=1)

        # time embeddings - expanded per token in sequence
        timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)
            if past_action_timestep is not None:
                timestep = torch.cat([timestep, past_action_timestep], dim=1)
        if ff_context_length:
            ff_timestep = torch.zeros(
                (B, ff_context_length),
                device=timestep.device,
                dtype=timestep.dtype,
            )
            timestep = torch.cat([timestep, ff_timestep], dim=1)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).float())
            e = e.unflatten(dim=0, sizes=(B, -1))
            e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # context
        context = self.text_embedding(context)
        image_context_len = 0

        if clip_feature is not None:
            image_context_proj = self.vae_img_emb if self.image_context_type == "vae" else self.img_emb
            clip_embedding = image_context_proj(clip_feature)
            image_context_len = clip_embedding.shape[1]
            context = torch.cat([clip_embedding, context], dim=1)

        image_context_query_mask = None
        if image_context_len > 0 and self.image_context_token_scope != "all":
            state_len = state_features.shape[1] if state_features is not None else 0
            image_context_query_mask = build_image_context_query_mask(
                seq_len=x.shape[1],
                video_len=seq_len,  # noisy，clean的在cache里
                action_len=action_length,
                state_len=state_len,
                extra_len=past_action_context_length + ff_context_length,
                token_scope=self.image_context_token_scope,
                device=x.device,
            )

        updated_kv_caches: list[torch.Tensor] = []
        for block_index, block in enumerate(self.blocks):
            x, updated_kv_cache = block(
                x=x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                image_context_len=image_context_len,
                action_register_length=action_register_length,
                past_action_context_length=past_action_context_length,
                image_context_query_mask=image_context_query_mask,
                kv_cache=kv_cache[block_index],
                current_start_frame=current_start_frame,
                ff_context_length=ff_context_length,
                ff_freqs=ff_freqs,
            )
            updated_kv_caches.append(updated_kv_cache)

        if action is not None:
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # Unpatchify video-only tokens
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]
        x_video = self.head(x_video, e_video.unsqueeze(2))

        return x_video, action_noise_pred, updated_kv_caches


    def _forward_inference_trt(
        self,
        x,
        timestep,
        context,
        kv_cache_packed: torch.Tensor,
        y,
        clip_feature,
        action,
        timestep_action,
        state,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:


        frame_seqlen = 880
        seq_len = 2*frame_seqlen
        kv_cache_seq_len = kv_cache_packed.shape[3]
        current_start_frame =  kv_cache_seq_len // frame_seqlen

        kv_cache_list = []
        for block_index in range(len(self.blocks)):
            kv_cache_list.append(kv_cache_packed[block_index])

        x_video, action_noise_pred, _ = self._forward_inference(
            x=x,
            timestep=timestep,
            context=context,
            seq_len=int(seq_len),
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=y,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            current_start_frame = current_start_frame,
        )

        return x_video, action_noise_pred

    def _forward_inference_trt_droid(
        self,
        x,
        timestep,
        context,
        kv_cache_packed: torch.Tensor,
        y,
        clip_feature,
        action,
        timestep_action,
        state,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:


        frame_seqlen = 880
        seq_len = 2*frame_seqlen
        kv_cache_seq_len = kv_cache_packed.shape[3]
        current_start_frame =  kv_cache_seq_len // frame_seqlen

        kv_cache_list = []
        for block_index in range(len(self.blocks)):
            kv_cache_list.append(kv_cache_packed[block_index])

        x_video, action_noise_pred, _ = self._forward_inference(
            x=x,
            timestep=timestep,
            context=context,
            seq_len=int(seq_len),
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=y,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            current_start_frame = current_start_frame,
        )

        return x_video, action_noise_pred


    def _forward_inference(
        self,
        x,
        timestep,
        context,
        seq_len,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[torch.Tensor],
        current_start_frame: int,
        y=None,
        clip_feature=None,
        action=None,
        timestep_action=None,
        state=None,
        embodiment_id=None,
        past_actions=None,
        ff_x=None,
        ff_rope_frame_index: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            timestep (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            action (Tensor, *optional*):
                Action tensor of shape [B, H, D]
            state (Tensor, *optional*):
                State tensor of shape [B, H, D]
            embodiment_id (Tensor, *optional*):
                Embodiment ID tensor of shape [B]
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            clip_feature (Tensor, *optional*):
                CLIP image features for image-to-video mode
            timestep_action (Tensor, *optional*):
                Action timestep tensor of shape [B]
            ff_x (Tensor, *optional*):
                Episode 首帧 VAE latent [B,C,T,H,W]；会 patchify 成尾部 context token。
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if clip_feature is not None:
            raise ValueError(
                "Image encoder conditioning is no longer supported; clip_feature must be None."
            )
        if ff_x is not None:
            raise ValueError(
                "Episode first-frame tokens are no longer supported; ff_x must be None."
            )
        if self.model_type == 'i2v':
            assert clip_feature is not None and y is not None
        assert context.shape[1] == self.text_len

        # Concat [x; y] only when pretrained that way (14B). 5B uses latent only, first-frame via CLIP.
        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # embeddings
        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)

        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=current_start_frame,
        )

        ff_tokens = None
        ff_freqs = None
        if ff_x is not None:
            if ff_rope_frame_index is None:
                raise ValueError("ff_x requires ff_rope_frame_index.")
            ff_x = self.patch_embedding(ff_x)
            ff_grid_size = torch.tensor(ff_x.shape[2:], dtype=torch.long)
            ff_frames = int(ff_grid_size[0].item())
            if ff_rope_frame_index < 0 or ff_rope_frame_index + ff_frames > self.freqs[0].shape[0]:
                raise ValueError(
                    "episode first-frame RoPE index is out of range: "
                    f"start={ff_rope_frame_index}, frames={ff_frames}, "
                    f"table={self.freqs[0].shape[0]}."
                )
            ff_freqs = self._create_freqs(
                grid_size=ff_grid_size,
                start_frame=int(ff_rope_frame_index),
            )
            ff_tokens = ff_x.flatten(start_dim=2).transpose(1, 2)

        x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(
            x=x,
            seq_len=seq_len,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            embodiment_id=embodiment_id,
            action=action,
            timestep_action=timestep_action,
            state=state,
            past_actions=past_actions,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
            ff_tokens=ff_tokens,
            ff_freqs=ff_freqs,
        )

        # Copy the updated KV caches back to the original KV cache.
        x_video = x_video.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()

        # Unpatchify outside the compiled region.
        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred, updated_kv_caches

    def _forward_train(
        self,
        x,
        timestep,
        timestep_action,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        y=None,
        clip_feature=None,
        action=None,
        state=None,
        embodiment_id=None,
        past_actions=None,
        past_action_keep_tokens=None,
        action_feature_mask=None,
    ):
        # RLinf SFT replaces this method at model construction time with
        # rlinf.models.embodiment.dreamzero.patch.wan_causal_model_forward_train._forward_train.
        # Keep the signature here as the patch target, but fail fast if this
        # DreamZero-dev fallback is reached directly.
        raise RuntimeError(
            "CausalWanModel._forward_train in DreamZero-dev is intentionally disabled. "
            "Use RLinf DreamZero SFT so this method is patched before training."
        )

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_size):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor):
                Patchified features, with shape [B, L, C_out * prod(patch_size)].
            grid_size (Tensor):
                Spatial-temporal grid dimensions before patching, with shape [3]
                (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            Tensor:
                Reconstructed video tensors with shape [B, C_out, F, H / 8, W / 8]
        """
        B = x.shape[0]
        c = self.out_dim
        grid_size = grid_size.tolist()
        assert x.shape[1] == math.prod(grid_size)

        # Keep the FP32 head output intact, instead of letting an outer BF16 autocast context quantize it.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float().view(B, *grid_size, *self.patch_size, c)
            x = torch.einsum('bfhwpqrc->bcfphqwr', x)
            x = x.reshape(B, c, *[i * j for i, j in zip(grid_size, self.patch_size)])
            return x

    def _create_freqs(
        self,
        grid_size: torch.Tensor,
        start_frame: int,
    ):
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state.device != device:
            self.freqs_state = self.freqs_state.to(device)

        f, h, w = grid_size.tolist()

        freqs = torch.cat(
            [
                self.freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1
        ).reshape(f * h * w, 1, -1)

        return freqs

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        if self.action_architecture == 'full':
            for m in self.action_time_embedding.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
