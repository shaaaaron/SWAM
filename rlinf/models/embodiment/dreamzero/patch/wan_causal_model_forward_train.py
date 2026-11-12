# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from groot.vla.model.dreamzero.modules.wan2_1_submodule import sinusoidal_embedding_1d
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    build_image_context_query_mask,
)

# RLinf replaces DreamZero's disabled training forward with this implementation.
# Gradient checkpointing supports both reentrant and non-reentrant execution and
# is controlled by the FSDP configuration. Registered in get_model.


def _full_action_time_embedding(self, timesteps, *, batch_size):
    with torch.amp.autocast('cuda', dtype=torch.float32):
        embedded = self.action_time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps.flatten()).float()
        )
        projected = self.action_time_projection(embedded)
    return projected.unflatten(0, (batch_size, -1)).unflatten(
        2, (6, self.action_hidden_size)
    )


def _forward_train_full(
    self,
    x,
    timestep,
    timestep_action,
    context,
    seq_len,
    clean_x=None,
    aug_t=None,
    y=None,
    action=None,
    state=None,
    embodiment_id=None,
    past_actions=None,
    past_action_keep_tokens=None,
    action_feature_mask=None,
):
    if action is None or timestep_action is None or state is None:
        raise ValueError("Full action architecture requires action, timestep_action, and state.")
    if self.model_type == "i2v":
        raise ValueError("Full action architecture does not support image-conditioned i2v.")

    if y is not None and self.concat_first_frame_latent:
        x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)
    noisy_patch = self.patch_embedding(x)
    noisy_video = noisy_patch.flatten(start_dim=2).transpose(1, 2)
    grid_size = torch.tensor(noisy_patch.shape[2:], dtype=torch.long, device=x.device)
    freqs = self._create_freqs(grid_size=grid_size, start_frame=0)
    assert noisy_video.shape[1] == seq_len

    batch_size = noisy_video.shape[0]
    video_x = noisy_video
    clean_video_length = 0
    if clean_x is not None:
        if y is not None and self.concat_first_frame_latent:
            clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
        clean_video = self.patch_embedding(clean_x).flatten(start_dim=2).transpose(1, 2)
        assert clean_video.shape[1] == seq_len
        clean_video_length = clean_video.shape[1]
        video_x = torch.cat([clean_video, noisy_video], dim=1)

    embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=noisy_video.device)
    action_x = self.action_encoder(
        action, timestep_action, embodiment_id, action_feature_mask=action_feature_mask
    )
    state_x = self.state_encoder(state, embodiment_id)
    past_x = None
    past_timestep = None
    if past_actions is not None:
        past_timestep = torch.zeros(
            past_actions.shape[:2], device=past_actions.device, dtype=timestep_action.dtype
        )
        past_x = self.action_encoder(past_actions, past_timestep, embodiment_id)

    frame_count = timestep.shape[1]
    video_timestep = timestep.unsqueeze(-1).expand(
        batch_size, frame_count, seq_len // frame_count
    ).reshape(batch_size, -1)
    if clean_video_length:
        if aug_t is None:
            aug_t = torch.zeros_like(video_timestep)
        video_timestep = torch.cat([aug_t, video_timestep], dim=1)
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
    action_e = _full_action_time_embedding(self, action_timestep, batch_size=batch_size)

    assert context.shape[1] == self.text_len
    context = self.text_embedding(context)

    for block in self.blocks:
        if not hasattr(block, "action_ffn"):
            raise RuntimeError("Full action architecture was paired with a split transformer block.")

        static_kwargs = dict(
            freqs=freqs,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            kv_cache=None,
            current_start_frame=0,
            is_tf=True,
            clean_video_length=clean_video_length,
            past_action_keep_tokens=past_action_keep_tokens,
        )

        # Bind the layer and pass every differentiable context tensor explicitly.
        # Reentrant recomputation must use this layer's inputs and FSDP hooks.
        def run_block(video_input, action_input, state_input, video_condition,
                      action_condition, text_context, past_action,
                      module=block, block_kwargs=static_kwargs):
            return module(
                video_x=video_input,
                action_x=action_input,
                state_x=state_input,
                e_video=video_condition,
                e_action=action_condition,
                context=text_context,
                past_action_x=past_action,
                **block_kwargs,
            )[:2]

        block_inputs = (video_x, action_x, state_x, video_e, action_e,
                        context, past_x)
        use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing
        if use_ckpt:
            video_result, action_result = torch.utils.checkpoint.checkpoint(
                run_block,
                *block_inputs,
                use_reentrant=getattr(self, "gradient_checkpointing_use_reentrant", True),
            )
        else:
            video_result, action_result = run_block(*block_inputs)

        video_x = video_result
        action_length = action_x.shape[1]
        state_length = state_x.shape[1]
        action_x = action_result[:, :action_length]
        state_x = action_result[:, action_length:action_length + state_length]
        if past_x is not None:
            past_x = action_result[:, action_length + state_length:]

    noisy_video_result = video_x[:, clean_video_length:clean_video_length + seq_len]
    action_noise_pred = self.action_decoder(action_x, embodiment_id)
    video_e_noisy = video_embedding[:, clean_video_length:clean_video_length + seq_len]
    x_video = self.head(noisy_video_result, video_e_noisy.unsqueeze(2))
    video_noise_pred = self.unpatchify(x_video, grid_size)
    return video_noise_pred, action_noise_pred


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
    ff_x=None,
    ff_rope_frame_index=1000,
    action_feature_mask=None,
):
    if clip_feature is not None:
        raise ValueError(
            "Image encoder conditioning is no longer supported; clip_feature must be None."
        )
    if ff_x is not None:
        raise ValueError(
            "Episode first-frame tokens are no longer supported; ff_x must be None."
        )
    if self.action_architecture == "full":
        return _forward_train_full(
            self,
            x=x,
            timestep=timestep,
            timestep_action=timestep_action,
            context=context,
            seq_len=seq_len,
            clean_x=clean_x,
            aug_t=aug_t,
            y=y,
            action=action,
            state=state,
            embodiment_id=embodiment_id,
            past_actions=past_actions,
            past_action_keep_tokens=past_action_keep_tokens,
            action_feature_mask=action_feature_mask,
        )
    if self.model_type == "i2v":
        assert clip_feature is not None and y is not None
    if self.action_architecture == "split" and (
        action is None or timestep_action is None or state is None
    ):
        raise ValueError(
            "Split-attention training requires action, timestep_action, and state."
        )
    if ff_x is not None and past_actions is not None:
        raise ValueError("episode first-frame token is not compatible with past_actions yet.")
    if ff_x is not None:
        assert self.image_context_token_scope == "all", (
            "episode first-frame token requires image_context_token_scope='all' in "
            "RLinf patched training."
        )

    if y is not None and self.concat_first_frame_latent:
        x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

    x = self.patch_embedding(x)
    grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
    freqs = self._create_freqs(grid_size=grid_size, start_frame=0)
    x = x.flatten(start_dim=2).transpose(1, 2)
    assert x.shape[1] == seq_len

    B = x.shape[0]
    F = timestep.shape[1]

    if action is not None:
        embodiment_id = (
            torch.tensor([0]).repeat(x.shape[0]).to(device=embodiment_id.device)
        )
        action_features = self.action_encoder(
            action,
            timestep_action,
            embodiment_id,
            action_feature_mask=action_feature_mask,
        )
        action_length = action_features.shape[1]
        state_features = self.state_encoder(state, embodiment_id)
        action_register = torch.cat([action_features, state_features], dim=1)
        action_register_length = action_register.shape[1]
        x = torch.cat([x, action_register], dim=1)
    else:
        action_features = None
        action_length = None
        state_features = None
        action_register = None
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
    ff_freqs = None
    if ff_x is not None:
        # 输入 ff_x 是 episode 原始第 0 帧的 VAE latent [B,C,T,H,W]；
        # 输出 ff_tokens append 到序列尾部，只作为 attention context，不参与预测 loss。
        ff_x = self.patch_embedding(ff_x)
        ff_grid_size = torch.tensor(ff_x.shape[2:], dtype=torch.long)
        ff_frames = int(ff_grid_size[0].item())
        if int(ff_rope_frame_index) + ff_frames > self.freqs[0].shape[0]:
            raise ValueError(
                "ff_rope_frame_index exceeds video RoPE table length: "
                f"index={ff_rope_frame_index}, ff_frames={ff_frames}, "
                f"table_len={self.freqs[0].shape[0]}."
            )
        ff_freqs = self._create_freqs(
            grid_size=ff_grid_size,
            start_frame=int(ff_rope_frame_index),
        )
        ff_tokens = ff_x.flatten(start_dim=2).transpose(1, 2)
        ff_context_length = ff_tokens.shape[1]
        x = torch.cat([x, ff_tokens], dim=1)

    timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
    timestep_original = timestep.clone()

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
            B,
            ff_context_length,
            device=x.device,
            dtype=timestep.dtype,
        )
        timestep = torch.cat([timestep, ff_timestep], dim=1)

    with torch.amp.autocast('cuda', dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).float()
        )
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
    e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

    assert context.shape[1] == self.text_len
    context = self.text_embedding(context)
    image_context_len = 0
    if clip_feature is not None:
        image_context_proj = self.vae_img_emb if self.image_context_type == "vae" else self.img_emb
        clip_embedding = image_context_proj(clip_feature)
        image_context_len = clip_embedding.shape[1]
        context = torch.cat([clip_embedding, context], dim=1)

    if clean_x is not None:
        if y is not None and self.concat_first_frame_latent:
            clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
        clean_x = self.patch_embedding(clean_x)
        clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
        assert clean_x.shape[1] == seq_len

        x = torch.cat([clean_x, x], dim=1)

        if aug_t is None:
            aug_t = torch.zeros_like(timestep_original)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).float()
            )
            e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
            e0_clean = self.time_projection(e_clean)
        e0_clean = e0_clean.unflatten(dim=2, sizes=(6, self.dim))
        e0 = torch.cat([e0_clean, e0], dim=1)

    image_context_query_mask = None
    if image_context_len > 0 and self.image_context_token_scope != "all":
        clean_len = clean_x.shape[1] if clean_x is not None else 0
        state_len = state_features.shape[1] if state_features is not None else 0
        image_context_query_mask = build_image_context_query_mask(
            seq_len=x.shape[1],
            video_len=clean_len + seq_len,
            action_len=action_length or 0,
            state_len=state_len,
            token_scope=self.image_context_token_scope,
            device=x.device,
        )

    kwargs = {
        "e": e0,
        "freqs": freqs,
        "freqs_action": self.freqs_action,
        "freqs_state": self.freqs_state,
        "action_register_length": action_register_length,
        "past_action_context_length": past_action_context_length,
        "context": context,
        "image_context_len": image_context_len,
        "image_context_query_mask": image_context_query_mask,
        "is_tf": clean_x is not None,
        "past_action_keep_tokens": past_action_keep_tokens,
        "ff_context_length": ff_context_length,
        "ff_freqs": ff_freqs,
    }

    def create_custom_forward(module):
        def custom_forward(*inputs, **kwargs):
            outputs, updated_kv_cache = module(*inputs, **kwargs)
            assert updated_kv_cache is None
            return outputs

        return custom_forward

    for block in self.blocks:
        use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing

        if use_ckpt:
            ckpt_use_reentrant = getattr(
                self, "gradient_checkpointing_use_reentrant", True
            )

            if ckpt_use_reentrant:
                # When gradient_checkpointing_use_reentrant=True,
                # torch.utils.checkpoint.checkpoint only accepts
                # positional arguments, not keyword arguments.
                # Positional order must match CausalWanAttentionBlock.forward:
                # x, e, freqs, freqs_action, freqs_state, action_register_length,
                # context, image_context_len, kv_cache, crossattn_cache,
                # current_start_frame, is_tf, past_action_context_length,
                # image_context_query_mask, past_action_keep_tokens,
                # ff_context_length, ff_freqs
                x, _ = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    e0,
                    freqs,
                    self.freqs_action,
                    self.freqs_state,
                    action_register_length,
                    context,
                    image_context_len,  # image_context_len
                    None,  # kv_cache
                    None,  # crossattn_cache
                    0,  # current_start_frame
                    clean_x is not None,  # is_tf
                    past_action_context_length,
                    image_context_query_mask,
                    past_action_keep_tokens,
                    ff_context_length,
                    ff_freqs,
                    use_reentrant=True,
                )
            else:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
        else:
            x, _ = block(x, **kwargs)

    if clean_x is not None:
        x = x[:, clean_x.shape[1] :]

    if action is not None:
        action_noise_pred = x[:, seq_len : seq_len + action_length]
        action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
    else:
        action_noise_pred = None

    x_video = x[:, :seq_len]
    e_video = e[:, :seq_len]
    x_video = self.head(x_video, e_video.unsqueeze(2))
    video_noise_pred = self.unpatchify(x_video, grid_size)

    return video_noise_pred, action_noise_pred
