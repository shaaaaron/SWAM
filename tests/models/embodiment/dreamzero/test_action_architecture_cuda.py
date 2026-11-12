"""Action regression checks with CPU fallback; FSDP requires torchrun and CUDA."""

import ast
import copy
import importlib.util
import inspect
import os
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.distributed as dist

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel
from groot.vla.model.dreamzero.modules import wan2_1_submodule

ROOT = Path(__file__).resolve().parents[4]
PATCH = ROOT / "rlinf/models/embodiment/dreamzero/patch/wan_causal_model_forward_train.py"
spec = importlib.util.spec_from_file_location("action_train_patch", PATCH)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)
CausalWanModel._forward_train = patch._forward_train


def make_model(architecture=None, *, action_condition_on_text=True):
    architecture_kwargs = {} if architecture is None else {"action_architecture": architecture}
    if architecture == "full":
        architecture_kwargs.update(
            action_hidden_size=8,
            action_ffn_dim=24,
            action_condition_on_text=action_condition_on_text,
        )
    with mock.patch.dict(os.environ, {"ATTENTION_BACKEND": "torch"} if DEVICE == "cpu" else {}):
        model = CausalWanModel(
            model_type="t2v", patch_size=(1, 1, 1), frame_seqlen=4,
            text_len=4, in_dim=2, dim=16, ffn_dim=32, freq_dim=8,
            text_dim=4, out_dim=2, num_heads=2, num_layers=2,
            max_chunk_size=2, num_frame_per_block=1, action_dim=2,
            max_state_dim=2, hidden_size=8, num_action_per_block=2,
            num_state_per_block=1, concat_first_frame_latent=False,
            **architecture_kwargs, video_condition_on_action_state=False,
        ).to(DEVICE)
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.1)
    return model


def inputs(first_frame=False):
    values = dict(
        x=torch.randn(1, 2, 3, 2, 2, device=DEVICE),
        clean_x=torch.randn(1, 2, 3, 2, 2, device=DEVICE),
        context=torch.randn(1, 4, 4, device=DEVICE),
        timestep=torch.rand(1, 3, device=DEVICE),
        timestep_action=torch.rand(1, 4, device=DEVICE),
        action=torch.randn(1, 4, 2, device=DEVICE),
        state=torch.randn(1, 2, 2, device=DEVICE), seq_len=12,
        embodiment_id=torch.zeros(1, device=DEVICE, dtype=torch.long),
    )
    if first_frame:
        values["ff_x"] = torch.randn(1, 2, 1, 2, 2, device=DEVICE)
    return values


def backward(model, values):
    model.zero_grad(set_to_none=True)
    video, action = model(**values)
    loss = video.square().mean() + action.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()
             if p.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    return loss.detach(), grads


def _cpu_text_attention(q, k, v, k_lens=None):
    assert k_lens is None
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2).contiguous()


def test_default_architecture_is_legacy_base():
    model = make_model()
    assert model.action_architecture == "base"
    assert model.config.action_architecture == "base"
    assert model.action_hidden_size == model.dim
    assert model.action_ffn_dim == model.ffn_dim
    assert all(not block.self_attn.split_video_control_attention for block in model.blocks)
    assert not hasattr(model, "action_time_embedding")
    assert not any(".self_attn.control_" in key for key in model.state_dict())
    make_model("base").load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize("preset", [
    "examples/embodiment/config/model/dreamzero_5b.yaml",
    "examples/sft/config/model/dreamzero_5b.yaml",
    "examples/sft/config/model/dreamzero_5b_wo_img.yaml",
])
def test_base_presets_omit_action_architecture_settings(preset):
    from omegaconf import OmegaConf

    diffusion = OmegaConf.load(ROOT / preset).action_head_cfg.config.diffusion_model_cfg
    assert not {"action_architecture", "action_hidden_size", "action_ffn_dim"}.intersection(diffusion)
    assert inspect.signature(CausalWanModel.__init__).parameters["action_architecture"].default == "base"


def test_full_cross_attention_norm_native_initialization():
    model = make_model("full")
    norm_keys = {key for key in model.state_dict() if ".action_norm3." in key}
    assert len(norm_keys) == 2 * len(model.blocks)
    for block in model.blocks:
        assert block.norm3.elementwise_affine
        assert block.action_norm3.elementwise_affine
        torch.testing.assert_close(block.action_norm3.weight, torch.ones_like(block.action_norm3.weight))
        torch.testing.assert_close(block.action_norm3.bias, torch.zeros_like(block.action_norm3.bias))
    old_full_weights = {key: value for key, value in model.state_dict().items() if key not in norm_keys}
    restored = make_model("full")
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(old_full_weights, strict=True)


@mock.patch.object(
    wan2_1_submodule, "flash_attention",
    _cpu_text_attention if DEVICE == "cpu" else wan2_1_submodule.flash_attention,
)
def test_full_can_skip_action_text_cross_attention():
    torch.manual_seed(7)
    reference = make_model("full")
    model = make_model("full", action_condition_on_text=False)
    model.load_state_dict(reference.state_dict(), strict=True)
    model.gradient_checkpointing = False

    assert reference.action_condition_on_text is True
    assert model.config.action_condition_on_text is False
    assert model.action_condition_on_text is False
    assert reference.state_dict().keys() == model.state_dict().keys()
    assert all(not block.action_condition_on_text for block in model.blocks)

    for block in model.blocks:
        block.action_cross_attn.forward = mock.Mock(
            side_effect=AssertionError("Action text cross-attention must be skipped")
        )

    values = inputs()
    model.zero_grad(set_to_none=True)
    _, action = model(**values)
    action.square().mean().backward()
    assert all(
        parameter.grad is None
        for block in model.blocks
        for parameter in block.action_cross_attn.parameters()
    )
    assert model.blocks[0].cross_attn.q.weight.grad is not None
    assert model.blocks[0].cross_attn.q.weight.grad.abs().sum() > 0

    model.eval()
    cache = [torch.empty(2, 1, 0, 2, 8, device=DEVICE) for _ in model.blocks]
    common = dict(
        x=values["x"][:, :, :1],
        timestep=values["timestep"][:, :1],
        context=values["context"],
        seq_len=4,
        crossattn_cache=[None] * len(model.blocks),
    )
    with torch.no_grad():
        _, action, cache = model(**common, kv_cache=cache, current_start_frame=0)
        assert action is None
        _, action, cache = model(
            **common,
            kv_cache=cache,
            current_start_frame=1,
            action=values["action"][:, :2],
            state=values["state"][:, :1],
            timestep_action=values["timestep_action"][:, :2],
        )
    assert action.shape == (1, 2, 2)
    assert torch.isfinite(action).all()


def test_full_no_action_text_config_is_explicit():
    from omegaconf import OmegaConf

    for name in (
        "20-4-v0a0_wo_img+umi_robo_ego_30k_full_no_action_text.yaml",
        "20-4-v0a0_wo_img+clean_adamw_4e-5_full_no_action_text.yaml",
    ):
        config = OmegaConf.load(ROOT / "examples/sft/config" / name)
        diffusion = config.actor.model.action_head_cfg.config.diffusion_model_cfg
        assert diffusion.action_architecture == "full"
        assert diffusion.action_condition_on_text is False
        assert config.actor.model.action_head_cfg.config.video_condition_on_action_state is False


@pytest.mark.parametrize("architecture", ["base", "split", "full"])
@mock.patch.object(
    wan2_1_submodule, "flash_attention",
    _cpu_text_attention if DEVICE == "cpu" else wan2_1_submodule.flash_attention,
)
def test_past_action_context_training_and_inference(architecture):
    torch.manual_seed(7)
    model = make_model(architecture)
    model.gradient_checkpointing = False
    values = inputs()
    history = torch.randn_like(values["action"], requires_grad=True)
    values["past_actions"] = history
    backward(model, values)
    assert history.grad is not None
    assert history.grad[:, :2].abs().sum() > 0
    # Only the second current block sees history, and only its preceding block.
    assert torch.count_nonzero(history.grad[:, 2:]) == 0

    model.eval()
    cache = [torch.empty(2, 1, 0, 2, 8, device=DEVICE) for _ in model.blocks]
    common = dict(x=values["x"][:, :, :1], timestep=values["timestep"][:, :1],
                  context=values["context"], seq_len=4, crossattn_cache=[None] * len(model.blocks))
    with torch.no_grad():
        for start in (0, 1):
            _, action, cache = model(**common, kv_cache=cache, current_start_frame=start)
            assert action is None
        _, action, cache = model(
            **common, kv_cache=cache, current_start_frame=2,
            action=values["action"][:, :2], state=values["state"][:, :1],
            timestep_action=values["timestep_action"][:, :2], past_actions=history[:, :2],
        )
    assert action.shape == (1, 2, 2) and torch.isfinite(action).all()
    # The history must not contribute tokens to the rolling video cache.
    assert all(item.shape == (2, 1, 12, 2, 8) for item in cache)


@mock.patch.object(
    wan2_1_submodule, "flash_attention",
    _cpu_text_attention if DEVICE == "cpu" else wan2_1_submodule.flash_attention,
)
def test_gradients_and_inference():
    for architecture in ("base", "split", "full"):
        torch.manual_seed(7)
        reference = make_model(architecture)
        reference.gradient_checkpointing = False
        values = inputs()
        expected_loss, expected_grads = backward(reference, values)
        for reentrant in (False, True):
            model = copy.deepcopy(reference)
            model.gradient_checkpointing = True
            model.gradient_checkpointing_use_reentrant = reentrant
            calls = []
            handles = [block.register_forward_pre_hook(
                lambda module, args, index=index: calls.append(index)
            ) for index, block in enumerate(model.blocks)]
            loss, grads = backward(model, values)
            torch.testing.assert_close(loss, expected_loss)
            assert grads.keys() == expected_grads.keys()
            # Reentrant BF16 recomputation can shift small text gradients by one quantization step.
            rtol, atol = (2e-3, 3e-4) if DEVICE == "cuda" and reentrant else (1e-4, 1e-5)
            for name, grad in grads.items():
                torch.testing.assert_close(grad, expected_grads[name],
                                           rtol=rtol, atol=atol, msg=name)
            assert all(calls.count(i) >= 2 for i in range(2)), calls
            for handle in handles:
                handle.remove()
            del model
        print(f"PASS {architecture}: gradients, hooks", flush=True)

        reference.eval()
        reference.gradient_checkpointing = False
        cache = [torch.empty(2, 1, 0, 2, 8, device=DEVICE) for _ in range(2)]
        common = dict(x=values["x"][:, :, :1], timestep=values["timestep"][:, :1],
                      context=values["context"], seq_len=4, crossattn_cache=[None] * 2)
        with torch.no_grad():
            video, action, cache = reference(**common, kv_cache=cache, current_start_frame=0)
            assert action is None and video.shape == (1, 2, 1, 2, 2)
            assert all(c.shape == (2, 1, 4, 2, 8) for c in cache)
            video, action, cache = reference(
                **common, kv_cache=cache, current_start_frame=1,
                action=values["action"][:, :2], state=values["state"][:, :1],
                timestep_action=values["timestep_action"][:, :2],
            )
            assert action.shape == (1, 2, 2) and torch.isfinite(action).all()
            assert all(c.shape == (2, 1, 8, 2, 8) for c in cache)
        restored = make_model(architecture)
        restored.load_state_dict(reference.state_dict(), strict=True)
        print(f"PASS {architecture}: observed-video cache, action inference, state roundtrip", flush=True)


@pytest.mark.parametrize("architecture", ["base", "split", "full"])
def test_rejects_first_frame_inputs(architecture):
    model = make_model(architecture)
    values = inputs(first_frame=True)
    message = "ff_x must be None"
    with pytest.raises(ValueError, match=message):
        model(**values)
    with pytest.raises(ValueError, match=message):
        model(
            x=values["x"][:, :, :1], timestep=values["timestep"][:, :1],
            context=values["context"], seq_len=4, current_start_frame=0,
            kv_cache=[], crossattn_cache=[], ff_x=values["ff_x"],
        )


def test_full_rejects_first_frame_inputs():
    model = make_model("full")
    message = "does not support episode first-frame tokens"
    for video_x in (None, torch.empty(0)):
        for key, value in (("ff_x", torch.empty(0)), ("ff_freqs", torch.empty(0)),
                           ("ff_context_length", 1)):
            with pytest.raises(ValueError, match=message):
                model.blocks[0](video_x=video_x, **{key: value})
    for function in (model._forward_blocks_full, model.blocks[0].self_attn.forward_full,
                     patch._forward_train_full):
        assert not any(name.startswith("ff_") for name in inspect.signature(function).parameters)


def test_checkpoint_rules():
    source = ROOT / "rlinf/models/embodiment/dreamzero/__init__.py"
    tree = ast.parse(source.read_text())
    wanted = {"_align_checkpoint_tensor_shapes", "_validate_checkpoint_weight_keys"}
    functions = ast.Module(body=[node for node in tree.body
                                if isinstance(node, ast.FunctionDef) and node.name in wanted],
                           type_ignores=[])
    namespace = {}
    exec(compile(functions, str(source), "exec"), namespace)

    def wrap(model):
        policy = torch.nn.Module()
        policy.action_head = torch.nn.Module()
        policy.action_head.model = model
        return policy

    def load(model, weights):
        namespace["_align_checkpoint_tensor_shapes"](model, weights)
        incompatible = model.load_state_dict(weights, strict=False)
        namespace["_validate_checkpoint_weight_keys"](
            "yaml_only", incompatible.missing_keys, incompatible.unexpected_keys
        )

    for architecture in ("base", "split", "full"):
        original = wrap(make_model(architecture))
        restored = wrap(make_model(architecture))
        weights = original.state_dict()
        load(restored, weights)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, weights[key])

        if architecture == "split":
            key = "action_head.model.blocks.0.self_attn.control_q.weight"
        elif architecture == "full":
            key = "action_head.model.blocks.0.action_norm3.weight"
        else:
            key = "action_head.model.blocks.0.self_attn.q.weight"
        with pytest.raises(RuntimeError, match="missing keys"):
            load(restored, {k: v for k, v in weights.items() if k != key})
        with pytest.raises(RuntimeError, match="unexpected keys"):
            load(restored, {**weights, "unexpected.weight": torch.ones(2, device=DEVICE)})
        for mismatch_key in ("action_head.model.action_encoder.W1.W",
                             "action_head.model.patch_embedding.weight"):
            with pytest.raises(RuntimeError, match="Checkpoint shape mismatch"):
                load(restored, {**weights, mismatch_key: torch.zeros(1, device=DEVICE)})

    base_weights = wrap(make_model("base")).state_dict()
    for architecture in ("split", "full"):
        with pytest.raises(RuntimeError):
            load(wrap(make_model(architecture)), dict(base_weights))
    full_weights = wrap(make_model("full")).state_dict()
    load(wrap(make_model("full", action_condition_on_text=False)), dict(full_weights))
    print("PASS same-architecture reload and incompatible checkpoint rejection", flush=True)


def test_fsdp():
    if DEVICE != "cuda" or "RANK" not in os.environ:
        pytest.skip("Run the FSDP check using torchrun on CUDA")
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    cases = (
        ("split", "split", True),
        ("full", "full", True),
        ("full_no_action_text", "full", False),
    )
    for label, architecture, action_condition_on_text in cases:
        torch.manual_seed(7)
        model = make_model(
            architecture,
            action_condition_on_text=action_condition_on_text,
        )
        model.gradient_checkpointing = True
        model.gradient_checkpointing_use_reentrant = True
        for block in model.blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                video, action = model(**inputs())
            loss = video.float().square().mean() + action.float().square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        dist.barrier()
        if dist.get_rank() == 0:
            print(f"PASS {label}: {dist.get_world_size()}-rank FSDP2 BF16 train, backward, optimizer", flush=True)
        del optimizer, model
    dist.destroy_process_group()


if __name__ == "__main__":
    if DEVICE == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if "RANK" in os.environ:
        test_fsdp()
    else:
        test_full_can_skip_action_text_cross_attention()
        test_full_no_action_text_config_is_explicit()
        test_gradients_and_inference()
        for architecture in ("base", "split", "full"):
            test_rejects_first_frame_inputs(architecture)
        test_full_rejects_first_frame_inputs()
        test_checkpoint_rules()
