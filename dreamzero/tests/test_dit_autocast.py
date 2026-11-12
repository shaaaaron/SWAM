from types import SimpleNamespace

import pytest
import torch

from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import WANPolicyHead
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CausalHead,
    CausalWanModel,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA autocast")
@pytest.mark.parametrize("inference", [False, True])
def test_model_forward_does_not_set_autocast(inference):
    def check_autocast(*args, **kwargs):
        assert not torch.is_autocast_enabled("cuda")
        return "forwarded"

    model = SimpleNamespace(
        _forward_train=check_autocast,
        _forward_inference=check_autocast,
    )
    with torch.amp.autocast("cuda", enabled=False):
        result = CausalWanModel.forward(
            model, kv_cache=[] if inference else None
        )
    assert result == "forwarded"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA autocast")
@pytest.mark.parametrize("use_trt, update_kv_cache", [
    (False, False), (True, False), (True, True),
])
def test_diffusion_model_call_owns_bf16_autocast(use_trt, update_kv_cache):
    calls = []

    def model(*args, **kwargs):
        calls.append("model")
        assert torch.is_autocast_enabled("cuda")
        assert torch.get_autocast_dtype("cuda") == torch.bfloat16
        return (
            torch.ones(1, device="cuda"),
            torch.ones(1, device="cuda"),
            [torch.ones(1, device="cuda")],
        )

    def trt_engine(*args, **kwargs):
        calls.append("trt")
        assert not torch.is_autocast_enabled("cuda")
        return torch.ones(1, device="cuda"), torch.ones(1, device="cuda")

    def exchange(predictions):
        assert not torch.is_autocast_enabled("cuda")
        return predictions

    policy = SimpleNamespace(
        _device="cuda", model=model, trt_engine=trt_engine if use_trt else None,
        _exchange_predictions=exchange,
    )
    value = torch.ones(1, device="cuda")
    kv_caches = [[torch.zeros(1, device="cuda")]]
    with torch.amp.autocast("cuda", enabled=False):
        predictions = WANPolicyHead._run_diffusion_steps(
            policy,
            noisy_input=value,
            timestep=value,
            action=value,
            timestep_action=value,
            state=value,
            embodiment_id=value,
            context=[value],
            seq_len=1,
            y=value,
            clip_feature=value,
            kv_caches=kv_caches,
            crossattn_caches=[[]],
            kv_cache_metadata={"start_frame": 0, "update_kv_cache": update_kv_cache},
        )
    assert calls == ["trt" if use_trt and not update_kv_cache else "model"]
    assert len(predictions) == 1
    assert kv_caches[0][0].item() == (1 if update_kv_cache else 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA autocast")
def test_video_head_keeps_fp32_output_under_bf16_autocast():
    head = CausalHead(dim=8, out_dim=2, patch_size=(1, 1, 1)).to(
        device="cuda", dtype=torch.bfloat16
    )
    x = torch.randn(1, 1, 8, device="cuda", dtype=torch.bfloat16)
    embedding = torch.randn(1, 1, 1, 8, device="cuda", dtype=torch.float32)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = head(x, embedding)

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
