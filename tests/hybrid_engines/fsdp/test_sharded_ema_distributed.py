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

"""Run with torchrun --standalone --nproc-per-node=8 on CUDA (PyTorch 2.6+)."""

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.tensor import DTensor

from rlinf.hybrid_engines.fsdp.ema import ShardedEMA, local_tensor
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
    validate_ema_config,
)
from rlinf.hybrid_engines.fsdp.strategy.fsdp2 import FSDP2Strategy
from rlinf.utils.ckpt_convertor.fsdp_convertor.convert_dcp_to_pt import (
    convert_state_dict,
)
from rlinf.utils.utils import warmup_optimizer_state


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(13, 31), torch.nn.Tanh(), torch.nn.Linear(31, 11)
        )
        self.frozen = torch.nn.Parameter(torch.randn(7, 9), requires_grad=False)
        self.register_buffer("sentinel", torch.tensor([123.0]))

    def forward(self, x):
        return self.net(x)


def expect_error(fn, match):
    try:
        fn()
    except ValueError as exc:
        assert match in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected failure: {match}")


def snapshot(model, ema):
    return (
        {n: local_tensor(p).clone() for n, p in model.named_parameters()},
        {k: {n: p.clone() for n, p in s.items()} for k, s in ema.shadows.items()},
    )


def check_config():
    cfg = OmegaConf.create(
        {
            "ema": {"enabled": True, "decays": [0.999, 0.9995]},
            "model": {"model_type": "dreamzero", "embodiment_tag": "robotwin"},
            "fsdp_config": {"strategy": "fsdp2", "grad_scaler": {"enabled": False}},
            "optim": {"optimizer": "adamw"},
        }
    )
    assert validate_ema_config(cfg)
    cfg.fsdp_config.grad_scaler.enabled = True
    expect_error(lambda: validate_ema_config(cfg), "grad_scaler.enabled=false")
    cfg.ema.enabled = False
    assert not validate_ema_config(cfg)  # 原有非 EMA 训练不受限制。
    cfg.ema.enabled = True
    cfg.fsdp_config.grad_scaler.enabled = False
    for key, value in [("start_step", 1000), ("dtype", "float64"), ("device", "cpu")]:
        cfg.ema[key] = value
        expect_error(lambda: validate_ema_config(cfg), "Unsupported EMA options")
        del cfg.ema[key]


def check_layout(shape, out):
    names = ("fsdp",) if len(shape) == 1 else ("ddp", "fsdp")
    mesh = init_device_mesh("cuda", shape, mesh_dim_names=names)
    torch.manual_seed(321)
    model = ToyModel().cuda()
    fully_shard(model, mesh=mesh)
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=0.02, fused=True
    )
    warmup_optimizer_state(optim)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
    ema = ShardedEMA(model, [0.999, 0.9995], mesh)
    strategy = FSDP2Strategy.__new__(FSDP2Strategy)
    strategy.cfg = SimpleNamespace(optim=SimpleNamespace(clip_grad=1.0))
    strategy._dp_group = mesh["ddp"].get_group() if len(shape) == 2 else None
    mgr = FSDPModelManager.__new__(FSDPModelManager)
    mgr.model, mgr.optimizer, mgr.ema = model, optim, ema
    mgr.grad_scaler = ShardedGradScaler(enabled=False)
    mgr.optimizer_steps, mgr.critic_warmup_steps, mgr.use_dmuon = 0, 0, False
    mgr._strategy = strategy
    mgr._logger = SimpleNamespace(warning=lambda *a: None)
    assert ema.num_updates == 0
    initial = snapshot(model, ema)
    ema.update()
    for n, p in model.named_parameters():
        torch.testing.assert_close(local_tensor(p), initial[0][n], rtol=0, atol=0)
    ema.num_updates = 0
    for decays in ([], [0.999, 0.999], [1.0], [float("nan")]):
        expect_error(lambda: ShardedEMA(model, decays, mesh), "decay")

    def step(i):
        optim.zero_grad()
        torch.manual_seed(1000 + i + dist.get_rank())
        loss = model(torch.randn(4, 13, device="cuda")).square().mean()
        loss.backward()
        prev = copy.deepcopy(ema.shadows)
        count = ema.num_updates
        mgr.optimizer_step()
        assert ema.num_updates == count + (mgr.ema is not None)
        sched.step()
        # FSDP 切换分片以及恢复后，缓存必须仍指向 optimizer 更新的参数。
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert ema.params[n] is p
        for key, decay in ema.decays.items():
            for n, p in model.named_parameters():
                if n in prev[key]:
                    reference = prev[key][n].double()
                    if mgr.ema is not None:
                        reference = reference * decay + local_tensor(p).double() * (
                            1 - decay
                        )
                    torch.testing.assert_close(
                        ema.shadows[key][n], reference.float(), rtol=1e-6, atol=1e-7
                    )
        return float(loss)

    for i in range(2):
        step(i)
    for invalid in (float("inf"), float("nan")):
        optim.zero_grad()
        model(torch.ones(4, 13, device="cuda")).square().mean().backward()
        grad = local_tensor(next(iter(ema.params.values())).grad)
        assert grad.numel() > 0  # 每个 rank 都必须实际注入异常梯度。
        grad.fill_(invalid)
        before = snapshot(model, ema)
        mgr.optimizer_step()
        after = snapshot(model, ema)
        assert ema.num_updates == 2
        for n in before[0]:
            torch.testing.assert_close(after[0][n], before[0][n], rtol=0, atol=0)
        for key in before[1]:
            for n in before[1][key]:
                torch.testing.assert_close(
                    after[1][key][n], before[1][key][n], rtol=0, atol=0
                )
    expected_exports = {}
    params = dict(model.named_parameters())
    for key, shadows in ema.shadows.items():
        expected_exports[key] = {}
        for name, shadow in shadows.items():
            p = params[name]
            expected_exports[key][name] = (
                DTensor.from_local(
                    shadow,
                    p.device_mesh,
                    p.placements,
                    run_check=False,
                    shape=p.shape,
                    stride=p.stride(),
                )
                .full_tensor()
                .cpu()
            )
    ckpt = out / ("x".join(map(str, shape)))
    FSDP2Strategy.save_checkpoint(
        model, optim, sched, str(ckpt), save_full_model_weights=False, ema=ema
    )
    assert ema.num_updates == 2
    metadata = json.loads(ema.state_dict()["metadata"])
    assert "grad_scaler" not in metadata and "grad_scaler_enabled" not in metadata
    for i in range(2, 4):
        step(i)
    expected = snapshot(model, ema)
    expected_lr = sched.get_last_lr()
    FSDP2Strategy.load_checkpoint(model, optim, sched, str(ckpt), ema=ema)
    assert ema.num_updates == 2
    for i in range(2, 4):
        step(i)
    actual = snapshot(model, ema)
    for n in expected[0]:
        torch.testing.assert_close(actual[0][n], expected[0][n], rtol=0, atol=0)
    for k in expected[1]:
        for n in expected[1][k]:
            torch.testing.assert_close(
                actual[1][k][n], expected[1][k][n], rtol=0, atol=0
            )
    assert ema.num_updates == 4 and sched.get_last_lr() == expected_lr

    mgr.ema = None
    FSDP2Strategy.load_checkpoint(model, optim, sched, str(ckpt))
    for i in range(2, 4):
        step(i)
    for n, p in model.named_parameters():
        torch.testing.assert_close(local_tensor(p), expected[0][n], rtol=0, atol=0)
    assert ema.num_updates == 4
    mgr.ema = ema

    saved_signature = copy.deepcopy(ema.signature)
    for field, bad in [
        ("decays", {"decay_0_9": 0.9}),
        ("mesh", [[0]]),
        ("parameters", {}),
    ]:
        ema.signature[field] = bad
        expect_error(lambda: ema.preflight(str(ckpt / "dcp_checkpoint")), field)
        ema.signature = copy.deepcopy(saved_signature)
    raw_ckpt = ckpt / "raw_only"
    FSDP2Strategy.save_checkpoint(
        model, optim, sched, str(raw_ckpt), save_full_model_weights=False
    )
    expect_error(
        lambda: ema.preflight(str(raw_ckpt / "dcp_checkpoint")), "no EMA state"
    )
    # All ranks read metadata collectively; conversion's tensor loading is local.
    raw, _ = convert_state_dict(str(ckpt / "dcp_checkpoint"))
    selected, metadata = convert_state_dict(str(ckpt / "dcp_checkpoint"), "ema", 0.999)
    assert metadata["num_updates"] == 2
    for name in ("frozen", "sentinel"):
        torch.testing.assert_close(raw[name], selected[name], rtol=0, atol=0)
    assert any(not torch.equal(raw[n], selected[n]) for n in ema.params)
    for key, decay in ema.decays.items():
        selected, _ = convert_state_dict(str(ckpt / "dcp_checkpoint"), "ema", decay)
        for name, expected_value in expected_exports[key].items():
            torch.testing.assert_close(selected[name], expected_value, rtol=0, atol=0)
    expect_error(
        lambda: convert_state_dict(str(ckpt / "dcp_checkpoint"), "ema", 0.9), "missing"
    )
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "layout": shape,
                    "result": "PASS",
                    "updates": ema.num_updates,
                    "checks": [
                        "fp32_vs_fp64_reference",
                        "raw_unchanged",
                        "manager_success_and_nonfinite_skip",
                        "cached_parameter_identity",
                        "scaler_and_unsupported_options_rejected",
                        "bitwise_resume",
                        "raw_resume_with_ema_disabled",
                        "strict_mismatch",
                        "raw_legacy",
                        "export_frozen_buffers",
                    ],
                }
            ),
            flush=True,
        )
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 8
    check_config()
    for shape in ((8,), (2, 4)):
        check_layout(shape, Path(args.output))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
