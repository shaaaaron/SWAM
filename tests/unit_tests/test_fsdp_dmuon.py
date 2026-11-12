import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp import dmuon as integration
from rlinf.hybrid_engines.fsdp import utils as fsdp_utils
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager


def _actor_cfg(optimizer: str = "dmuon"):
    return OmegaConf.create(
        {
            "model": {"precision": "fp32"},
            "enable_offload": False,
            "optim": {
                "optimizer": optimizer,
                "lr": 5.0e-5,
                "value_lr": 5.0e-5,
                "adam_beta1": 0.95,
                "adam_beta2": 0.999,
                "adam_eps": 1.0e-8,
                "weight_decay": 1.0e-5,
                "clip_grad": 1.0,
                "dmuon": {
                    "lr": 0.02,
                    "momentum": 0.95,
                    "weight_decay": 0.0,
                    "ns_steps": 5,
                    "nesterov": True,
                    "param_name_pattern": r"^blocks\.\d+\.weight$",
                    "expected_param_count": 2,
                    "master_dtype": "fp32",
                    "optim_dtype": "bf16",
                },
            },
            "fsdp_config": {
                "strategy": "fsdp2",
                "cpu_offload": False,
                "save_full_model_weights": False,
                "grad_scaler": {"enabled": False},
                "mixed_precision": {
                    "param_dtype": "bf16",
                    "reduce_dtype": "bf16",
                    "buffer_dtype": "bf16",
                },
                "reshard_after_forward": True,
                "enable_gradient_accumulation": True,
            },
        }
    )


@pytest.mark.parametrize("value", [None, "adam", "adamw"])
def test_legacy_optimizer_names_stay_on_adamw(value):
    cfg = _actor_cfg("adamw")
    if value is None:
        del cfg.optim.optimizer
    else:
        cfg.optim.optimizer = value
    assert integration.get_optimizer_type(cfg) == "adamw"
    integration.validate_dmuon_config(cfg)


def test_unknown_optimizer_is_rejected():
    cfg = _actor_cfg("dmuonn")
    with pytest.raises(ValueError, match="Unsupported FSDP optimizer"):
        integration.get_optimizer_type(cfg)


def test_adamw_builder_keeps_existing_warmup_behavior():
    cfg = _actor_cfg("adamw")
    model = torch.nn.Linear(4, 3)
    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager._cfg = cfg
    manager.use_dmuon = False
    manager.store_requires_grad_param_name = []

    optimizer = manager.build_optimizer(model)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == cfg.optim.lr
    assert len(optimizer.state) == 2
    assert all(int(state["step"].item()) == 1 for state in optimizer.state.values())


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("enable_offload", True, "offload is not supported"),
        ("fsdp_config.cpu_offload", True, "offload is not supported"),
        ("fsdp_config.grad_scaler.enabled", True, "grad_scaler.enabled=false"),
        (
            "fsdp_config.save_full_model_weights",
            True,
            "save_full_model_weights=false",
        ),
        ("fsdp_config.strategy", "fsdp", "only with FSDP2"),
    ],
)
def test_dmuon_capability_guards(path, value, message):
    cfg = _actor_cfg()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError, match=message):
        integration.validate_dmuon_config(cfg)


def test_dmuon_requires_explicit_route_count():
    cfg = _actor_cfg()
    del cfg.optim.dmuon.expected_param_count
    with pytest.raises(ValueError, match="expected_param_count"):
        integration.validate_dmuon_config(cfg)


def test_dmuon_checkpoint_rejects_implicit_full_model_export():
    with pytest.raises(ValueError, match="merge the sharded checkpoint"):
        integration.save_dmuon_checkpoint(
            torch.nn.Linear(2, 2),
            object(),
            object(),
            "/unused",
            optimizer_steps=0,
            training_step=0,
            save_full_model_weights=True,
        )


def test_dmuon_checkpoint_requires_device_mesh():
    with pytest.raises(ValueError, match="requires the current device mesh"):
        integration.save_dmuon_checkpoint(
            torch.nn.Linear(2, 2),
            object(),
            object(),
            "/unused",
            optimizer_steps=0,
            training_step=0,
            save_full_model_weights=False,
        )


def test_dmuon_checkpoint_serializes_device_mesh():
    mesh = SimpleNamespace(
        mesh=torch.tensor([[0, 1], [2, 3]]),
        mesh_dim_names=("ddp", "fsdp"),
    )

    assert integration._checkpoint_device_mesh(mesh) == {
        "shape": (2, 2),
        "dim_names": ("ddp", "fsdp"),
        "rank_layout": [[0, 1], [2, 3]],
    }


def test_dmuon_checkpoint_validates_device_mesh():
    mesh = SimpleNamespace(
        mesh=torch.tensor([[0, 1], [2, 3]]),
        mesh_dim_names=("ddp", "fsdp"),
    )
    saved = integration._checkpoint_device_mesh(mesh)

    integration._validate_checkpoint_device_mesh(saved, mesh)
    with pytest.raises(ValueError, match="missing device_mesh"):
        integration._validate_checkpoint_device_mesh(None, mesh)
    with pytest.raises(ValueError, match="device mesh mismatch"):
        integration._validate_checkpoint_device_mesh(
            {**saved, "rank_layout": [[0, 2], [1, 3]]},
            mesh,
        )


def test_hsdp_sync_broadcasts_ordinary_states_globally(monkeypatch):
    class FakeMesh:
        mesh_dim_names = ("ddp", "fsdp")

        def __getitem__(self, name):
            assert name == "ddp"
            return SimpleNamespace(get_group=lambda: "ddp_group")

    calls = []
    monkeypatch.setattr(fsdp_utils.dist, "get_global_rank", lambda group, rank: 0)
    monkeypatch.setattr(
        fsdp_utils.dist,
        "broadcast",
        lambda tensor, *, src, group=None: calls.append((tensor, src, group)),
    )
    model = torch.nn.Linear(2, 2)
    model.register_buffer("scale", torch.ones(1))

    fsdp_utils.sync_fsdp2_hybrid_module_states(model, FakeMesh())

    assert len(calls) == 3
    assert all(src == 0 and group is None for _, src, group in calls)


def test_optimizer_state_uses_stable_fqn_after_model_parameter_replacement():
    model = torch.nn.Linear(3, 2)
    optimizer_param = model.weight
    optimizer = SimpleNamespace(
        _fsdp_params=[optimizer_param],
        _rlinf_fsdp_param_fqns={id(optimizer_param): "weight"},
        state={optimizer_param: {"step": 7}},
    )

    model.weight = torch.nn.Parameter(model.weight.detach().clone())
    saved = integration._local_optimizer_state(optimizer, {})

    assert saved == {"dedicated": {}, "regular": {"weight": {"step": 7}}}


def test_optimizer_state_restore_uses_stable_fqn():
    model = torch.nn.Linear(3, 2)
    optimizer_param = model.weight
    optimizer = SimpleNamespace(
        _fsdp_params=[optimizer_param],
        _rlinf_fsdp_param_fqns={id(optimizer_param): "weight"},
        state={optimizer_param: {"step": 1}},
    )
    model.weight = torch.nn.Parameter(model.weight.detach().clone())

    integration._restore_optimizer_state(
        model,
        optimizer,
        {},
        {"dedicated": {}, "regular": {"weight": {"step": 9}}},
    )

    assert optimizer.state == {optimizer_param: {"step": 9}}


def test_optimizer_state_requires_stable_fqn_mapping():
    optimizer = SimpleNamespace(_fsdp_params=[], state={})
    with pytest.raises(RuntimeError, match="stable FSDP parameter FQN mapping"):
        integration._local_optimizer_state(optimizer, {})


def test_model_checkpoint_uses_stable_fsdp_parameter_after_replacement():
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer_param = model.weight
    optimizer_param.data.fill_(1.0)
    optimizer = SimpleNamespace(
        _fsdp_params=[optimizer_param],
        _rlinf_fsdp_param_fqns={id(optimizer_param): "weight"},
    )
    model.weight = torch.nn.Parameter(torch.full_like(model.weight, 2.0))

    saved = integration._local_model_state(model, optimizer, {})
    metadata = integration._checkpoint_model_metadata(model, optimizer, {})

    assert torch.equal(saved["regular"]["weight"], torch.ones_like(model.weight))
    assert metadata["regular"] == {
        "weight": {"shape": (2, 3), "is_sharded": False}
    }

    optimizer_param.data.zero_()
    integration._restore_model_state(model, optimizer, {}, saved)
    assert torch.equal(optimizer_param, torch.ones_like(optimizer_param))


def test_no_sync_does_not_import_dmuon_when_disabled(monkeypatch):
    monkeypatch.setattr(
        integration,
        "_load_dmuon",
        lambda: pytest.fail("DMuon should not be imported"),
    )
    assert isinstance(
        integration.dmuon_no_sync(
            torch.nn.Linear(2, 2), enabled=False, is_last_micro_batch=False
        ),
        type(nullcontext()),
    )


@pytest.mark.parametrize("low_dtype", [torch.bfloat16, torch.float16])
def test_clip_grad_by_total_norm_updates_low_precision_gradients(low_dtype):
    fp32_param = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    low_precision_param = torch.nn.Parameter(torch.ones(1, dtype=low_dtype))
    fp32_param.grad = torch.ones_like(fp32_param)
    low_precision_param.grad = torch.ones_like(low_precision_param)

    integration.clip_grad_by_total_norm_(
        [fp32_param, low_precision_param],
        max_grad_norm=1.0,
        total_norm=2.0,
    )

    expected = 1.0 / (2.0 + 1.0e-6)
    assert fp32_param.grad.item() == pytest.approx(expected)
    assert low_precision_param.grad.float().item() == pytest.approx(
        expected, abs=1.0e-3
    )
    assert low_precision_param.grad.dtype == low_dtype


def test_combined_clip_uses_one_coefficient(monkeypatch):
    calls = []
    fake_dmuon = SimpleNamespace(
        clip_grad_norm_=lambda optimizer, max_norm: (
            calls.append(("muon", max_norm))
            or SimpleNamespace(total_norm=3.0)
        )
    )
    monkeypatch.setattr(integration, "_load_dmuon", lambda: fake_dmuon)
    monkeypatch.setattr(integration, "get_grad_norm", lambda *args, **kwargs: 4.0)
    monkeypatch.setattr(
        integration,
        "clip_grad_by_total_norm_",
        lambda *args, **kwargs: calls.append(("regular", kwargs)),
    )

    stats = integration.clip_dmuon_combined_grad_norm_(
        torch.nn.Linear(2, 2),
        object(),
        dp_group=None,
        max_grad_norm=1.0,
    )

    assert stats.total == 5.0
    assert stats.coefficient == pytest.approx(1.0 / (5.0 + 1.0e-6))
    assert calls[1][0] == "regular"
    assert calls[1][1]["total_norm"] == 5.0
    assert calls[2][0] == "muon"
    assert calls[2][1] == pytest.approx(stats.coefficient * (3.0 + 1.0e-6))


def test_combined_clip_reports_nonfinite_without_scaling(monkeypatch):
    calls = []
    fake_dmuon = SimpleNamespace(
        clip_grad_norm_=lambda optimizer, max_norm: SimpleNamespace(
            total_norm=math.inf
        )
    )
    monkeypatch.setattr(integration, "_load_dmuon", lambda: fake_dmuon)
    monkeypatch.setattr(integration, "get_grad_norm", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        integration,
        "clip_grad_by_total_norm_",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    stats = integration.clip_dmuon_combined_grad_norm_(
        torch.nn.Linear(2, 2),
        object(),
        dp_group=None,
        max_grad_norm=1.0,
    )
    assert math.isinf(stats.total)
    assert stats.coefficient == 0.0
    assert calls == []
