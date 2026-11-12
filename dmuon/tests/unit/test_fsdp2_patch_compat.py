from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import pytest
import torch


def _load_patch_module():
    path = Path(__file__).resolve().parents[2] / "dmuon" / "_backends" / "fsdp2" / "patch.py"
    spec = importlib.util.spec_from_file_location("dmuon_fsdp2_patch_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_managed_states_patch_supports_torch26_signature() -> None:
    patch = _load_patch_module()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    dedicated = model[0].weight
    dedicated._dedicated_owner_rank = 0

    def original(modules):
        params = []
        for module in modules:
            params.extend(list(module.parameters(recurse=False)))
        return params, ["buffer"]

    params, buffers = patch._call_get_managed_states(original, [model[0], model[1]])

    assert all(param is not dedicated for param in params)
    assert any(param is model[0].bias for param in params)
    assert any(param is model[1].weight for param in params)
    assert buffers == ["buffer"]


def test_get_managed_states_patch_supports_legacy_ignored_params_signature() -> None:
    patch = _load_patch_module()
    model = torch.nn.Linear(2, 2)
    model.weight._dedicated_owner_rank = 0

    def original(modules, ignored_params):
        params = []
        for module in modules:
            params.extend(
                param
                for param in module.parameters(recurse=False)
                if param not in ignored_params
            )
        return params, []

    params, buffers = patch._call_get_managed_states(original, [model])

    assert all(param is not model.weight for param in params)
    assert any(param is model.bias for param in params)
    assert buffers == []


@pytest.mark.parametrize(
    "missing_module",
    [
        "torch.distributed.fsdp._fully_shard._fsdp_init",
        "torch.distributed.fsdp._fully_shard._fully_shard",
    ],
)
def test_install_patch_fails_when_required_fsdp2_internals_are_missing(
    monkeypatch, missing_module
) -> None:
    patch = _load_patch_module()
    real_import = builtins.__import__

    def missing_fsdp2_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == missing_module:
            raise ModuleNotFoundError(
                "simulated missing FSDP2 internals",
                name=name,
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_fsdp2_import)

    with pytest.raises(
        RuntimeError, match="required FSDP2 dedicated-parameter patch"
    ) as exc_info:
        patch.install_patch()

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
    assert patch._installed is False
    assert patch._original_fn is None
