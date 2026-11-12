"""Compatibility helpers and the FSDP2 dedicated-parameter patch."""

import contextlib
import inspect

_installed = False
_original_fn = None


def _collect_dedicated_params(modules):
    dedicated_params = set()
    for module in modules:
        for _, param in module.named_parameters(recurse=False):
            if hasattr(param, "_dedicated_owner_rank"):
                dedicated_params.add(param)
    return dedicated_params


def _call_get_managed_states(original_fn, modules, ignored_params=None):
    modules = tuple(modules)
    ignored_params = set() if ignored_params is None else set(ignored_params)
    ignored_params.update(_collect_dedicated_params(modules))

    signature = inspect.signature(original_fn)
    if len(signature.parameters) >= 2:
        return original_fn(modules, ignored_params)

    params, buffers = original_fn(modules)
    if ignored_params:
        params = [param for param in params if param not in ignored_params]
    return params, buffers


@contextlib.contextmanager
def coalescing_manager(*, group, device, async_ops: bool = False):
    """Coalesce collectives without the ``torch<=2.7`` None-work crash.

    PyTorch 2.6 and 2.7 unconditionally call ``work.wait()`` when exiting
    ``dist._coalescing_manager(device=...)``. Owner-rank reductions are not
    part of its Python fast path, so the process group may return ``None`` and
    trigger ``AttributeError``. PyTorch 2.8 made this wait conditional.

    Drive the same process-group coalescing API directly and wait only when
    the backend returns an actual work handle.
    """
    group._start_coalescing(device)
    try:
        yield
    except BaseException:
        try:
            group._end_coalescing(device)
        except BaseException:
            pass
        raise

    work = group._end_coalescing(device)
    if work is not None and not async_ops:
        work.wait()


def install_patch():
    """Install the FSDP2 monkey-patch that makes ``fully_shard`` skip
    dedicated parameters.

    Called automatically on ``import dmuon`` — users do not normally
    invoke this directly. After patching, any subsequent call to
    ``fully_shard()`` filters out parameters previously marked with
    ``_dedicated_owner_rank`` by :func:`dedicate_params`, leaving them
    under DMuon's ownership instead of FSDP2's uniform sharding.

    Safe to call repeatedly (idempotent). The reverse operation is
    :func:`uninstall_patch`.

    Patched function: ``torch.distributed.fsdp._fully_shard._fsdp_init.
    _get_managed_states``.
    """
    global _installed, _original_fn
    if _installed:
        return

    try:
        import torch.distributed.fsdp._fully_shard._fsdp_init as _fsdp_init
        import torch.distributed.fsdp._fully_shard._fully_shard as _fully_shard_mod
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DMuon could not install its required FSDP2 dedicated-parameter "
            "patch because the installed PyTorch does not expose the expected "
            "FSDP2 internals. Install a supported PyTorch version."
        ) from exc

    _original_fn = _fsdp_init._get_managed_states

    def _patched_get_managed_states(modules, ignored_params=None):
        return _call_get_managed_states(_original_fn, modules, ignored_params)

    # Patch both the definition site and the import site
    _fsdp_init._get_managed_states = _patched_get_managed_states
    _fully_shard_mod._get_managed_states = _patched_get_managed_states
    _installed = True


def uninstall_patch():
    """Uninstall monkey-patch, restoring original FSDP2 behavior."""
    global _installed, _original_fn
    if not _installed:
        return

    try:
        import torch.distributed.fsdp._fully_shard._fsdp_init as _fsdp_init
        import torch.distributed.fsdp._fully_shard._fully_shard as _fully_shard_mod
    except ModuleNotFoundError:
        return

    _fsdp_init._get_managed_states = _original_fn
    _fully_shard_mod._get_managed_states = _original_fn
    _original_fn = None
    _installed = False
