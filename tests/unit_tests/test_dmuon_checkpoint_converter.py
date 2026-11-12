from pathlib import Path

import pytest
import torch

from rlinf.utils.ckpt_convertor.fsdp_convertor.convert_dmuon_to_pt import (
    merge_dmuon_model_state,
)


def _write_checkpoint(
    root: Path,
    *,
    duplicate_owner: bool = False,
    include_device_mesh: bool = True,
) -> None:
    metadata = {
        "dedicated": {
            "dedicated.weight": {
                "shape": (2, 3),
                "owner_rank": (1,),
                "route": "muon",
            }
        },
        "regular": {
            "sharded.weight": {"shape": (3, 2), "is_sharded": True},
            "replicated.bias": {"shape": (2,), "is_sharded": False},
        },
        "buffers": {"scale": {"shape": (1,)}},
    }
    manifest = {
        "format": "rlinf-dmuon-sharded-v2",
        "world_size": 2,
        "rank_files": ["rank_00000.pt", "rank_00001.pt"],
        "model_metadata": metadata,
    }
    if include_device_mesh:
        manifest["device_mesh"] = {
            "shape": (2,),
            "dim_names": ("fsdp",),
            "rank_layout": [0, 1],
        }
    torch.save(manifest, root / "manifest.pt")

    dedicated = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    sharded = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    for rank in range(2):
        local_dedicated = {}
        if rank == 1 or duplicate_owner:
            local_dedicated["dedicated.weight"] = {"_owned_data": dedicated.clone()}
        if rank == 0:
            local_sharded = sharded[:2].clone()
        else:
            local_sharded = sharded[2:].clone()
        payload = {
            "format": "rlinf-dmuon-sharded-v2",
            "rank": rank,
            "world_size": 2,
            "model": {
                "dedicated": local_dedicated,
                "regular": {
                    "sharded.weight": local_sharded,
                    "replicated.bias": torch.tensor([7.0, 8.0]),
                },
                "buffers": {"scale": torch.tensor([0.5])},
            },
        }
        torch.save(payload, root / f"rank_{rank:05d}.pt")


def test_merge_dmuon_model_state(tmp_path):
    _write_checkpoint(tmp_path)

    state = merge_dmuon_model_state(str(tmp_path))

    assert set(state) == {
        "dedicated.weight",
        "sharded.weight",
        "replicated.bias",
        "scale",
    }
    torch.testing.assert_close(
        state["dedicated.weight"],
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
    )
    torch.testing.assert_close(
        state["sharded.weight"],
        torch.arange(6, dtype=torch.float32).reshape(3, 2),
    )
    torch.testing.assert_close(state["replicated.bias"], torch.tensor([7.0, 8.0]))
    torch.testing.assert_close(state["scale"], torch.tensor([0.5]))


def test_merge_dmuon_model_state_rejects_missing_device_mesh(tmp_path):
    _write_checkpoint(tmp_path, include_device_mesh=False)

    with pytest.raises(ValueError, match="missing device_mesh"):
        merge_dmuon_model_state(str(tmp_path))


def test_merge_dmuon_model_state_rejects_multiple_owner_copies(tmp_path):
    _write_checkpoint(tmp_path, duplicate_owner=True)

    with pytest.raises(ValueError, match="one owner copy"):
        merge_dmuon_model_state(str(tmp_path))


def test_merge_dmuon_model_state_one_dimensional_mesh(tmp_path):
    _write_checkpoint(tmp_path, include_device_mesh=True)

    state = merge_dmuon_model_state(str(tmp_path))

    torch.testing.assert_close(
        state["sharded.weight"],
        torch.arange(6, dtype=torch.float32).reshape(3, 2),
    )


def _write_hybrid_checkpoint(
    root: Path,
    *,
    dedicated_mismatch: bool = False,
    sharded_mismatch: bool = False,
    replicated_mismatch: bool = False,
    buffer_mismatch: bool = False,
) -> None:
    metadata = {
        "dedicated": {
            "dedicated.weight": {
                "shape": (2, 2),
                "owner_rank": (0, 0),
                "route": "muon",
            }
        },
        "regular": {
            "sharded.weight": {"shape": (4, 2), "is_sharded": True},
            "replicated.bias": {"shape": (2,), "is_sharded": False},
        },
        "buffers": {"scale": {"shape": (1,)}},
    }
    manifest = {
        "format": "rlinf-dmuon-sharded-v2",
        "world_size": 4,
        "rank_files": [f"rank_{rank:05d}.pt" for rank in range(4)],
        "device_mesh": {
            "shape": (2, 2),
            "dim_names": ("ddp", "fsdp"),
            "rank_layout": [[0, 1], [2, 3]],
        },
        "model_metadata": metadata,
    }
    torch.save(manifest, root / "manifest.pt")

    dedicated = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    sharded = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    for rank in range(4):
        local_dedicated = {}
        if rank in (0, 2):
            value = dedicated + (1 if dedicated_mismatch and rank == 2 else 0)
            local_dedicated["dedicated.weight"] = {"_owned_data": value}
        local_sharded = sharded[:2].clone() if rank % 2 == 0 else sharded[2:].clone()
        if sharded_mismatch and rank == 2:
            local_sharded += 1
        replicated = torch.tensor([7.0, 8.0])
        if replicated_mismatch and rank == 2:
            replicated += 1
        scale = torch.tensor([0.5 + (1 if buffer_mismatch and rank == 3 else 0)])
        payload = {
            "format": "rlinf-dmuon-sharded-v2",
            "rank": rank,
            "world_size": 4,
            "model": {
                "dedicated": local_dedicated,
                "regular": {
                    "sharded.weight": local_sharded,
                    "replicated.bias": replicated,
                },
                "buffers": {"scale": scale},
            },
        }
        torch.save(payload, root / f"rank_{rank:05d}.pt")


def test_merge_dmuon_model_state_hybrid_shard(tmp_path):
    _write_hybrid_checkpoint(tmp_path)

    state = merge_dmuon_model_state(str(tmp_path))

    torch.testing.assert_close(
        state["dedicated.weight"],
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
    )
    torch.testing.assert_close(
        state["sharded.weight"],
        torch.arange(8, dtype=torch.float32).reshape(4, 2),
    )
    torch.testing.assert_close(state["replicated.bias"], torch.tensor([7.0, 8.0]))
    torch.testing.assert_close(state["scale"], torch.tensor([0.5]))


def test_merge_dmuon_model_state_rejects_dedicated_replica_mismatch(tmp_path):
    _write_hybrid_checkpoint(tmp_path, dedicated_mismatch=True)

    with pytest.raises(ValueError, match="dedicated parameter"):
        merge_dmuon_model_state(str(tmp_path))


def test_merge_dmuon_model_state_rejects_sharded_replica_mismatch(tmp_path):
    _write_hybrid_checkpoint(tmp_path, sharded_mismatch=True)

    with pytest.raises(ValueError, match="regular parameter"):
        merge_dmuon_model_state(str(tmp_path))


def test_merge_dmuon_model_state_rejects_replicated_param_mismatch(tmp_path):
    _write_hybrid_checkpoint(tmp_path, replicated_mismatch=True)

    with pytest.raises(ValueError, match="regular parameter"):
        merge_dmuon_model_state(str(tmp_path))


def test_merge_dmuon_model_state_rejects_buffer_mismatch(tmp_path):
    _write_hybrid_checkpoint(tmp_path, buffer_mismatch=True)

    with pytest.raises(ValueError, match="buffer"):
        merge_dmuon_model_state(str(tmp_path))
