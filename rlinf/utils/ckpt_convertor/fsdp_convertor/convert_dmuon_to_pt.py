# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge an FSDP2 + DMuon rank-local checkpoint into a model state dict."""

import argparse
import io
import math
import os
import shutil
import time
from typing import Any, BinaryIO

import torch

_CHECKPOINT_FORMAT = "rlinf-dmuon-sharded-v2"
_MANIFEST_FILE = "manifest.pt"


def _load(path: str | BinaryIO, *, mmap: bool = False) -> dict[str, Any]:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=mmap,
    )


def _load_rank_sequential(path: str) -> dict[str, Any]:
    """Read one shard sequentially before parsing its tensor records in memory."""
    started = time.monotonic()
    print(f"[DMuon] Reading {path}", flush=True)
    with open(path, "rb") as source, io.BytesIO() as buffer:
        shutil.copyfileobj(source, buffer, length=16 * 1024 * 1024)
        buffer.seek(0)
        print(f"[DMuon] Read complete; parsing {path}", flush=True)
        payload = _load(buffer)
    print(f"[DMuon] Loaded {path} in {time.monotonic() - started:.1f}s", flush=True)
    return payload


def _validate_shape(tensor: torch.Tensor, shape: tuple[int, ...], label: str) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"DMuon checkpoint shape mismatch for {label}: "
            f"saved={tuple(tensor.shape)}, expected={shape}"
        )


def _replica_groups(
    manifest: dict[str, Any], world_size: int
) -> tuple[tuple[int, ...], ...]:
    """
    恢复分组((0, 1, ...), (8, 9, ...))，并且验证分组是否合法
    """
    device_mesh = manifest.get("device_mesh")
    if device_mesh is None:
        raise ValueError("DMuon checkpoint manifest is missing device_mesh")

    dim_names = tuple(device_mesh.get("dim_names", ()))
    shape = tuple(int(size) for size in device_mesh.get("shape", ()))
    rank_layout = device_mesh.get("rank_layout", ())
    # device_mesh = {
    #     "dim_names": ("ddp", "fsdp"),
    #     "shape": (2, 8),
    #     "rank_layout": [
    #         [0, 1, 2, 3, 4, 5, 6, 7],
    #         [8, 9, 10, 11, 12, 13, 14, 15],
    #     ],
    # }
    if dim_names == ("fsdp",):
        ranks = tuple(int(rank) for rank in rank_layout)
        if (
            shape != (world_size,)
            or len(ranks) != world_size
            or set(ranks) != set(range(world_size))
        ):
            raise ValueError(
                f"Invalid DMuon FSDP device mesh: shape={shape}, ranks={ranks}"
            )
        return (ranks,)
    if dim_names != ("ddp", "fsdp"):
        raise ValueError(
            "DMuon checkpoint conversion only supports a 1D FSDP mesh or "
            f"a 2D ('ddp', 'fsdp') mesh, got {dim_names!r}"
        )

    groups = tuple(tuple(int(rank) for rank in row) for row in rank_layout)
    if (
        not groups
        or any(not group for group in groups)           # 检查空group
        or len({len(group) for group in groups}) != 1   # 所有group长度需要相同
        or shape != (len(groups), len(groups[0]))       # mesh shape和实际分布需要一致
    ):
        raise ValueError(
            f"Invalid DMuon hybrid device mesh: shape={shape}, groups={groups}"
        )
    ranks = tuple(rank for group in groups for rank in group)
    if len(ranks) != world_size or set(ranks) != set(range(world_size)):
        raise ValueError(
            "DMuon hybrid device mesh does not cover each rank exactly once"
        )
    return groups


def merge_dmuon_model_state(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Return a full model state dict from a 1D or hybrid-shard checkpoint."""

    manifest_path = os.path.join(checkpoint_path, _MANIFEST_FILE)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"DMuon checkpoint manifest not found: {manifest_path}")
    manifest = _load(manifest_path)
    if manifest.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported DMuon checkpoint format: {manifest.get('format')!r}"
        )

    world_size = int(manifest.get("world_size", -1))
    rank_files = manifest.get("rank_files", [])
    if world_size <= 0 or len(rank_files) != world_size:
        raise ValueError(
            "DMuon checkpoint rank file count mismatch: "
            f"world_size={world_size}, rank_files={len(rank_files)}"
        )

    replica_groups = _replica_groups(manifest, world_size)
    metadata = manifest["model_metadata"]
    dedicated_parts = {fqn: [] for fqn in metadata["dedicated"]}
    regular_parts = {fqn: [] for fqn in metadata["regular"]}
    buffer_parts = {fqn: [] for fqn in metadata["buffers"]}

    for rank, filename in enumerate(rank_files):
        rank_path = os.path.join(checkpoint_path, filename)
        if not os.path.isfile(rank_path):
            raise FileNotFoundError(f"DMuon rank checkpoint not found: {rank_path}")
        payload = _load_rank_sequential(rank_path)
        if payload.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported DMuon rank checkpoint format in {filename}: "
                f"{payload.get('format')!r}"
            )
        if int(payload.get("rank", -1)) != rank:
            raise ValueError(
                f"DMuon rank checkpoint mismatch in {filename}: "
                f"saved={payload.get('rank')}, expected={rank}"
            )
        if int(payload.get("world_size", -1)) != world_size:
            raise ValueError(
                f"DMuon world size mismatch in {filename}: "
                f"saved={payload.get('world_size')}, expected={world_size}"
            )

        model_state = payload["model"]
        unexpected_dedicated = set(model_state["dedicated"]) - set(dedicated_parts)
        if unexpected_dedicated:
            raise ValueError(
                "Unexpected dedicated model keys in DMuon rank checkpoint: "
                f"{sorted(unexpected_dedicated)[:8]}"
            )
        for fqn, entry in model_state["dedicated"].items():
            if "_sharded_adamw_data" in entry:
                raise ValueError(
                    "DMuon checkpoint conversion does not support sharded AdamW "
                    f"parameter {fqn!r}"
                )
            if "_owned_data" in entry:
                dedicated_parts[fqn].append((rank, entry["_owned_data"]))

        if set(model_state["regular"]) != set(regular_parts):
            raise ValueError(
                f"Regular model FQNs differ in DMuon rank checkpoint {filename}"
            )
        for fqn, tensor in model_state["regular"].items():
            regular_parts[fqn].append(tensor)

        if set(model_state["buffers"]) != set(buffer_parts):
            raise ValueError(
                f"Model buffer FQNs differ in DMuon rank checkpoint {filename}"
            )
        for fqn, tensor in model_state["buffers"].items():
            buffer_parts[fqn].append(tensor)

    state_dict: dict[str, torch.Tensor] = {}
    for fqn, ranked_parts in dedicated_parts.items():
        # fqn = "blocks.0.attn.weight"
        # ranked_parts = [
        #     (0, tensor_a),
        #     (8, tensor_b),
        # ]
        parts_by_rank = dict(ranked_parts)
        owner_ranks = []
        owner_positions = []
        for group in replica_groups:
            group_owners = [rank for rank in group if rank in parts_by_rank]
            if len(group_owners) != 1:
                owner_count = len(group_owners)
                raise ValueError(
                    "DMuon checkpoint requires one owner copy per replica: "
                    f"{fqn!r} has {owner_count} owners in group {group}"
                )
            owner_ranks.append(group_owners[0])  # golbal rank
            owner_positions.append(group.index(group_owners[0]))    # local rank
        if len(set(owner_positions)) != 1:
            raise ValueError(
                "DMuon checkpoint owner copies use different FSDP positions: "
                f"{fqn!r} positions={owner_positions}"
            )
        if set(parts_by_rank) != set(owner_ranks):
            raise ValueError(
                "DMuon checkpoint contains unexpected owner copies: "
                f"{fqn!r} saved on {sorted(parts_by_rank)}, "
                f"expected {sorted(owner_ranks)}"
            )

        # 检查所有副本形状相同
        shape = tuple(metadata["dedicated"][fqn]["shape"])
        reference = parts_by_rank[owner_ranks[0]]
        _validate_shape(reference, shape, f"{fqn} rank {owner_ranks[0]}")
        for rank in owner_ranks[1:]:
            replica = parts_by_rank[rank]
            _validate_shape(replica, shape, f"{fqn} rank {rank}")
            if not torch.equal(reference, replica):
                raise ValueError(
                    f"DMuon replica mismatch for dedicated parameter {fqn!r}"
                )
        state_dict[fqn] = reference

    # 检查所有副本数值相同
    for fqn, parts in regular_parts.items():
        shape = tuple(metadata["regular"][fqn]["shape"])
        if not metadata["regular"][fqn]["is_sharded"]:
            for rank, tensor in enumerate(parts):
                _validate_shape(tensor, shape, f"{fqn} rank {rank}")
            reference = parts[0]
            for replica in parts[1:]:
                if not torch.equal(reference, replica):
                    raise ValueError(
                        f"DMuon replica mismatch for regular parameter {fqn!r}"
                    )
            state_dict[fqn] = reference
            continue
        if not shape:
            raise ValueError(f"Cannot shard scalar parameter {fqn!r}")

        # 检查Dmuon ckpt中的fsdp参数
        shard_world_size = len(replica_groups[0])
        chunk_size = math.ceil(shape[0] / shard_world_size)
        reference_parts = []
        for shard_index, reference_rank in enumerate(replica_groups[0]):
            chunk_start = shard_index * chunk_size
            local_dim0 = min(chunk_size, max(shape[0] - chunk_start, 0))
            local_shape = (local_dim0, *shape[1:])
            reference = parts[reference_rank]
            _validate_shape(reference, local_shape, f"{fqn} rank {reference_rank}")
            reference_parts.append(reference)
            for group in replica_groups[1:]:
                replica_rank = group[shard_index]
                _validate_shape(
                    parts[replica_rank],
                    local_shape,
                    f"{fqn} rank {replica_rank}",
                )
                if not torch.equal(reference, parts[replica_rank]):
                    raise ValueError(
                        f"DMuon replica mismatch for regular parameter {fqn!r}"
                    )
        full = torch.cat(reference_parts, dim=0)
        _validate_shape(full, shape, fqn)
        state_dict[fqn] = full.contiguous()

    # 检查buffer部分
    for fqn, parts in buffer_parts.items():
        shape = tuple(metadata["buffers"][fqn]["shape"])
        for rank, tensor in enumerate(parts):
            _validate_shape(tensor, shape, f"{fqn} rank {rank}")
        reference = parts[0]
        for replica in parts[1:]:
            if not torch.equal(reference, replica):
                raise ValueError(f"DMuon replica mismatch for buffer {fqn!r}")
        state_dict[fqn] = reference

    expected_keys = set(dedicated_parts) | set(regular_parts) | set(buffer_parts)
    if set(state_dict) != expected_keys:
        raise RuntimeError("DMuon checkpoint conversion produced incomplete model state")
    return state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a DMuon rank-local checkpoint into model.pt"
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to the dmuon_checkpoint directory",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path for the merged model state dict",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dict = merge_dmuon_model_state(args.checkpoint_path)
    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = f"{output_path}.tmp"
    torch.save(state_dict, temporary_path)
    os.replace(temporary_path, output_path)
    print(f"Merged DMuon checkpoint to {output_path}")


if __name__ == "__main__":
    main()
