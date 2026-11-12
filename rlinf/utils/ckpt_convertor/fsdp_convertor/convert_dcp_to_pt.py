# Copyright 2025 The RLinf Authors.
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

import argparse
import json
import os
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

from rlinf.hybrid_engines.fsdp.ema import decay_key, read_ema_metadata

"""
python rlinf/utils/ckpt_convertor/fsdp_convertor/convert_dcp_to_pt.py\
    --dcp_path /path/to/dcp_checkpoint/ \
    --output_path /path/to/save_path/model.pt
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DCP checkpoint to state_dict checkpoint"
    )
    parser.add_argument(
        "--dcp_path",
        type=str,
        required=True,
        help="Path to the DCP checkpoint directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted state_dict checkpoint",
    )
    parser.add_argument("--weight-source", choices=("raw", "ema"), default="raw")
    parser.add_argument("--ema-decay", type=float)
    args = parser.parse_args()
    if (args.weight_source == "ema") != (args.ema_decay is not None):
        parser.error("--ema-decay is required exactly when --weight-source=ema")
    return args


def convert_state_dict(dcp_path, weight_source="raw", ema_decay=None):
    if weight_source not in {"raw", "ema"}:
        raise ValueError(f"Unknown weight source: {weight_source}")
    if (weight_source == "ema") != (ema_decay is not None):
        raise ValueError("Select raw, or ema with an explicit decay")
    keys = {"fsdp_checkpoint.model"}
    reader = FileSystemReader(dcp_path)
    metadata = None
    if weight_source == "ema":
        metadata, _ = read_ema_metadata(reader)
        key = decay_key(ema_decay)
        if key not in metadata["decays"]:
            raise ValueError(
                f"EMA decay {ema_decay} missing; available: {metadata['decays']}"
            )
        keys.add(f"ema.weights.{key}")  # 同时读取 raw 和 ema
    checkpoint = {}
    _load_state_dict(
        checkpoint,
        storage_reader=reader,
        planner=_EmptyStateDictLoadPlanner(keys=keys),
        no_dist=True,
    )

    try:
        model_state_dict = checkpoint["fsdp_checkpoint"]["model"]
    except KeyError as e:
        raise KeyError(
            "Could not find 'fsdp_checkpoint.model' in the DCP checkpoint. "
            f"Loaded top-level keys: {list(checkpoint.keys())}"
        ) from e

    if metadata is not None:
        selected = checkpoint.get("ema", {}).get("weights", {}).get(key, {})
        if set(selected) != set(metadata["parameters"]):
            raise ValueError(
                "EMA checkpoint has missing/unexpected trainable parameters"
            )
        for name, value in selected.items():
            if (
                name not in model_state_dict
                or value.shape != model_state_dict[name].shape
            ):
                raise ValueError(f"EMA/raw parameter mismatch: {name}")
            if (
                value.dtype != torch.float32
                or list(value.shape) != metadata["parameters"][name]["shape"]
            ):
                raise ValueError(f"EMA shape/dtype mismatch: {name}")
        model_state_dict.update(selected)   # 冻结参数 和 buffer 使用原始的
    return model_state_dict, metadata


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    model_state_dict, metadata = convert_state_dict(
        args.dcp_path, args.weight_source, args.ema_decay
    )

    torch.save(model_state_dict, args.output_path)
    source = {  # 整理为字典
        "checkpoint_path": str(Path(args.dcp_path).resolve()),
        "weight_source": args.weight_source,
        "ema_decay": args.ema_decay,
        "ema_updates": metadata["num_updates"] if metadata is not None else None,
    }
    # 导出为 JSON，仅说明权重来源，不参与后续转换。
    with open(f"{args.output_path}.weight_source.json", "w", encoding="utf-8") as f:
        json.dump(source, f, indent=2)
        f.write("\n")

    print(
        f"Converted DCP checkpoint from {args.dcp_path} to state_dict at {args.output_path}"
    )
