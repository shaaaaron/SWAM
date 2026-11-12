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

"""可训练参数的 FP32 分片 EMA，以及严格匹配的 DCP 保存恢复。"""

import json
import math

import torch
from torch.distributed import checkpoint as dcp
from torch.distributed.tensor import DTensor


def decay_key(decay):
    """校验单个 decay，返回稳定的 checkpoint 键名。"""
    value = float(decay)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"EMA decay must be finite and in (0, 1), got {decay!r}")
    return "decay_" + repr(value).replace(".", "_")


def local_tensor(tensor):
    """分布式张量，就取当前 rank 持有的本地分片；否则原样返回。"""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def read_ema_metadata(reader):
    """输入 DCP reader，返回 EMA 元数据和张量目录，不修改训练状态。"""
    stored = reader.read_metadata().state_dict_metadata # 包含module的size、dtype的信息
    if "ema.metadata" not in stored:
        raise ValueError(
            f"EMA resume/export requested, but {reader.path} has no EMA state. "
            "Use a checkpoint saved with EMA enabled; history cannot be rebuilt."
        )
    state = {"ema": {"metadata": ""}}
    dcp.load(state, storage_reader=reader)  # 将reader的内容填充到state中
    metadata = json.loads(state["ema"]["metadata"])
    if metadata.get("version") != 1:
        raise ValueError("Unsupported EMA checkpoint version")
    if type(metadata.get("num_updates")) is not int or metadata["num_updates"] < 0:
        raise ValueError("EMA checkpoint has an invalid/missing num_updates")
    return metadata, stored


class ShardedEMA:
    """维护本地影子参数，保存时恢复 DTensor 分片信息。

    在模型加载、FSDP 包装及 optimizer 初始化后构造；只缓存参数引用，
    不缓存本地 tensor view。更新不修改 raw 参数，也不产生跨卡通信。
    """

    def __init__(self, model, decays, device_mesh):
        self.decays = {decay_key(d): float(d) for d in decays}
        if not self.decays or len(self.decays) != len(decays):
            raise ValueError("EMA decays must be nonempty and unique")
        self.num_updates = 0
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not self.params:
            raise ValueError("EMA requires trainable parameters")
        for name, param in self.params.items():
            if local_tensor(param).device.type != "cuda":
                raise ValueError(f"EMA requires GPU resident parameters: {name}")
        self.signature = {
            "version": 1,
            "decays": self.decays,
            "dtype": "float32",
            "start_step": 0,
            "mesh": device_mesh.mesh.tolist(),                  # 如 [2, 8]
            "mesh_dim_names": list(device_mesh.mesh_dim_names), # 如 ["ddp", "fsdp"]
            "parameters": {
                n: {
                    "shape": list(param.shape),
                    "placements": [str(p) for p in param.placements]
                    if isinstance(param, DTensor)
                    else [],
                }
                for n, param in self.params.items()
            },
        }
        self.shadows = {
            key: {
                n: local_tensor(param).detach().to(torch.float32).clone()   # 取本地分片、切断梯度传播、强制转为fp32、创建独立存储
                for n, param in self.params.items()
            }
            for key in self.decays
        }

        self.local_bytes = sum(
            t.numel() * t.element_size()
            for shadows in self.shadows.values()
            for t in shadows.values()
        )

    @torch.no_grad()
    def update(self):
        """成功执行 optimizer step 后调用，更新所有 decay 和共享计数。"""
        # 逐参数转为 FP32，限制临时显存；每次获取当前的本地分片。
        for name, param in self.params.items():
            value = local_tensor(param).detach().to(torch.float32)
            for key, decay in self.decays.items():
                self.shadows[key][name].lerp_(value, 1.0 - decay)
        self.num_updates += 1

    def validate_metadata(self, metadata):
        """检查 decay、参数和拓扑一致，不允许重新构造缺失的历史。"""
        for field, expected in self.signature.items():
            if metadata.get(field) != expected:
                actual = metadata.get(field)
                if field == "parameters":  # 找到具体的差异
                    actual = actual if isinstance(actual, dict) else {}
                    changed = [
                        name for name in expected if actual.get(name) != expected[name]
                    ]
                    extra = sorted(set(actual) - set(expected))
                    detail = f"missing/changed={changed[:8]}, unexpected={extra[:8]}"
                else:
                    detail = f"configured={expected!r}, saved={actual!r}"
                raise ValueError(
                    f"EMA resume mismatch in {field}: {detail}; requires the same decays, "
                    "trainable parameters and FSDP topology as the saved run."
                )

    def preflight(self, path):
        """加载训练状态前，校验 EMA 元数据和每个张量的 shape/dtype。"""
        metadata, stored = read_ema_metadata(dcp.FileSystemReader(path))
        self.validate_metadata(metadata)
        for key in self.decays:
            for name, info in self.signature["parameters"].items():
                fqn = f"ema.weights.{key}.{name}"
                entry = stored.get(fqn)
                if entry is None:
                    raise ValueError(f"EMA checkpoint is missing {fqn}")
                if (
                    list(entry.size) != info["shape"]
                    or entry.properties.dtype != torch.float32
                ):
                    raise ValueError(f"EMA checkpoint shape/dtype mismatch: {fqn}")

    def state_dict(self):
        """返回供 DCP 保存的 CPU 分片及元数据，保留原参数的分片布局。"""
        weights = {}
        for key, shadows in self.shadows.items():
            weights[key] = {}
            for name, shadow in shadows.items():    # 遍历每一个ema
                param = self.params[name]
                if isinstance(param, DTensor):
                    value = DTensor.from_local(
                        shadow,
                        param.device_mesh,
                        param.placements,
                        run_check=False,
                        shape=param.shape,
                        stride=param.stride(),
                    )
                else:
                    value = shadow
                weights[key][name] = value.cpu()
        return {
            "metadata": json.dumps(
                {**self.signature, "num_updates": self.num_updates}, sort_keys=True
            ),
            "weights": weights,
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        """将 DCP 加载的 EMA 分片复制回 GPU，并恢复成功更新次数。"""
        metadata = json.loads(state["metadata"])
        self.validate_metadata(metadata)
        for key, shadows in self.shadows.items():
            for name, shadow in shadows.items():
                shadow.copy_(local_tensor(state["weights"][key][name]))
        self.num_updates = metadata["num_updates"]
