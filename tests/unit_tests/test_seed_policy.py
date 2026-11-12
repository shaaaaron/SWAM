# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import numpy as np
import torch
from omegaconf import OmegaConf

import rlinf.workers.sft.fsdp_sft_worker as worker_module
from rlinf.utils.utils import seed_everything
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


def _two_phase_draw(base_seed: int, rank: int):
    seed_everything(base_seed)
    model = torch.nn.Linear(4, 4)
    parameters = torch.cat([value.detach().flatten() for value in model.parameters()])

    runtime_seed = base_seed + rank
    seed_everything(runtime_seed)
    runtime_values = (
        random.random(),
        float(np.random.rand()),
        torch.randn(8),
        torch.randint(0, 1000, (8,)),
    )
    return parameters, runtime_values


def test_two_phase_seed_keeps_model_equal_and_runtime_rank_specific():
    rank0_model, rank0_runtime = _two_phase_draw(42, rank=0)
    rank1_model, rank1_runtime = _two_phase_draw(42, rank=1)
    rank0_model_repeat, rank0_runtime_repeat = _two_phase_draw(42, rank=0)

    torch.testing.assert_close(rank0_model, rank1_model, rtol=0, atol=0)
    torch.testing.assert_close(rank0_model, rank0_model_repeat, rtol=0, atol=0)
    assert rank0_runtime[0] != rank1_runtime[0]
    assert rank0_runtime[1] != rank1_runtime[1]
    assert not torch.equal(rank0_runtime[2], rank1_runtime[2])
    assert not torch.equal(rank0_runtime[3], rank1_runtime[3])
    assert rank0_runtime[0] == rank0_runtime_repeat[0]
    assert rank0_runtime[1] == rank0_runtime_repeat[1]
    torch.testing.assert_close(
        rank0_runtime[2], rank0_runtime_repeat[2], rtol=0, atol=0
    )
    torch.testing.assert_close(
        rank0_runtime[3], rank0_runtime_repeat[3], rtol=0, atol=0
    )


def test_sft_worker_switches_seed_after_model_setup(monkeypatch):
    events = []
    worker = object.__new__(FSDPSftWorker)
    worker._rank = 3
    worker._model_initialization_seed = 42
    worker._training_runtime_seed = 42 + 3
    worker.cfg = OmegaConf.create({"actor": {"enable_offload": False}})
    worker.setup_model_and_optimizer = lambda: events.append("setup")

    def record_seed(seed):
        events.append(("seed", seed))
        return seed

    monkeypatch.setattr(worker_module, "seed_everything", record_seed)

    worker.init_worker()

    assert events == [
        ("seed", 42),
        "setup",
        ("seed", 42 + 3),
    ]
