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

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

import rlinf.workers.sft.fsdp_vla_sft_worker as worker_module
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


class _RangeDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return index


class _EpochSampler(Sampler):
    def __init__(self, events):
        self.events = events
        self.epoch = 0

    def __iter__(self):
        start = self.epoch * 100
        return iter(range(start, start + 12))

    def __len__(self):
        return 12

    def set_epoch(self, epoch):
        self.events.append(("set_epoch", epoch))
        self.epoch = epoch


def _build_worker(events, num_workers=0):
    sampler = _EpochSampler(events)
    loader = StatefulDataLoader(
        _RangeDataset(),
        batch_size=2,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    original_load_state_dict = loader.load_state_dict

    def record_load_state_dict(state):
        events.append(("load_state_dict", sampler.epoch))
        return original_load_state_dict(state)

    loader.load_state_dict = record_load_state_dict
    worker = object.__new__(FSDPVlaSftWorker)
    worker.data_loader = loader
    worker.data_iter = iter(loader)
    worker._rank = 0
    worker._world_size = 1
    worker._data_epoch = 0
    worker._data_iter_offset = 0
    return worker, loader


def _patch_base_checkpoint(monkeypatch):
    monkeypatch.setattr(
        FSDPSftWorker, "load_checkpoint", lambda self, path: None
    )
    monkeypatch.setattr(
        FSDPSftWorker, "save_checkpoint", lambda self, path, step: None
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)


def test_save_checkpoint_writes_rank_data_progress(tmp_path, monkeypatch):
    _patch_base_checkpoint(monkeypatch)

    def gather_single_rank(output, value):
        output[0] = value

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_single_rank)
    monkeypatch.setattr(worker_module, "get_rng_state", lambda: {"seed": 7})
    worker, _loader = _build_worker([])
    worker._data_epoch = 2
    worker._data_iter_offset = 3

    worker.save_checkpoint(str(tmp_path), step=11)

    progress = torch.load(tmp_path / "data_progress.pt", weights_only=False)
    assert progress == [{"data_epoch": 2, "data_iter_offset": 3}]
    assert (tmp_path / "data.pt").is_file()
    assert (tmp_path / "rng.pt").is_file()


@pytest.mark.parametrize("num_workers", [0, 2])
def test_load_checkpoint_restores_epoch_before_loader_state(
    tmp_path, monkeypatch, num_workers
):
    _patch_base_checkpoint(monkeypatch)
    events = []
    worker, loader = _build_worker(events, num_workers=num_workers)
    loader.sampler.set_epoch(2)
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    torch.save([loader.state_dict()], tmp_path / "data.pt")
    torch.save(
        [{"data_epoch": 2, "data_iter_offset": 2}],
        tmp_path / "data_progress.pt",
    )
    torch.save([{"seed": 9}], tmp_path / "rng.pt")
    events.clear()
    worker.data_loader.sampler.set_epoch(0)
    events.clear()
    monkeypatch.setattr(worker_module, "set_rng_state", lambda state: None)

    worker.load_checkpoint(str(tmp_path))

    assert events == [("set_epoch", 2), ("load_state_dict", 2)]
    assert worker._data_epoch == 2
    assert worker._data_iter_offset == 2
    assert next(worker.data_iter).tolist() == [204, 205]


def test_load_checkpoint_at_epoch_end_advances_to_next_epoch(
    tmp_path, monkeypatch
):
    _patch_base_checkpoint(monkeypatch)
    worker, loader = _build_worker([])
    loader.sampler.set_epoch(2)
    iterator = iter(loader)
    for _ in range(len(loader)):
        next(iterator)
    torch.save([loader.state_dict()], tmp_path / "data.pt")
    torch.save(
        [{"data_epoch": 2, "data_iter_offset": len(loader)}],
        tmp_path / "data_progress.pt",
    )
    torch.save([{"seed": 9}], tmp_path / "rng.pt")
    monkeypatch.setattr(worker_module, "set_rng_state", lambda state: None)

    worker.load_checkpoint(str(tmp_path))

    with pytest.raises(StopIteration):
        next(worker.data_iter)
    worker._data_epoch += 1
    worker.data_loader.sampler.set_epoch(worker._data_epoch)
    worker.data_iter = iter(worker.data_loader)
    worker._data_iter_offset = 1

    assert worker._data_epoch == 3
    assert worker._data_iter_offset == 1
    assert next(worker.data_iter).tolist() == [300, 301]


def test_load_checkpoint_requires_data_progress(tmp_path, monkeypatch):
    _patch_base_checkpoint(monkeypatch)
    base_load_calls = []
    monkeypatch.setattr(
        FSDPSftWorker,
        "load_checkpoint",
        lambda self, path: base_load_calls.append(path),
    )
    worker, loader = _build_worker([])
    torch.save([loader.state_dict()], tmp_path / "data.pt")

    with pytest.raises(FileNotFoundError, match="data_progress.pt"):
        worker.load_checkpoint(str(tmp_path))
    assert base_load_calls == []


@pytest.mark.parametrize(
    ("progress", "error", "message"),
    [
        ([], ValueError, "world size mismatch"),
        (
            [{"data_epoch": -1, "data_iter_offset": 0}],
            ValueError,
            "must be non-negative",
        ),
    ],
)
def test_load_checkpoint_rejects_invalid_progress(
    tmp_path: Path, monkeypatch, progress, error, message
):
    _patch_base_checkpoint(monkeypatch)
    base_load_calls = []
    monkeypatch.setattr(
        FSDPSftWorker,
        "load_checkpoint",
        lambda self, path: base_load_calls.append(path),
    )
    worker, loader = _build_worker([])
    torch.save([loader.state_dict()], tmp_path / "data.pt")
    torch.save(progress, tmp_path / "data_progress.pt")

    with pytest.raises(error, match=message):
        worker.load_checkpoint(str(tmp_path))
    assert base_load_calls == []


def test_load_checkpoint_rejects_data_state_world_size_before_model_load(
    tmp_path, monkeypatch
):
    _patch_base_checkpoint(monkeypatch)
    base_load_calls = []
    monkeypatch.setattr(
        FSDPSftWorker,
        "load_checkpoint",
        lambda self, path: base_load_calls.append(path),
    )
    worker, _loader = _build_worker([])
    torch.save([], tmp_path / "data.pt")
    torch.save(
        [{"data_epoch": 2, "data_iter_offset": 3}],
        tmp_path / "data_progress.pt",
    )

    with pytest.raises(ValueError, match="Dataloader state world size mismatch"):
        worker.load_checkpoint(str(tmp_path))
    assert base_load_calls == []


def test_load_checkpoint_requires_epoch_aware_sampler_before_model_load(
    tmp_path, monkeypatch
):
    _patch_base_checkpoint(monkeypatch)
    base_load_calls = []
    monkeypatch.setattr(
        FSDPSftWorker,
        "load_checkpoint",
        lambda self, path: base_load_calls.append(path),
    )
    worker, _loader = _build_worker([])
    worker.data_loader.sampler.set_epoch = None

    with pytest.raises(TypeError, match="does not support set_epoch"):
        worker.load_checkpoint(str(tmp_path))
    assert base_load_calls == []
