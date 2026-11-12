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

import logging
import os
import time
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import clear_memory, seed_everything


def _cuda_sync_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class FSDPSftWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()

        self._model_initialization_seed = int(self.cfg.actor.get("seed", 0))
        self._training_runtime_seed = self._model_initialization_seed + self._rank
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # set the global batch size, micro batch size, eval batch size and gradient accumulation
        self.global_batch_size = self.cfg.actor.global_batch_size
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.eval_batch_size = self.cfg.actor.get("eval_batch_size", 1)

        assert (
            self.global_batch_size % (self.micro_batch_size * self._world_size) == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"
        self.gradient_accumulation = (
            self.global_batch_size // self.micro_batch_size // self._world_size
        )

        # if train_data_paths is not set, the code will just eval the model
        if self.cfg.data.get("train_data_paths") is None:
            logging.warning("train_data_paths is not set, will just eval the model")
            assert self.cfg.data.get("val_data_paths") is not None, (
                "train_data_paths is not set, val_data_paths must be set"
            )
            self.data_loader = None
            self.data_iter = None
        else:
            self.data_loader, self.data_config = self.build_dataloader(
                self.cfg.data.train_data_paths, eval_dataset=False
            )
            self.data_iter = iter(self.data_loader)

        if self.cfg.data.get("val_data_paths") is not None:
            self.eval_data_loader, self.eval_data_config = self.build_dataloader(
                self.cfg.data.val_data_paths, eval_dataset=True
            )
        else:
            self.eval_data_loader = None

        self.global_step = 0
        # set the dataloader epoch and data iter offset
        self._data_epoch = 0
        self._data_iter_offset = 0
        self.profile_sft_step = bool(
            self.cfg.runner.get("profile_sft_step", False)
            or os.environ.get("RLINF_PROFILE_SFT_STEP", "0") == "1"
        )

    def init_worker(self):
        # DataLoader 构建可能消耗进程级 RNG 状态。模型创建前重新设置共享 seed，
        # 确保所有 FSDP rank 使用完全相同的随机初始化参数。
        seed_everything(self._model_initialization_seed)
        self.setup_model_and_optimizer()

        # 模型初始化和 FSDP 同步完成后，切换为各 rank 独立且可复现的 seed，
        # 用于 dropout、diffusion noise 和 timestep 等训练期随机过程。
        seed_everything(self._training_runtime_seed)
        logging.info(
            "SFT RNG policy: rank=%s model_seed=%s runtime_seed=%s",
            self._rank,
            self._model_initialization_seed,
            self._training_runtime_seed,
        )

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def set_global_step(self, global_step):
        self.global_step = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def get_max_steps_per_epoch(self):
        if self.data_loader is not None:
            return max(1, len(self.data_loader) // self.gradient_accumulation)
        return 0

    def run_eval(self):
        assert self.eval_data_loader is not None, "eval_data_loader is not set"

        # reset the eval_data_iter
        eval_data_iter = iter(self.eval_data_loader)

        with self.worker_timer():
            eval_step = len(eval_data_iter)
            eval_pbar = tqdm(
                initial=0,
                total=eval_step,
                desc="Evaluate Step",
                dynamic_ncols=True,
            )
            self.model.eval()
            total = eval_step * self.eval_batch_size
            correct = 0

            # get the next batch
            for _ in range(eval_step):
                correct += self.get_eval_model_output(next(eval_data_iter))
                eval_pbar.update(1)

            metrics = {
                "eval_accuracy": float(correct / max(1, total)),
            }
            metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
            return metrics

    def run_training(self):
        with self.worker_timer():
            self.model.train()

            metrics = {}
            if self.profile_sft_step:
                perf_metrics: dict[str, float] = {
                    "perf_data_time": 0.0,
                    "perf_forward_time": 0.0,
                    "perf_backward_time": 0.0,
                }
                step_start = time.perf_counter()

            for idx in range(self.gradient_accumulation):
                # set the gradient accumulation backward_ctx
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )

                if self.profile_sft_step:
                    data_start = time.perf_counter()
                try:
                    batch = next(self.data_iter)
                    self._data_iter_offset += 1
                except StopIteration:
                    self._data_epoch += 1
                    logging.info(
                        f"[INFO] data_iter exhausted, reset iterator self._data_epoch {self._data_epoch}"
                    )
                    if hasattr(self.data_loader, "sampler") and hasattr(
                        self.data_loader.sampler, "set_epoch"
                    ):
                        self.data_loader.sampler.set_epoch(self._data_epoch)
                    self.data_iter = iter(self.data_loader)
                    batch = next(self.data_iter)
                    self._data_iter_offset = 1
                if self.profile_sft_step:
                    perf_metrics["perf_data_time"] += time.perf_counter() - data_start

                if self.profile_sft_step:
                    _cuda_sync_if_available()
                    forward_start = time.perf_counter()
                loss, step_metrics = self.get_train_model_output(batch)
                if self.profile_sft_step:
                    _cuda_sync_if_available()
                    perf_metrics["perf_forward_time"] += (
                        time.perf_counter() - forward_start
                    )
                append_to_dict(metrics, step_metrics)

                loss = loss / self.gradient_accumulation
                if self.profile_sft_step:
                    _cuda_sync_if_available()
                    backward_start = time.perf_counter()
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()
                if self.profile_sft_step:
                    _cuda_sync_if_available()
                    perf_metrics["perf_backward_time"] += (
                        time.perf_counter() - backward_start
                    )

            # in one step do the optimizer step
            if self.profile_sft_step:
                _cuda_sync_if_available()
                optimizer_start = time.perf_counter()
            grad_norm, lr_list = self.optimizer_step()
            if self.profile_sft_step:
                _cuda_sync_if_available()
                perf_metrics["perf_optimizer_time"] = (
                    time.perf_counter() - optimizer_start
                )
                zero_grad_start = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            if self.profile_sft_step:
                perf_metrics["perf_zero_grad_time"] = (
                    time.perf_counter() - zero_grad_start
                )

            if self.profile_sft_step:
                scheduler_start = time.perf_counter()
            self.lr_scheduler.step()
            if self.profile_sft_step:
                perf_metrics["perf_scheduler_time"] = (
                    time.perf_counter() - scheduler_start
                )
            lr_value = self.optimizer.param_groups[0]["lr"]
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            extra_metrics = {}
            if self.profile_sft_step:
                perf_metrics["perf_total_time"] = time.perf_counter() - step_start
                perf_metrics["perf_micro_batches"] = float(self.gradient_accumulation)
                perf_metrics["perf_micro_batch_size"] = float(self.micro_batch_size)
                perf_metrics["perf_global_batch_size"] = float(self.global_batch_size)
                perf_metrics["perf_samples_per_sec_per_rank"] = (
                    self.global_batch_size / self._world_size
                ) / max(perf_metrics["perf_total_time"], 1.0e-12)
                extra_metrics = perf_metrics
            if self.last_dmuon_grad_norm is not None:
                extra_metrics.update(
                    {
                        "grad_norm_regular": self.last_dmuon_grad_norm.regular,
                        "grad_norm_muon": self.last_dmuon_grad_norm.muon,
                        "grad_clip_coefficient": self.last_dmuon_grad_norm.coefficient,
                    }
                )
            if self.ema is not None:
                extra_metrics["ema_updates"] = self.ema.num_updates
                extra_metrics["ema_local_gib"] = self.ema.local_bytes / 2**30
            append_to_dict(
                metrics,
                {
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                    **extra_metrics,
                },
            )

            if self.global_step > 0 and self.global_step % 1000 == 0:
                clear_memory()

            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            return train_metrics

    @abstractmethod
    def build_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_model_output(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_eval_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError
