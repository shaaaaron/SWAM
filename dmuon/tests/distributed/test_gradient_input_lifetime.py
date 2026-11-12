"""Stress DMuon's real reduce path while the source allocation is reused.

Run with torchrun (two or four CUDA ranks), selecting dmuon via PYTHONPATH.
Pass an output directory as the first argument. Set TEST_ROUTE=sharded_adamw
to cover reduce-scatter input copies; the default covers owner-managed Muon.
The four-rank case uses a 2x2 HSDP mesh. Delaying the reduce stream makes a
missing source-buffer lifetime dependency reproducible without model training.
"""

import json
import os
import sys
from pathlib import Path

import dmuon
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
hybrid = world == 4
mesh = init_device_mesh(
    "cuda",
    (2, 2) if hybrid else (world,),
    mesh_dim_names=("replicate", "shard") if hybrid else ("shard",),
)
model = torch.nn.Sequential(torch.nn.Linear(1024, 1024, bias=False)).cuda()
dmuon.dedicate_params(
    model,
    mesh=mesh["shard"] if hybrid else mesh,
    replicate_mesh=mesh["replicate"] if hybrid else None,
    predicate=lambda name, param: True,
    compute_dtype=torch.bfloat16,
    param_policy={
        "defaults": {
            "route": os.environ.get("TEST_ROUTE", "muon"),
            "param_dtype": torch.bfloat16,
            "grad_dtype": torch.float32,
            "master_dtype": torch.float32,
        }
    },
    process_group_policy="isolated",
)
group = dmuon.get_comm_ctx(model).all_states[0].group
group.unshard()
group.wait_for_unshard()
param = group.params[0]
rows = []
for trial in range(4):
    param._reduced_grad = None
    param._unsharded_param.grad = torch.full(
        (1024, 1024), rank + 1, dtype=torch.bfloat16, device="cuda"
    )
    original_pointer = param._unsharded_param.grad.data_ptr()
    torch.cuda.synchronize()
    with torch.cuda.stream(group.comm_ctx.reduce_stream):
        torch.cuda._sleep(100_000_000)
    group.reduce_grads()
    # An unrelated computation may immediately reuse a freed default-stream
    # allocation. The reduction must retain its input until the cast finishes.
    poison = [torch.full((1024, 1024), -99, dtype=torch.bfloat16, device="cuda") for _ in range(8)]
    reused = any(t.data_ptr() == original_pointer for t in poison)
    group.wait_for_reduce()
    torch.cuda.synchronize()
    error = 0.0
    if os.environ.get("TEST_ROUTE") == "sharded_adamw":
        error = float((param._sharded_adamw_grad - (world + 1) / 2).abs().max())
    elif param.is_owner:
        error = float((param._reduced_grad - (world + 1) / 2).abs().max())
    rows.append({"trial": trial, "source_allocation_reused": reused, "max_error": error})
    del poison
report = {
    "rank": rank,
    "world_size": world,
    "owner": param.is_owner,
    "dependency": dmuon.__file__,
    "trials": rows,
    "passed": all(row["max_error"] == 0 for row in rows),
}
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
(destination / f"rank-{rank}.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
dist.barrier()
dist.destroy_process_group()
assert report["passed"], "reduce read a reused gradient allocation"
