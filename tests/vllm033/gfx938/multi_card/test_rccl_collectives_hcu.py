# SPDX-License-Identifier: Apache-2.0
"""Exercise real single-node collectives on the two HCU cards assigned to CI."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


pytestmark = [pytest.mark.hcu, pytest.mark.multi_hcu, pytest.mark.hcu_count(2)]


def _check_collectives(rank, init_method):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    # torch uses the NCCL backend name for RCCL on ROCm.
    dist.init_process_group(
        "nccl", init_method=init_method, rank=rank, world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        value = torch.tensor([rank + 1], device=device, dtype=torch.float32)
        gathered = [torch.empty_like(value) for _ in range(2)]
        dist.all_gather(gathered, value)
        assert [item.item() for item in gathered] == [1.0, 2.0]
        dist.all_reduce(value)
        assert value.item() == 3.0
    finally:
        dist.destroy_process_group()


def test_rccl_two_card_collectives(tmp_path):
    assert torch.version.hip, "This contract requires the HCU/ROCm build of Torch"
    assert torch.cuda.device_count() == 2, "Allocate exactly two visible HCU cards"
    assert dist.is_nccl_available(), "RCCL backend is required"
    mp.spawn(_check_collectives, args=((tmp_path / "rccl-rendezvous").as_uri(),),
             nprocs=2, join=True)
