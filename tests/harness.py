import os
import socket
import sys
from collections.abc import Callable
from datetime import timedelta

import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int, fn: Callable, args: tuple) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=30))
    try:
        fn(rank, world, *args)
        dist.barrier()
    except BaseException:
        dist.destroy_process_group()
        raise
    # Gloo's teardown can abort on Linux after the work is done; every rank is past the barrier.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def run_distributed(world: int, fn: Callable, *args) -> None:
    for attempt in range(3):
        try:
            mp.spawn(_worker, args=(world, _free_port(), fn, args), nprocs=world, join=True)
            return
        except mp.ProcessRaisedException as error:
            if "EADDRINUSE" not in str(error) or attempt == 2:
                raise
