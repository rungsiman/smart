import os
import torch.distributed as dist


def init_process(rank, world_size, experiment, backend='nccl'):
    os.environ['MASTER_ADDR'] = experiment.ddp_master_address
    os.environ['MASTER_PORT'] = experiment.ddp_master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
