import multiprocessing as mp
import os
import torch
import torch.distributed as dist
from ._slurm_env import *


def _get_slurm_addr():

    node_list = get_nodelist()
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0: pos1 = len(node_list)
        pos2 = node_list.find(',', beg)
        if pos2 < 0: pos2 = len(node_list)
        node_list = node_list[:min(pos1, pos2)].replace('[', '')

    addr = node_list.replace('-', '.').split(',')[0]

    return addr


def slurm_data_parallel_arch(port=23032, backend='nccl', method='spawn'):
    os.environ['DISTRIBUTED_BACKEND'] = backend
    if mp.get_start_method(allow_none=True) != method:
        mp.set_start_method(method, force=True)

    rank, world_size = get_rank(), get_world_size()
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)

    if world_size == 1:
        rank, world_size = 0, 1
    else:
        os.environ['MASTER_PORT'] = str(port)
        os.environ['MASTER_ADDR'] = _get_slurm_addr()
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)

        # print(os.environ['MASTER_PORT'], os.environ['MASTER_ADDR'], os.environ['WORLD_SIZE'], os.environ['RANK'])

        dist.init_process_group(backend=backend)

    return rank, world_size


