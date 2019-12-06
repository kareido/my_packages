import torch
import torch.distributed as dist
from ._slurm_env import *


def all_reduce_mean(tensor_list, group=None):
    if group is None:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.SUM)
    else:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.SUM, group)
    if isinstance(tensor_list, torch.Tensor):
        raise TypeError('tensor_list should be list of tensors')
    if get_world_size() == 1: return
    for tensor in tensor_list:
        _allreduce(tensor)
        tensor.div_(dist.get_world_size())


def all_reduce_sum(tensor_list, group=None):
    if group is None:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.SUM)
    else:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.SUM, group)
    if isinstance(tensor_list, torch.Tensor):
        raise TypeError('tensor_list should be list of tensors')
    if get_world_size() == 1: return
    for tensor in tensor_list:
         _allreduce(tensor)


def all_reduce_max(tensor_list, group=None):
    if group is None:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.MAX)
    else:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.MAX, group)
    if isinstance(tensor_list, torch.Tensor):
        raise TypeError('tensor_list should be list of tensors')
    if get_world_size() == 1: return
    for tensor in tensor_list:
        _allreduce(tensor)


def all_reduce_min(tensor_list, group=None):
    if group is None:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.MAX)
    else:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.MAX, group)
    if isinstance(tensor_list, torch.Tensor):
        raise TypeError('tensor_list should be list of tensors')
    if get_world_size() == 1: return
    for tensor in tensor_list:
        tensor.neg_()
        _allreduce(tensor)
        tensor.neg_()


def broadcast(tensor_list, src, group=None):
    if group is None:
        _broadcast = lambda tensor: dist.broadcast(tensor, src)
    else:
        _broadcast = lambda tensor: dist.broadcast(tensor, src, group)
    if get_world_size() == 1: return
    for tensor in tensor_list:
        _broadcast(tensor)


def barrier(group=None):
    if get_world_size() == 1: return
    if group is None:
        _barrier = dist.barrier
    else:
        _barrier = lambda: dist.barrier(group)
    _barrier()


def all_reduce(tensor_list, op='SUM', group=None):
    switch_dict = {
        'SUM': all_reduce_sum,
        'MEAN': all_reduce_mean,
        'MAX': all_reduce_max,
        'MIN': all_reduce_min,
    }
    assert type(op) == str, 'op should be a string'
    assert op in switch_dict, 'unknown operation {}'.format(op)

    switch_dict['op'](tensor_list, group=group)


def sync_state(network, src=0):
    if get_world_size() == 1: return
    tensor_list = list(network.state_dict().values())
    if src == 'all':
        all_reduce_mean(tensor_list)
    else:
        broadcast(tensor_list, src)


def sync_grad_mean(network):
    if get_world_size() == 1: return
    all_reduce_mean([param.grad.data for param in network.parameters() if param.grad is not None])


def sync_grad_sum(network):
    if get_world_size() == 1: return
    all_reduce_sum([param.grad.data for param in network.parameters() if param.grad is not None])


def sync_bn_stat(network):
    if get_world_size() == 1: return
    tensor_list = []
    for mod in network.modules():
        if type(mod) == torch.nn.BatchNorm2d:
            tensor_list.append(mod.running_mean)
            tensor_list.append(mod.running_var)
    all_reduce_mean(tensor_list)



