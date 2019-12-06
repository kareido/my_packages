import torch
import euler_distributed as edist

"""distributed environment initialization and retrieve environment variables"""
edist.slurm_data_parallel_arch(port=23032, backend='nccl')
rank, world_size = edist.get_rank(), edist.get_world_size()

"""rank == 0 ensures only 1 print to the console"""
if rank == 0:
    print('this is an example for the package euler_distributed', flush=True)
    print('you can launch it via srun or lrun (with GPU enabled)', flush=True)
    print('e.g. srun -p batch_default -n4 -ntasks-per-node2 --gres gpu:2 python example_dist.py', flush=True)
    print('e.g. lrun -n4 python example_dist.py', flush=True)

    print('\nenvironment summaries:', flush=True)
    print(' |- world size: {}'.format(world_size), flush=True)

"""this to ensure all other ranks are waiting until rank 0 finishes above printings """

edist.barrier()
print(
    ' |- this is rank [{}/{}] with CUDA ID [{}/{}] speaking'.format(
    rank, world_size, torch.cuda.current_device(), torch.cuda.device_count()),
    flush=True,
)
edist.barrier()

if rank == 0:
    print('creating sample tensors on each rank ...', flush=True)
edist.barrier()

test_tensor = torch.randn(6).cuda()
print('[rank {}] my test_tensor is: {}'.format(rank, test_tensor), flush=True)
edist.barrier()

test_copy = test_tensor.clone().cuda()
edist.all_reduce_sum([test_copy])
print('[rank {}] after all_reduce_sum: {}'.format(rank, test_copy), flush=True)
edist.barrier()

test_copy = test_tensor.clone().cuda()
edist.all_reduce_mean([test_copy])
print('[rank {}] after all_reduce_mean: {}'.format(rank, test_copy), flush=True)
edist.barrier()

test_copy = test_tensor.clone().cuda()
edist.all_reduce_max([test_copy])
print('[rank {}] after all_reduce_max: {}'.format(rank,  test_copy), flush=True)

test_copy = test_tensor.clone().cuda()
edist.all_reduce_min([test_copy])
print('[rank {}] after all_reduce_min: {}'.format(rank,  test_copy), flush=True)
edist.barrier()

print('all tests finished.')

