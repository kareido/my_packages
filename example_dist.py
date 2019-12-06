import torch
import euler_distributed as edist

"""distributed environment initialization and retrieve environment variables"""    
edist.slurm_data_parallel_arch(port=23032, backend='nccl')
rank, world_size = edist.get_rank(), edist.get_world_size()

"""rank == 0 ensures only 1 print to the console"""    
if rank == 0:
    print('this is an example for the package euler_distributed')
    print('you can launch it via srun or lrun (with GPU enabled)')
    print('e.g. srun -p batch_default -n4 -ntasks-per-node2 --gres gpu:2 python example_dist.py')
    print('e.g. lrun -n4 python example_dist.py')

    print('\nenvironment summaries:')
    print(' |- world size: {}'.format(world_size))

"""this to ensure all other ranks are waiting until rank 0 finishes above printings """    
edist.barrier()
print(
    ' |- this is rank [{}/{}] with CUDA ID [{}/{}] speaking'.format(
    rank, world_size, torch.cuda.current_device(), torch.cuda.device_count())
)

if rank == 0:
    print('creating sample tensors on each rank ...')
edist.barrier()

test_tensor = torch.randn(8).cuda()
print('[rank {}/{}] my test_tensor is: ', test_tensor)

test_copy = test_tensor.clone().cuda()
all_reduce_sum(test_copy)
print('[rank {}/{}] after all_reduce_sum: ', test_copy)

test_copy = test_tensor.clone().cuda()
all_reduce_mean(test_copy)
print('[rank {}/{}] after all_reduce_mean: ', test_copy)

test_copy = test_tensor.clone().cuda()
all_reduce_max(test_copy)
print('[rank {}/{}] after all_reduce_max: ', test_copy)

test_copy = test_tensor.clone().cuda()
all_reduce_min(test_copy)
print('[rank {}/{}] after all_reduce_min: ', test_copy)
