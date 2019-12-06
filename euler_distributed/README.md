## euler_distributed  
  
  
**Note**: currently using NCCL only, thus requiring GPU(s).  
  
### common examples: 
```sh
srun -p batch_default -n8 -t2-0 --ntasks-per-node 4 --gres gpu:4 python test.py  
srun -J my_job --exclusive -n4 -t2-0 --gres gpu:4 python train.py  

# if you want to run your scripts locally (i.e. without Slurm environment), use lrun:  
lrun -n4 python test.py  
```  
