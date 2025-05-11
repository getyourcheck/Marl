import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    tensor = torch.ones(1).cuda(rank)
    dist.all_reduce(tensor)
    print(f'Rank {rank}: {tensor}')

def init_process(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    dist.destroy_process_group()

if __name__ == "__main__":
    size = torch.cuda.device_count()
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join() 