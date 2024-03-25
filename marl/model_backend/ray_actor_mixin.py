import os
import torch
from typing import Optional
from .cuda_memory_stats import CudaMemoryStats
from .net_utils import get_ip_hostname, get_free_port, get_ip
from .ray_actor_metadata import RayActorMetadata

class RayActorMixin:
    def inject_distribute_env(
        self,
        master_ip: Optional[str] = None,
        master_port: int = 0,
        rank_id: int = 0,
        world_size: int = 0,
    ) -> None:
        """
        Inject Environment Variables before training.

        Args:
            master_ip (Optional[str]): The ip address of the master node.
            master_port (int): The port on the master node used for dist_init.
            rank_id (int): The rank id of this actor.
            world_size (int): Number of Actors for DDP training.

        """
        os.environ["MASTER_ADDR"] = master_ip
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank_id)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"

    def get_metadata(self) -> RayActorMetadata:
        node_ip, hostname = get_ip_hostname()
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_num = torch.cuda.device_count()

        return RayActorMetadata(
            node_ip=node_ip,
            hostname=hostname,
            gpu_ids=gpu_ids,
            gpu_num=gpu_num,
        )

    def get_free_port(self):
        return get_free_port()

    def get_memory_stats_of_visible_devices(self) -> CudaMemoryStats:
        visible_gpu_ids = []
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        else:
            visible_gpu_ids = [str(index) for index in range(torch.cuda.device_count())]

        cuda_memory_stats = CudaMemoryStats()
        for index, gpu_id in enumerate(visible_gpu_ids):
            status = torch.cuda.memory_stats(device=index)
            node_ip = get_ip()
            cuda_memory_stats[f"ip{node_ip}-gpu{gpu_id}"] = status
        return cuda_memory_stats
