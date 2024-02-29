import os
import torch
from typing import Optional
from .ray_actor_metadata import RayActorMetadata
from .net_utils import get_ip_hostname, get_free_port


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
