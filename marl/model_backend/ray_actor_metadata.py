import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class RayActorMetadata:
    """Metadata for Ray actor.

    This information is expected to stay the same throughout the lifetime of actor.

    Args:
        node_ip (str): Node IP address that this actor is on.
        hostname (str): Hostname that this actor is on.
        gpu_ids (Optional[list[int]]): List of CUDA IDs available to this actor.
        gpu_num (int): Number of used GPUs of this actor.
    """

    node_ip: str
    hostname: str
    gpu_ids: Optional[list[int]]
    gpu_num: int

    def __str__(self) -> str:
        info = {
            "Node_IP": self.node_ip,
            "Hostname": self.hostname,
            "GPU_IDs": self.gpu_ids,
            "GPU_Num": self.gpu_num,
        }
        return json.dumps(info, indent=4, sort_keys=True)
