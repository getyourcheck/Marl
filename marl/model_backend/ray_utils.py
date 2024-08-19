import uuid
import ray
from typing import TypeVar
from loguru import logger
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

DEFAULT_NUM_CPUS = 1
DEFAULT_NUM_GPUS = 1
T = TypeVar('T')
UUID = uuid.uuid4()  # may called multiple times in different ray instances


# Create Ray Actors
def create_ray_actors(
    name_prefix: str,
    config: dict,
    placement_group: PlacementGroup,
    trainer_class: T,
) -> list[T]:
    ray_actors = [_ for _ in range(placement_group.bundle_count)]
    for index in range(placement_group.bundle_count):
        ray_actors[index] = trainer_class.options(
            name=f'{name_prefix}_rank_{index}',
            namespace=f'{UUID}_{trainer_class.__class__.__name__}',
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=index,
            ),
            runtime_env=set_runtime_env(),
        ).remote(config)
    ray_actors = sort_ray_actors(name_prefix, ray_actors)
    return ray_actors


def set_runtime_env():
    runtime_env = {'env_vars': {'HF_ENDPOINT': 'https://hf-mirror.com'}}
    return runtime_env

def sort_ray_actors(name_prefix, unsorted_actors: list[T]) -> list[T]:
    sorted_actors = []
    metadata_refs = [actor.get_metadata.remote() for actor in unsorted_actors]
    metadatas = ray.get(metadata_refs)
    ip_index_map = {}
    for index, metadata in enumerate(metadatas):
        node_ip = metadata.node_ip
        if node_ip in ip_index_map:
            ip_index_map[node_ip].append(index)
        else:
            ip_index_map[node_ip] = [index]
    rank = -1
    for node_ip, indexes in ip_index_map.items():
        for index in indexes:
            rank = rank + 1
            sorted_actors.append(unsorted_actors[index])
            logger.info(f"rank_mapping: actor_name:[{name_prefix}_rank_{index}] -> rank:[{rank}]")
    return sorted_actors
