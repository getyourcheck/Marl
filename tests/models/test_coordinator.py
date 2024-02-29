import os
import pytest
from marl.coordinator import Coordinator
from marl.config import Config
import ray


@pytest.mark.parametrize(
    "configs",
    [
        ["auto", "projects/ppo/internlm2/1B/actor_2gpu.py", 1],
        ["ray://172.31.2.142:10001", "projects/ppo/internlm2/1B/actor_2gpu.py", 1],
    ],
)
def test_coordinator(
    configs: list,
):
    cluster_address = configs[0]
    model_configs_path = configs[1]
    gpu_num_per_actor = configs[2]
    model_configs = Config.from_file(model_configs_path)
    coordinator = Coordinator(cluster_address, model_configs)
    model_dict = coordinator.create_models()
    actor_model = model_dict["actor"]
    trainer = actor_model.trainer
    for actor in trainer.ray_actors:
        meta_data = ray.get(actor.get_metadata.remote())
        assert meta_data.gpu_num == gpu_num_per_actor
    coordinator.clean_up()
