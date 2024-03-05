import os
import pytest
from marl.coordinator import Coordinator
from marl.config import Config
import ray


@pytest.mark.parametrize(
    "configs",
    [
        ["auto", "projects/ppo/internlm2/1B/actor_2gpu.py", 1]
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

def test_coordinator_fsdp():
    configs_fsdp = Config.from_file("projects/ppo/internlm2/1B/actor_2gpu_fsdp.py")
    coordinator_fsdp = Coordinator("auto", configs_fsdp)
    models_fdsp = coordinator_fsdp.create_models()
    trainer_fdsp = models_fdsp["actor"].trainer
    memory_stats_fdsp = ray.get(trainer_fdsp.ray_actors[0].get_memory_stats_of_visible_devices.remote())
    allocated_fdsp = list(memory_stats_fdsp.values())[0]["requested_bytes.all.allocated"]
    coordinator_fsdp.clean_up()
    
    configs = Config.from_file("projects/ppo/internlm2/1B/actor_2gpu.py")
    coordinator = Coordinator("auto", configs)
    models = coordinator.create_models()
    trainer = models["actor"].trainer
    memory_stats = ray.get(trainer.ray_actors[0].get_memory_stats_of_visible_devices.remote())
    allocated = list(memory_stats.values())[0]["requested_bytes.all.allocated"]
    coordinator.clean_up()

    assert (allocated_fdsp < allocated), f"fdsp: {allocated_fdsp}\n no_fsdp: {allocated}"