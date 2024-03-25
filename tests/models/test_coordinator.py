import ray
import torch
import pytest
from marl.coordinator import Coordinator
from marl.config import Config
from marl.model_backend.cuda_memory_stats import (
    CudaMemoryStats,
    merge_cuda_memory_stats_list,
)

DDP_1B_CONFIG = Config(
    dict(
        actor=dict(
            model_path="internlm/internlm2-chat-1_8b-sft",
            model_type="actor",
            torch_dtype=torch.float16,
            trainer_config=dict(
                trainer_type="huggingface",
                train_kwargs=dict(
                    micro_bsz=1,
                    lr=1e-6,
                    total_steps=1e9,
                    lr_decay_rate=1,
                    loss_type="per_token",
                ),
                parallel=dict(
                    data=dict(size=2, mode="ddp"),
                    tensor=dict(size=1, mode="1d"),
                    pipeline=dict(size=1, interleaved_overlap=False),
                    sequence=False,
                ),
            ),
            generator_config=dict(
                shared_with_trainer=True,
                generate_kwargs=dict(
                    max_new_tokens=64,
                ),
            ),
        ),
    )
)

DDP_7B_CONFIG = Config(DDP_1B_CONFIG)  # deepcopy
DDP_7B_CONFIG["actor"]["model_path"] = "internlm/internlm2-chat-7b-sft"

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
    cuda_mem_stats: CudaMemoryStats = merge_cuda_memory_stats_list(
        ray.get(
            [
                ray_actor.get_memory_stats_of_visible_devices.remote()
                for ray_actor in trainer_fdsp.ray_actors
            ]
        )
    )
    allocated_fdsp = cuda_mem_stats.total_current_bytes
    print("allocated_fdsp:", allocated_fdsp)
    coordinator_fsdp.clean_up()

    configs = Config.from_file("projects/ppo/internlm2/1B/actor_2gpu.py")
    coordinator = Coordinator("auto", configs)
    models = coordinator.create_models()
    trainer = models["actor"].trainer
    cuda_mem_stats: CudaMemoryStats = merge_cuda_memory_stats_list(
        ray.get(
            [
                ray_actor.get_memory_stats_of_visible_devices.remote()
                for ray_actor in trainer.ray_actors
            ]
        )
    )
    allocated = cuda_mem_stats.total_current_bytes
    print("allocated:", allocated)
    coordinator.clean_up()

    assert (allocated_fdsp < allocated), f"fdsp: {allocated_fdsp}\n no_fsdp: {allocated}"


DEEPSPEED_1B_CONFIG = Config(
    dict(
        actor=dict(
            model_path="internlm/internlm2-chat-1_8b-sft",
            torch_dtype=torch.float16,
            model_type="actor",
            trainer_config=dict(
                trainer_type="huggingface",
                train_kwargs=dict(
                    micro_bsz=1,
                    lr=1e-6,
                    total_steps=1e9,
                    lr_decay_rate=1,
                    loss_type="per_token",
                ),
                parallel=dict(
                    data=dict(size=2, mode="deepspeed"),
                    tensor=dict(size=1, mode="1d"),
                    pipeline=dict(size=1, interleaved_overlap=False),
                    sequence=False,
                ),
                deepspeed_config={
                    "fp16": {"enabled": True},
                    "bf16": {"enabled": False},
                    "zero_optimization": {
                        "stage": 3,
                        "offload_optimizer": {
                            "device": "cpu",
                        },  # default: {}
                        "offload_param": {
                            "device": "cpu",
                        },  # default: {}
                        "stage3_max_live_parameters": 1,  # default: 1e9
                        "stage3_max_reuse_distance": 1,  # default: 1e9
                        "stage3_prefetch_bucket_size": 1,  # default: 5e8
                        "stage3_param_persistence_threshold": 1,  # default: 1e9
                        "sub_group_size": 1,  # default: 1e12
                        "zero_quantized_gradients": True,  # default: False
                        "zero_quantized_weights": True,  # default: False
                    },
                    "gradient_accumulation_steps": 1,
                    "train_micro_batch_size_per_gpu": 1,
                    # "train_batch_size" == "gradient_accumulation_steps" * "train_micro_batch_size_per_gpu"
                    "gradient_clipping": 1.0,
                    "steps_per_print": 10,
                    "wall_clock_breakdown": False,
                },
            ),
            generator_config=dict(
                shared_with_trainer=True,
                generate_kwargs=dict(
                    max_new_tokens=64,
                ),
            ),
        )
    )
)

DEEPSPEED_7B_CONFIG = Config(DEEPSPEED_1B_CONFIG)  # deepcopy
DEEPSPEED_7B_CONFIG["actor"]["model_path"] = "internlm/internlm2-chat-7b-sft"


@pytest.mark.parametrize(
    "configs",
    [
        [DDP_1B_CONFIG, DEEPSPEED_1B_CONFIG],
        # [DDP_7B_CONFIG, DEEPSPEED_7B_CONFIG],
    ],
)
def test_coordinator_deepspeed(configs: list):
    configs_ddp = configs[0]
    coordinator = Coordinator("auto", configs_ddp)
    models = coordinator.create_models()
    trainer = models["actor"].trainer
    cuda_mem_stats: CudaMemoryStats = merge_cuda_memory_stats_list(
        ray.get(
            [
                ray_actor.get_memory_stats_of_visible_devices.remote()
                for ray_actor in trainer.ray_actors
            ]
        )
    )
    cuda_mem_ddp = cuda_mem_stats.total_current_mb
    print(f"cuda_mem_ddp: {cuda_mem_ddp} MB")
    coordinator.clean_up()

    # configs_deepspeed = Config.from_file("projects/ppo/internlm2/1B/actor_2gpu_deepspeed.py")
    configs_deepspeed = configs[1]
    coordinator_deepspeed = Coordinator("auto", configs_deepspeed)
    models_deepspeed = coordinator_deepspeed.create_models()
    trainer_deepspeed = models_deepspeed["actor"].trainer
    cuda_mem_stats: CudaMemoryStats = merge_cuda_memory_stats_list(
        ray.get(
            [
                ray_actor.get_memory_stats_of_visible_devices.remote()
                for ray_actor in trainer_deepspeed.ray_actors
            ]
        )
    )
    cuda_mem_ds = cuda_mem_stats.total_current_mb
    print(f"cuda_mem_ds: {cuda_mem_ds} MB")
    coordinator_deepspeed.clean_up()

    assert (
        cuda_mem_ds <= cuda_mem_ddp
    ), f"deepspeed's cuda mem: {cuda_mem_ds} MB should be lower than ddp's: {cuda_mem_ddp} MB"
