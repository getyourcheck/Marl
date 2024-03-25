import torch

actor = dict(
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
# deepspeed configs, see:
# https://www.deepspeed.ai/docs/config-json
# https://lightning.ai/docs/fabric/stable/api/generated/lightning.fabric.strategies.DeepSpeedStrategy.html
