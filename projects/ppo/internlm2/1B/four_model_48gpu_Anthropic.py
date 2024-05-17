import torch
from marl.model_backend.models.internlm2_reward import (
    InternLM2ForRewardModel,
    InternLM2ForCriticModel,
)
from marl.model_backend.models.modeling_internlm2_p import InternLM2ForCausalLM

rollout_config=dict(
    actor_micro_bs=32,
    reward_micro_bs=32,
    clip_reward_min=-5,
    clip_reward_max=5,
    max_new_tokens=1024,
    generate_kwargs={
        "do_sample":True,
        "temperature":1.0,
        "top_k": 0,
        "top_p": 0.9,
        "pad_token_id": 0,
        "eos_token_id": 92542,
        "early_stopping": True,
        "num_beams":1,
        "min_new_tokens":1,
    }
)

repeater_config=dict(
    actor_micro_bs=8,
    ref_micro_bs=8,
    critic_micro_bs=32,
    reward_scale=False,
    fine_grained_rm=False,
    value_ema=False,
    kl_coeff = 0.01,
    gamma = 1.0,
    gae_lambda = 0.99,
    answer_end_id = 92542,
    norm_rewards = True,
)
train_config=dict(
    ppo_minibatch=512,
    value_minibatch=512,
    actor_micro_bs=2,
    critic_micro_bs=2,
    pretrain_step=20,
    save_interval=800,
)

model_configs=dict(
    actor = dict(
        model_path="/cpfs01/shared/public/public_hdd/lishuaibin/models/1.8B_baseline/sft/Luyou_1B_FT_0.19_130_avg5/",
        model_type="actor",
        model_class=InternLM2ForCausalLM,
        use_flash_attn=False,
        trainer_config=dict(
            trainer_type="huggingface",
            torch_dtype=torch.float32,
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_seq",
            ),
            parallel=dict(
                data=dict(size=32, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                "zero_optimization": {
                    "stage": 2, 
                    "offload_param": {
                        "device": "none"
                    },
                    "reduce_bucket_size": "auto", 
                    "zero_hpz_partition_size": 1, 
                    "zero_quantized_weights": False, 
                    "zero_quantized_gradients": False
                }, 
                "bf16": {
                    "enabled": True
                }, 
                "gradient_clipping": 1.0, 
                "prescale_gradients": False, 
                "wall_clock_breakdown": False, 
                "data_types": {
                    "grad_accum_dtype": "fp32"
                }, 
                "train_micro_batch_size_per_gpu": 2, 
                "gradient_accumulation_steps": 8,
                "train_batch_size": 512
            }
        ),
        generator_config=dict(
            shared_with_trainer=True,
        ),
    ),

    reference = dict(
        model_path="/cpfs01/shared/public/public_hdd/lishuaibin/models/1.8B_baseline/sft/Luyou_1B_FT_0.19_130_avg5/",
        model_type="reference",
        model_class=InternLM2ForCausalLM,
        use_flash_attn=False,
        trainer_config=dict(
            torch_dtype=torch.float32,
            trainer_type="huggingface",
            parallel=dict(
                data=dict(size=4, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),

    critic = dict(
        model_path="/cpfs01/shared/public/public_hdd/lvchengqi/ckpts/reward_model/R-Luyou-1B-8k-D20240130-v1/819_hf_bf16/",
        model_type="critic",
        model_class=InternLM2ForCriticModel,
        use_flash_attn=False,
        trainer_config=dict(
            torch_dtype="auto",
            trainer_type="huggingface",
            train_kwargs=dict(
                micro_bsz=1,
                lr=5e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_seq",
            ),
            parallel=dict(
                data=dict(size=8, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                "zero_optimization": {
                    "stage": 2, 
                    "offload_param": {
                        "device": "none"
                    },
                    "reduce_bucket_size": "auto", 
                    "zero_hpz_partition_size": 1, 
                    "zero_quantized_weights": False, 
                    "zero_quantized_gradients": False
                }, 
                "bf16": {
                    "enabled": True
                }, 
                "gradient_clipping": 1.0, 
                "prescale_gradients": False, 
                "wall_clock_breakdown": False, 
                "data_types": {
                    "grad_accum_dtype": "fp32"
                }, 
                "train_micro_batch_size_per_gpu": 2, 
                "gradient_accumulation_steps": 32,
                "train_batch_size": 512
            }
        ),
    ),

    reward = dict(
        model_path="/cpfs01/shared/public/public_hdd/lvchengqi/ckpts/reward_model/R-Luyou-1B-8k-D20240130-v1/819_hf_bf16/",
        model_type="reward",
        model_class=InternLM2ForRewardModel,
        use_flash_attn=False,
        trainer_config=dict(
            trainer_type="huggingface",
            torch_dtype="auto",
            parallel=dict(
                data=dict(size=4, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),
)


dataset_config = {
        "num_samples_each_epoch": 512,
        "max_seq_len": 1024,
        "random_seed": 1024,
        "ppo_datas": [
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/new_arrow_messages_data/Anthropic_hh-rlhf_helpful-base-train.json::1.0",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/new_arrow_messages_data/Anthropic_hh-rlhf_harmless-base-train.json::0.5",
            ],
    }