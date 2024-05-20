import torch

MAX_PROMPT_LEN=1536
MAX_ANSWER_LEN=512

DATA_BATCH_SIZE=512
GENERATE_MICRO_BATCH_SIZE=16  # 512 / 16dp = 32, 32 / 16 = 2 pass
AC_INFER_MICRO_BATCH_SIZE=8  # 512 / 16dp = 32, 32 / 8 = 4 pass
REF_INFER_MICRO_BATCH_SIZE=8 # 512 / 8 = 64 pass
TRAIN_MICRO_BATCH_SIZE=2  # 512 / 16dp = 32, 32 / 2 = 16 pass

ZERO_STAGE=3
ACTOR_DP_SIZE=16
CRITIC_DP_SIZE=8
ACTOR_GRADIENT_ACC_STEP=DATA_BATCH_SIZE // ACTOR_DP_SIZE // TRAIN_MICRO_BATCH_SIZE
CRITIC_GRADIENT_ACC_STEP=DATA_BATCH_SIZE // CRITIC_DP_SIZE // TRAIN_MICRO_BATCH_SIZE

MODEL_DTYPE="auto"

tokenizer_config = dict(
    pad_token_id = 0,
    eos_token_id = 92542,
    padding_side = 'left',
)

rollout_config = dict(
    actor_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    reward_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    clip_reward_min=-5,
    clip_reward_max=5,
    max_new_tokens=MAX_ANSWER_LEN,
    write_to_file=False,
    generate_kwargs={
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.9,
        "min_new_tokens": 1,
        "num_beams": 1,
        "early_stopping": True,
        "eos_token_id": 92542,
        "pad_token_id": 0,
    },
)

repeater_config = dict(
    actor_micro_bs=AC_INFER_MICRO_BATCH_SIZE,
    critic_micro_bs=AC_INFER_MICRO_BATCH_SIZE,
    ref_micro_bs=REF_INFER_MICRO_BATCH_SIZE,
    reward_scale=False,
    fine_grained_rm=False,
    value_ema=False,
    kl_coeff = 0.01,
    gamma = 1.0,
    gae_lambda = 0.99,
    answer_end_id = 92542,
    norm_rewards = True,
)

train_config = dict(
    ppo_minibatch=DATA_BATCH_SIZE,
    value_minibatch=DATA_BATCH_SIZE,
    actor_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    critic_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    pretrain_step=20,
    save_interval=80,
)

model_configs = dict(
    actor=dict(
        model_path="/fs-computility/llm/shared/models/internlm2-chat-20b-sft",
        model_type="actor",
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type="huggingface",
            use_flash_attn=True,
            gradient_checkpointing=True,
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_seq",
            ),
            parallel=dict(
                data=dict(size=ACTOR_DP_SIZE, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                "zero_optimization": {
                    "stage": ZERO_STAGE, 
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
                "gradient_accumulation_steps": ACTOR_GRADIENT_ACC_STEP,
                "train_micro_batch_size_per_gpu": TRAIN_MICRO_BATCH_SIZE,
            },
        ),
        generator_config=dict(
            shared_with_trainer=False,
            generator_type="vllm",
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=2, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),

    critic=dict(
        model_path="/fs-computility/llm/shared/marl/models/internlm2/20B/hf/R-Gauss_20B-8k-D20240204-v1_hf/",
        model_type="critic",
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type="huggingface",
            use_flash_attn=True,
            gradient_checkpointing=True,
            train_kwargs=dict(
                micro_bsz=1,
                lr=5e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_seq",
            ),
            parallel=dict(
                data=dict(size=CRITIC_DP_SIZE, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                "zero_optimization": {
                    "stage": ZERO_STAGE, 
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
                "gradient_accumulation_steps": CRITIC_GRADIENT_ACC_STEP,
                "train_micro_batch_size_per_gpu": TRAIN_MICRO_BATCH_SIZE,
            },
        ),
    ),

    reference=dict(
        model_path="/fs-computility/llm/shared/models/internlm2-chat-20b-sft",
        model_type="reference",
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type="huggingface",
            use_flash_attn=True,
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),

    reward=dict(
        model_path="/fs-computility/llm/shared/marl/models/internlm2/20B/hf/R-Gauss_20B-8k-D20240204-v1_hf/",
        model_type="reward",
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type="huggingface",
            use_flash_attn=True,
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),
)

dataset_config = {
    "num_samples_each_epoch": DATA_BATCH_SIZE,
    "max_seq_len": MAX_PROMPT_LEN,
    "random_seed": 1024,
    "ppo_datas": [
        "/fs-computility/llm/shared/marl/datasets/lishuaibin/ppo_data/new_arrow_messages_data/Anthropic_hh-rlhf_helpful-base-train.json::1.0",
        "/fs-computility/llm/shared/marl/datasets/lishuaibin/ppo_data/new_arrow_messages_data/Anthropic_hh-rlhf_harmless-base-train.json::0.5",
    ],
}