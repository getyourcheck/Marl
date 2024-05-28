import torch

MAX_PROMPT_LEN=1536
MAX_ANSWER_LEN=512

DATA_BATCH_SIZE=512
GENERATE_MICRO_BATCH_SIZE=8
INFER_MICRO_BATCH_SIZE=8
TRAIN_MICRO_BATCH_SIZE=2

ZERO_STAGE=3
TRAIN_DP_SIZE=8
INFER_DP_SIZE=1
VLLM_TP=2
GRADIENT_ACC_STEP=DATA_BATCH_SIZE // TRAIN_DP_SIZE // TRAIN_MICRO_BATCH_SIZE

MODEL_DTYPE=torch.bfloat16

tokenizer_config = dict(
    pad_token_id = 0,
    eos_token_id = 92542,
    padding_side = 'left',
)

rollout_config = dict(
    write_to_file=False,
    actor_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    reward_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    clip_reward_min=-5,
    clip_reward_max=5,
    max_new_tokens=MAX_ANSWER_LEN,
    async_reward = True,
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
    actor_micro_bs=INFER_MICRO_BATCH_SIZE,
    ref_micro_bs=INFER_MICRO_BATCH_SIZE,
    critic_micro_bs=INFER_MICRO_BATCH_SIZE,
    reward_scale=False,
    fine_grained_rm=False,
    value_ema=False,
    kl_coeff=0.01,
    gamma=1.0,
    gae_lambda=0.99,
    answer_end_id=92542,
    norm_adv=True,
)

train_config = dict(
    ppo_minibatch=DATA_BATCH_SIZE,
    value_minibatch=DATA_BATCH_SIZE,
    actor_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    critic_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    pretrain_step=0,
    save_interval=80,
    step_interval=1,
)

model_configs = dict(
    actor=dict(
        model_path="internlm/internlm2-chat-7b-sft",
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
                data=dict(size=TRAIN_DP_SIZE, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                "bf16": {"enable": True},
                "fp16": {"enable": False},
                "zero_optimization": {
                    "stage": ZERO_STAGE,
                    "stage3_gather_16bit_weights_on_model_save": True,
                },
                "gradient_accumulation_steps": GRADIENT_ACC_STEP,
                "train_micro_batch_size_per_gpu": TRAIN_MICRO_BATCH_SIZE,
            },
        ),
        generator_config=dict(
            shared_with_trainer=False,
            generator_type="vllm",
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=VLLM_TP, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),

    critic=dict(
        model_path="/fs-computility/llm/shared/marl/models/internlm2/7B/hf/R-Ampere-7B-8k-D20240126-v1_hf/",
        model_type="critic",
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
                data=dict(size=TRAIN_DP_SIZE, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                "bf16": {"enable": True},
                "fp16": {"enable": False},
                "zero_optimization": {
                    "stage": ZERO_STAGE,
                },
                "gradient_accumulation_steps": GRADIENT_ACC_STEP,
                "train_micro_batch_size_per_gpu": TRAIN_MICRO_BATCH_SIZE,
            },
        ),
    ),

    reward=dict(
        model_path="/fs-computility/llm/shared/marl/models/internlm2/7B/hf/R-Ampere-7B-8k-D20240126-v1_hf/",
        model_type="reward",
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type="huggingface",
            parallel=dict(
                data=dict(size=INFER_DP_SIZE, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),

    reference=dict(
        model_path="internlm/internlm2-chat-7b-sft",
        model_type="reference",
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type="huggingface",
            parallel=dict(
                data=dict(size=INFER_DP_SIZE, mode="ddp"),
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