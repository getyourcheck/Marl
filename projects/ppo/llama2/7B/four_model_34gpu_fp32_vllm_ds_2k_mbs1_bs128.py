import torch

MAX_PROMPT_LEN=1024
MAX_ANSWER_LEN=1024

DATA_BATCH_SIZE=128
GENERATE_MICRO_BATCH_SIZE=8
INFER_MICRO_BATCH_SIZE=GENERATE_MICRO_BATCH_SIZE
TRAIN_MICRO_BATCH_SIZE=1

ZERO_STAGE=3
DP_SIZE=8
GRADIENT_ACC_STEP=DATA_BATCH_SIZE // DP_SIZE // TRAIN_MICRO_BATCH_SIZE

tokenizer_config = dict(
    pad_token_id = 2,
    eos_token_id = 2,
    padding_side = 'left',
    chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{'Human:\n' + message['content'] + '\n'}}{% elif message['role'] == 'assistant' %}{{'Assistant:\n' + message['content'] + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:\n' }}{% endif %}",
)

rollout_config = dict(
    actor_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    reward_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    clip_reward_min=-5,
    clip_reward_max=5,
    max_new_tokens=MAX_ANSWER_LEN,
    generate_kwargs={
        "do_sample":True,
        "temperature":1.0,
        "top_k": 0,
        "top_p": 0.9,
        "pad_token_id": 2,
        "eos_token_id": 2,
        "early_stopping": True,
        "num_beams":1,
        "min_new_tokens":1,
    }
)

repeater_config = dict(
    actor_micro_bs=INFER_MICRO_BATCH_SIZE,
    ref_micro_bs=INFER_MICRO_BATCH_SIZE,
    critic_micro_bs=INFER_MICRO_BATCH_SIZE,
    reward_scale=False,
    fine_grained_rm=False,
    value_ema=False,
    kl_coeff = 0.01,
    gamma = 1.0,
    gae_lambda = 0.99,
    answer_end_id = 2,
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
        model_path="OpenLLMAI/Llama-2-7b-sft-model-ocra-500k",
        model_type="actor",
        trainer_config=dict(
            trainer_type="huggingface",
            torch_dtype="auto",
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_seq",
            ),
            parallel=dict(
                data=dict(size=DP_SIZE, mode="deepspeed"),
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
                "train_micro_batch_size_per_gpu": 1, 
                "gradient_accumulation_steps": 16,
                "train_batch_size": DATA_BATCH_SIZE
            }
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
    reference=dict(
        model_path="OpenLLMAI/Llama-2-7b-sft-model-ocra-500k",
        model_type="reference",
        trainer_config=dict(
            trainer_type="huggingface",
            torch_dtype="auto",
            parallel=dict(
                data=dict(size=DP_SIZE, mode="deepspeed"),
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
                "train_micro_batch_size_per_gpu": 1, 
                "gradient_accumulation_steps": 16,
                "train_batch_size": DATA_BATCH_SIZE
            }
        ),
    ),
    critic=dict(
        model_path="OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt",
        model_type="critic",
        head_name="value_head",
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
                data=dict(size=DP_SIZE, mode="deepspeed"),
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
                "train_micro_batch_size_per_gpu": 1, 
                "gradient_accumulation_steps": 16,
                "train_batch_size": DATA_BATCH_SIZE
            }
        ),
    ),
    reward=dict(
        model_path="OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt",
        model_type="reward",
        head_name="value_head",
        trainer_config=dict(
            trainer_type="huggingface",
            torch_dtype="auto",
            parallel=dict(
                data=dict(size=DP_SIZE, mode="deepspeed"),
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
                "train_micro_batch_size_per_gpu": 1, 
                "gradient_accumulation_steps": 16,
                "train_batch_size": DATA_BATCH_SIZE
            }
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
