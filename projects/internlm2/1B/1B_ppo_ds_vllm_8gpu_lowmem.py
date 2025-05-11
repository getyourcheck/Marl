#######################################################################
#                 Settings(RLHF/PPO训练配置文件 - 低显存版本)                #
#######################################################################
import torch

MAX_PROMPT_LEN = 1024  # Prompt最大输入长度
MAX_ANSWER_LEN = 1024  # 生成回答最大长度
MAX_PRETRAIN_LEN = 8192  # 预训练数据最大长度

PROMPT_BATCH_SIZE = 8  # 减小批次大小
PRETRAIN_BATCH_SIZE = 4

GENERATE_MICRO_BATCH_SIZE = 1
INFER_MICRO_BATCH_SIZE = 1
TRAIN_MICRO_BATCH_SIZE = 1

ZERO_STAGE = 3
POLICY_DP_SIZE = 1  # 保持与原配置一致
reference_dp_size = 1
CRITIC_DP_SIZE = 1
reward_dp_size = 1
use_flash_attn = True
gradient_checkpointing = True
vllm_dp_size = 1
vllm_tp_size = 1

# 并行设置
TENSOR_PARALLEL_SIZE = 4  # 4路张量并行

reference_model_path = '/ssd/zhaohui/baiyao/internlm2-chat-1_8b-sft/'
reward_model_path = '/ssd/zhaohui/baiyao/819_hf_bf16/'
policy_model_path = reference_model_path
critic_model_path = reward_model_path
resume_step = -1

MODEL_DTYPE = 'auto'

tokenizer_config = dict(
    pad_token_id=0,
    eos_token_id=92542,
    padding_side='left',
)

rollout_config = dict(
    policy_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    reward_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    max_new_tokens=MAX_ANSWER_LEN,
    write_to_file=True,
    generate_kwargs={
        'do_sample': True,
        'temperature': 1.0,
        'top_k': -1,
        'top_p': 0.9,
        'min_new_tokens': 1,
        'num_beams': 1,
        'early_stopping': False,
        'eos_token_id': 92542,
        'pad_token_id': 0,
    },
)

repeater_config = dict(
    policy_micro_bs=INFER_MICRO_BATCH_SIZE,
    critic_micro_bs=INFER_MICRO_BATCH_SIZE,
    ref_micro_bs=INFER_MICRO_BATCH_SIZE,
    kl_coeff=0.01,
    gamma=1.0,
    gae_lambda=0.99,
    clip_reward_min=-5,
    clip_reward_max=5,
    norm_rewards=True,
)

train_config = dict(
    policy_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    critic_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    ppo_loss_weight=1.0,
    pretrain_loss_weight=0.5,
    critic_warmup_step=40,
    save_interval=40,
    max_train_step=400,
    resume_step=resume_step
)

prompt_dataset_config = dict(
    samples_each_epoch=PROMPT_BATCH_SIZE,
    max_len=MAX_PROMPT_LEN,
    message_type='prompt',
    random_seed=1024,
    sample_strategy='in_batch',
    message_datasets=[
        "/ssd/zhaohui/baiyao/Anthropic_hh-rlhf_harmless-base-train.json::0.5",
        "/ssd/zhaohui/baiyao/Anthropic_hh-rlhf_helpful-base-train.json::0.5",
    ])

model_configs = dict(
    policy=dict(
        model_path=policy_model_path,
        model_type='policy',
        model_config=dict(
            attn_implementation='flash_attention_2' if use_flash_attn else 'eager',
            use_cache=True,
        ),
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=use_flash_attn,
            gradient_checkpointing=gradient_checkpointing,
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
            ),
            parallel=dict(
                data=dict(size=POLICY_DP_SIZE, mode='deepspeed'),
                tensor=dict(size=TENSOR_PARALLEL_SIZE, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
            deepspeed_config={
                'zero_optimization': {
                    'stage': ZERO_STAGE,
                    'offload_param': {
                        'device': 'none'
                    },
                    'reduce_bucket_size': 'auto',
                    'zero_hpz_partition_size': 1,
                    'zero_quantized_weights': False,
                    'zero_quantized_gradients': False,
                    'stage3_gather_16bit_weights_on_model_save': True,
                },
                'bf16': {
                    'enabled': True
                },
                'gradient_clipping': 1.0,
                'prescale_gradients': False,
                'wall_clock_breakdown': False,
                'data_types': {
                    'grad_accum_dtype': 'fp32'
                },
                'train_micro_batch_size_per_gpu': TRAIN_MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': 128,
                'train_batch_size': PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE,
            },
        ),
        generator_config=dict(
            shared_with_trainer=False,
            generator_type='vllm',
            parallel=dict(
                data=dict(size=vllm_dp_size, mode='ddp'),
                tensor=dict(size=vllm_tp_size, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
        ),
    ),
    critic=dict(
        model_path=reward_model_path,
        model_type='critic',
        model_config=dict(
            attn_implementation='flash_attention_2' if use_flash_attn else 'eager',
        ),
        trainer_config=dict(
            head_name='v_head',
            two_linear=False,
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=use_flash_attn,
            gradient_checkpointing=gradient_checkpointing,
            train_kwargs=dict(
                micro_bsz=1,
                lr=5e-6,
                total_steps=1e9,
                lr_decay_rate=1,
            ),
            parallel=dict(
                data=dict(size=CRITIC_DP_SIZE, mode='deepspeed'),
                tensor=dict(size=TENSOR_PARALLEL_SIZE, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
            deepspeed_config={
                'zero_optimization': {
                    'stage': ZERO_STAGE,
                    'offload_param': {
                        'device': 'none'
                    },
                    'reduce_bucket_size': 'auto',
                    'zero_hpz_partition_size': 1,
                    'zero_quantized_weights': False,
                    'zero_quantized_gradients': False,
                    'stage3_gather_16bit_weights_on_model_save': True,
                },
                'bf16': {
                    'enabled': True
                },
                'gradient_clipping': 1.0,
                'prescale_gradients': False,
                'wall_clock_breakdown': False,
                'data_types': {
                    'grad_accum_dtype': 'fp32'
                },
                'train_micro_batch_size_per_gpu': TRAIN_MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': 128,
                'train_batch_size': PROMPT_BATCH_SIZE,
            },
        ),
    ),
    reference=dict(
        model_path=policy_model_path,
        model_type='reference',
        model_config=dict(
            attn_implementation='flash_attention_2' if use_flash_attn else 'eager',
        ),
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type="huggingface",
            use_flash_attn=use_flash_attn,
            gradient_checkpointing=gradient_checkpointing,
            parallel=dict(
                data=dict(size=reference_dp_size, mode='deepspeed'),
                tensor=dict(size=TENSOR_PARALLEL_SIZE, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
            deepspeed_config={
                'zero_optimization': {
                    'stage': ZERO_STAGE,
                    'offload_param': {
                        'device': 'none'
                    },
                    'reduce_bucket_size': 'auto',
                    'zero_hpz_partition_size': 1,
                    'zero_quantized_weights': False,
                    'zero_quantized_gradients': False,
                    'stage3_gather_16bit_weights_on_model_save': True,
                },
                'bf16': {
                    'enabled': True
                },
                'gradient_clipping': 1.0,
                'prescale_gradients': False,
                'wall_clock_breakdown': False,
                'data_types': {
                    'grad_accum_dtype': 'fp32'
                },
                'train_micro_batch_size_per_gpu': TRAIN_MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': 128,
                'train_batch_size': PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE,
            },
        ),
    ),
    reward=dict(
        model_path=reward_model_path,
        model_type='reward',
        model_config=dict(
            attn_implementation='flash_attention_2' if use_flash_attn else 'eager',
        ),
        trainer_config=dict(
            head_name='v_head',
            two_linear=False,
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=use_flash_attn,
            gradient_checkpointing=gradient_checkpointing,
            parallel=dict(
                data=dict(size=reward_dp_size, mode='deepspeed'),
                tensor=dict(size=TENSOR_PARALLEL_SIZE, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
            deepspeed_config={
                'zero_optimization': {
                    'stage': ZERO_STAGE,
                    'offload_param': {
                        'device': 'none'
                    },
                    'reduce_bucket_size': 'auto',
                    'zero_hpz_partition_size': 1,
                    'zero_quantized_weights': False,
                    'zero_quantized_gradients': False,
                    'stage3_gather_16bit_weights_on_model_save': True,
                },
                'bf16': {
                    'enabled': True
                },
                'gradient_clipping': 1.0,
                'prescale_gradients': False,
                'wall_clock_breakdown': False,
                'data_types': {
                    'grad_accum_dtype': 'fp32'
                },
                'train_micro_batch_size_per_gpu': TRAIN_MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': 128,
                'train_batch_size': PROMPT_BATCH_SIZE,
            },
        ),
    ),
) 