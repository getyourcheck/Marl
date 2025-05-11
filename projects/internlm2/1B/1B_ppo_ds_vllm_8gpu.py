#######################################################################
#                 Settings(RLHF/PPO训练配置文件)                       #
#######################################################################

MAX_PROMPT_LEN = 1024  # Prompt最大输入长度
MAX_ANSWER_LEN = 1024  # 生成回答最大长度
MAX_PRETRAIN_LEN = 8192  # 预训练数据最大长度

PROMPT_BATCH_SIZE = 128 # 训练时每个batch的prompt数量
PRETRAIN_BATCH_SIZE = 32 # 预训练每个batch的pretrain数据数量

GENERATE_MICRO_BATCH_SIZE = 4 # 生成时每个batch的prompt数量(rollout阶段生成回答和reward打分)
INFER_MICRO_BATCH_SIZE = 2 # 推理时每个batch的prompt数量(repeater阶段计算kl散度，计算value值，生成优势函数值)
TRAIN_MICRO_BATCH_SIZE = 1 # 训练时每个batch的prompt数量(ppo损失计算，模型参数更新)

ZERO_STAGE = 2 # 设置DeepSpeed的zero stage
POLICY_DP_SIZE = 2 # 设置DeepSpeed的policy的dp size
reference_dp_size = 1 # 设置DeepSpeed的reference的dp size
CRITIC_DP_SIZE = 2 # 设置DeepSpeed的critic的dp size
reward_dp_size = 1 # 设置DeepSpeed的reward的dp size
use_flash_attn = True # 是否使用flash attention
gradient_checkpointing = True # 是否使用gradient checkpointing
vllm_dp_size = 1  # 保持为1，避免跨设备通信
vllm_tp_size = 2  # 保持为1

# reference_model_path = '/fs-computility/llm/shared/marl/models/internlm2/1.8B/hf/internlm2-chat-1_8b-sft/'
# reward_model_path = '/fs-computility/llm/shared/marl/models/internlm2/1.8B/hf/R-Luyou-1B-8k-D20240130-v1/819_hf_bf16/'
reference_model_path = '/ssd/zhaohui/baiyao/internlm2-chat-7b-sft/'
reward_model_path = '/ssd/zhaohui/baiyao/819_hf_bf16/'
policy_model_path = reference_model_path    # resume 时需设置
critic_model_path = reward_model_path       # resume 时需设置
resume_step = -1                            # 断点续训，需将policy、critic model path设置为对应路径

MODEL_DTYPE = 'auto'  # 设置模型参数和计算所使用的数据类型,自动选择精度

tokenizer_config = dict(  # 分词器配置
    pad_token_id=0,       # padding ID
    eos_token_id=92542,   # 结束符ID
    padding_side='left',  # 序列较短的句子在左边进行 padding, 对齐序列
)

rollout_config = dict(  # 策略生成配置
    policy_micro_bs=GENERATE_MICRO_BATCH_SIZE,  # policy模型进行回答生成批次
    reward_micro_bs=GENERATE_MICRO_BATCH_SIZE,  # reward模型对回答打分计算批次
    max_new_tokens=MAX_ANSWER_LEN,  # 最大生成tokens数
    write_to_file=True,  # 保存生成结果
    generate_kwargs={  # 生成参数
        'do_sample': True,   # 启用采样（非贪婪）
        'temperature': 1.0,  # 温度参数（平衡多样性）
        'top_k': -1,         # 禁用Top-K采样
        'top_p': 0.9,        # Nucleus采样（保留概率质量前90%）
        'min_new_tokens': 1, # 最小生成长度
        'num_beams': 1,      # 不使用 beam search，仅依赖随机采样
        'early_stopping': False,  # 不使用beam search时必须设为False
        'eos_token_id': 92542,
        'pad_token_id': 0,
    },
)

repeater_config = dict(  # PPO参数
    policy_micro_bs=INFER_MICRO_BATCH_SIZE,
    critic_micro_bs=INFER_MICRO_BATCH_SIZE,
    ref_micro_bs=INFER_MICRO_BATCH_SIZE,
    kl_coeff=0.01,  # KL散度惩罚系数,用于限制新策略与旧策略的差异
    gamma=1.0,      # 折扣因子（无折扣）
    gae_lambda=0.99,# GAE参数（平衡偏差-方差）
    clip_reward_min=-5,  # 奖励裁剪下限
    clip_reward_max=5,   # 奖励裁剪上限
    norm_rewards=True,   # 标准化奖励
)

train_config = dict(  # PPO训练配置
    policy_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    critic_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    ppo_loss_weight=1.0,  # PPO损失权重
    pretrain_loss_weight=0.5,  # 预训练损失权重（混合训练）
    critic_warmup_step=4,  # 评价模型预热步数
    save_interval=4,  # 每 40 步保存一次 checkpoint
    max_train_step=20,  # 训练步数上限
    resume_step=resume_step
)

prompt_dataset_config = dict(  # prompt数据集配置
    samples_each_epoch=PROMPT_BATCH_SIZE,  # 在每个 epoch 中从 prompt 数据集中采样 512 个样本
    max_len=MAX_PROMPT_LEN,  # 截断长度
    message_type='prompt',
    random_seed=1024,
    sample_strategy='in_batch',  # 批次内采样 'in_data'
    message_datasets=[
        "/ssd/zhaohui/baiyao/Anthropic_hh-rlhf_harmless-base-train.json::0.5",
        "/ssd/zhaohui/baiyao/Anthropic_hh-rlhf_helpful-base-train.json::0.5",
    ])

# pretrain_dataset_config = dict(
#     samples_each_epoch=PRETRAIN_BATCH_SIZE,
#     max_len=MAX_PRETRAIN_LEN,
#     message_type='pretrain',
#     random_seed=1024,
#     sample_strategy='in_batch',  # 'in_data'
#     message_datasets=[
#         './demo_datas/pretrain_data.json::0.01',
#         '[HF]Anthropic/hh-rlhf/helpful-base::0.5',
#         '[HF]HuggingFaceH4/summarize_from_feedback::0.5',
#     ],
# )
# pretrain_dataset_config = dict(  # 预训练数据集配置
#         folder='/fs-computility/llm/shared/zhaoqian/dataset/pretrain/1226-mix-v13-complete-watermark-pjx50/train',
#         packed=False,
#         max_length=MAX_PRETRAIN_LEN,
#         batch_size=PRETRAIN_BATCH_SIZE,
# )


model_configs = dict(  # 模型配置
    policy=dict(
        model_path=policy_model_path,
        model_type='policy',
        model_config=dict(
            attn_implementation='flash_attention_2' if use_flash_attn else 'eager',
            use_cache=True,  # 启用KV缓存
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
                tensor=dict(size=1, mode='1d'),  # 保持张量并行为1
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
            deepspeed_config={
                'zero_optimization': {
                    'stage': ZERO_STAGE,
                    'offload_param': {
                        'device': 'cpu',  # 改为CPU offload以节省显存
                        'pin_memory': True
                    },
                    'offload_optimizer': {
                        'device': 'cpu',
                        'pin_memory': True
                    },
                    'contiguous_gradients': True,
                    'overlap_comm': True,
                    'reduce_scatter': True,
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
                'gradient_accumulation_steps': (PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE) // POLICY_DP_SIZE // TRAIN_MICRO_BATCH_SIZE, 
                'train_batch_size': PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE,
            },
        ),
        generator_config=dict(
            shared_with_trainer=False,  # 是否与训练器共享生成器
            generator_type='vllm',  # 使用vLLM高效推理
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
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
            deepspeed_config={
                'zero_optimization': {
                    'stage': ZERO_STAGE,
                    'offload_param': {
                        'device': 'cpu',  # 改为CPU offload以节省显存
                        'pin_memory': True
                    },
                    'offload_optimizer': {
                        'device': 'cpu',
                        'pin_memory': True
                    },
                    'contiguous_gradients': True,
                    'overlap_comm': True,
                    'reduce_scatter': True,
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
                'gradient_accumulation_steps': PROMPT_BATCH_SIZE // CRITIC_DP_SIZE // TRAIN_MICRO_BATCH_SIZE, 
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
                data=dict(size=reference_dp_size, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
            deepspeed_config={
                "zero_optimization": {
                    "stage": ZERO_STAGE, 
                    "offload_param": {
                        "device": "cpu",  # 改为CPU offload以节省显存
                        "pin_memory": True
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
                'train_micro_batch_size_per_gpu': TRAIN_MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': (PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE) // reference_dp_size // TRAIN_MICRO_BATCH_SIZE,
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
            trainer_type="huggingface",
            use_flash_attn=use_flash_attn,
            gradient_checkpointing=gradient_checkpointing,
            parallel=dict(
                data=dict(size=reward_dp_size, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
            deepspeed_config={
                "zero_optimization": {
                    "stage": ZERO_STAGE, 
                    "offload_param": {
                        "device": "cpu",  # 改为CPU offload以节省显存
                        "pin_memory": True
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
                'train_micro_batch_size_per_gpu': TRAIN_MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': PROMPT_BATCH_SIZE // (reward_dp_size * TRAIN_MICRO_BATCH_SIZE),  # 修改为16
                'train_batch_size': PROMPT_BATCH_SIZE,  # 保持为16
            },
        ),
    ),
)


# checkout generate config
assert PROMPT_BATCH_SIZE % GENERATE_MICRO_BATCH_SIZE == 0
assert PROMPT_BATCH_SIZE % POLICY_DP_SIZE == 0
# checkout infer config
assert PROMPT_BATCH_SIZE % (INFER_MICRO_BATCH_SIZE * POLICY_DP_SIZE) == 0
assert PROMPT_BATCH_SIZE % (INFER_MICRO_BATCH_SIZE * CRITIC_DP_SIZE) == 0
# checkout learn config
assert (PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE) % (TRAIN_MICRO_BATCH_SIZE *
                                                    POLICY_DP_SIZE) == 0
assert (PROMPT_BATCH_SIZE) % (TRAIN_MICRO_BATCH_SIZE * CRITIC_DP_SIZE) == 0
