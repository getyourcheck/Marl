#######################################################################
#                              Settings                               #
#######################################################################
MAX_PROMPT_LEN = 1024
MAX_ANSWER_LEN = 1024
MAX_PRETRAIN_LEN = 32 * 1024
packed_data = False
use_varlen_attn = False

PROMPT_BATCH_SIZE = 512
PRETRAIN_BATCH_SIZE = 0

GENERATE_MICRO_BATCH_SIZE = 8
INFER_MICRO_BATCH_SIZE = 8
TRAIN_MICRO_BATCH_SIZE = 1


ZERO_STAGE = 2
POLICY_DP_SIZE = 16
reference_dp_size = 8
CRITIC_DP_SIZE = 8
reward_dp_size = 8
use_flash_attn = True
gradient_checkpointing = True
vllm_dp_size = 8
vllm_tp_size = 1
sp_size = 1


resume_step = -1
reference_model_path = '/fs-computility/llm/shared/lishuaibin/ckpts/gitlab_marl/7B/internlm2_5/aliyun_internlm2_5_boost1_7B_FT_s1_20240621rc11_s2_20240612rc13_379_hf/'
policy_model_path = reference_model_path
reward_model_path = '/fs-computility/llm/shared/lishuaibin/ckpts/gitlab_marl/7B/internlm2_5/R-Ampere-7B-8k-D20240620-v1_615_hf/'
critic_model_path = reward_model_path


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
        'top_k': 0,
        'top_p': 0.9,
        'min_new_tokens': 1,
        'num_beams': 1,
        'early_stopping': True,
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
    resume_step=resume_step,
    save_interval=40,
    max_train_step=400,
    use_varlen_attn=use_varlen_attn,
)

prompt_dataset_config = dict(
    samples_each_epoch=PROMPT_BATCH_SIZE,
    max_len=MAX_PROMPT_LEN,
    message_type='prompt',
    random_seed=1024,
    sample_strategy='in_data',  # 'in_data'
    message_datasets=[
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/Anthropic_hh-rlhf_harmless-base-train.json::0.2",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/Anthropic_hh-rlhf_helpful-base-train.json::0.5",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/non_toxic_single_turn_tie_both_good-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/non_toxic_single_turn_tie_both_bad-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/split_0_2_prompt.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/toxic_single_turn-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/zephyr-ultrachat-200k_ppo_train_1t.json::0.08",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/lmsys-chat-english-chat-format-100char-1t.json::0.04",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/airoboros_reward.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/out_instinwild_cn_origin_18_26-rd.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/split_0_prompt-refined.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/split_1_prompt.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/indomain_writing_2k.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/zhihu_177k_outline_to_artical-with-sys.json::0.1",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/subeval_writing_prompt_only_v2.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/gaokao_essay_prompt.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/wangrui6_Zhihu-KOL-train.json::0.05",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/0801-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/shezheng_52230_20230905_prompts-train-rd.json::0.2[RM_PROMPT]:cn-safety",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/yuqing_5817_20230831_prompts-train-rd.json::0.2[RM_PROMPT]:cn-safety",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/shezheng_adv_7549_20230913-train-rd.json::0.2",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/yuqing_adv_5817_20230913-train-rd.json::0.2",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/lingdaoren_adv_4963_20230913-train-rd.json::0.2",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/safety_reward_merged_20230609_newformat_prompt.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/0801-train-unfair.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/0801-train-illegal.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/0801-train-privacy.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/0801-train-immoral.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/openai_summarize_from_feedback-train.json::0.04[SYS_PROMPT]:summarization",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/puyu_chat_format_v2-train.json::1.0[RM_PROMPT]:puyu",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/identity_200_sft_for_ppo_prompt_only.json::1.0[RM_PROMPT]:puyu",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/COIG-0906-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/ANLI-0904-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/SFT6W-prompts-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/if_cn_en_gpt4vs1and7b.json::0.25",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/prm800k_ppo_prompt_1212.json::1.0[SYS_PROMPT]:latex",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/math_shepherd_ppo_prompt_augmented.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/40k_sampled_prompt_for_ppo.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/maxmin_sample200_prompt_only.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/gsm8k_sample200_prompt_only.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/reward_patch_20240103_prompt_only.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/haochen_data.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/data_reflow_2w.json::0.75",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/retrieval_refined_bench_no_alpaca.json::0.5",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/airoboros_reward_ocra_math.json::0.5",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/gaokao_preference_processed.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/reward_patch_20240407.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/gongwuyuan_logic_prompt_for_ppo.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/questions_of_name.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/questions_of_objects.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/questions_of_task.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/safety_positive_patch_china_politic.jsonl::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5/lmsys-arena-human-preference-55k_single_round_train.jsonl::1.0[RM_PROMPT]:lmsys_arena",
    ])

pretrain_dataset_config = dict(
        folder='/fs-computility/llm/shared/zhaoqian/dataset/pretrain/1226-mix-v13-complete-watermark-pjx50/train',
        packed=packed_data,
        max_length=MAX_PRETRAIN_LEN,
        batch_size=PRETRAIN_BATCH_SIZE,
)

model_configs = dict(
    policy=dict(
        model_path=policy_model_path,
        model_type='policy',
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
                data=dict(size=POLICY_DP_SIZE // sp_size, mode='deepspeed'),
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=sp_size),
            ),
            envs=dict(
                ACCELERATE_USE_DEEPSPEED="true",
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
                'gradient_accumulation_steps': sp_size * (PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE) // POLICY_DP_SIZE // TRAIN_MICRO_BATCH_SIZE,
                'train_batch_size': sp_size * (PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE),
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
        model_path=critic_model_path,
        model_type='critic',
        trainer_config=dict(
            head_name='v_head',
            two_linear=True,
            exclude_keys=["v_head.0.weight", ],
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
            envs=dict(
                ACCELERATE_USE_DEEPSPEED="true",
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
                'gradient_accumulation_steps': PROMPT_BATCH_SIZE // CRITIC_DP_SIZE // TRAIN_MICRO_BATCH_SIZE,
                'train_batch_size': PROMPT_BATCH_SIZE,
            },
        ),
    ),
    reference=dict(
        model_path=reference_model_path,
        model_type='reference',
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            use_flash_attn=use_flash_attn,
            # gradient_checkpointing=gradient_checkpointing,
            trainer_type="huggingface",
            parallel=dict(
                data=dict(size=reference_dp_size, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
            envs=dict(
                ACCELERATE_USE_DEEPSPEED="true",
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
                'train_micro_batch_size_per_gpu': TRAIN_MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': (PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE) // reference_dp_size // TRAIN_MICRO_BATCH_SIZE,
                'train_batch_size': PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE,
            },
        ),
    ),
    reward=dict(
        model_path=reward_model_path,
        model_type='reward',
        trainer_config=dict(
            head_name='v_head',
            two_linear=True,
            torch_dtype=MODEL_DTYPE,
            use_flash_attn=use_flash_attn,
            # gradient_checkpointing=gradient_checkpointing,
            trainer_type="huggingface",
            parallel=dict(
                data=dict(size=reward_dp_size, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
            envs=dict(
                ACCELERATE_USE_DEEPSPEED="true",
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
                'train_micro_batch_size_per_gpu': TRAIN_MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': PROMPT_BATCH_SIZE // reward_dp_size // TRAIN_MICRO_BATCH_SIZE,
                'train_batch_size': PROMPT_BATCH_SIZE,
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
