#######################################################################
#                              Settings                               #
#######################################################################
MAX_PROMPT_LEN = 1024
MAX_ANSWER_LEN = 1024
MAX_PRETRAIN_LEN = 32 * 1024
packed_data = True
use_varlen_attn = True

PROMPT_BATCH_SIZE = 512
PRETRAIN_BATCH_SIZE = 16

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
sp_size = 2


resume_step = -1
reference_model_path = '/fs-computility/llm/shared/llm_data/yehaochen/ckpts/workflow/regression/aliyun_official_Ampere2_9_1_7B_enhance_0_0_0_16000_FT_s2_4k_internlm2_5_baseline_414_hf/'
policy_model_path = reference_model_path
reward_model_path = '/fs-computility/llm/shared/lishuaibin/ckpts/gitlab_marl/7B/internlm2_5/R-Ampere-7B-8k-D20240617-v1_607_hf/'
# reward_model_path = '/fs-computility/llm/shared/lvchengqi/ckpts/xtuner/R-Ampere-7B-8k-D20241009-1e-5lr-decay-09-RRM/iter_6141_hf/'
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

from marl.loss import PretrainLoss, PPOPolicyLoss, CriticLoss
train_config = dict(
    policy_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    critic_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    ppo_loss_weight=1.0,
    pretrain_loss_weight=0.5,
    critic_warmup_step=40,  # TODO
    resume_step=resume_step,
    save_interval=40,
    max_train_step=400,
    use_varlen_attn=use_varlen_attn,
    pretrain_criterion=PretrainLoss(label_smoothing=0),
    policy_criterion=PPOPolicyLoss(cliprange=0.2, loss_type="per_token"),
    critic_criterion=CriticLoss(cliprange_value=0.5, loss_type="per_token"),
)

prompt_dataset_config = dict(
    samples_each_epoch=PROMPT_BATCH_SIZE,
    max_len=MAX_PROMPT_LEN,
    message_type='prompt',
    random_seed=1024,
    sample_strategy='in_data',  # 'in_data'
    message_datasets=[
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/Anthropic_hh-rlhf_helpful-base-train-deduplicate.json::0.5",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/non_toxic_single_turn_tie_both_good-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/non_toxic_single_turn_tie_both_bad-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/WizardLM-0718_split_0_2_prompt.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/toxic_single_turn-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/zephyr-ultrachat-200k_ppo_train_1t.json::0.08",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/lmsys-chat-english-chat-format-100char-1t.json::0.04",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/airoboros_reward.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/Anthropic_hh-rlhf_harmless-base-train-deduplicate.json::0.2",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/out_instinwild_cn_origin_18_26-rd.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/instinwild_ch_500-end.json::0.05",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/alpaca-chinese-0711_split_0_prompt-refined.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/alpaca-chinese-0711_split_1_prompt.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/indomain_writing_2k.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/zhihu_177k_outline_to_artical-with-sys.json::0.08", # 0.1*0.8
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/subeval_writing_prompt_only_v2.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/gaokao_essay_prompt.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/wangrui6_Zhihu-KOL-train-deduplicate.json::0.05",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/zijianwenti-2k_0801-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/shezheng_52230_20230905_prompts-train-rd.json::0.2[RM_PROMPT]:cn-safety",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/yuqing_5817_20230831_prompts-train-rd.json::0.2[RM_PROMPT]:cn-safety",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/shezheng_adv_7549_20230913-train-rd.json::0.2",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/yuqing_adv_5817_20230913-train-rd.json::0.2",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/lingdaoren_adv_4963_20230913-train-rd.json::0.2",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/safety_reward_merged_20230609_newformat_prompt.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/thu-safety_0801-train-unfair.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/thu-safety_0801-train-illegal.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/thu-safety_0801-train-privacy.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/thu-safety_0801-train-immoral.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/openai_summarize_from_feedback-train.json::0.04[SYS_PROMPT]:summarization",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/puyu_chat_format_v2-train.json::1.0[RM_PROMPT]:puyu",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/identity_200_sft_for_ppo_prompt_only.json::1.0[RM_PROMPT]:puyu",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/COIG-0906-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/ANLI-0904-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/SFT6W-prompts-train.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/if_cn_en_gpt4vs1and7b_train_v2.json::0.25",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/prm800k_ppo_prompt_1212.json::1.0[SYS_PROMPT]:latex",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/math_shepherd_ppo_prompt_augmented.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/40k_sampled_prompt_for_ppo.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/maxmin_sample200_prompt_only.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/gsm8k_sample200_prompt_only.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/reward_patch_20240103_prompt_only.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/subjective_haochen_data.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/subjective_data_reflow_2w.json::0.75",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/subjective_retrieval_refined_bench_no_alpaca.json::0.5",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/airoboros_reward_ocra_math.json::0.5",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/gaokao_preference_processed.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/reward_patch_20240407.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/gongwuyuan_logic_prompt_for_ppo.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/questions_of_name.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/questions_of_objects.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/questions_of_task.json::1.0",
        "/fs-computility/llm/shared/lishuaibin/datasets/internlm2_5_baseline/safety_positive_patch_china_politic.json::1.0",
    ])

pretrain_dataset_config = dict(
        folder='/fs-computility/llm/shared/zhaoqian/dataset/pretrain/L1-240621/training/chatml_llamav13_32k/train',
        packed=packed_data,
        max_length=MAX_PRETRAIN_LEN,
        batch_size=PRETRAIN_BATCH_SIZE,
)

model_configs = dict(
    policy=dict(
        model_path=policy_model_path,
        model_type='policy',
        tokenizer_config=tokenizer_config,
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
                # "activation_checkpointing": {
                #     "partition_activations": True,
                #     "cpu_checkpointing": True,
                #     "contiguous_memory_optimization": True,
                #     "number_checkpoints": 8,
                #     "synchronize_checkpoint_boundary": True,
                #     "profile": True
                #     },
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
