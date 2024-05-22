import torch
from marl.model_backend.models.internlm2_reward import (
    InternLM2ForRewardModel,
    InternLM2ForCriticModel,
)
from marl.model_backend.models.modeling_internlm2_p import InternLM2ForCausalLM

tokenizer_config = dict(
    pad_token_id = 0,
    eos_token_id = 92542,
    padding_side = 'left',
)

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
    pt_minibatch=128, # length=4096, 
    actor_micro_bs=1,
    critic_micro_bs=2,
    pretrain_step=40,
    save_interval=800,
)
# length=4096, 
# batch_size=128
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
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 80,
                "train_batch_size": 640,
            }
        ),
        generator_config=dict(
            shared_with_trainer=False,
            generator_type="vllm",
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=4, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
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
                data=dict(size=2, mode="ddp"),
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
                data=dict(size=2, mode="ddp"),
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
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/openai_summarize_from_feedback-train.json::0.17886468602360783[META]:summarization",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/prm800k_prm800k_ppo_prompt_1212.json::0.3259440481981093[META]:latex",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/chinese-sensitive_shezheng_52230_20230905_prompts-train-rd.json::0.45274822652340796[REWARD_META]:cn-safety",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/chinese-sensitive_yuqing_5817_20230831_prompts-train-rd.json::0.05042308192426424[REWARD_META]:cn-safety",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/experiments_identity_puyu_chat_format_v2-train.json::0.17337449372239855[REWARD_META]:puyu",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/lcq_share_identity_200_sft_for_ppo_prompt_only.json::0.009583756736321475[REWARD_META]:puyu",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/Anthropic_hh-rlhf_harmless-base-train.json::1.0555135496982802",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/Anthropic_hh-rlhf_helpful-base-train.json::0.4096935605823457",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/IF-ANLI_ANLI-0904-train.json::0.04329546384901008",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/IF-COIG_COIG-0906-train.json::0.04329546384901008",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/IF-SFT6W_SFT6W-prompts-train.json::0.0866390872796097",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/WizardLM-0718_split_0_2_prompt.json::0.27903661572988253",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/alpaca-chinese-0711_split_0_2_prompt.json::0.2738835404998001",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/chatbot_arena_non_toxic_single_turn_tie_both_bad-train.json::0.2085309882827738",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/chatbot_arena_non_toxic_single_turn_tie_both_good-train.json::0.1154385170701637",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/chatbot_arena_toxic_single_turn-train.json::0.026873046526971773",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/chinese-sensitive_lingdaoren_adv_4963_20230913-train-rd.json::0.0429583467778832",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/chinese-sensitive_shezheng_adv_7549_20230913-train-rd.json::0.0654007117986159",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/chinese-sensitive_yuqing_adv_5817_20230913-train-rd.json::0.04941173071088358",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/experiments_gaokao_essay_gaokao_essay_prompt.json::0.04777430493683871",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/experiments_mathci_gsm8k_ci.json::0.12786368912026894",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/experiments_mathci_math_ci.json::0.14948734125397917",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/experiments_subjective_prompts_data_reflow_2w.json::0.552342241250608",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/experiments_subjective_prompts_haochen_data.json::0.10551764326271534",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/experiments_subjective_prompts_retrieval_refined_bench_no_alpaca.json::0.2534157183242392",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/experiments_writing_indomain_writing_2k.json::0.1014722384091927",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/experiments_writing_subeval_writing_prompt_only_v2.json::0.024850344100210458",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/experiments_writing_zhihu_177k_outline_to_artical-with-sys.json::0.7809557750562263",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/instinwild_evol_instinwild_evol_v3_out_instinwild_cn_origin_15_26-rd.json::1.0075947660166729",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/lcq_share_gsm8k_sample200_prompt_only.json::0.00963191631791103",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/lcq_share_maxmin_sample200_prompt_only.json::0.00963191631791103",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/lcq_share_reward_patch_20240103_prompt_only.json::0.004912277322134625",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/vicuna_lmsys-chat-english-chat-format-100char-1t.json::1.1365661255135016",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/wangrui6_Zhihu-KOL-train.json::0.48458170995410393",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/zephyr_zephyr-ultrachat-200k_ppo_train_1t.json::1.2330297674373805",
            "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data_v2/1B_messages_data/zijianwenti-2k_0801-train.json::0.079463309622766",
            ],
    }