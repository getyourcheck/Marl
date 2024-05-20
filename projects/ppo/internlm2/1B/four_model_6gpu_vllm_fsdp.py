import torch

tokenizer_config = dict(
    pad_token_id = 0,
    eos_token_id = 92542,
    padding_side = 'left',
)

rollout_config = dict(
    actor_micro_bs=32,
    reward_micro_bs=32,
    clip_reward_min=-1.5,
    clip_reward_max=1.5,
    max_new_tokens=1024,
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
    actor_micro_bs=8,
    ref_micro_bs=8,
    critic_micro_bs=32,
    reward_scale=False,
    fine_grained_rm=False,
    value_ema=False,
    kl_coeff=0.02,
    gamma=1.0,
    gae_lambda=0.95,
    answer_end_id=92542,
    norm_adv=True,
)

train_config = dict(
    ppo_minibatch=512,
    value_minibatch=512,
    actor_micro_bs=2,
    critic_micro_bs=2,
    pretrain_step=0,
    save_interval=80,
)

model_configs = dict(
    actor=dict(
        model_path="internlm/internlm2-chat-1_8b-sft",
        model_type="actor",
        torch_dtype=torch.bfloat16,
        trainer_config=dict(
            trainer_type="huggingface",
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_seq",
            ),
            parallel=dict(
                # data=dict(size=2, mode="fsdp"), deepspeed
                data=dict(size=2, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
        generator_config=dict(
            shared_with_trainer=False,
            generator_type="vllm",
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),
    reference=dict(
        model_path="internlm/internlm2-chat-1_8b-sft",
        model_type="reference",
        torch_dtype=torch.bfloat16,
        trainer_config=dict(
            trainer_type="huggingface",
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),
    critic=dict(
        model_path="/cpfs01/shared/public/llm_model/ckpt/Luyou_1B/R-Luyou-1B-8k-D20240130-v1-hf/",
        model_type="critic",
        torch_dtype=torch.bfloat16,
        trainer_config=dict(
            trainer_type="huggingface",
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_seq",
            ),
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),
    reward=dict(
        model_path="/cpfs01/shared/public/llm_model/ckpt/Luyou_1B/R-Luyou-1B-8k-D20240130-v1-hf/",
        model_type="reward",
        torch_dtype=torch.bfloat16,
        trainer_config=dict(
            trainer_type="huggingface",
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
    "num_samples_each_epoch": 512,
    "max_seq_len": 1024,
    "random_seed": 1024,
    "ppo_datas": [
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/0801-train.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/ANLI-0904-train.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/COIG-0906-train.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/SFT6W-prompts-train.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/data_reflow_2w.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/gaokao_essay_prompt.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/gsm8k_ci.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/gsm8k_sample200_prompt_only.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/haochen_data.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/identity_200_sft_for_ppo_prompt_only.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/indomain_writing_2k.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/lingdaoren_adv_4963_20230913-train-rd.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/lmsys-chat-english-chat-format-100char-1t.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/math_ci.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/maxmin_sample200_prompt_only.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/non_toxic_single_turn_tie_both_bad-train.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/non_toxic_single_turn_tie_both_good-train.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_15-rd.json::0.01",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_16-rd.json::0.01",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_17-rd.json::0.01",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_18-rd.json::0.01",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_19-rd.json::0.01",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_20-rd.json::0.01",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_21-rd.json::0.01",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_22-rd.json::0.01",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_23-rd.json::0.01",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_24-rd.json::0.01",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_25-rd.json::0.01",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_26-rd.json::0.01",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/prm800k_ppo_prompt_1212.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/puyu_chat_format_v2-train.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/retrieval_refined_bench_no_alpaca.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/reward_patch_20240103_prompt_only.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/shezheng_52230_20230905_prompts-train-rd.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/shezheng_adv_7549_20230913-train-rd.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/split_0_prompt-refined.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/split_0_prompt.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/split_1_prompt.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/split_2_prompt.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/subeval_writing_prompt_only_v2.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/toxic_single_turn-train.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/yuqing_5817_20230831_prompts-train-rd.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/yuqing_adv_5817_20230913-train-rd.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/zephyr-ultrachat-200k_ppo_train_1t.json::0.1",
        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/zhihu_177k_outline_to_artical-with-sys.json::0.1",
    ],
}
