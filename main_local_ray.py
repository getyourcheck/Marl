from marl.dataset.txt_loader import TxtMessageDataset
from marl.tokenizer.tokenizer_utils import get_tokenizer
from marl.envs.txt_env import TxtEnv
from marl.repeaters.base import BaseRepeater
from marl.trainer.ppo import PPOTrainer
from marl.coordinator import Coordinator
from marl.config import Config
import numpy as np
from loguru import logger

if __name__ == "__main__":

    logger.add("train_train.log", filter=lambda record: record["extra"].get("name") == "train")
    logger.add("train_rollout.log", filter=lambda record: record["extra"].get("name") == "rollout")
    logger_train = logger.bind(name="train")
    logger_rollout = logger.bind(name="rollout")

    model_path = "internlm/internlm2-chat-1_8b-sft"
    tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"

    dataset_config = {
        "ppo_datas": ["/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/0801-train.json::0.1",
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
        # "pt_datas": ["./data/pt_data/pt_data_0.json::0.9",
        #              "./data/pt_data/pt_data_1.json::0.3",
        #              ],
        "num_samples_each_epoch": 512,
        # "pt_data_samples": 2,
        "tokenizer": tokenizer,
        "max_seq_len": 1024,
        "random_seed": 1024,
    }

    # init model
    cluster_address = "ray://172.31.0.40:10001"
    print(f"cluster_address={cluster_address}")
    model_configs_path = "projects/ppo/internlm2/1B/four_model_24gpu.py"
    model_configs = Config.from_file(model_configs_path)
    coordinator = Coordinator(cluster_address, model_configs)
    model_dict = coordinator.create_models()
    sft_model = model_dict["reference"]
    actor_model = model_dict["actor"]
    reward_model = model_dict["reward"]
    critic_model = model_dict["critic"]

    # init txt env
    txt_loader = TxtMessageDataset(**dataset_config)
    txt_env = TxtEnv(dataloader=txt_loader, reward_function=reward_model)
    # init repeater
    rl_repeater = BaseRepeater(sft_model=sft_model, reward_scale=False, fine_grained_rm=False, value_ema=False)
    # init trainer
    train_config = {
        "ppo_minibatch": 512,
        "train_minibatch": 2,
        "value_minibatch": 512
    }
    ppo = PPOTrainer(policy_model=actor_model, value_model=None, train_cfg=train_config)
    
    pretrain_step = 40
    import time
    np.set_printoptions(threshold=np.inf)
    step = 1
    while True:
        s_t = time.time()
        trajectories = txt_env.rollout(policy_model=actor_model)
        # deal with trajectories
        trajectories = rl_repeater.process(trajectories, policy_model=actor_model, value_model=critic_model, sft_model=None)

        # # for policy & critic learn
        if pretrain_step <= 0:
            ppo_loss, pt_loss = ppo.policy_learn(trajectories, actor_model)
            logger_train.info(f"[Policy Train] Step: {step}, ppo loss: {ppo_loss}, pretrain loss: {pt_loss}")
            logger_train.info(f"[Policy Train] Step: {step}, kl: {np.mean(trajectories.kl_distance)}")
        
        logger_train.info(f"rewards: {trajectories.rewards.mean()}")

        value_loss = ppo.value_learn(trajectories, critic_model)
        logger_train.info(f"[Value Train] step: {step}, value loss: {value_loss}")
        pretrain_step -= 1

        logger_rollout.info(f"generates: {trajectories.output_str}")
        step += 1
        logger_train.info(f"End to End time: {time.time() - s_t}s")
        if step % 40 == 0:
            actor_model.save_model(f"/cpfs01/shared/public/llm_model/ckpt/test_0326/{step}/")

        if step > 380:
            break