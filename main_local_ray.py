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
        "ppo_data_filename": "data/config/1.8B_ppo.json",
        # "sft_data_filename": "data/config/1.8B_sft.json",
        # "ppo_data_filename": "data/config/task_ppo.json",
        # "sft_data_filename": "data/config/task_sft.json",
        "num_samples_each_epoch": 16,
        # "sft_data_samples": 2,
        "tokenizer": tokenizer,
        "max_seq_len": 4096,
        "random_seed": 1024,
    }

    # init model
    cluster_address = "auto"
    print(f"cluster_address={cluster_address}")
    model_configs_path = "projects/ppo/internlm2/1B/four_model_4gpu.py"
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
        "ppo_minibatch": 16,
        "train_minibatch": 1,
        "value_minibatch": 16
    }
    ppo = PPOTrainer(policy_model=actor_model, value_model=None, train_cfg=train_config)
    
    pretrain_step = 0#40
    import time
    np.set_printoptions(threshold=np.inf)
    step = 1
    while True:
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
        if step % 20 == 0:
            actor_model.save_model(f"/cpfs01/shared/public/llm_model/ckpt/test_0326/{step}/")
