import argparse
import os
import time

from loguru import logger
import torch
import numpy as np

from marl.config import Config
from marl.coordinator import Coordinator
from marl.dataset.txt_loader import TxtMessageDataset
from marl.envs.txt_env import TxtEnv
from marl.tokenizer.tokenizer_utils import get_tokenizer
from marl.repeaters.base import BaseRepeater
from marl.trainer.ppo import PPOTrainer
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')
    parser.add_argument('-c','--config', help='config file name or path.', type=str, default='projects/ppo/internlm2/1B/four_model_24gpu_alldata_vllm.py')
    parser.add_argument('-w','--work_dir', help='the dir to save logs and models', type=str, default=None)
    parser.add_argument('-a','--address', help='ray head address', type=str, default="auto")
    args = parser.parse_args()
    return args

def validate_config(config:Config):
    assert config["model_configs"] is not None
    assert config["model_configs"]["actor"] is not None
    assert config["model_configs"]["actor"]["model_path"] is not None
    assert config["dataset_config"] is not None
    assert config["rollout_config"] is not None
    assert config["rollout_config"]["generate_kwargs"] is not None
    assert config["rollout_config"]["max_new_tokens"] is not None
    
if __name__ == "__main__":
    args = parse_args()
    assert args.config is not None ,"config should not be None"
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.abspath(work_dir)
    logger.info(f"using work_dir: {work_dir}")
    os.makedirs(work_dir, exist_ok=True)

    logger.add(f"{work_dir}/train.log", filter=lambda record: record["extra"].get("name") == "train")
    logger.add(f"{work_dir}/rollout.log", filter=lambda record: record["extra"].get("name") == "rollout")
    logger_train = logger.bind(name="train")

    configs_path = args.config
    config = Config.from_file(configs_path)

    #init dataset
    model_path = config["model_configs"]["actor"]["model_path"]
    tokenizer_config = config.get("tokenizer_config",{})
    for model_type in config["model_configs"].keys():
        if "tokenizer_config" not in config["model_configs"][model_type]:
            config["model_configs"][model_type]["tokenizer_config"] = tokenizer_config
    tokenizer = get_tokenizer(model_path, trust_remote_code=True,**tokenizer_config)
    dataset_config = config["dataset_config"]
    dataset_config["tokenizer"] = tokenizer
    txt_loader = TxtMessageDataset(**dataset_config)

    # init model
    cluster_address = args.address
    if cluster_address != "auto":
        cluster_address = f"ray://{cluster_address}:10001"
    logger.info(f"cluster_address={cluster_address}")
    coordinator = Coordinator(cluster_address, config["model_configs"])
    model_dict = coordinator.create_models()
    sft_model = model_dict["reference"]
    actor_model = model_dict["actor"]
    reward_model = model_dict["reward"]
    critic_model = model_dict["critic"]

    # init txt env

    rollout_config = config.get("rollout_config",{})
    txt_env = TxtEnv(
        dataloader=txt_loader,
        reward_function=reward_model,
        **rollout_config,
    )
    # init repeater
    repeater_config = config.get("repeater_config",{})
    rl_repeater = BaseRepeater(
        sft_model=sft_model, 
        **repeater_config,
    )
    # init trainer
    train_config = config.get("train_config",{})
    ppo = PPOTrainer(
        policy_model=actor_model, 
        value_model=None, 
        **train_config
    )
    pretrain_step = train_config["pretrain_step"]
    save_interval = train_config["save_interval"]
    np.set_printoptions(threshold=np.inf)
    step = 1
    while True:
        s_t = time.time()
        trajectories = txt_env.rollout(policy_model=actor_model)
        # deal with trajectories
        trajectories = rl_repeater.process(trajectories, policy_model=actor_model, value_model=critic_model, sft_model=None, env=txt_env)

        # # for value & policy learn
        value_loss_ref = ppo.value_learn_async(trajectories, critic_model)

        ppo_loss, pt_loss = None, None
        if pretrain_step <= 0:
            ppo_loss, pt_loss = ppo.policy_learn(trajectories, actor_model)
            logger_train.info(f"[Policy Train] Step: {step}, ppo loss: {ppo_loss}, pretrain loss: {pt_loss}")
        
        value_loss = ppo.value_learn_get(value_loss_ref, critic_model)
        logger_train.info(f"[Value Train] step: {step}, value loss: {value_loss}")
        logger_train.info(f"rewards: {trajectories.rewards.mean()}")
        pretrain_step -= 1

        if config["rollout_config"].get("write_to_file", True):
            with open(f"{work_dir}/rollout.log", "a") as file:
                file.write(f"generates: {trajectories.output_str}")
        summaries = dict(
            reward_mean=trajectories.rewards.mean().item(),
            reward_std=trajectories.rewards.std().item(),
            new_tokens_mean=trajectories.action_mask.sum(-1).float().mean().item(),
            new_tokens_std=trajectories.action_mask.sum(-1).float().std().item(),
            kl=trajectories.kl.mean().item(),
            entropy=trajectories.entropy.mean().item(),
            step=step,
            policy_loss=ppo_loss,
            pretrain_loss=pt_loss,
            critic_loss=value_loss,
        )
        with open(f"{work_dir}/train.log.jsonl", "a") as f:
            f.write(json.dumps(summaries) + "\n")

        step += 1
        logger_train.info(f"[end to end] duration: {time.time() - s_t} s")
        if step % save_interval == 0:
            actor_model.save_model(f"{work_dir}/ckpt/{step}/")
