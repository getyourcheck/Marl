"""Test txt env

Run `python tests/trainers/test_hf_ppotrainer.py`.
"""
# import pytest
import sys
sys.path.extend(["./", "marl/dataset"])
from collections import defaultdict
from marl.dataset.txt_loader import TxtMessageDataset
from marl.tokenizer.tokenizer_utils import get_tokenizer
from marl.envs.txt_env import TxtEnv
from marl.repeaters.base import BaseRepeater
from marl.trainer.ppo import PPOTrainer
import torch
import numpy as np
from loguru import logger


if __name__ == "__main__":
    # logger.remove(handler_id=None)  # 清除之前的设置
    logger.add("train_policy.log", filter=lambda record: record["extra"].get("name") == "policy")
    logger.add("train_value.log", filter=lambda record: record["extra"].get("name") == "value")
    # logger.add("rollout.log", filter=lambda record: record["extra"].get("name") == "value")

    logger_policy = logger.bind(name="policy")
    logger_value = logger.bind(name="value")

    """ppo reader test here"""
    model_path = "internlm/internlm2-chat-1_8b-sft"
    tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"

    dataset_config = {
        "task_group_filename": "data/config/task_ppo.json",
        "tokenizer": tokenizer,
        "max_seq_len": 1024,
        "num_samples_each_epoch": 8,
        "random_seed": 1,
        "start_token": "[UNUSED_TOKEN_146]user\n",
        "end_token": "[UNUSED_TOKEN_145]\n",
    }

    # init model
    from marl.config.config import Config
    from marl.model_backend.hf_model_runner import HfModelRunner
    from marl.config.config_consts import ENGINE_HUGGINGFACE
    trainer_config = Config(
        dict(
            model_path="internlm/internlm2-chat-1_8b-sft",
            torch_dtype=torch.bfloat16,
            model_type="reference",
            trainer_type=ENGINE_HUGGINGFACE,
            parallel=dict(
                data=dict(size=1),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
            ),
        ),
    )
    sft_model = HfModelRunner(model_config=trainer_config)
    sft_model.initialize()
    # policy model
    trainer_config = Config(
        dict(
            model_path="internlm/internlm2-chat-1_8b-sft",
            torch_dtype=torch.bfloat16,
            model_type="actor",
            trainer_type=ENGINE_HUGGINGFACE,
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                # loss_type="per_token",
                loss_type="per_seq",
            ),
            parallel=dict(
                data=dict(size=1),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
            ),
        ),
    )
    actor_model = HfModelRunner(model_config=trainer_config)
    actor_model.initialize()
    # reward model 
    from marl.config.config_consts import MODEL_TYPE_REWARD, ENGINE_HUGGINGFACE
    reward_trainer_config = Config(
        dict(
            model_path="/cpfs01/shared/public/llm_model/ckpt/Luyou_1B/R-Luyou-1B-8k-D20240130-v1-hf/",
            model_type=MODEL_TYPE_REWARD,
            trainer_type=ENGINE_HUGGINGFACE,
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    )
    reward_model = HfModelRunner(model_config=reward_trainer_config)
    reward_model.initialize()
    # critic model
    from marl.config.config_consts import MODEL_TYPE_CRITIC, ENGINE_HUGGINGFACE
    critic_trainer_config = Config(
        dict(
            model_path="/cpfs01/shared/public/llm_model/ckpt/Luyou_1B/R-Luyou-1B-8k-D20240130-v1-hf/",
            model_type=MODEL_TYPE_CRITIC,
            trainer_type=ENGINE_HUGGINGFACE,
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
    )
    critic_model = HfModelRunner(model_config=critic_trainer_config)
    critic_model.initialize()

    # init txt env
    txt_loader = TxtMessageDataset(**dataset_config)
    txt_env = TxtEnv(dataloader=txt_loader, reward_function=reward_model)
    # init repeater
    rl_repeater = BaseRepeater(sft_model=sft_model, reward_scale=False, fine_grained_rm=False, value_ema=False)
    # init trainer
    ppo = PPOTrainer(policy_model=actor_model, value_model=None)
    
    import time
    while True:
        for _ in range(2):
            trajectories = txt_env.rollout(policy_model=actor_model)
            # deal with trajectories
            trajectories = rl_repeater.process(trajectories, policy_model=actor_model, value_model=critic_model, sft_model=None)

            # s_t = time.time()
            # np.set_printoptions(threshold=np.inf)
            # logger_policy.info(f"rewards: {trajectories.rewards.numpy()}")
            # logger_value.info(f"kl_rewards: {trajectories.kl_rewards.numpy()}")
            # print(time.time() - s_t)
            # # for policy & critic learn
            ppo_loss = ppo.policy_learn(trajectories, actor_model)
            value_loss = ppo.value_learn(trajectories, critic_model)
        break
