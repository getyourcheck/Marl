"""Run

Run `python main.py`.
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
    # logger.remove(handler_id=None)
    logger.add("train_train.log", filter=lambda record: record["extra"].get("name") == "train")
    logger.add("train_rollout.log", filter=lambda record: record["extra"].get("name") == "rollout")

    logger_train = logger.bind(name="train")
    logger_rollout = logger.bind(name="rollout")

    """ppo reader test here"""
    model_path = "internlm/internlm2-chat-1_8b-sft"
    tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"

    dataset_config = {
        "ppo_datas": ["/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/0801-train.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/ANLI-0904-train.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/COIG-0906-train.json::0.1",
                      ],
        "pt_datas": ["./data/pt_data/pt_data_0.json::0.9",
                     "./data/pt_data/pt_data_1.json::0.3",
                     ],
        "num_samples_each_epoch": 10,
        "pt_data_samples": 2,
        "tokenizer": tokenizer,
        "max_seq_len": 1024,
        "random_seed": 1024,
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
    train_config = {
        "ppo_minibatch": 8,
        # "pt_minibatch": 1,
        "value_minibatch": 8
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
