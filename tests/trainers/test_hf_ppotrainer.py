"""Test txt env

Run `python tests/trainers/test_hf_ppotrainer.py`.
"""
# import pytest
import sys
sys.path.extend(["./", "marl/dataset"])
from collections import defaultdict
from marl.dataset.txt_loader import TxtSequenceDataset
from marl.tokenizer.tokenizer_utils import get_tokenizer
from marl.envs.txt_env import TxtEnv
from marl.repeaters.base import BaseRepeater
from marl.trainer.ppo import PPOTrainer
import torch

if __name__ == "__main__":
    """ppo reader test here"""
    model_path = "internlm/internlm2-chat-1_8b-sft"
    tokenizer_path = model_path
    tokenizer_path = "/cpfs01/shared/public/llm_model/tokenizer/v13.model" # also work
    tokenizer = get_tokenizer(tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    dataset_config = {
        "task_group_filename": "data/config/task_ppo.json",
        "tokenizer": tokenizer,
        "max_seq_len": 1024,
        "num_samples_each_epoch": 7,
        "random_seed": 1,
        "start_token": "[UNUSED_TOKEN_146]user\n",
        "end_token": "[UNUSED_TOKEN_145]\n",
    }
    # init txt env
    txt_loader = TxtSequenceDataset(**dataset_config)

    # init repeater
    rl_repeater = BaseRepeater(reward_scale=False, fine_grained_rm=False, value_ema=False)

    # init model
    from marl.config import Config
    from marl.model_backend.hf_model_runner import HfModelRunner
    from marl.config_consts import ENGINE_HUGGINGFACE
    trainer_config = Config(
        dict(
            model_path="internlm/internlm2-chat-1_8b-sft",
            torch_dtype=torch.float16,
            model_type="actor",
            trainer_type=ENGINE_HUGGINGFACE,
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_token",
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
    from marl.config_consts import MODEL_TYPE_REWARD, ENGINE_HUGGINGFACE
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
    from marl.config_consts import MODEL_TYPE_CRITIC, ENGINE_HUGGINGFACE
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
                loss_type="per_token",
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

    txt_env = TxtEnv(dataloader=txt_loader, reward_function=reward_model)
    models_dict = {"policy": actor_model, "rm_model": reward_model, "value_model": critic_model, "sft_model": actor_model}

    ppo = PPOTrainer(policy_model=actor_model, value_model=None)
    
    while True:
        for _ in range(2):
            trajectories = txt_env.rollout(policy=models_dict["policy"])
            # deal with trajectories
            trajectories = rl_repeater.process(trajectories, models_dict)
            # print(dir(trajectories))
            # 'advs', 'attentions', 'kl_rewards', 'logits', 'logits_entropy', 'logprobs', 'output_ids', 
            # 'output_str', 'policy_logprobs', 'returns', 'rewards', 'sft_logprobs', 'values'
            # # for policy & critic learn
            ppo_loss = ppo.policy_learn(trajectories, actor_model)
            print(_, ppo_loss)
            print("===========")
            value_loss = ppo.value_learn(trajectories, critic_model)
            print(_, value_loss)
        break
