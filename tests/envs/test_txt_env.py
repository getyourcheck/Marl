"""Test txt env

Run `python tests/env/test_txt_env.py`.
"""
# import pytest
import sys
sys.path.extend(["./", "marl/dataset"])
from collections import defaultdict
from transformers import AutoTokenizer
from marl.dataset.txt_loader import MultiSourceDatset, TxtSequenceDataset
from marl.envs.txt_env import TxtEnv
import torch

if __name__ == "__main__":
    """ppo reader test here
    """
    model_path = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset_config = {
        "task_group_filename": "data/config/task_ppo.json",
        "tokenizer": tokenizer,
        "max_seq_len": 1024,
        "num_samples_each_epoch": 7,
        "random_seed": 1,
        "start_token": "[UNUSED_TOKEN_146]user\n",
        "end_token": "[UNUSED_TOKEN_145]\n",
    }

    """Create txt env for PPO """
    txt_loader = TxtSequenceDataset(**dataset_config)
    txt_env = TxtEnv(dataloader=txt_loader, reward_function=None)

    # model
    from marl.config import Config
    trainer_config = Config(
                            dict(
                                model_path = "facebook/opt-1.3b",
                                trainer_type = "huggingface",
                                parallel = dict(
                                    data = dict(size = 1),
                                    tensor = dict(size = 1, mode = "1d"),
                                    pipeline = dict(size = 1, interleaved_overlap = False),
                                ),
                            ),
                        )

    from marl.model_backend.hf_model_trainer import HfModelTrainer
    actor_model = HfModelTrainer(model_config=trainer_config)

    trajectories = txt_env.rollout(policy=actor_model, generate_kwargs={"max_new_tokens": 128})

    print((trajectories.output_ids))
