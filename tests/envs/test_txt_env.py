"""Test txt env

Run `python tests/env/test_txt_env.py`.
"""
# import pytest
import sys

sys.path.extend(["./", "marl/dataset"])
from collections import defaultdict
from marl.dataset.txt_loader import TxtMessageDataset
from marl.tokenizer.tokenizer_utils import get_tokenizer
from marl.envs.txt_env import TxtEnv
import torch

if __name__ == "__main__":
    """ppo reader test here"""
    model_path = "internlm/internlm2-chat-1_8b-sft"
    tokenizer_path = model_path
    # tokenizer_path = "/cpfs01/shared/public/llm_model/tokenizer/v13.model" # also work
    tokenizer = get_tokenizer(tokenizer_path, trust_remote_code=True)

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
    txt_loader = TxtMessageDataset(**dataset_config)
    txt_env = TxtEnv(dataloader=txt_loader, reward_function=None)

    # model
    from marl.config.config import Config

    trainer_config = Config(
        dict(
            model_path=model_path,
            torch_dtype=torch.float16,
            trainer_type="huggingface",
            parallel=dict(
                data=dict(size=1),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
            ),
        ),
    )

    from marl.model_backend.hf_model_runner import HfModelRunner

    actor_model = HfModelRunner(model_config=trainer_config)
    actor_model.initialize()

    trajectories = txt_env.rollout(
        policy=actor_model, generate_kwargs={"max_new_tokens": 2048}
    )

    print((trajectories.output_ids))
    for i, s in enumerate(trajectories.output_str):
        print(f"[REPLY {i} BGN] {'#' * 20}\n{s}\n[REPLY {i} END] {'#' * 20}\n")
