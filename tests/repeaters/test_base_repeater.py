"""Test txt env

Run `python tests/repeaters/test_base_repeater.py`.
"""
# import pytest
import sys
sys.path.extend(["./", "marl/dataset"])
from collections import defaultdict
from marl.dataset.txt_loader import TxtMessageDataset
from marl.tokenizer.tokenizer_utils import get_tokenizer
from marl.envs.txt_env import TxtEnv
from marl.repeaters.base import BaseRepeater
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
    txt_loader = TxtMessageDataset(**dataset_config)
    txt_env = TxtEnv(dataloader=txt_loader, reward_function=None)

    # init repeater
    rl_repeater = BaseRepeater(reward_scale=False, fine_grained_rm=False, value_ema=False)

    # init model
    from marl.config import Config

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

    models_dict = {"policy": actor_model, "rm_model": None, "value_model": None, "sft_model": None}

    eoc_token_id = tokenizer.sp_model.piece_to_id("[UNUSED_TOKEN_145]")
    generate_kwargs = dict(
                        do_sample=True,
                        temperature=1.0,
                        top_k=0,
                        top_p=0.9,
                        min_new_tokens=1,
                        # max_length=2048,
                        max_new_tokens=2048,
                        eos_token_id=eoc_token_id,
                        pad_token_id=tokenizer.pad_token_id,)

    trajectories = txt_env.rollout(policy=models_dict["policy"], generate_kwargs=generate_kwargs)
    # deal with trajectories
    trajectories = rl_repeater.process(trajectories, models_dict)
    print(dir(trajectories))

    # for policy & critic learn
    print(trajectories.output_ids.shape)
    print(trajectories.question_mask.shape)
    print(trajectories.answer_mask.shape)
    print(trajectories.kl_rewards.shape)
    print(trajectories.values.shape)
    print(trajectories.advs.shape)
    print(trajectories.returns.shape)
