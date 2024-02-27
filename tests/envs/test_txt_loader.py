"""Test txt env

Run `python tests/env/test_txt_loader.py`.
"""
# import pytest
import sys
sys.path.extend(["marl/dataset"])
from collections import defaultdict
from transformers import AutoTokenizer
from txt_loader import MultiSourceDatset, TxtSequenceDataset

if __name__ == "__main__":
    """ppo reader test here
    """
    model_path = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset_config = {
        "task_group_filename": "data/config/task_ppo.json",
        "tokenizer": tokenizer,
        "max_seq_len": 4096,
        "num_samples_each_epoch": 8,
        "random_seed": 1024,
        "start_token": "[UNUSED_TOKEN_146]user\n",
        "end_token": "[UNUSED_TOKEN_145]\n",
    }

    """Create txt env for PPO """
    ppo_dataset = MultiSourceDatset(
        task_group_filename=dataset_config["task_group_filename"],
        sub_dataset_type="file",
    )
    # print(ppo_dataset, ppo_dataset._task_group)

    dataset_config["dataset"] = ppo_dataset
    txt_iter = TxtSequenceDataset(**dataset_config)

    test_iter = 0
    for data in txt_iter:
        # print(data[0].token_ids)
        groups = defaultdict(int)
        for d in data:
            groups[d.group] += 1
        assert sum(groups.values()) == dataset_config['num_samples_each_epoch'], "epoch data num error"
        # print(groups.values())
        test_iter += 1
        if test_iter == 100:
            print(data)
            break
