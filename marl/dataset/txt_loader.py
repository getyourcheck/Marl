""" Finetuning dataset. """
import random
from typing import List
import numpy as np
import logging
from dataclasses import dataclass
import torch
from torch.utils.data import IterableDataset
from .base import MultiSourceDatset, InfiniteDataset


@dataclass
class Message:
    message: List[dict]
    token_ids: List[int] = None


class TxtMessageDataset(IterableDataset):
    """ Create sequences from dataset.
    """
    def __init__(self,
                 task_group_filename: str = None,
                 tokenizer=None,
                 max_seq_len: int = 4096,
                 num_samples_each_epoch: int = 64,
                 is_valid: bool = False,
                 random_seed: int = 110,
                 **kwargs
                 ):
        
        assert task_group_filename is not None, "[Data error] Specify your data task config"
        self.message_dataset = MultiSourceDatset(task_group_filename=task_group_filename,
                                                sub_dataset_type="file",
                                                )
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.unk_token
        assert self.tokenizer.chat_template is not None, "Make sure tokenizer has chat_template."

        self.max_seq_len = max_seq_len
        self.num_samples_each_epoch = num_samples_each_epoch
        self.is_valid = is_valid
        
        self.random_seed = random_seed
        self.rng = random.Random(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        if not is_valid:
            samples_cnts = []
            for task in self.message_dataset._task_group:
                task["target_num_each_epoch"] = int(task["prob"] * num_samples_each_epoch + 0.5)
                inner_dataset = InfiniteDataset(task["dataset"], self.rng)
                task["iterator"] = iter(inner_dataset)
                samples_cnts.append(task["target_num_each_epoch"])
                print(f"{task['filepath']}: task prob: {task['prob']}, "
                        f"ori number of messages: {len(inner_dataset.data)}, "
                        f"target_num_each_epoch: {task['target_num_each_epoch']}")
            assert sum(samples_cnts) == num_samples_each_epoch, "[Dataset init] sample num error"
            print(f"[Txt] Training dataset initialized, random seed {self.random_seed}.")
            print()
        self.epoch_index = 0

    def __iter__(self):
        while True:
            # epoch_rng only use in this epoch.
            epoch_rng = np.random.RandomState(self.epoch_index)
            # prepare epoch data
            # print(f"prepare TxtMessageDataset for epoch {self.epoch_index}...")
            messages_all = []
            for task in self.message_dataset._task_group:
                if self.is_valid:
                    messages = [ex for ex in task["dataset"]]
                else:
                    messages = [next(task["iterator"]) for _ in range(task["target_num_each_epoch"])]
                print(f"prepare {len(messages)} data from {task['filepath']}")
                epoch_rng.shuffle(messages)
                messages_all.extend(messages)
            epoch_rng.shuffle(messages_all)
            print(f"prepare TxtMessageDataset done: total number of messages is {len(messages_all)} for epoch {self.epoch_index}.")

            batch_sequence = []
            for index, message in enumerate(messages_all):
                sequence = self._postprocess_sequence(message)
                batch_sequence.append(sequence)
            assert len(batch_sequence) == self.num_samples_each_epoch, f"[Epoch {self.epoch_index}] Wrong data len"
            yield batch_sequence

            self.epoch_index += 1

    def _postprocess_sequence(self, message):
        """Post process sequence: tokenization & truncation."""
        token_ids = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")

        if token_ids.shape[-1] <= 4:
            return None

        assert token_ids.shape[-1] <= self.max_seq_len, "{}-{}".format(token_ids.shape[-1], self.max_seq_len)
        if token_ids.shape[-1] > self.max_seq_len:
            # TODO truncation
            raise RuntimeError(f"token_ids is too long: {token_ids.shape[-1]}")
        return Message(message=message,
                       token_ids=token_ids,)


if __name__ == "__main__":
    import sys

    sys.path.extend(["./"])
    from collections import defaultdict
    from transformers import AutoTokenizer
    from marl.dataset.txt_loader import TxtMessageDataset

    """ppo reader test here"""
    model_path = "internlm/internlm2-chat-1_8b-sft"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dataset_config = {
        "task_group_filename": "data/config/task_ppo.json",
        "tokenizer": tokenizer,
        "max_seq_len": 4096,
        "num_samples_each_epoch": 8,
        "random_seed": 1024,
    }

    """Create txt env for PPO """
    txt_iter = TxtMessageDataset(**dataset_config)

    for data in txt_iter:
        print(data)
        exit()
