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
    mes_type: str = "ppo"


class TxtMessageDataset(IterableDataset):
    """ Create sequences from dataset.
    """
    def __init__(self,
                 ppo_data_filename: str = None,
                 sft_data_filename: str = None,
                 tokenizer=None,
                 max_seq_len: int = 4096,
                 num_samples_each_epoch: int = 64,
                 sft_data_samples: int = 0,
                 is_valid: bool = False,
                 random_seed: int = 110,
                 **kwargs
                 ):
        
        assert ppo_data_filename is not None, "[Data error] Specify your data task config"
        self.ppo_message_dataset = MultiSourceDatset(task_group_filename=ppo_data_filename,
                                                sub_dataset_type="file",
                                                )
        if sft_data_filename is not None:
            self.sft_message_dataset = MultiSourceDatset(task_group_filename=sft_data_filename,
                                                        sub_dataset_type="file",
                                                        )
            assert sft_data_samples is not None, "[SFT DATA error] samples num"
            self.sft_data_per_epoch = sft_data_samples
            self.ppo_data_per_epoch = num_samples_each_epoch - self.sft_data_per_epoch
        else:
            self.sft_message_dataset = None
            self.sft_data_per_epoch = 0
            self.ppo_data_per_epoch = num_samples_each_epoch

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
            sft_data_len = 0
            if self.sft_message_dataset is not None:
                for task in self.sft_message_dataset._task_group:
                    task["target_num_each_epoch"] = int(task["prob"] * self.sft_data_per_epoch + 0.5) + 1
                    inner_dataset = InfiniteDataset(task["dataset"], self.rng)
                    task["iterator"] = iter(inner_dataset)
                    samples_cnts.append(task["target_num_each_epoch"])
                    print(f"[SFT data] {task['filepath']}: task prob: {task['prob']}, "
                            f"ori number of messages: {len(inner_dataset.data)}, "
                            f"target_num_each_epoch: {task['target_num_each_epoch']}")
                sft_data_len = sum(samples_cnts)
                # TODO
                assert sft_data_len >= self.sft_data_per_epoch, f"Make sure there are enough sft datas, {sft_data_len} >= {self.sft_data_per_epoch}"
                print(f"[SFT data] sft data per epoch: {self.sft_data_per_epoch}, sampled {sft_data_len}")

            for task in self.ppo_message_dataset._task_group:
                task["target_num_each_epoch"] = int(task["prob"] * self.ppo_data_per_epoch + 0.5) + 1
                inner_dataset = InfiniteDataset(task["dataset"], self.rng)
                task["iterator"] = iter(inner_dataset)
                samples_cnts.append(task["target_num_each_epoch"])
                print(f"{task['filepath']}: task prob: {task['prob']}, "
                        f"ori number of messages: {len(inner_dataset.data)}, "
                        f"target_num_each_epoch: {task['target_num_each_epoch']}")
            assert (sum(samples_cnts) - sft_data_len) >= self.ppo_data_per_epoch, "Make sure there are enough ppo datas"
            print(f"[PPO data] ppo data per epoch: {self.ppo_data_per_epoch}, sampled: {sum(samples_cnts) - sft_data_len}")

            # assert sum(samples_cnts) >= num_samples_each_epoch, "[Dataset init] sample num error"
            if sum(samples_cnts) <= num_samples_each_epoch:
                print(f"[Txt loader] Warning!!! sample nums {sum(samples_cnts)} <= samples {num_samples_each_epoch}")
            print(f"[Txt] Training dataset initialized, random seed {self.random_seed}.")
            print()
        self.epoch_index = 0

    def __iter__(self):
        while True:
            batch_sequence = []
            ppo_sequence, sft_sequence = [], []

            # epoch_rng only use in this epoch.
            epoch_rng = np.random.RandomState(self.epoch_index)
            # prepare epoch data
            # print(f"prepare TxtMessageDataset for epoch {self.epoch_index}...")
            if self.sft_message_dataset is not None:
                sft_batch_messages = []
                for task in self.sft_message_dataset._task_group:
                    if self.is_valid:
                        messages = [ex for ex in task["dataset"]]
                    else:
                        messages = [next(task["iterator"]) for _ in range(task["target_num_each_epoch"])]
                    print(f"[SFT] prepare {len(messages)} data from {task['filepath']}")
                    epoch_rng.shuffle(messages)
                    sft_batch_messages.extend(messages)
                    # if len(sft_batch_messages) == self.sft_data_per_epoch:
                    #     break
                epoch_rng.shuffle(sft_batch_messages)
                for index, message in enumerate(sft_batch_messages):
                    sequence = self._postprocess_sequence(message)
                    if sequence is not None:
                        assert sequence.mes_type == 'sft', f"Data type should in sft, but get {sequence.mes_type}"
                        sft_sequence.append(sequence)
                        if len(sft_sequence) == self.sft_data_per_epoch:
                            break
            assert len(sft_sequence) == self.sft_data_per_epoch, ""

            ppo_batch_messages = []
            for task in self.ppo_message_dataset._task_group:
                if self.is_valid:
                    messages = [ex for ex in task["dataset"]]
                else:
                    messages = [next(task["iterator"]) for _ in range(task["target_num_each_epoch"])]
                print(f"[PPO] prepare {len(messages)} data from {task['filepath']}")
                epoch_rng.shuffle(messages)
                ppo_batch_messages.extend(messages)
            epoch_rng.shuffle(ppo_batch_messages)
            for index, message in enumerate(ppo_batch_messages):
                sequence = self._postprocess_sequence(message)
                if sequence is not None:
                    assert sequence.mes_type == 'ppo', f"Data type should in ppo. but get {sequence.mes_type}"
                    ppo_sequence.append(sequence)
                    if len(ppo_sequence) == self.ppo_data_per_epoch:
                        break
            assert len(ppo_sequence) == self.ppo_data_per_epoch, ""

            print(f"prepare TxtMessageDataset done: {len(ppo_sequence)} ppo & {len(sft_sequence)} sft, for epoch {self.epoch_index}.")
            batch_sequence = ppo_sequence + sft_sequence
            assert len(batch_sequence) == self.num_samples_each_epoch, "[Epoch {self.epoch_index}] Wrong data len"
            yield batch_sequence
            self.epoch_index += 1

    def _postprocess_sequence(self, message):
        """Post process sequence: tokenization & truncation."""
        mes_type = "ppo"
        if message[-1]["role"] == "assistant":
            if message[-1]["content"] != '':
                token_ids = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=False, return_tensors="pt")
                mes_type = "sft"
            else:
                message = message[:-1]
                token_ids = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        else:
            token_ids = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if token_ids.shape[-1] <= 4:
            return None

        assert token_ids.shape[-1] <= self.max_seq_len, "{}-{}".format(token_ids.shape[-1], self.max_seq_len)
        if token_ids.shape[-1] > self.max_seq_len:
            # TODO truncation
            # raise RuntimeError(f"token_ids is too long: {token_ids.shape[-1]}")
            print(f"[TXT Loader] Warning, {mes_type} message {message} is too long, skipped...")
            return None
        return Message(message=message,
                       token_ids=token_ids,
                       mes_type=mes_type)


if __name__ == "__main__":
    import sys
    import time
    sys.path.extend(["./"])
    from collections import defaultdict
    from transformers import AutoTokenizer

    """ppo reader test here"""
    model_path = "internlm/internlm2-chat-1_8b-sft"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dataset_config = {
        "ppo_data_filename": "data/config/1.8B_ppo.json",
        "sft_data_filename": "data/config/1.8B_sft.json",
        "num_samples_each_epoch": 10,
        "sft_data_samples": 2,
        "tokenizer": tokenizer,
        "max_seq_len": 4096,
        "random_seed": 1024,
    }

    """Create txt env for PPO """
    txt_iter = TxtMessageDataset(**dataset_config)

    for data in txt_iter:
        types = [d.mes_type for d in data]
        print(types)
        time.sleep(2)

