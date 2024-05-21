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
    sys_meta: str = "default"
    rm_meta: str = "default"
    token_ids: List[int] = None
    mes_type: str = "ppo"


class TxtMessageDataset(IterableDataset):
    """ Create sequences from dataset.
    """
    def __init__(self,
                 ppo_datas: list[str] = None,
                 pt_datas: list[str] = None,
                 tokenizer=None,
                 max_seq_len: int = 4096,
                 num_samples_each_epoch: int = 64,
                 pt_data_samples: int = 0,
                 is_valid: bool = False,
                 random_seed: int = 110,
                 sample_strategy: str = "batch", # TODO, sample data by ratio in "batch" or "all"
                 **kwargs
                 ):
        
        assert ppo_datas is not None, "[Data error] Specify your data task config"
        self.ppo_message_dataset = MultiSourceDatset(task_groups=ppo_datas,
                                                    sub_dataset_type="file",
                                                    )
        if pt_datas is not None:
            self.pt_message_dataset = MultiSourceDatset(task_groups=pt_datas,
                                                        sub_dataset_type="file",
                                                        )
            assert pt_data_samples is not None, "[PT DATA error] samples num"
            self.pt_data_per_epoch = pt_data_samples
            self.ppo_data_per_epoch = num_samples_each_epoch - self.pt_data_per_epoch
        else:
            self.pt_message_dataset = None
            self.pt_data_per_epoch = 0
            self.ppo_data_per_epoch = num_samples_each_epoch

        self.tokenizer = tokenizer
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
            pt_data_len = 0
            if self.pt_message_dataset is not None:
                for task in self.pt_message_dataset._task_group:
                    task["target_num_each_epoch"] = int(task["prob"] * self.pt_data_per_epoch + 0.5) + 1
                    inner_dataset = InfiniteDataset(task["dataset"], self.rng)
                    task["iterator"] = iter(inner_dataset)
                    samples_cnts.append(task["target_num_each_epoch"])
                    print(f"[PT data] {task['filepath']}: task prob: {task['prob']}, "
                            f"ori number of messages: {len(inner_dataset.data)}, "
                            f"target_num_each_epoch: {task['target_num_each_epoch']}")
                pt_data_len = sum(samples_cnts)
                # TODO
                assert pt_data_len >= self.pt_data_per_epoch, f"Make sure there are enough pretrain datas, {pt_data_len} >= {self.pt_data_per_epoch}"
                print(f"[PT data] pretrain data per epoch: {self.pt_data_per_epoch}, sampled {pt_data_len}")

            for task in self.ppo_message_dataset._task_group:
                task["target_num_each_epoch"] = int(task["prob"] * self.ppo_data_per_epoch + 0.5) + 1
                inner_dataset = InfiniteDataset(task["dataset"], self.rng)
                task["iterator"] = iter(inner_dataset)
                samples_cnts.append(task["target_num_each_epoch"])
                print(f"{task['filepath']}: task prob: {task['prob']}, "
                        f"ori number of messages: {len(inner_dataset.data)}, "
                        f"target_num_each_epoch: {task['target_num_each_epoch']}")
            assert (sum(samples_cnts) - pt_data_len) >= self.ppo_data_per_epoch, "Make sure there are enough ppo datas"
            print(f"[PPO data] ppo data per epoch: {self.ppo_data_per_epoch}, sampled: {sum(samples_cnts) - pt_data_len}")

            # assert sum(samples_cnts) >= num_samples_each_epoch, "[Dataset init] sample num error"
            if sum(samples_cnts) <= num_samples_each_epoch:
                print(f"[Txt loader] Warning!!! sample nums {sum(samples_cnts)} <= samples {num_samples_each_epoch}")
            print(f"[Txt] Training dataset initialized, random seed {self.random_seed}.")
            print()
        self.epoch_index = 0

    def __iter__(self):
        while True:
            batch_sequence = []
            ppo_sequence, pt_sequence = [], []

            # epoch_rng only use in this epoch.
            epoch_rng = np.random.RandomState(self.epoch_index)
            # prepare epoch data
            # print(f"prepare TxtMessageDataset for epoch {self.epoch_index}...")
            if self.pt_message_dataset is not None:
                pt_batch_messages = []
                for task in self.pt_message_dataset._task_group:
                    if self.is_valid:
                        messages = [ex for ex in task["dataset"]]
                    else:
                        messages = [next(task["iterator"]) for _ in range(task["target_num_each_epoch"])]
                    print(f"[PT] prepare {len(messages)} data from {task['filepath']}")
                    epoch_rng.shuffle(messages)
                    pt_batch_messages.extend(messages)
                    # if len(pt_batch_messages) == self.pt_data_per_epoch:
                    #     break
                epoch_rng.shuffle(pt_batch_messages)
                for index, message in enumerate(pt_batch_messages):
                    sequence = self._postprocess_sequence(message)
                    if sequence is not None:
                        assert sequence.mes_type == 'pt', f"Data type should in pt, but get {sequence.mes_type}"
                        pt_sequence.append(sequence)
                        if len(pt_sequence) == self.pt_data_per_epoch:
                            break
            assert len(pt_sequence) == self.pt_data_per_epoch, ""

            ppo_batch_messages = []
            for task in self.ppo_message_dataset._task_group:
                messages = []
                if self.is_valid:
                    # messages = [ex for ex in task["dataset"]]
                    pass
                else:
                    for _ in range(task["target_num_each_epoch"]):
                        messages.append({"data": next(task["iterator"]),
                                         "sys_meta": task['sys_meta'],
                                         'rm_meta': task['rm_meta']})
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
            assert len(ppo_sequence) == self.ppo_data_per_epoch, f"{len(ppo_sequence)} == {self.ppo_data_per_epoch}"

            print(f"prepare TxtMessageDataset done: {len(ppo_sequence)} ppo & {len(pt_sequence)} pretrain, for epoch {self.epoch_index}.")
            batch_sequence = ppo_sequence + pt_sequence
            assert len(batch_sequence) == self.num_samples_each_epoch, "[Epoch {self.epoch_index}] Wrong data len"
            yield batch_sequence
            self.epoch_index += 1

    def _postprocess_sequence(self, message):
        """Post process sequence: tokenization & truncation."""
        mes_type = "ppo"
        message_data = message['data']
        if message_data[-1]["role"] == "assistant":
            if message_data[-1]["content"] != '':
                token_ids = self.tokenizer.apply_chat_template(message_data, tokenize=True, add_generation_prompt=False, return_tensors="pt")
                mes_type = "pt"
            else:
                message_data = message_data[:-1]
                token_ids = self.tokenizer.apply_chat_template(message_data, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        else:
            token_ids = self.tokenizer.apply_chat_template(message_data, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if token_ids.shape[-1] <= 4:
            return None

        # assert token_ids.shape[-1] <= self.max_seq_len, "{}-{}".format(token_ids.shape[-1], self.max_seq_len)
        if token_ids.shape[-1] > self.max_seq_len:
            # TODO truncation??
            # raise RuntimeError(f"token_ids is too long: {token_ids.shape[-1]}")
            # print(f"[TXT Loader] Warning, {mes_type} message {message} is too long, skipped...")
            return None
        return Message(message=message_data,
                       token_ids=token_ids,
                       sys_meta=message['sys_meta'],
                       rm_meta=message['rm_meta'],
                       mes_type=mes_type)


if __name__ == "__main__":
    import sys
    import time
    sys.path.extend(["./"])
    from collections import defaultdict
    from transformers import AutoTokenizer

    """ppo reader test here"""
    model_path = "/cpfs01/shared/public/public_hdd/lishuaibin/models/1.8B_baseline/sft/Luyou_1B_FT_0.19_130_avg5/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dataset_config = {
        "ppo_datas": ["/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/0801-train.json::0.1[META]:summarization[REWARD_META]:cn-safety",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/ANLI-0904-train.json::0.1[META]:summarization",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/COIG-0906-train.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/SFT6W-prompts-train.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/data_reflow_2w.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/gaokao_essay_prompt.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/gsm8k_ci.json::0.1",
                      ],
        "pt_datas": ["./data/pt_data/pt_data_0.json::0.9",
                     "./data/pt_data/pt_data_1.json::0.3",
                     ],
        "num_samples_each_epoch": 512,
        "pt_data_samples": 2,
        "tokenizer": tokenizer,
        "max_seq_len": 4096,
        "random_seed": 1024,
    }

    """Create txt env for PPO """
    txt_iter = TxtMessageDataset(**dataset_config)

    for data in txt_iter:
        types = [d for d in data]
        print(types)
        time.sleep(2)

