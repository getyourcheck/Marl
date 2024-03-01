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
class Sequence:
    token_ids: List[int]
    prompt: str = None
    group: str = None
    idx: str = None


class TxtSequenceDataset(IterableDataset):
    """ Create sequences from dataset.
    """
    def __init__(self,
                 task_group_filename: str = None,
                 tokenizer=None,
                 start_token = "[UNUSED_TOKEN_146]user\n",
                 end_token = "[UNUSED_TOKEN_145]\n",
                 max_seq_len: int = 4096,
                 num_samples_each_epoch: int = 64,
                 is_valid: bool = False,
                 random_seed: int = 110,
                 **kwargs
                 ):
        
        assert task_group_filename is not None, "[Data error] Specify your data task config"
        self.example_dataset = MultiSourceDatset(task_group_filename=task_group_filename,
                                                sub_dataset_type="file",
                                                )
        self.tokenizer = tokenizer
        self.start_token = start_token
        self.end_token = end_token
        self.max_seq_len = max_seq_len
        self.num_samples_each_epoch = num_samples_each_epoch
        self.is_valid = is_valid
        
        self.random_seed = random_seed
        self.rng = random.Random(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        if not is_valid:
            samples_cnts = []
            for task in self.example_dataset._task_group:
                task["target_num_each_epoch"] = int(task["prob"] * num_samples_each_epoch + 0.5)
                inner_dataset = InfiniteDataset(task["dataset"], self.rng)
                task["iterator"] = iter(inner_dataset)
                samples_cnts.append(task["target_num_each_epoch"])
                print(f"{task['filepath']}: task prob: {task['prob']}, "
                        f"ori number of examples: {len(inner_dataset.data)}, "
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
            # print(f"prepare TxtSequenceDataset for epoch {self.epoch_index}...")
            examples_all = []
            for task in self.example_dataset._task_group:
                if self.is_valid:
                    examples = [ex for ex in task["dataset"]]
                else:
                    examples = [next(task["iterator"]) for _ in range(task["target_num_each_epoch"])]
                print(f"prepare {len(examples)} data from {task['filepath']}")
                epoch_rng.shuffle(examples)
                examples_all.extend(examples)
            epoch_rng.shuffle(examples_all)
            print(f"prepare TxtSequenceDataset done: total number of examples is {len(examples_all)} for epoch {self.epoch_index}.")

            batch_sequence = []
            for index, example in enumerate(examples_all):
                sequence = self._postprocess_sequence(example)
                batch_sequence.append(sequence)
            assert len(batch_sequence) == self.num_samples_each_epoch, f"[Epoch {self.epoch_index}] Wrong data len"
            yield batch_sequence

            self.epoch_index += 1

    def _postprocess_sequence(self, example):
        """Post process sequence: tokenization & truncation."""
        # print(example)
        question, answer = example["question"].strip(), example.get("answer", "").strip()
        
        tokens_question, tokens_answer = self.tokenizer.tokenize(question), self.tokenizer.tokenize(answer)
        
        tokens = (tokens_question + tokens_answer)

        if len(tokens) <= 4:
            return None
        
        tokens = [self.start_token] + tokens + [self.end_token]

        assert len(tokens) <= self.max_seq_len, "{}-{}".format(len(tokens), self.max_seq_len)
        if len(tokens) > self.max_seq_len:
            raise RuntimeError(f"token_ids is too long: {len(tokens)}")
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # pos_ids = list(range(len(token_ids)))
        token_ids = self._padding_data(torch.tensor(token_ids))
        
        return Sequence(group=example["group"],
                        idx=example["id"],
                        prompt=question + answer,
                        token_ids=token_ids,
                        )

    def _padding_data(self, data):
        data = np.array(data)
        shape = data.shape
        if len(shape) == 3:
            res = np.zeros([1, self.max_seq_len, self.max_seq_len], dtype=data.dtype)
            res[:,:shape[1], :shape[2]] = data
        elif len(shape) == 2:
            res = np.zeros([1, self.max_seq_len], dtype=data.dtype)
            res[:,:shape[1]] = data
        else:
            res = np.zeros([self.max_seq_len], dtype=data.dtype)
            res[:len(data)] = data
        return res 
