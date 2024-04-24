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
                 ppo_datas: list[str] = None,
                 pt_datas: list[str] = None,
                 tokenizer=None,
                 max_seq_len: int = 4096,
                 num_samples_each_epoch: int = 64,
                 pt_data_samples: int = 0,
                 is_valid: bool = False,
                 random_seed: int = 110,
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
            assert len(ppo_sequence) == self.ppo_data_per_epoch, f"{len(ppo_sequence)} == {self.ppo_data_per_epoch}"

            print(f"prepare TxtMessageDataset done: {len(ppo_sequence)} ppo & {len(pt_sequence)} pretrain, for epoch {self.epoch_index}.")
            batch_sequence = ppo_sequence + pt_sequence
            assert len(batch_sequence) == self.num_samples_each_epoch, "[Epoch {self.epoch_index}] Wrong data len"
            yield batch_sequence
            self.epoch_index += 1

    def _postprocess_sequence(self, message):
        """Post process sequence: tokenization & truncation."""
        mes_type = "ppo"
        if message[-1]["role"] == "assistant":
            if message[-1]["content"] != '':
                token_ids = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=False, return_tensors="pt")
                mes_type = "pt"
            else:
                message = message[:-1]
                token_ids = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        else:
            token_ids = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if token_ids.shape[-1] <= 4:
            return None

        # assert token_ids.shape[-1] <= self.max_seq_len, "{}-{}".format(token_ids.shape[-1], self.max_seq_len)
        if token_ids.shape[-1] > self.max_seq_len:
            # TODO truncation
            # raise RuntimeError(f"token_ids is too long: {token_ids.shape[-1]}")
            # print(f"[TXT Loader] Warning, {mes_type} message {message} is too long, skipped...")
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
        "ppo_datas": ["/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/0801-train.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/ANLI-0904-train.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/COIG-0906-train.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/SFT6W-prompts-train.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/data_reflow_2w.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/gaokao_essay_prompt.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/gsm8k_ci.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/gsm8k_sample200_prompt_only.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/haochen_data.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/identity_200_sft_for_ppo_prompt_only.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/indomain_writing_2k.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/lingdaoren_adv_4963_20230913-train-rd.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/lmsys-chat-english-chat-format-100char-1t.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/math_ci.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/maxmin_sample200_prompt_only.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/non_toxic_single_turn_tie_both_bad-train.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/non_toxic_single_turn_tie_both_good-train.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_15-rd.json::0.01",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_16-rd.json::0.01",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_17-rd.json::0.01",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_18-rd.json::0.01",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_19-rd.json::0.01",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_20-rd.json::0.01",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_21-rd.json::0.01",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_22-rd.json::0.01",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_23-rd.json::0.01",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_24-rd.json::0.01",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_25-rd.json::0.01",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/out_instinwild_cn_origin_26-rd.json::0.01",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/prm800k_ppo_prompt_1212.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/puyu_chat_format_v2-train.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/retrieval_refined_bench_no_alpaca.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/reward_patch_20240103_prompt_only.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/shezheng_52230_20230905_prompts-train-rd.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/shezheng_adv_7549_20230913-train-rd.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/split_0_prompt-refined.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/split_0_prompt.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/split_1_prompt.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/split_2_prompt.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/subeval_writing_prompt_only_v2.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/toxic_single_turn-train.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/yuqing_5817_20230831_prompts-train-rd.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/yuqing_adv_5817_20230913-train-rd.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/zephyr-ultrachat-200k_ppo_train_1t.json::0.1",
                        "/cpfs01/shared/public/public_hdd/lishuaibin/ppo_data/messages_data/zhihu_177k_outline_to_artical-with-sys.json::0.1",
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
        types = [d.mes_type for d in data]
        print(types)
        time.sleep(2)

