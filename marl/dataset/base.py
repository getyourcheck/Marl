""" Basic datasets implement. """

import gzip
import json
import random
from contextlib import contextmanager
from torch.utils.data import IterableDataset


@contextmanager
def open_file(filename):
    """Construct a file handler.

    The handler can read a normal file or a file compressed by `gzip`.
    """
    if filename.endswith(".gz"):
        fp = gzip.open(filename, "rt")
    else:
        fp = open(filename, "r", encoding="utf-8")
    yield fp
    fp.close()


class InfiniteDataset(IterableDataset):
    """ Load infinite data from original dataset with shuffle. """

    def __init__(self, dataset, rng=None):
        self.data = list(iter(dataset))
        self.indices = list(range(len(self.data)))
        if rng is None:
            rng = random.Random()
        self.rng = rng

    def __iter__(self):
        while True:
            self.rng.shuffle(self.indices)
            for i in self.indices:
                yield self.data[i]


class FileDataset(IterableDataset):
    """Single json file dataset."""

    def __init__(self, filename, tokenizer):
        self._filename = filename
        self.tokenizer = tokenizer

    def __iter__(self):
        with open_file(self._filename) as fin:
            for lineno, line in enumerate(fin):
                data = json.loads(line)
                try:
                    self.tokenizer.apply_chat_template(data, tokenize=True)
                except:
                    print(f"[data tokenize check] skip dirty data: {data}")
                    continue
                if data is None:
                    continue
                yield data


class OpensourceDataset(IterableDataset):
    """Opensource dataset."""

    def __init__(self, filename, tokenizer):
        self._filename = filename
        self.tokenizer = tokenizer
        assert "Anthropic" in filename or "openai" in filename, "[Coming soon] currently only support loading Anthropic and openai opensource datasets..."
        if "Anthropic" in filename:
            from .open_datasets.Anthropic_hh_rlhf import AnthropicHhrlhf
            self.data_list = AnthropicHhrlhf(path=filename)
        elif "openai" in filename:
            pass
        else:
            raise NotImplementedError()

    def __iter__(self):
        for lineno, data in enumerate(self.data_list):
            if data is None:
                continue
            try:
                self.tokenizer.apply_chat_template(data, tokenize=True)
            except:
                print(f"[data tokenize check] skip dirty data: {data}")
                continue
            yield data


class MultiSourceDatset(IterableDataset):
    """Multiple source dataset."""

    def __init__(self, task_groups, sub_dataset_type="file", tokenizer=None, random_seed=1024):
        self._task_group = []
        for _task in task_groups:
            file_path, extra_info = _task.split("::")[0], _task.split("::")[1]
            prob = float(extra_info.split("[")[0])
            sys_meta = "default"
            rm_meta = "default"
            if '[META]:' in extra_info:
                sys_meta = extra_info.split("[META]:")[-1].split("[")[0]
            if "[REWARD_META]:" in extra_info:
                rm_meta = extra_info.split("[REWARD_META]:")[-1].split("[")[0]
            if prob > 0:
                self._task_group.append({'prob': prob, 'filepath': file_path, "sys_meta": sys_meta, "rm_meta": rm_meta})
                print(f"[DataLoader] Load {_task} with prob:{prob}, sys_meta type: {sys_meta}, reward meta: {rm_meta}")
            else:
                print(f"[DataLoader] Warning skip file, prob of {file_path} is {prob} ...")
        assert len(self._task_group) > 0, "No data to be trained"
        if sub_dataset_type == "file":
            for task in self._task_group:
                filepath = task["filepath"]
                if ".json" in filepath:
                    task["dataset"] = FileDataset(filepath, tokenizer)
                else:
                    # loading opensource datasets
                    print(f"Try loading {filepath} from huggingface ...")
                    task["dataset"] = OpensourceDataset(filepath, tokenizer)
        else:
            raise NotImplementedError("Cannot support filelist now.")
        sum_prob = sum([task["prob"] for task in self._task_group])
        for task in self._task_group:
            task["prob"] = task["prob"] / sum_prob

        self.random_seed = random_seed

    def __iter__(self):
        """ sample data one task by probs
        """
        rng = random.Random(self.random_seed)
        probs = [task["prob"] for task in self._task_group]
        # Initialize task iterator
        for task in self._task_group:
            task["iterator"] = iter(task["dataset"])
        while True:
            task = rng.choices(self._task_group, weights=probs)[0]
            try:
                yield from task["iterator"]
            except StopIteration:
                task["iterator"] = iter(task["dataset"])
                yield from task["iterator"]


if __name__ == "__main__":
    dataset_config = {
        "task_groups": ["./data/ppo_data/ppo_data_1.json::0.1[META]:summarization[REWARD_META]:cn-safety",
                        "./data/ppo_data/ppo_data_0.json::0.9[META]:summarization",
                        "./data/ppo_data/ppo_data_1.json::0.1[REWARD_META]:cn-safety",
                        "./data/ppo_data/ppo_data_1.json::0.1",
                        "./data/ppo_data/ppo_data_1.json::0.0",
                        "Anthropic/hh-rlhf::0.5"
                        ],
        "max_seq_len": 4096,
        "num_samples_each_epoch": 8,
        "random_seed": 1
    }
    from transformers import AutoTokenizer

    """ppo reader test here"""
    model_path = "/cpfs01/shared/public/public_hdd/lishuaibin/models/1.8B_baseline/sft/Luyou_1B_FT_0.19_130_avg5/"
    model_path = "/fs-computility/llm/shared/marl/models/internlm2/1.8B/hf/Luyou_1B_FT_0.19_130_avg5/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    example_dataset = iter(MultiSourceDatset(
        task_groups=dataset_config["task_groups"],
        sub_dataset_type="file",
        tokenizer=tokenizer
    ))
    print(next(example_dataset))
