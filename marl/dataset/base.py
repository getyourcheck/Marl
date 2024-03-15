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
    """Single file dataset."""

    def __init__(self, filename):
        self._filename = filename

    def __iter__(self):
        with open_file(self._filename) as fin:
            for lineno, line in enumerate(fin):
                ex = json.loads(line)
                if ex is None:
                    continue
                yield ex


class MultiSourceDatset(IterableDataset):
    """Multiple source dataset."""

    def __init__(self, task_group_filename, sub_dataset_type="file", random_seed=1024):
        with open(task_group_filename) as fin:
            tasks = json.load(fin)
            # filter zero probability task
            tasks = [task for task in tasks if task["prob"] > 0]
            self._task_group = tasks
        if sub_dataset_type == "file":
            for task in self._task_group:
                task["dataset"] = FileDataset(task["filepath"])
        else:
            raise NotImplementedError("Cannot support filelist now.")
        sum_prob = sum([task["prob"] for task in self._task_group])
        for task in self._task_group:
            task["prob"] = task["prob"] / sum_prob

        self.random_seed = random_seed

    def __iter__(self):
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
        "task_group_filename": "data/config/task_ppo.json",
        "max_seq_len": 4096,
        "num_samples_each_epoch": 8,
        "random_seed": 1
    }

    example_dataset = MultiSourceDatset(
        task_group_filename=dataset_config["task_group_filename"],
        sub_dataset_type="file",
    )
    print(example_dataset)
