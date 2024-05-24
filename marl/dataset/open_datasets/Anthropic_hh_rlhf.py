import re
import os
import datasets
from datasets import load_dataset
from torch.utils.data import ConcatDataset, Dataset, Subset
from typing import Dict, List


def deduplicate(data: List[str]):
    """
    Deduplicate data while preserving order.
    Refer to https://stackoverflow.com/questions/9792664/converting-a-list-to-a-set-changes-element-order
    """
    return list(dict.fromkeys(data).keys())


class AnthropicHhrlhf(Dataset):
    """
    helpful-base:
        train: 42,537
        test: 2,312

    harmless-base:
        train: 43,835
        test: 2,354

    helpful-online:
        train: 22,007
        test: 1,137

    helpful-rejection-sampled:
        train: 52,421
        test: 2,749

    red-team-attempts:
        train: 38,961
    """

    def __init__(self, path: str="Anthropic/hh-rlhf/helpful-base", test=False):
        super().__init__()
        parts = path.split("/")
        assert "Anthropic" in parts and "hh-rlhf" in parts, f"{self.__class__.__name__}: {path}"
        if parts.index('hh-rlhf') == len(parts) - 1:
            data_dir = None
        else:
            data_dir = parts[-1]
        if os.path.exists(path):
            raw_datasets = datasets.load_from_disk(path)
        else:
            print(f"loading Anthropic/hh-rlhf data_dir={data_dir} from huggingface ...")
            raw_datasets = datasets.load_dataset('Anthropic/hh-rlhf', data_dir=data_dir, trust_remote_code=True)
            raw_datasets.save_to_disk("data/" + path)
        if test:
            raw_data_list = raw_datasets["test"]["chosen"]
        else:
            raw_data_list = raw_datasets["train"]["chosen"]
        raw_data_list = [d for d in raw_data_list if d is not None]
        raw_data_list = deduplicate(raw_data_list)
        self.data_list = [self.format_chatml(prompt) for prompt in raw_data_list]
        self.name = self.__class__.__name__ + '-' + data_dir if data_dir else self.__class__.__name__

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        return self.data_list[index]

    @staticmethod
    def format_chatml(string):
        pattern = r"(\n\nHuman|\n\nAssistant)(.+?)(?=(\n\nHuman|\n\nAssistant|$))"
        matches = re.findall(pattern, string, re.DOTALL)
        messages = []
        for match in matches:
            role, content = match[0].strip(), match[1].strip()
            if role == 'Human':
                messages.append({"role": "user", "content": content[2:]})
            elif role == "Assistant":
                messages.append({"role": "assistant", "content": content[2:]})
            else:
                raise NotImplementedError("role must in Human or Assistant")
        return messages

if __name__ == "__main__":
    path = "Anthropic/hh-rlhf/helpful-base"
    path = "/fs-computility/llm/lishuaibin/0523/marl/marl/dataset/open_datasets/Anthropic/hh-rlhf/helpful-base"
    data_list = AnthropicHhrlhf(path=path)
    print(len(data_list))

    # # data = load_dataset("Anthropic/hh-rlhf", split="train", trust_remote_code=True)
    # data = load_dataset("Anthropic/hh-rlhf", data_dir=None, trust_remote_code=True)
    # print(data["train"])

    # data = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", trust_remote_code=True)
    # print(data["train"])

