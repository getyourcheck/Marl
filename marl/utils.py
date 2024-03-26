import torch
import random
import numpy as np
import os
from typing import Optional, Union

DEFAULT_SEED_NUMBER = 1234


def set_seed(seed: int = DEFAULT_SEED_NUMBER):
    if seed is None or not isinstance(seed, int):
        seed = DEFAULT_SEED_NUMBER
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # refer to https://pytorch.org/docs/1.13/notes/randomness.html#reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn_deterministic = True
        torch.backends.cudnn_benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # refer to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.putenv(
        "CUBLAS_WORKSPACE_CONFIG", os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    )

def encode(inputs: Union[list[str], list[list[dict]]], tokenizer, add_generation_prompt:bool=False):
    if isinstance(inputs[0], list):
        inputs = [tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=add_generation_prompt, return_tensors="pt") for input in inputs]
    output = tokenizer(inputs, return_tensors="pt", padding=True)
    return output.input_ids, output.attention_mask

def expand_reward_token_id(reward_token_id:int, input_ids:torch.Tensor, attention_mask:Optional[torch.Tensor]=None):
    input_ids = torch.cat(
        [
            input_ids,
            torch.tensor([[reward_token_id]], dtype=torch.long).expand(input_ids.shape[0], 1),
        ],
        dim=1,
    ).to(input_ids.device)
    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(attention_mask.shape[0], 1, dtype=torch.bool),
            ],
            dim=1,
        ).to(attention_mask.device)
    return input_ids, attention_mask