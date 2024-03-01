import torch
import random
import numpy as np
import os

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
