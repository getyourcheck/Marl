from abc import abstractmethod
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset


class TxtEnv(object):
    """
    A generic RL environment to generate textual sequences.
    """

    def __init__(self, dataloader: IterableDataset, reward_function=None, ):
        """
        Args:
            dataloader (IterableDataset): generate rl data iteratively
            reward_function: reward function that computes scalar reward for each episode
        """
        self.dataloader = iter(dataloader)
        self.reward_function = reward_function
        self._cur_episodes = []

    def rollout(self, policy, generate_kwargs=None, display=False):
        s_t = time.time()
        sample_data = next(self.dataloader)

        input_ids = []
        for data in sample_data:
            input_ids.append(data.token_ids)
        input_ids = torch.from_numpy(np.array(input_ids))

        generate_kwargs = dict() if generate_kwargs is None else generate_kwargs
        output_gen = policy.generate(input_ids, **generate_kwargs)

        print(f"[TxtEnv & {policy.__class__.__name__}] {round(time.time() - s_t, 2)}s generate {len(sample_data)} episodes.")
        return output_gen


    # # Standard gym methods
    # def step(self, action):
    #     pass

    # def render(self):
    #     pass
