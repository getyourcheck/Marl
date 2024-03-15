from abc import abstractmethod

# import gym
import numpy as np
from torch.utils.data import IterableDataset


# class BaseEnv(gym.Env):
class BaseEnv(object):
    """
    A base class for all the environments
    """

    def __init__(self, dataloader: IterableDataset, reward_function=None, ):
        """
        Args:
            dataloader (IterableDataset): generate rl data iteratively
            reward_function: reward function that computes scalar reward for each episode
        """
        self.dataloader = dataloader
        self.rm_fc = rm_fc

    @abstractmethod
    def rollout(self, policy, episodes=1):

        raise NotImplementedError

    # Standard gym methods
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass
