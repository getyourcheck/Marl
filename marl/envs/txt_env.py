from abc import abstractmethod
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset
from copy import deepcopy


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
        self.max_step = 1024

    def rollout(self, policy, display=False):
        s_t = time.time()
        sample_data = next(self.dataloader)
        # print(sample_data) [sequence, ...]
        # Sequence(token_ids_padding=array([    0, 60836, ...,     0,     0,     0]), token_ids=array([    0, 60836, 68855, ...]), question_mask=array([1, 1, 1, ..., 0, 0, 0]), prompt='请回答下面的问题：\n小孩身上起湿疹，该怎么办我家女儿今年5岁了，身上过敏，快半个月了，到几家医院看了。都说是湿疹，刚开始就脸上有，现在身上也有请问这种情况要怎么办，该用什么药效果比较好\n答案：\n\n在你的整个回答中，在你的答案中至少突出显示 4 个部分，例如用markdown语法 `` 符号突出显示。同时，首先，不加改变地重复问题，然后给出你的回答（在重复求之前不要说任何话；你需要重复的不包括这句话）。', group='test_0', idx='zh-1015')

        input_questions = []
        input_ids = []
        questions_masks = []
        for data in sample_data:
            input_ids.append(data.token_ids_padding)
            questions_masks.append(data.question_mask)

            input_questions.append(data.prompt)
        input_ids = torch.from_numpy(np.array(input_ids))

        # TODO deal with answer & mask
        output_gen = policy.generate(input_questions, step=self.max_step, output_logits=True, output_str=True)
        # question/ans mask
        # _question_mask = np.ones((output_gen.logprobs.shape))
        # output_gen["question_mask"] = torch.from_numpy(np.array(_question_mask))
        # output_gen["answer_mask"] = deepcopy(output_gen["question_mask"])

        print(f"[TxtEnv & {policy.__class__.__name__}] {round(time.time() - s_t, 2)}s generate {len(sample_data)} episodes.")
        output_gen["rewards"] = self._get_reward(output_gen)

        # print(dir(trajectories))
        # print(trajectories.rewards) # [0.435546875, -2.140625, -0.69140625, -1.6796875, -1.6796875, 1.5859375, -1.9296875]
        # # print(trajectories.logits[0].shape) # torch.Size([7, 92544])
        # print("output_ids", trajectories.output_ids.shape) # torch.Size([7, 1145]
        # print("output_str", trajectories.output_str) # None
        # print("logprobs", trajectories.logprobs) # None
        return output_gen
    
    def _get_reward(self, policyout):
        if self.reward_function is not None:
            rewards = []
            rm_strs = policyout.output_str
            # TODO deal with rm data
            # chat_1_str = "<s><|im_start|>user\nHello! What's your name?<|im_end|>\n<|im_start|>assistant\nMy name is InternLM2!<|im_end|>\n"
            for rm_s in rm_strs:
                r = self.reward_function.infer(rm_s)
                rewards.append(r.logits.item())
            return rewards
        print(f"[TxtEnv] No reward funtion, no reward provided.")
        return None

    # # Standard gym methods
    # def step(self, action):
    #     pass

    # def render(self):
    #     pass
