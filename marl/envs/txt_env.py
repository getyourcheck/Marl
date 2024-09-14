from collections.abc import Iterable
from copy import deepcopy

import torch
from loguru import logger

from marl.model_server import BaseModelServer
from marl.timer import Timer
from marl.envs.base import EnvBase
from marl.envs.utils import SYSTEM_PROMPT


class TxtEnv(EnvBase):
    """A generic RL environment to generate textual sequences."""

    def __init__(
        self,
        policy_model: BaseModelServer,
        reward_model: BaseModelServer,
        prompt_mes_iter: Iterable,
        pretrain_mes_iter: Iterable = None,
        max_new_tokens: int = 1024,
        policy_micro_bs: int = 32,
        reward_micro_bs: int = 32,
        async_reward: bool = True,
        generate_kwargs: dict = None,
        resume_step=-1,
        **_ignored,
    ):
        self.policy_model = policy_model
        self.reward_model = reward_model

        self.prompt_mes_iter = iter(prompt_mes_iter)
        self.pretrain_mes_iter =  None

        self.max_new_tokens = max_new_tokens
        self.policy_micro_bs = policy_micro_bs
        self.reward_micro_bs = reward_micro_bs
        self.async_reward = async_reward
        self.generate_kwargs: dict = generate_kwargs
        self.resume_step = resume_step

    def rollout(self, display=True):
        while self.resume_step > 0:
            logger.info(f'[Resume] {self.resume_step} consuming data...')
            next(self.prompt_mes_iter)
            if self.pretrain_mes_iter is not None:
                next(self.pretrain_mes_iter)
            self.resume_step -= 1
        prompt_datas = deepcopy(next(self.prompt_mes_iter))
        prompt_input_messages = []
        for data in prompt_datas:
            assert data['mes_type'] == 'prompt'
            if data['sys_prompt'] != 'default':
                message = deepcopy([
                    dict(
                        role='system', content=SYSTEM_PROMPT[data['sys_prompt']])
                ] + data['message'])
            else:
                message = deepcopy(data['message'])
            prompt_input_messages.append(message)
        # prompt data
        if display:
            logger.info(
                f'[TXT_ENV For Generate]: \n{prompt_input_messages[0]}')
        with Timer('policy_model.generate'):
            trajectories = self.policy_model.generate(
                inputs=prompt_input_messages,
                micro_batch_size=self.policy_micro_bs,
                step=self.max_new_tokens,
                output_str=True,
                output_logprobs=True,
                generate_kwargs=self.generate_kwargs)
        logger.info(f'[Generate] len: {len(prompt_input_messages)}')

        if self.async_reward:
            reward_output_ref = self.get_reward_async(prompt_datas,
                                                      trajectories)
            trajectories['reward_output_ref'] = reward_output_ref
        else:
            rewards = self.get_reward(prompt_datas, trajectories)
            trajectories['rewards'] = rewards

        return trajectories

    # default get_reward() is blocking.
    # get_reward_async() needs to call get_reward_collect()
    def get_reward_async(self, prompt_datas, policyout):
        rm_input_messages = []
        for i in range(len(prompt_datas)):
            if prompt_datas[i]['mes_type'] != 'prompt':
                continue
            if (prompt_datas[i]['rm_prompt'] !=
                    'default') or (prompt_datas[i]['sys_prompt'] != 'default'):
                # Conditional Reward Model
                # for queries from different domains, use appropriate conditional system prompts  # noqa: E501
                # From Alignment section of the InternLM2 Technical Report:
                # https://arxiv.org/pdf/2403.17297
                if prompt_datas[i]['rm_prompt'] != 'default':
                    prompt = prompt_datas[i]['rm_prompt']
                else:
                    prompt = prompt_datas[i]['sys_prompt']
                cur_rm_data = [
                    dict(role='system', content=SYSTEM_PROMPT[prompt])
                ] + prompt_datas[i]['message'] + [
                    dict(
                        role='assistant', content=policyout.output_ans_str[i])
                ]
            else:
                cur_rm_data = prompt_datas[i]['message'] + [
                    dict(
                        role='assistant', content=policyout.output_ans_str[i])
                ]
            rm_input_messages.append(cur_rm_data)

        logger.info(f'[For Reward]: {rm_input_messages[0]}')
        with Timer('reward_model.infer_async'):
            reward_output_ref = self.reward_model.infer_async(
                rm_input_messages,
                output_logprobs=False,
                micro_batch_size=self.reward_micro_bs)
        return reward_output_ref

    def get_reward_collect(self, reward_output_ref):
        with Timer('reward_model.infer_get'):
            rm_out = self.reward_model.infer_get(reward_output_ref)
        rewards = rm_out.logits.squeeze(-1)
        return rewards

    def get_reward(self, prompt_datas, policyout):
        rm_input_messages = []
        for i in range(len(prompt_datas)):
            if prompt_datas[i]['mes_type'] != 'prompt':
                continue
            if prompt_datas[i]['rm_prompt'] != 'default':
                cur_rm_data = [
                    dict(
                        role='system',
                        content=SYSTEM_PROMPT[prompt_datas[i]['rm_prompt']])
                ] + prompt_datas[i]['message'] + [
                    dict(
                        role='assistant', content=policyout.output_ans_str[i])
                ]
            else:
                cur_rm_data = prompt_datas[i]['message'] + [
                    dict(
                        role='assistant', content=policyout.output_ans_str[i])
                ]
            rm_input_messages.append(cur_rm_data)

        logger.info(f'[For Reward]: {rm_input_messages[0]}')
        with Timer('reward_model.infer'):
            rm_out = self.reward_model.infer(
                rm_input_messages,
                output_logprobs=False,
                micro_batch_size=self.reward_micro_bs)
        rewards = rm_out.logits.squeeze(-1)
        return rewards
