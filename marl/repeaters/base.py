from ..policy_output import PolicyOutput
import numpy as np
import torch
from loguru import logger
from marl.model_server.base_model_server import BaseModelServer
import time

def find_mask_begin(padded_datas, mask_id=0):
    """
    finding the mask id begin index and it's length
    """
    begin_indexs = []
    lengths = []

    for padded_data in padded_datas:
        is_flag = 0
        for index, data in enumerate(padded_data):
            if data != mask_id:
                is_flag = 1
                begin_indexs.append(index)
                length = (np.array(padded_data)!=mask_id).sum()
                lengths.append(length)
                break
        assert is_flag
    return begin_indexs, lengths


class BaseRepeater(object):
    def __init__(
            self, 
            sft_model, 
            reward_scale: bool = False,
            fine_grained_rm: bool = False,
            value_ema: bool = False,
            actor_micro_bs:int=8,
            ref_micro_bs:int=8,
            critic_micro_bs:int=32,
            kl_coeff = 0.02,
            gamma = 1.0,
            gae_lambda = 0.95,
            answer_end_id = 92542,
            norm_adv = True,
            **kwargs,
        ):
        self.sft_model = sft_model
        self.actor_micro_bs = actor_micro_bs
        self.ref_micro_bs = ref_micro_bs
        self.critic_micro_bs = critic_micro_bs
        self.reward_scale = reward_scale
        self.fine_grained_rm = fine_grained_rm
        self.value_ema = value_ema
        self.kl_coeff = kl_coeff
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.answer_end_id = answer_end_id
        self.norm_adv = norm_adv

    def process(
            self, 
            trajectories:PolicyOutput, 
            policy_model:BaseModelServer, 
            value_model:BaseModelServer, 
            sft_model:BaseModelServer=None
        ):
        if sft_model is not None:
            self.sft_model:BaseModelServer = sft_model
        kl_rewards, policy_logprobs, sft_logprobs, kl_distance = self._get_kl_rewards(trajectories, policy_model)
        trajectories["kl_distance"] = kl_distance
        trajectories["kl_rewards"] = kl_rewards
        trajectories["policy_logprobs"] = policy_logprobs
        trajectories["sft_logprobs"] = sft_logprobs

        values_with_last_value = self._get_values(trajectories, value_model)
        trajectories["values_with_last_value"] = values_with_last_value

        advantages, returns = self._get_advantages_and_returns(trajectories)
        answer_mask = trajectories["answer_mask"].cpu()
        if self.norm_adv:
            mean =  torch.sum(advantages) / torch.sum(answer_mask + 1e-8)
            var = torch.sum(((advantages - mean) ** 2) * answer_mask) / (torch.sum(answer_mask) + 1e-8)
            trajectories["advantages"] = (advantages - mean) * torch.rsqrt(var + 1e-8)
        else:
            trajectories["advantages"] = advantages
        trajectories["returns"] = returns

        return trajectories

    def _get_kl_rewards(self, trajectories: PolicyOutput, policy_model:BaseModelServer):
        # rewards = trajectories.rewards
        rewards = trajectories.clipped_rewards
        answer_mask = trajectories.answer_mask.cpu()
        attention_mask = trajectories.attention_mask.cpu()
        s_t = time.time()
        policy_output = policy_model.infer_async(
            inputs=trajectories.output_ids, 
            micro_batch_size=self.actor_micro_bs, 
            attention_mask=attention_mask, 
            output_logits=False, 
            output_logprobs=True
        )
        sft_output = self.sft_model.infer_async(
            inputs=trajectories.output_ids, 
            micro_batch_size=self.ref_micro_bs, 
            attention_mask=attention_mask, 
            output_logits=False, 
            output_logprobs=True
        )
        policy_output = policy_model.infer_get(policy_output)
        sft_output = policy_model.infer_get(sft_output)
        logger.info(f"[actor & ref infer_async] duration: {round(time.time() - s_t, 2)} s")

        policy_logprobs = policy_output.logprobs.cpu() * answer_mask
        sft_logprobs = sft_output.logprobs.cpu() * answer_mask

        kl_div = policy_logprobs - sft_logprobs

        kl_divergence_estimate = - 1.0 * self.kl_coeff * kl_div
        # C1rN09: some tokens are supressed which lead log_probs to be -Inf. This will make ratios `nan`
        is_inf = torch.logical_and(policy_logprobs.isinf(), sft_logprobs.isinf())
        kl_divergence_estimate = torch.where(is_inf, 0, kl_divergence_estimate)
        # C1rN09: <pad> tokens should have 0 reward
        kl_rewards = kl_divergence_estimate * answer_mask

        # (bs, 1), Note that in the answerr_mask, the padding is 0
        begins_index, answers_length = find_mask_begin(answer_mask, 0)
        finnal_rewards = kl_rewards.clone()
        count = 0
        kl_distance = []
        # (bs, max_total_len) only add the rewards in the last.
        for begin_index, ans_len in zip(begins_index, answers_length):
            kl_distance.append(finnal_rewards[count, begin_index:begin_index+ans_len].mean().item())
            finnal_rewards[count, begin_index+ans_len-1] += rewards[count]
            count += 1

        return finnal_rewards, policy_logprobs, sft_logprobs, kl_distance

    def _get_values(self, trajectories: PolicyOutput, value_model:BaseModelServer):
        s_t = time.time()
        value_output = value_model.infer(
            inputs = trajectories.output_ids, 
            attention_mask=trajectories.answer_mask, 
            output_logits=True, 
            micro_batch_size=self.critic_micro_bs,
        )
        logger.info(f"[critic infer] duration: {round(time.time() - s_t, 2)} s")
        values_with_last_value = value_output.logits.to(torch.float32).cpu()
        return values_with_last_value

    def _get_advantages_and_returns(self, trajectories):
        output_ids = trajectories.output_ids.cpu()
        answer_mask = trajectories.answer_mask.cpu()
        values_with_last_value = trajectories.values_with_last_value
        kl_rewards = trajectories.kl_rewards

        begins_index, answers_length = find_mask_begin(answer_mask, 0)
        count = 0
        advantages_padded, returns_padded = torch.zeros_like(kl_rewards, dtype=values_with_last_value.dtype), torch.zeros_like(kl_rewards, dtype=values_with_last_value.dtype)
        for begin_index, ans_len, value_with_last_value, reward, output_id in zip(\
            begins_index, answers_length, values_with_last_value, kl_rewards, output_ids):
            # shape :ans_len + 1
            value_with_last_value = value_with_last_value[begin_index-1:begin_index+ans_len]
            # shape :ans_len
            reward = reward[begin_index:begin_index+ans_len]
            last_gae_lam = torch.zeros((1), dtype=values_with_last_value.dtype)
            # shape :ans_len
            advantages = torch.zeros_like(reward, dtype=values_with_last_value.dtype)
            step_nums = advantages.shape[-1]
            # shape:ans_len + 1
            dones = self._build_dones(output_id[begin_index:begin_index+ans_len])
            for step in reversed(range(step_nums)):
                next_non_terminal = 1 - dones[step + 1]
                next_values = value_with_last_value[step+1]
                # delta and last_gae_lam using value and reward 
                delta = reward[step] + self.gamma * next_values * next_non_terminal - value_with_last_value[step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                advantages[step] = last_gae_lam[0]
            returns = advantages + value_with_last_value[:-1]
            advantages_padded[count, begin_index:begin_index+ans_len] = advantages
            returns_padded[count, begin_index:begin_index+ans_len] = returns
            count += 1
        return advantages_padded, returns_padded

    # ans_len + 1: dones
    def _build_dones(self, answer_ids):
        dones = torch.tensor((answer_ids == self.answer_end_id).numpy().astype(np.float32))
        # (1, )the first one is not done, so obs_0_dones=0
        obs_0_dones = torch.zeros((1), dtype=torch.float32)
        # (ans_len + 1)，
        dones = torch.concat((obs_0_dones, dones), axis=0)
        return dones

    # TODO
    def _get_fine_grained_values(self, trajectories):
        pass
