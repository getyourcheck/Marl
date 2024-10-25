import torch
from loguru import logger

from ..model_server.base_model_server import BaseModelServer
from ..policy_output import PolicyOutput
from ..timer import Timer
from .base import RepeaterBase
from .utils import RunningStates


class LOORepeater(RepeaterBase):
    def __init__(
        self,
        ref_model: BaseModelServer,
        policy_model: BaseModelServer,
        policy_micro_bs: int = 8,
        ref_micro_bs: int = 8,
        kl_coeff=0.05,
        clip_reward_min: int = -5,
        clip_reward_max: int = 5,
        norm_adv=False,
        env=None,
        loo_k=4,
        sum_kl=True,
        group_advantage=False,
        non_eos_penalty=True,
        penalty_reward_value=-3,
        **_ignored,
    ):
        # models
        self.ref_model = ref_model
        self.policy_model = policy_model

        self.policy_micro_bs = policy_micro_bs
        self.ref_micro_bs = ref_micro_bs
        self.kl_coeff = kl_coeff
        # rewards
        self.clip_reward_min = clip_reward_min
        self.clip_reward_max = clip_reward_max

        self.norm_adv = norm_adv
        # only used for async reward model.infer_get() in _get_kl_rewards
        self.env = env
        self.loo_k = loo_k
        self.sum_kl = sum_kl
        self.group_advantage = group_advantage
        assert not (self.group_advantage and self.norm_adv), "goup advantage and norm_adv should not be used together"
        self.non_eos_penalty = non_eos_penalty
        self.penalty_reward_value = penalty_reward_value

    def process(self, trajectories: PolicyOutput):
        action_mask = trajectories["action_mask"]
        num_actions = action_mask.size(1)
        (
            kl_rewards,
            entropy,
            kl_distance,
            policy_logprobs,
            ref_logprobs,
        ) = self._get_kl_rewards(trajectories)
        trajectories["kl"] = (kl_distance * action_mask).sum(axis=-1) / action_mask.sum(
            axis=-1
        )
        trajectories["seq_kl"] = kl_distance.sum(1)
        trajectories["entropy"] = entropy
        trajectories["kl_rewards"] = kl_rewards
        trajectories["policy_logprobs"] = policy_logprobs
        trajectories["ref_logprobs"] = ref_logprobs

        # vectorized RLOO advantages implementation
        kl_rewards = kl_rewards.reshape(-1, self.loo_k).T
        if not self.group_advantage:
            baseline = (kl_rewards.sum(0) - kl_rewards) / (self.loo_k - 1)
            advantages = kl_rewards - baseline
        else:
            advantages = (kl_rewards - kl_rewards.mean(0)) / (kl_rewards.std(0) + 1e-8)
        advantages = advantages.T.flatten()

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        trajectories["rloo_advantages"] = advantages
        return trajectories

    def _get_kl_rewards(self, trajectories: PolicyOutput):
        with Timer("policy_model.infer_async"):
            policy_output = self.policy_model.infer_async(
                inputs=trajectories.output_ids,
                micro_batch_size=self.policy_micro_bs,
                attention_mask=trajectories.attention_mask,
                output_logits=False,
                output_logprobs=True,
            )
        with Timer("ref_model.infer_async"):
            ref_output = self.ref_model.infer_async(
                inputs=trajectories.output_ids,
                micro_batch_size=self.ref_micro_bs,
                attention_mask=trajectories.attention_mask,
                output_logits=False,
                output_logprobs=True,
            )
        with Timer("policy_model.infer_get"):
            policy_output = self.policy_model.infer_get(policy_output)
        with Timer("ref_model.infer_get"):
            ref_output = self.ref_model.infer_get(ref_output)

        # Experimental
        if self.env.async_reward:
            rewards = self.env.get_reward_collect(trajectories["reward_output_ref"])
            trajectories["reward_output_ref"] = None
            trajectories["rewards"] = rewards
        # Experimental
        clipped_rewards = torch.clamp(
            rewards, min=self.clip_reward_min, max=self.clip_reward_max
        )

        # non eos penalty
        if self.non_eos_penalty:
            assert "finish_reasons" in trajectories, "finish_reasons should be in trajectories"
            logger.info(f"[LOORepeater]: generate finish reasons: {trajectories.finish_reasons}")
            finish_by_eos = [reason == "stop" for reason in trajectories.finish_reasons]
            finish_by_eos = torch.tensor(finish_by_eos, dtype=torch.bool)
            clipped_rewards = torch.where(finish_by_eos, clipped_rewards, self.penalty_reward_value)
        
        trajectories["clipped_rewards"] = clipped_rewards

        action_mask = trajectories.action_mask
        num_actions = action_mask.size(1)

        policy_logprobs = policy_output.logprobs[:, -num_actions:]
        ref_logprobs = ref_output.logprobs[:, -num_actions:]
        INVALID_LOGPROB = 1.0
        policy_logprobs = torch.masked_fill(
            policy_logprobs, action_mask == 0, INVALID_LOGPROB
        )
        ref_logprobs = torch.masked_fill(
            ref_logprobs, action_mask == 0, INVALID_LOGPROB
        )

        if self.kl_coeff <= 0.0:
            self.kl_coeff = 0.0
        # compute_approx_kl
        log_ratio = policy_logprobs - ref_logprobs
        kl = log_ratio * action_mask
        if self.sum_kl:
            kl_reward = (-self.kl_coeff * kl).sum(1)
        else:
            kl_reward = -self.kl_coeff * kl.sum(1) / action_mask.sum(1)

        reward = clipped_rewards + kl_reward

        entropy = -(policy_logprobs * action_mask).sum(axis=-1) / action_mask.sum(
            axis=-1
        )
        return reward, entropy, kl, policy_logprobs, ref_logprobs
