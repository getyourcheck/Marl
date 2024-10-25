import torch
from loguru import logger

from ..model_server.base_model_server import BaseModelServer
from ..policy_output import PolicyOutput
from ..timer import Timer
from .base import RepeaterBase
from .utils import RunningStates


class GRPORepeater(RepeaterBase):
    def __init__(
        self,
        ref_model: BaseModelServer,
        policy_model: BaseModelServer,
        policy_micro_bs: int = 8,
        ref_micro_bs: int = 8,
        clip_reward_min: int = -5,
        clip_reward_max: int = 5,
        norm_adv=False,
        env=None,
        grpo_k=4,
        use_loo=False,
        kl_reward_penalty=None,
        kl_coeff=0.01,
        non_eos_penalty=True,
        penalty_reward_value=-3,
        **_ignored,
    ):
        # models
        self.ref_model = ref_model
        self.policy_model = policy_model

        self.policy_micro_bs = policy_micro_bs
        self.ref_micro_bs = ref_micro_bs
        # rewards
        self.clip_reward_min = clip_reward_min
        self.clip_reward_max = clip_reward_max

        self.norm_adv = norm_adv
        # only used for async reward model.infer_get() in _get_kl_rewards
        self.env = env
        self.grpo_k = grpo_k
        self.use_loo = use_loo
        assert kl_reward_penalty in (None, "sum", "mean", "per_token")
        self.kl_reward_penalty = kl_reward_penalty
        self.kl_coeff = kl_coeff
        self.non_eos_penalty = non_eos_penalty
        self.penalty_reward_value = penalty_reward_value

    def process(self, trajectories: PolicyOutput):
        action_mask = trajectories["action_mask"]
        num_actions = action_mask.size(1)
        (
            rewards,
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
        trajectories["kl_rewards"] = rewards
        trajectories["policy_logprobs"] = policy_logprobs
        trajectories["ref_logprobs"] = ref_logprobs

        if self.kl_reward_penalty == "sum":
            kl_reward = (-self.kl_coeff * kl_distance).sum(1)
            rewards = rewards + kl_reward
        elif self.kl_reward_penalty == "mean":
            kl_reward = -self.kl_coeff * kl_distance.sum(1) / action_mask.sum(1)
            rewards = rewards + kl_reward

        # vectorized GRPO advantages implementation
        rewards = rewards.reshape(-1, self.grpo_k).T
        if self.use_loo:
            baseline = (rewards.sum(0) - rewards) / (self.grpo_k - 1)
            advantages = rewards - baseline
        else:
            advantages = (rewards - rewards.mean(0)) / (rewards.std(0) + 1e-8)
        advantages = advantages.T.flatten()

        per_token_advantages = torch.ones_like(kl_distance).float()
        per_token_advantages = per_token_advantages * advantages[:, None]

        if self.kl_reward_penalty == "per_token":
            per_token_advantages = per_token_advantages - self.kl_coeff * kl_distance

        trajectories["advantages"] = per_token_advantages  # only for outcome reward
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

        # compute_approx_kl
        log_ratio = policy_logprobs - ref_logprobs
        non_negtive_kl = torch.exp(log_ratio) - log_ratio -1

        kl = log_ratio * action_mask

        reward = clipped_rewards

        entropy = -(policy_logprobs * action_mask).sum(axis=-1) / action_mask.sum(
            axis=-1
        )
        return reward, entropy, kl, policy_logprobs, ref_logprobs
