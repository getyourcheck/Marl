from ..policy_output import PolicyOutput
import numpy as np
import torch
from copy import deepcopy


class BaseRepeater(object):
    def __init__(self, reward_scale: bool = False,
                        fine_grained_rm: bool = False,
                        value_ema: bool = False,
                        **kwargs):
        
        self.reward_scale = reward_scale
        self.fine_grained_rm = fine_grained_rm
        self.value_ema = value_ema
        self.kl_coeff = 0.02
        self.gamma = 1.0
        self.gae_lambda = 0.95
        self.device = None

    # TODO
    def process(self, trajectories: PolicyOutput, models):
        assert "policy" in models.keys(), "[Repeater] make sure models have ... "
        # models_dict = {"policy": actor_model, "rm_model": reward_model, "value_model": critic_model, "sft_model": actor_model}
        # self.device = models['value_model'].device

        kl_rewards, policy_logprobs, sft_logprobs = self._get_kl_rewards(trajectories, models['policy'], models['sft_model'])
        trajectories["kl_rewards"] = kl_rewards
        trajectories["policy_logprobs"] = policy_logprobs
        trajectories["sft_logprobs"] = sft_logprobs

        values = self._get_values(trajectories, models['value_model'])

        # rewards = self._new_rewards(kl_rewards, values)
        adv, returns = self._get_advantages_and_returns(values[:, :-1].cpu(), kl_rewards.cpu(), last_values=None)
        
        trajectories["values"] = values
        trajectories["advs"] = adv
        trajectories["returns"] = values # returns # TODO

        # fake 
        _question_mask = np.ones((trajectories.values.shape))
        trajectories["question_mask"] = torch.from_numpy(np.array(_question_mask))
        trajectories["answer_mask"] = deepcopy(trajectories["question_mask"][:, :-1])
        # print(values.shape, adv.shape, returns.shape) 
        # torch.Size([7, 1145]) torch.Size([7, 1144]) torch.Size([7, 1144])
        return trajectories

    def _get_kl_rewards(self, trajectories: PolicyOutput, policy_model, sft_model):
        # print(trajectories.output_ids.shape) # torch.Size([7, 1145])
        policy_output = policy_model.infer(trajectories.output_ids, output_logprobs=True)
        sft_output = sft_model.infer(trajectories.output_ids, output_logprobs=True)

        kl_div = policy_output.logprobs - sft_output.logprobs
        # print(policy_output.logprobs.shape, sft_output.logprobs.shape) # torch.Size([7, 1144]) torch.Size([7, 1144])
        kl_rewards = - 1.0 * self.kl_coeff * kl_div
        # print(kl_rewards.shape) # torch.Size([7, 1144])
        return kl_rewards, policy_output.logprobs, sft_output.logprobs

    def _get_values(self, trajectories: PolicyOutput, value_model):
        value_output = value_model.infer(trajectories.output_ids)
        values = value_output.logits
        # print(values.shape) # torch.Size([7, 1145])
        return values

    def _get_advantages_and_returns(self, values, rewards, last_values):
        if last_values is None:
            last_values = torch.tensor([0] * rewards.shape[0])
            last_values = last_values
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        # print(values.shape, rewards.shape, ) # (7, 1144) (7, 1144)
        length = rewards.shape[-1]
        for t in reversed(range(0, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else last_values
            # Since old_rewards and old_values are masked with action_mask, i.e. they have
            # 0's at pad tokens, delta will be 0 if current t is at a pad token, so will
            # lastgaelam
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.gae_lambda * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    # TODO
    def _get_fine_grained_values(self, trajectories):
        pass
