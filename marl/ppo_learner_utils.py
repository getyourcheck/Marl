import torch

def calc_kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty="full") -> torch.FloatTensor:
    # Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1116
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty == "full":
        # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
        return torch.nn.functional.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError

def get_advantages_and_returns(values, rewards, last_values, gamma, lam):
    # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size()[-1]
    for t in reversed(range(0, length)):
        nextvalues = values[:, t + 1] if t < length - 1 else last_values
        # Since old_rewards and old_values are masked with action_mask, i.e. they have
        # 0's at pad tokens, delta will be 0 if current t is at a pad token, so will
        # lastgaelam
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages.detach(), returns

def compute_rewards(prompts, log_probs, ref_log_probs, reward_score, action_mask, kl_ctl):
    kl_divergence_estimate = -kl_ctl * (log_probs - ref_log_probs)
    # C1rN09: some tokens are supressed which lead log_probs to be -Inf. This will make ratios `nan`
    is_inf = torch.logical_and(log_probs.isinf(), ref_log_probs.isinf())
    kl_divergence_estimate = torch.where(is_inf, 0, kl_divergence_estimate)
    # C1rN09: <pad> tokens should have 0 reward
    rewards = kl_divergence_estimate * action_mask
    ends = action_mask.sum(axis=1)
    batch_size = log_probs.shape[0]
    for j in range(batch_size):
        rewards[j, ends[j]-1] += reward_score[j]

    return rewards