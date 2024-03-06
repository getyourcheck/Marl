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