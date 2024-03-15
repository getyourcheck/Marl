import torch
from typing import Any
from ..policy_output import PolicyOutput, logprobs_from_logits


class ActorLoss(torch.nn.Module):
    """
    Loss function for actor model.
    """

    def __init__(self, cliprange: float = 0.2, loss_type: str = "per_seq"):
        super().__init__()
        self.cliprange = cliprange
        self.loss_type = loss_type
        assert self.loss_type in ["per_token", "per_seq"]

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask, loss_factor):
        # policy gradient loss
        log_ratio = (logprobs - old_logprobs)
        # C1rN09: some tokens are supressed which lead log_probs to be -Inf. This will make ratios `nan`
        is_supressed = torch.logical_and(logprobs.isinf(), old_logprobs.isinf())
        log_ratio = torch.where(is_supressed, 0, log_ratio)
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        if self.loss_type == "per_seq":
            pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        elif self.loss_type == "per_token":
            pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) * loss_factor
        else:
            raise RuntimeError(f"ActorLoss.loss_type must be ['per_seq', 'per_token'], got {self.loss_type}")
        return pg_loss

    def forward(self, logits: torch.Tensor, labels: dict[str, Any]):
        """Forward function of ActorLoss.

        Args:
            logits (Tensor): Forward result of the model. Its shape may be varied.
                For packed forward: (micro_bsz * seqlen, 1), where micro_bsz = 1
                For non packed forward: (micro_bsz, seqlen, 1)

            labels (tuple[dict]): Label values which are split by pipeline
                schedule into pieces. The length of the list is micro_bsz. Each
                element is a dict, representing labels to a batch.

        Note:
            The parameter `labels` seems strange because of pj-colossalai's
            pipeline schedule machanism. Labels are delivered to colosslai.Engine
            in List format, so pipeline schedule split it into micro_bsz pieces,
            and deliver them to loss_fn by `*args`.

        Returns:
            Tensor: Return the final loss
        """
        assert logits.ndim == 2 or logits.ndim == 3
        mask = labels["mask"]  # (micro_bsz, seqlen)
        # print("Loss", labels)
        micro_bsz = labels["input_ids"].shape[0]
        # logits = gather_output(logits)  # TODO: handle tp.
        if logits.ndim == 2:  # (seqlen, vocab_size)
            logits = logits.reshape(micro_bsz, -1, logits.shape[-1])  # (micro_bsz, seqlen, vocab_size)
        assert logits.shape[0] == labels["input_ids"].shape[0]
        input_ids = labels["input_ids"]  # (micro_bsz, seqlen)
        old_logprobs = labels["old_logprobs"] * mask  # (micro_bsz, seqlen)
        advantages = labels["advantages"] * mask  # (micro_bsz, seqlen)
        loss_factor = labels["loss_factor"]
        
        logpy = logprobs_from_logits(logits=logits[:, :-1, :], labels=input_ids[:, 1:], gather=True)
        logpy_shift_right_byone = torch.zeros_like(input_ids, dtype=logits.dtype)
        logpy_shift_right_byone[:, 1:] = logpy
        logprobs = logpy_shift_right_byone * mask

        loss = self.actor_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            loss_factor=loss_factor,
        )
        return loss