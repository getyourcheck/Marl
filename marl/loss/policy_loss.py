from typing import Any

import torch
from loguru import logger

from ..policy_output import logprobs_from_logits


class PretrainLoss(torch.nn.Module):
    """Loss function for flash GPT Language Model."""

    def __init__(self, label_smoothing=0, loss_type="per_seq"):
        super().__init__()

        if label_smoothing is not None and label_smoothing != 0:
            logger.warning(f'Use label_smoothing: {label_smoothing}')
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        assert self.loss_type in ["per_token", "per_seq"]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, loss_factor=None):
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)
        
        if self.loss_type == "per_seq":
            # the output will gather output is set in the model,
            # so use ordinary loss
            self.loss_fn = torch.nn.CrossEntropyLoss(
                reduction='mean', ignore_index=-100, label_smoothing=self.label_smoothing)
            loss = self.loss_fn(shift_logits, shift_labels)
            # There is no need to consider the ignore_index problem here,
            # because the loss calculation will be calculated through the calculation range,  # noqa: E501
            # and -100 must be outside this range,
            # so there is no problem
        elif self.loss_type == "per_token":
            assert loss_factor is not None
            logger.info(f"[loss_factor] pretrain: {loss_factor}")
            # pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) * loss_factor
            self.loss_fn = torch.nn.CrossEntropyLoss(
                reduction='sum', ignore_index=-100, label_smoothing=self.label_smoothing)
            loss = self.loss_fn(shift_logits, shift_labels) * loss_factor

        return loss


class PPOPolicyLoss(torch.nn.Module):
    """Loss function for policy model."""

    def __init__(self, cliprange: float = 0.2, loss_type: str = "per_seq"):
        super().__init__()
        self.cliprange = cliprange
        self.loss_type = loss_type
        assert self.loss_type in ["per_token", "per_seq"]

    def policy_loss_fn(self, logprobs, old_logprobs, advantages, mask, loss_factor=None):
        ratio = (logprobs - old_logprobs).exp()
        pg_loss1 = -ratio * advantages
        pg_loss2 = -ratio.clamp(1 - self.cliprange,
                                1 + self.cliprange) * advantages
        if self.loss_type == "per_seq":
            pg_loss = (torch.max(pg_loss1, pg_loss2) * mask).sum() / (mask.sum() + 1e-8)
        elif self.loss_type == "per_token":
            assert loss_factor is not None
            logger.info(f"[loss_factor] ppo: {loss_factor}")
            pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) * loss_factor
        return pg_loss

    def forward(self, logits: torch.Tensor, labels: dict[str, Any], loss_factor=None):
        assert logits.ndim == 3
        mask = labels['mask']

        assert logits.shape[0] == labels['input_ids'].shape[0]
        input_ids = labels['input_ids']
        old_logprobs = labels['old_logprobs']
        advantages = labels['advantages']

        logpy = logprobs_from_logits(
            logits=logits[:, :-1, :], labels=input_ids[:, 1:], gather=True)
        num_actions = mask.size(1)
        logprobs = logpy[:, -num_actions:]

        loss = self.policy_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            loss_factor=loss_factor)
        return loss


class RLOOLoss(torch.nn.Module):
    """Loss function for policy model."""

    def __init__(self, cliprange: float = 0.2, loss_type: str = "per_seq"):
        super().__init__()
        self.cliprange = cliprange
        self.loss_type = loss_type
        assert self.loss_type == "per_seq", "RLOO Loss only supports per_seq loss"

    def policy_loss_fn(self, logprobs, old_logprobs, advantages, mask, loss_factor=None):
        INVALID_LOGPROB = 1.0
        logprobs = torch.masked_fill(logprobs, mask==0, INVALID_LOGPROB)
        old_logprobs = torch.masked_fill(old_logprobs, mask==0, INVALID_LOGPROB)

        logprobs = logprobs.sum(1)
        old_logprobs = old_logprobs.sum(1)
        logprobs_diff = logprobs - old_logprobs
        ratio = torch.exp(logprobs_diff)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        pg_loss = pg_loss_max.mean()
        return pg_loss

    def forward(self, logits: torch.Tensor, labels: dict[str, Any], loss_factor=None):
        assert logits.ndim == 3
        mask = labels['mask']

        assert logits.shape[0] == labels['input_ids'].shape[0]
        input_ids = labels['input_ids']
        old_logprobs = labels['old_logprobs']
        advantages = labels['rloo_advantages']

        logpy = logprobs_from_logits(
            logits=logits[:, :-1, :], labels=input_ids[:, 1:], gather=True)
        num_actions = mask.size(1)
        logprobs = logpy[:, -num_actions:]

        loss = self.policy_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            loss_factor=loss_factor)
        return loss


class GRPOPolicyLoss(torch.nn.Module):
    """Loss function for policy model."""

    def __init__(self, 
                 cliprange: float = 0.2, 
                 kl_coef=0.01, 
                 kl_penalty_loss=True,
                 sampling_group_size=4, 
                 loss_type: str = "per_seq"):
        super().__init__()
        self.cliprange = cliprange
        self.kl_coef = kl_coef
        self.kl_penalty_loss = kl_penalty_loss
        self.sampling_group_size = sampling_group_size
        self.loss_type = loss_type
        assert self.loss_type in ["per_token", "per_seq"]

    def policy_loss_fn(self, logprobs, old_logprobs, ref_logprobs, advantages, mask, loss_factor=None):
        kl_estimate = ref_logprobs - logprobs
        approx_kl = torch.exp(kl_estimate) - kl_estimate -1
        kl_coef_approx_kl = self.kl_coef * approx_kl
        ratio = (logprobs - old_logprobs).exp()
        pg_loss1 = -ratio * advantages
        pg_loss2 = -ratio.clamp(1 - self.cliprange,
                                1 + self.cliprange) * advantages
        if self.kl_penalty_loss:
            pg_loss_max = torch.max(pg_loss1, pg_loss2) + kl_coef_approx_kl / self.sampling_group_size
        else:
            pg_loss_max = torch.max(pg_loss1, pg_loss2)
        if self.loss_type == "per_seq":
            pg_loss = (pg_loss_max * mask).sum() / (mask.sum() + 1e-8)
        elif self.loss_type == "per_token":
            assert loss_factor is not None
            pg_loss = torch.sum(pg_loss_max * mask) * loss_factor
        return pg_loss

    def forward(self, logits: torch.Tensor, labels: dict[str, Any], loss_factor=None):
        assert logits.ndim == 3
        mask = labels['mask']

        assert logits.shape[0] == labels['input_ids'].shape[0]
        input_ids = labels['input_ids']
        old_logprobs = labels['old_logprobs']
        ref_logprobs = labels['ref_logprobs']
        advantages = labels['advantages']

        logpy = logprobs_from_logits(
            logits=logits[:, :-1, :], labels=input_ids[:, 1:], gather=True)
        num_actions = mask.size(1)
        logprobs = logpy[:, -num_actions:]

        loss = self.policy_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            mask=mask,
            loss_factor=loss_factor)
        return loss
