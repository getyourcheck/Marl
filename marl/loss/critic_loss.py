from typing import Any

import torch
from loguru import logger


class CriticLoss(torch.nn.Module):
    """Loss function for critic model."""

    def __init__(self, cliprange_value: float = 0.5, loss_type: str = "per_seq"):
        super().__init__()
        self.cliprange_value = cliprange_value
        self.loss_type = loss_type
        assert self.loss_type in ["per_token", "per_seq"]

    def critic_loss_fn(self, values, old_values, returns, mask, loss_factor=None):
        values_clipped = old_values + (values - old_values).clamp(
            -self.cliprange_value, self.cliprange_value)
        vf_loss1 = (values_clipped - returns)**2
        vf_loss2 = (values - returns)**2
        if self.loss_type == "per_seq":
            vf_loss = (torch.max(vf_loss1, vf_loss2) * mask).sum() / mask.sum()
        elif self.loss_type == "per_token":
            assert loss_factor is not None
            logger.info(f"[loss_factor] value: {loss_factor}")
            vf_loss = torch.sum(torch.max(vf_loss1, vf_loss2) * mask * loss_factor)
        return 0.5 * vf_loss

    def forward(self, values: torch.Tensor, labels: dict[str, Any], loss_factor=None):
        assert values.ndim == 2
        mask = labels['mask']
        num_actions = mask.size(1)
        values = values[:, -num_actions:]

        old_values = labels['old_values']
        returns = labels['returns']
        loss = self.critic_loss_fn(
            values=values, old_values=old_values, returns=returns, mask=mask, loss_factor=loss_factor)
        return loss
