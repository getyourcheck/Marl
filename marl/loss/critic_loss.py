import torch
from typing import Any


class CriticLoss(torch.nn.Module):
    """
    Loss function for critic model.
    """

    def __init__(self, cliprange_value: float = 100, loss_type: str = "per_seq"):
        super().__init__()
        self.cliprange_value = cliprange_value
        self.loss_type = loss_type
        assert self.loss_type in ["per_token", "per_seq"]

    def critic_loss_fn(self, values, old_values, returns, mask, loss_factor):
        # value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        if self.loss_type == "per_seq":
            vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask / mask.sum())
        elif self.loss_type == "per_token":
            vf_loss = 0.5 * torch.sum(
                torch.max(vf_loss1, vf_loss2) * mask * loss_factor
            )
        else:
            raise RuntimeError(
                f"CriticLoss.loss_type must be ['per_seq', 'per_token'], got {self.loss_type}"
            )
        return vf_loss

    def forward(self, values: torch.Tensor, *labels: tuple[dict[str, Any]]):
        """Forward function of CriticLoss.

        Args:
            values (Tensor): Forward result of the model. Its shape may be varied.
                For packed forward: (micro_bsz * seqlen, 1), where micro_bsz = 1
                For non packed forward: (micro_bsz, seqlen, 1)

            labels (Tuple[dict]): Label values which are split by pipeline
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
        assert values.ndim == 2 or values.ndim == 3
        values = values[:, :-1]  # (micro_bsz, seqlen - 1)
        old_values = labels[0]["old_values"][:, :-1]  # (micro_bsz, seqlen - 1)
        returns = labels[0]["returns"][:, :-1]  # (micro_bsz, seqlen - promptlen)
        mask = labels[0]["mask"][:, :-1]  # (micro_bsz, seqlen - 1)
        loss_factor = labels[0]["loss_factor"]
        print(values.shape, old_values.shape, returns.shape, mask.shape)
        loss = self.critic_loss_fn(
            values=values,
            old_values=old_values,
            returns=returns,
            mask=mask,
            loss_factor=loss_factor,
        )
        return loss
