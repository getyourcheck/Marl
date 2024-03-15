# Adopted from: https://github.com/huggingface/transformers/blob/HEAD/src/transformers/generation/utils.py
from typing import Optional
import torch
from dataclasses import dataclass
from transformers.utils.generic import ModelOutput


@dataclass
class PolicyOutput(ModelOutput):
    output_ids: Optional[torch.Tensor] = None
    output_str: Optional[list[str]] = None
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    logits_entropy: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None
    top_logprobs: Optional[torch.Tensor] = None
    question_mask: Optional[torch.Tensor] = None
    answer_mask: Optional[torch.Tensor] = None

    def __eq__(self, other: ModelOutput):
        if len(self.keys()) != len(other.keys()):
            return False
        for k, v in self.items():
            if k not in other:
                return False
            vother = other[k]

            if isinstance(v, torch.Tensor):
                if not torch.equal(v, vother):
                    return False
            elif isinstance(v, tuple):  # tuple(torch.Tensor)
                for i, j in zip(v, vother):
                    if isinstance(i, torch.Tensor):
                        if not torch.equal(i, j):
                            return False
                    else:
                        if i != j:
                            return False
            else:
                if v != vother:
                    return False
        return True
    
    def to(self, device):
        for _, v in self.items():
            if isinstance(v, torch.Tensor):
                v.to(device)

def concat_policy_outputs(inputs: list[PolicyOutput]) -> PolicyOutput:
    if inputs == None or len(inputs) == 0:
        return PolicyOutput(None)
    elif len(inputs) == 1:
        return inputs[0]

    concated = PolicyOutput()
    for key, value in inputs[0].items():
        if value == None:
            continue
        elif isinstance(value, torch.Tensor):
            concated[key] = torch.cat([po[key] for po in inputs], dim=0)
        elif isinstance(value, list):
            concated[key] = []
            for po in inputs:
                concated[key].extend(po[key])
        else:
            raise TypeError(f"value: {value} with unsupported type: {type(value)}.")
    return concated


def logprobs_from_logits(
    logits: torch.Tensor, labels: torch.Tensor = None, gather: bool = True
) -> torch.Tensor:
    r"""
    Adapted from: https://github.com/huggingface/trl/blob/main/trl/core.py#L95

    Example:

    ```python
    >>> logits, _, values = model(**input_kwargs)
    >>> input_ids = input_kwargs["input_ids"]
    >>> logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
    ```"""

    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    if not gather or labels is None:
        return logp
    logpy = torch.gather(logp, -1, labels.unsqueeze(2)).squeeze(-1)
    return logpy
