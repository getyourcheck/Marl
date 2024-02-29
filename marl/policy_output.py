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
    logits_entropy: Optional[torch.Tensor] = None
    log_probs: Optional[torch.Tensor] = None
    top_logprobs: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None

    def __eq__(self, other: ModelOutput):
        if len(self.keys()) != len(other.keys()):
            return False
        for k, v in self.items():
            if k not in other:
                return False
            vother = other[k]
            if type(v) == torch.Tensor:
                eq = torch.equal(v, vother)
            else:
                eq = (v == vother)
            if eq is False:
                return False
        return True


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
