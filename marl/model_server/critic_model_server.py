import torch
from .base_model_server import BaseModelServer
from ..policy_output import PolicyOutput
from transformers.modeling_outputs import (
    SequenceClassifierOutputWithPast,
)
import marl.utils as marl_utils

class CriticModelServer(BaseModelServer):
    # def initialize(self):
    #     pass

    def infer_async(self, inputs, *args, **infer_kwargs):
        print(f"[{self.__class__.__name__}] self.critic_model.forward(str)")
        input_ids, attention_mask = marl_utils.encode(inputs,self.tokenizer)
        model_output: SequenceClassifierOutputWithPast = self.trainer.infer_async(
            inputs = input_ids,
            attention_mask = attention_mask,
            *args
            **infer_kwargs,
        )
        return model_output