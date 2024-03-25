import torch
from .base_model_server import BaseModelServer
from ..policy_output import PolicyOutput
from transformers.modeling_outputs import (
    SequenceClassifierOutputWithPast,
)

class CriticModelServer(BaseModelServer):
    # def initialize(self):
    #     pass

    def infer_async(self, inputs, *args, **infer_kwargs):
        print(f"[{self.__class__.__name__}] self.critic_model.forward(str)")
        input_ids: torch.Tensor = self.tokenize_str_input(inputs=inputs)
        model_output: SequenceClassifierOutputWithPast = self.trainer.infer_async(
            input_ids,
            **infer_kwargs,
        )
        return model_output