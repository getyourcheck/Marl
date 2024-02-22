import ray
import torch

from typing import Optional, List, Union
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    get_linear_schedule_with_warmup,
)

from .policy_output import PolicyOutput

DEFAULT_MAX_NEW_TOKENS=64

"""
HfModelRunner can be individually called by other process
HfModelRunnerRayActor is called by ModelServer
"""

class HfModelRunner():
    def __init__(self, model_config):
        # 1. Model
        model_path = model_config.get("model_path")
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device_map="auto",
            trust_remote_code=True
        )
        self.device = self.model.device

        # 2. Tokenizer
        tokenizer_path = model_config.get("tokenizer_path", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side='left')

        # 3. Trainer
        train_kwargs = model_config.get("train_kwargs", {})
        optimizer_type = train_kwargs.get("optimizer", torch.optim.AdamW)
        learning_rate = train_kwargs.get("lr", 1e-5)
        self.optimizer = optimizer_type(params=self.model.parameters(), lr=learning_rate)
        lr_scheduler_type = train_kwargs.get("lr_scheduler", "linear")
        lr_scheduler_kwargs = train_kwargs.get("lr_scheduler_kwargs", {"num_warmup_steps": 5, "num_training_steps": 10})
        self.lr_scheduler = get_scheduler(lr_scheduler_type, optimizer=self.optimizer, **lr_scheduler_kwargs)
        self.trainer = Accelerator()

        # 4. Inferer
        self.inferer = self.model  # outputs = model(input_ids)
        print(f"[{self.__class__.__name__}] __init__() done")

    def train(
        self,
        input_ids: Union[List[torch.Tensor], torch.Tensor],
        labels: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **train_kwargs,
    ):
        print(f"[{self.__class__.__name__}] self.trainer.train({train_kwargs})")
        # TODO
        # reward_loss = actor_model(inputs_ids, reward_model_labels)
        # pretrained_loss = actor_model(pretrained_inputs_ids, pretrained_labels)
        # loss = reward_loss + 0.1 * pretrained_loss

        labels = input_ids.clone() if labels is None else labels
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        self.model.train()
        loss = self.model(**batch, use_cache=False).loss
        self.accelerator.backward(loss)  # i.e., loss.backward()
        self.optimizer.step()
        success = True
        return success, loss

    def infer(
        self,
        input_ids: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = -1,
        tokenizer = None,
        chat_template = None,
        step: Optional[int] = -1,
        **infer_kwargs,
    ) -> PolicyOutput:
        if step == 1:
            output_inf = self.inferer(input_ids) # CausalLMOutputWithPast with `loss`, `past_key_values`
            return PolicyOutput(logits=output_inf.logits)
        else:
            return self.generate(input_ids, step, tokenizer, **infer_kwargs)

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        step=-1,
        tokenizer=None,
        **infer_kwargs
    ) -> PolicyOutput:
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS if step <= 0 else step
        infer_kwargs.setdefault("max_new_tokens", max_new_tokens)
        # infer_kwargs.setdefault("return_dict_in_generate", True)
        # infer_kwargs.setdefault("output_scores", True)
        # infer_kwargs.setdefault("output_hidden_states", True)
        # infer_kwargs.setdefault("output_attentions", True)
        print(f"[{self.__class__.__name__}] infer_kwargs: {infer_kwargs}")

        output_ids = self.model.generate(
            input_ids.to(self.device),
            use_cache=True,
            **infer_kwargs,
        )

        output_str = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return PolicyOutput(
            output_ids=output_ids,
            output_str=output_str
        )

    def get_model(self):
        return self.model


@ray.remote(num_cpus=0.1, num_gpus=0.1)
class HfModelRunnerRayActor(HfModelRunner):
    pass
