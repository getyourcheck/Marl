import ray
import torch

from typing import Optional, Union
from accelerate import Accelerator

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler as transformers_get_scheduler,
)

from ..policy_output import PolicyOutput

DEFAULT_MAX_NEW_TOKENS=64

"""
HfModelTrainer can be individually called by other process
HfModelTrainerRayActor is called by ModelServer with .remote()
"""

class HfModelTrainer():
    """
    ModelTrainer is capable of training, inference, and generation
    """
    def __init__(self, model_config):
        # 1. Model
        model_path = model_config.get("model_path")
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device_map="auto",
            trust_remote_code=True,
        )
        self.device = self.model.device

        # 2. Tokenizer
        tokenizer_path = model_config.get("tokenizer_path", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            padding_side='left',
        )

        # 3. Trainer
        parallel: dict = model_config["parallel"]
        assert parallel['zero1']['size'] == 1
        assert parallel['tensor']['size'] == 1
        assert parallel['pipeline']['size'] == 1
        # self.trainer = Accelerator()  # TODO: multi-GPU training

        train_kwargs = model_config.get("train_kwargs")
        if train_kwargs is None:  # requires no training
            print(f"[{self.__class__.__name__}] __init__() done without train_kwargs.")
            return

        optimizer_type = train_kwargs.get("optimizer", torch.optim.AdamW)
        learning_rate = train_kwargs.get("lr", 1e-5)
        self.optimizer = optimizer_type(
            params=self.model.parameters(),
            lr=learning_rate,
        )

        lr_scheduler_type = train_kwargs.get("lr_scheduler", "linear")
        lr_scheduler_kwargs = train_kwargs.get("lr_scheduler_kwargs", {"num_warmup_steps": 5, "num_training_steps": 10})
        self.lr_scheduler = transformers_get_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            **lr_scheduler_kwargs,
        )

        print(f"[{self.__class__.__name__}] __init__() done with train_kwargs.")

    # TODO: change train() to 2-step training: compute_loss(), optimizer_step()
    def train(
        self,
        input_ids: Union[list[torch.Tensor], torch.Tensor],
        labels: Optional[Union[list[torch.Tensor], torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_fn = None,
        **_ignored,
    ):
        print(f"[{self.__class__.__name__}] self.train()")
        # TODO: input_ids as a list of torch.Tensor
        # reward_loss = actor_model(inputs_ids, reward_model_labels)
        # pretrained_loss = actor_model(pretrained_inputs_ids, pretrained_labels)
        # loss = reward_loss + 0.1 * pretrained_loss

        input_ids = input_ids.to(self.device)
        labels = input_ids.clone() if labels is None else labels.to(self.device)
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        self.model.train()
        loss = self.model(**batch, use_cache=False).loss
        loss.backward()  # TODO: self.accelerator.backward(loss)
        self.optimizer.step()
        return loss

    # TODO: decouple infer() to infer() and generate()
    def infer(
        self,
        input_ids: Union[torch.Tensor, list[torch.Tensor]],
        batch_size: Optional[int] = -1,
        tokenizer = None,
        chat_template = None,
        step: Optional[int] = -1,
        generate_kwargs: Optional[dict] = {},
        **_ignored,
    ) -> PolicyOutput:
        if step == 1:
            with torch.no_grad():
                output_inf = self.model(input_ids) # outputs: CausalLMOutputWithPast with `loss`, `past_key_values`
                return PolicyOutput(logits=output_inf.logits)
        else:
            return self.generate(input_ids, step, tokenizer, **generate_kwargs)

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        step=-1,
        tokenizer=None,
        **generate_kwargs
    ) -> PolicyOutput:
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS if step <= 0 else step
        generate_kwargs.setdefault("max_new_tokens", max_new_tokens)
        print(f"[{self.__class__.__name__}] generate_kwargs: {generate_kwargs}")

        output_ids = self.model.generate(
            input_ids.to(self.device),
            use_cache=True,
            **generate_kwargs,
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
class HfModelTrainerRayActor(HfModelTrainer):
    pass
