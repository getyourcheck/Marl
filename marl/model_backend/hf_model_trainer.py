import ray
import torch

from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler

from typing import Optional, Union
from accelerate import Accelerator

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
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
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
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
        # assert parallel['data']['size'] == 1
        assert parallel['tensor']['size'] == 1
        assert parallel['pipeline']['size'] == 1
        self.step = 0
        self.accelerator = Accelerator()  # TODO: multi-GPU training

        train_kwargs = model_config.get("train_kwargs")
        if train_kwargs is None:  # requires no training
            print(f"[{self.__class__.__name__}] __init__() done without train_kwargs.")
            return

        optimizer_type = train_kwargs.get("optimizer", torch.optim.AdamW)
        learning_rate = train_kwargs.get("lr", 1e-5)
        self.optimizer: torch.optim.Optimizer = optimizer_type(
            params=self.model.parameters(),
            lr=learning_rate,
        )

        lr_scheduler_type = train_kwargs.get("lr_scheduler", "linear")
        lr_scheduler_kwargs = train_kwargs.get("lr_scheduler_kwargs", {"num_warmup_steps": 5, "num_training_steps": 10})
        self.lr_scheduler: _LRScheduler = transformers_get_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            **lr_scheduler_kwargs,
        )

        print(f"[{self.__class__.__name__}] __init__() done with train_kwargs.")

    # TODO: change train() to 2-step training: compute_loss(), optimizer_step()
    def compute_loss(
        self,
        input_ids: Union[list[torch.Tensor], torch.Tensor],
        labels: Optional[Union[list[torch.Tensor], torch.Tensor]] = None,
        attention_mask: Optional[Union[list[torch.Tensor], torch.Tensor]] = None,
        criterion: Optional[Union[list[_Loss], _Loss]] = None,
        loss_weights: Optional[list[float]] = None,
        **_ignored,
    ):
        """
        criterion: _Loss class, e.g., torch.nn.CrossEntropyLoss()
        """
        if type(input_ids) == list:  # multiple inputs grouped to compute loss
            # https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
            print(f"[{self.__class__.__name__}] self.compute_loss() for multiple input batches")
            assert len(input_ids) == len(labels) == len(criterion) == len(attention_mask) == len(loss_weights), \
                f"{len(input_ids)} {len(labels)} {len(criterion)} {len(attention_mask)} {len(loss_weights)} must equal"
            loss_cache = [0 for _ in range(len(input_ids))]
            loss_weights = [x / float(len(loss_weights)) for x in loss_weights]  # normalized to 1

            for i in range(len(input_ids)):
                loss = self.compute_loss_one(input_ids[i], labels[i], attention_mask[i], criterion[i])
                loss_cache[i] = loss * loss_weights[i]
            loss_sum = sum(loss_cache)
            return loss_sum
        else:
            print(f"[{self.__class__.__name__}] self.compute_loss() for single input batch")
            loss = self.compute_loss_one(input_ids, labels, attention_mask, criterion)
            return loss

    def compute_loss_one(
        self,
        input_ids:torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        criterion: Optional[_Loss] = torch.nn.CrossEntropyLoss,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        labels = input_ids.clone() if labels is None else labels.to(self.device)
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        self.model.train()

        if criterion is None:
            loss = self.model(**batch, use_cache=False).loss
            return loss
        else:
            # Adopted from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1200
            logits: torch.Tensor = self.model(**batch, use_cache=False).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = criterion()  # default: torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)  # Enable model parallelism
            loss = loss_fct(shift_logits, shift_labels)
            return loss

    def backward_step(self, loss: torch.Tensor, step_interval = 1):
        print(f"[{self.__class__.__name__}] self.backward_step()")
        loss.backward()  # TODO: self.accelerator.backward(loss)
        self.step += 1
        if self.step % step_interval == 0:
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        return loss

    def train(
        self,
        input_ids: Union[list[torch.Tensor], torch.Tensor],
        labels: Optional[Union[list[torch.Tensor], torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        criterion = None,
        step_interval: int = 1,
        **_ignored,
    ):
        print(f"[{self.__class__.__name__}] self.train()")
        criterion = torch.nn.CrossEntropyLoss
        loss = self.compute_loss(input_ids, labels, attention_mask, criterion)
        self.backward_step(loss, step_interval=step_interval)
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

    @property
    def vocab_size(self):
        return self.model.config.vocab_size

@ray.remote(num_cpus=0.1, num_gpus=0.1)
class HfModelTrainerRayActor(HfModelTrainer):
    pass

# class HfModelTrainerRayActor:
#     def __init__(self):
#         from ray.actor import ActorClass
#         return ActorClass.remote(HfModelTrainer)
