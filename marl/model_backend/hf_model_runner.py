import torch

from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler

from typing import Optional, Union
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    get_scheduler as transformers_get_scheduler,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..policy_output import PolicyOutput
from ..utils import set_seed
from ..tokenizer.tokenizer_utils import get_tokenizer

DEFAULT_MAX_NEW_TOKENS = 64

"""
HfModelRunner can be individually called by other process
HfModelRunnerRayActor is called by ModelServer with .remote()
"""


class HfModelRunner:
    """
    ModelTrainer is capable of training, inference, and generation
    """

    def __init__(self, model_config):
        self.model_config: dict = model_config

    def initialize(self):
        # 1. Model
        model_path = self.model_config.get("model_path")
        torch_dtype = self.model_config.get("torch_dtype", "auto")
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.vocab_size = self.model.config.vocab_size

        # 2. Tokenizer
        tokenizer_path = self.model_config.get("tokenizer_path", model_path)
        self.tokenizer = get_tokenizer(tokenizer_path, trust_remote_code=True)

        # 3. Trainer
        parallel: dict = self.model_config["parallel"]
        assert parallel["tensor"]["size"] == 1  # TODO: support TP
        assert parallel["pipeline"]["size"] == 1  # TODO: support PP
        self.step = 0
        self.accelerator = Accelerator()

        train_kwargs = self.model_config.get("train_kwargs")
        if train_kwargs is None:  # requires no training
            self.device = self.accelerator.device
            print(f"[{self.__class__.__name__}] __init__() done without train_kwargs.")
            return

        optimizer_type = train_kwargs.get("optimizer", torch.optim.AdamW)
        learning_rate = train_kwargs.get("lr", 1e-5)
        self.optimizer: torch.optim.Optimizer = optimizer_type(
            params=self.model.parameters(),
            lr=learning_rate,
        )

        lr_scheduler_type = train_kwargs.get("lr_scheduler", "linear")
        lr_scheduler_kwargs = train_kwargs.get(
            "lr_scheduler_kwargs", {"num_warmup_steps": 5, "num_training_steps": 10}
        )
        self.lr_scheduler: _LRScheduler = transformers_get_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            **lr_scheduler_kwargs,
        )
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

        # Others
        self.device = self.accelerator.device
        set_seed(self.model_config.get("seed"))

        print(f"[{self.__class__.__name__}] __init__() done with train_kwargs.")

    # Training
    def compute_loss(
        self,
        input_ids: Union[list[torch.Tensor], torch.Tensor],
        labels: Optional[Union[list[torch.Tensor], torch.Tensor]] = None,
        attention_mask: Optional[Union[list[torch.Tensor], torch.Tensor]] = None,
        criterion: Optional[Union[list[_Loss], _Loss]] = None,
        loss_weights: Optional[list[float]] = None,
        **_ignored,
    ) -> torch.Tensor:
        """
        criterion: _Loss class, e.g., torch.nn.CrossEntropyLoss()
        """
        if type(input_ids) == list:  # multiple inputs grouped to compute loss
            # https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
            print(
                f"[{self.__class__.__name__}] self.compute_loss() for multiple input batches"
            )
            assert (
                len(input_ids)
                == len(labels)
                == len(criterion)
                == len(attention_mask)
                == len(loss_weights)
            ), f"{len(input_ids)} {len(labels)} {len(criterion)} {len(attention_mask)} {len(loss_weights)} must equal"
            loss_cache = [0 for _ in range(len(input_ids))]
            loss_weights = [
                x / float(len(loss_weights)) for x in loss_weights
            ]  # normalized to 1

            for i in range(len(input_ids)):
                loss = self.compute_loss_one(
                    input_ids[i], labels[i], attention_mask[i], criterion[i]
                )
                loss_cache[i] = loss * loss_weights[i]
            loss_sum = sum(loss_cache)
            return loss_sum
        else:
            print(
                f"[{self.__class__.__name__}] self.compute_loss() for single input batch"
            )
            loss = self.compute_loss_one(input_ids, labels, attention_mask, criterion)
            return loss

    def compute_loss_one(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        criterion: Optional[_Loss] = torch.nn.CrossEntropyLoss,
        loss_weight: Optional[float] = None,
        **_ignored,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        labels = input_ids.clone() if labels is None else labels.to(self.device)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        self.model.train()

        if criterion is None:
            loss = self.model(**batch, use_cache=False).loss
        else:
            # Adopted from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1200
            logits: torch.Tensor = self.model(**batch, use_cache=False).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = criterion()  # default: torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)  # Enable model parallel
            loss = loss_fct(shift_logits, shift_labels)

        if loss_weight is not None:
            loss *= loss_weight
        return loss

    def backward_step(self, loss: torch.Tensor, step_interval=1):
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
        attention_mask: Optional[Union[list[torch.Tensor], torch.Tensor]] = None,
        criterion: Optional[Union[list[_Loss], _Loss]] = None,
        loss_weights: Optional[Union[list[float], float]] = None,
        step_interval: int = 1,
        **_ignored,
    ):
        print(f"[{self.__class__.__name__}] self.train()")
        loss = self.compute_loss(
            input_ids, labels, attention_mask, criterion, loss_weights
        )
        self.backward_step(loss, step_interval)
        return loss

    # Inference
    # TODO: decouple infer() to infer() and generate()
    def infer(
        self,
        input_ids: Union[torch.Tensor, str, list[str]],
        batch_size: Optional[int] = -1,
        tokenizer=None,
        chat_template=None,
        step: Optional[int] = -1,
        generate_kwargs: Optional[dict] = {},
        **_ignored,
    ) -> PolicyOutput:
        input_ids = self.tokenize_str_input(inputs=input_ids)
        if step == 1:
            with torch.no_grad():
                output_inf: CausalLMOutputWithPast = self.model(input_ids)
                return PolicyOutput(logits=output_inf.logits)
        else:
            return self.generate(input_ids, step, tokenizer, **generate_kwargs)

    @torch.inference_mode()
    def generate(
        self, input_ids, step=-1, tokenizer=None, **generate_kwargs
    ) -> PolicyOutput:
        model = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.accelerator.unwrap_model(self.model)
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS if step <= 0 else step
        generate_kwargs.setdefault("max_new_tokens", max_new_tokens)
        print(f"[{self.__class__.__name__}] generate_kwargs: {generate_kwargs}")

        output_ids = model.generate(
            input_ids.to(model.device),
            use_cache=True,
            **generate_kwargs,
        )

        output_str = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return PolicyOutput(output_ids=output_ids, output_str=output_str)

    def get_model(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.accelerator.unwrap_model(self.model)
        return self.model

    def set_seed(self, seed=None):
        set_seed(seed)

    def tokenize_str_input(self, inputs: Union[list[str], str]) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs
        elif isinstance(inputs, str):
            input_strs = list(inputs)
        elif isinstance(inputs, list):
            topping = inputs[0]
            if isinstance(topping, torch.Tensor):
                print(f"[{self.__class__.__name__}] Cat list[torch.Tensor]: {inputs}")
                return torch.cat(inputs, dim=0)
            if not isinstance(topping, str):
                raise TypeError(f"Unsupported type: type({topping}) inputs({inputs})")
            input_strs = inputs

        print(f"[{self.__class__.__name__}] encode string input into input_ids ...")
        output = self.tokenizer(input_strs, return_tensors="pt")
        return output.input_ids


import ray
from ray.util.placement_group import (
    remove_placement_group,
    placement_group as create_placement_group,
)
from .ray_utils import create_ray_actors
from .ray_actor_mixin import RayActorMixin
from .ray_utils import DEFAULT_NUM_CPUS, DEFAULT_NUM_GPUS
from ..config_utils import get_gpu_requirement
from ..policy_output import concat_policy_outputs


class HfModelRunnerRayActor(HfModelRunner, RayActorMixin):
    """
    A ray.remote Actor Class initialized by HfModelRunnerRayActorGroup,
    extending HfModelRunner with ray related method via RayActorMixin
    """

    pass


class HfModelRunnerRayActorGroup:
    """
    HfModelRunnerRayActorGroup manages a list of HfModelRunnerRayActor
    create ray actors
    """

    def __init__(self, name: str, config: dict):
        self.released = True
        num_gpus = get_gpu_requirement(config)
        bundles = [
            {"CPU": DEFAULT_NUM_CPUS, "GPU": DEFAULT_NUM_GPUS} for _ in range(num_gpus)
        ]
        self.placement_group = create_placement_group(bundles)
        self.ray_actors: list[HfModelRunnerRayActor] = create_ray_actors(
            name_prefix=name,
            config=config,
            placement_group=self.placement_group,
            trainer_class=ray.remote(
                num_cpus=DEFAULT_NUM_CPUS, num_gpus=DEFAULT_NUM_GPUS
            )(HfModelRunnerRayActor),
        )
        self.released = False

        master_ip = ray.get(self.ray_actors[0].get_metadata.remote()).node_ip
        master_port = ray.get(self.ray_actors[0].get_free_port.remote())
        ray.get(
            [
                actor.inject_distribute_env.remote(
                    master_ip=master_ip,
                    master_port=master_port,
                    rank_id=rank,
                    world_size=len(self.ray_actors),
                )
                for rank, actor in enumerate(self.ray_actors)
            ]
        )
        ray.get([actor.initialize.remote() for actor in self.ray_actors])

    # Training
    def train_async(self, *args, **kwargs):
        return [actor.train.remote(*args, **kwargs) for actor in self.ray_actors]

    def train_get(self, object_refs, timeout=None):
        losses = ray.get(object_refs, timeout=timeout)
        return sum(losses) / len(losses)

    def train(self, *args, **kwargs):
        object_refs = self.train_async(*args, **kwargs)
        return self.train_get(object_refs)

    # Inference
    def infer_async(self, *args, **kwargs):
        return [actor.infer.remote(*args, **kwargs) for actor in self.ray_actors]

    def infer_get(self, object_refs, timeout=None):
        outputs = ray.get(object_refs, timeout=timeout)
        return concat_policy_outputs(outputs)

    def infer(self, *args, **kwargs):
        object_refs = self.infer_async(*args, **kwargs)
        return self.infer_get(object_refs)

    # Others
    def get_model(self):
        return self.ray_actors[0].get_model.remote()

    def set_seed(self, seed=None):
        ray.get([actor.set_seed.remote(seed) for actor in self.ray_actors])

    def release_resources(self):
        """
        release ray resources.

        """
        if self.released:
            return
        for actor in self.ray_actors:
            try:
                ray.kill(actor=actor, no_restart=True)
            except BaseException as exp:
                print(f"failed to kill ray actor {actor}. {exp}")
        remove_placement_group(self.placement_group)
        self.released = True
