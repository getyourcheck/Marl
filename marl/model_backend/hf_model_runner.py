import torch
import os
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler

from typing import Optional, Union
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    get_scheduler as transformers_get_scheduler,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput

from ..policy_output import PolicyOutput, logprobs_from_logits
from ..utils import set_seed
from ..tokenizer.tokenizer_utils import get_tokenizer
from .models.internlm2_reward import (
    InternLM2ForRewardModel,
    InternLM2ForCriticModel,
)
from ..config_consts import *
from .generate_utils import get_question_answer_mask

DEFAULT_NEW_TOKENS = 64
MAXIMUM_NEW_TOKENS = 1024

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
        # 0. Environment
        envs = self.model_config.get("envs", {})
        for key, value in envs.items():
            os.environ[key] = value

        # 1. Model
        model_path = self.model_config.get("model_path")
        model_type = self.model_config.get("model_type", "").lower()
        torch_dtype = self.model_config.get("torch_dtype", "auto")
        if model_type == MODEL_TYPE_REWARD:
            # TODO: support reward model from other classes
            self.model: PreTrainedModel = InternLM2ForRewardModel.from_pretrained(
                pretrained_model_name_or_path=model_path,
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        elif model_type == MODEL_TYPE_CRITIC:
            self.model: PreTrainedModel = InternLM2ForCriticModel.from_pretrained(
                pretrained_model_name_or_path=model_path,
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        else:
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
        self.tokenizer.pad_token = self.tokenizer.unk_token
        if self.tokenizer.chat_template is None:
            raise NotImplementedError("Make sure tokenizer has chat_template.")

        # 3. Trainer
        parallel: dict = self.model_config["parallel"]
        assert parallel["tensor"]["size"] == 1  # TODO: support TP
        assert parallel["pipeline"]["size"] == 1  # TODO: support PP
        self.step = 0
        if "mode" in parallel["data"] and parallel["data"]["mode"] == "fsdp":
            self.accelerator = Accelerator(fsdp_plugin=FullyShardedDataParallelPlugin())
        else:
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
        labels: Optional[
            Union[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]
        ] = None,
        attention_mask: Optional[Union[list[torch.Tensor], torch.Tensor]] = None,
        criterion: Optional[Union[list[_Loss], _Loss]] = None,
        loss_weights: Optional[list[float]] = None,
        **_ignored,
    ) -> torch.Tensor:
        """
        criterion: _Loss class, e.g., torch.nn.CrossEntropyLoss()
        """
        if isinstance(input_ids, torch.Tensor):
            print(f"[{self.__class__.__name__}] self.compute_loss() for 1 input batch")
            loss = self.compute_loss_one(input_ids, labels, attention_mask, criterion)
            return loss
        elif type(input_ids) == list:  # multiple inputs grouped to compute loss
            # https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
            print(f"[{self.__class__.__name__}] self.compute_loss() for  input batches")
            assert (
                len(input_ids)
                == len(labels)
                == len(criterion)
                == len(attention_mask)
                == len(loss_weights)
            ), f"{len(input_ids)} {len(labels)} {len(criterion)} {len(attention_mask)} {len(loss_weights)} must equal"
            loss_cache = [0 for _ in range(len(input_ids))]
            loss_weights = [x / float(len(loss_weights)) for x in loss_weights]  # to 1

            for i in range(len(input_ids)):
                loss = self.compute_loss_one(
                    input_ids[i], labels[i], attention_mask[i], criterion[i]
                )
                loss_cache[i] = loss * loss_weights[i]
            loss_sum = sum(loss_cache)
            return loss_sum
        else:
            raise NotImplementedError

    def compute_loss_one(
        self,
        input_ids: torch.Tensor,
        labels: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        criterion: Optional[_Loss] = None,
        loss_weight: Optional[float] = None,
        **_ignored,
    ) -> torch.Tensor:

        input_ids = input_ids.to(self.device)
        labels = input_ids.clone() if labels is None else labels
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        self.model.train()

        if criterion is None:
            # OPT. A) Default settings
            assert isinstance(
                labels, torch.Tensor
            ), "Please pass in `criterion` for non-tensor labels"
            batch["labels"] = labels.to(self.device)
            fwd_output = self.model(**batch, use_cache=False)
            loss = fwd_output.loss
        elif isinstance(labels, torch.Tensor):
            # OPT. B) Use preset loss functions, e.g., torch.nn.CrossEntropyLoss()
            # Adopted from: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L1199
            logits: torch.Tensor = self.model(**batch, use_cache=False).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)  # enable model para
            loss_fct = criterion()
            loss = loss_fct(shift_logits, shift_labels)
        elif isinstance(labels, dict):
            # OPT. C) Use customized loss function, see loss/actor_loss.py
            logits: torch.Tensor = self.model(
                **batch, use_cache=False, return_dict=True
            ).logits
            # loss_fct = criterion()
            for k, v in labels.items():
                labels[k] = v.to(self.device)
            loss = criterion(logits, labels)
        else:
            raise ValueError(f"labels of unsupported type: {type(labels)}")

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
        labels: Optional[
            Union[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]
        ] = None,
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
    @torch.no_grad()
    def infer(
        self,
        inputs: Union[torch.Tensor, list[dict], list[list[dict]]],
        batch_size: Optional[int] = -1,
        tokenizer=None,  # reward model may use tokenizer in inference
        attention_mask=None,
        output_logprobs=False,
        output_logits=True,
        output_attentions=False,
        output_hidden_states=False,
        infer_kwargs: Optional[dict] = {},
        **_ignored,
    ) -> PolicyOutput:
        model_type = self.model_config.get("model_type", "").lower()

        # OPT. A) Reward Model
        if model_type == MODEL_TYPE_REWARD:
            self.model: InternLM2ForRewardModel = self.model
            if isinstance(inputs, torch.Tensor):
                input_ids = inputs
                print(f"[{self.__class__.__name__}] self.reward_model.forward(ids)")
                fwd_output: SequenceClassifierOutputWithPast = self.model(
                    input_ids=input_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    attention_mask=attention_mask,
                    **infer_kwargs,
                )
                return PolicyOutput(**fwd_output)
            elif isinstance(inputs, str):
                print(f"[{self.__class__.__name__}] self.reward_model.forward(str)")
                tokenizer = self.tokenizer if tokenizer is None else tokenizer
                input_ids = tokenizer.encode(inputs, return_tensors="pt")
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.tensor(
                            [[self.model.reward_token_id]], dtype=torch.long
                        ).expand(input_ids.shape[0], 1),
                    ],
                    dim=1,
                ).to(self.device)

                fwd_output: SequenceClassifierOutputWithPast = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, **infer_kwargs
                )
                score = fwd_output.logits.cpu().item()
                return PolicyOutput(logits=torch.Tensor([score]))
            elif isinstance(inputs, list):
                if isinstance(inputs[0], dict):
                    print(f"[{self.__class__.__name__}] self.reward_model.score(conv)")
                    score: float = self.model.get_score(
                        conversation=inputs, tokenizer=self.tokenizer, **infer_kwargs
                    )
                    return PolicyOutput(logits=torch.Tensor([score]))
                elif isinstance(inputs[0], list):
                    assert isinstance(inputs[0][0], dict)
                    print(f"[{self.__class__.__name__}] self.reward_model.scores(conv)")
                    scores: list = self.model.get_scores(
                        conversations=inputs, tokenizer=self.tokenizer, **infer_kwargs
                    )
                    return PolicyOutput(logits=torch.Tensor(scores))
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        # OPT. B) Critic Model
        elif model_type == MODEL_TYPE_CRITIC:
            print(f"[{self.__class__.__name__}] self.critic_model.forward(str)")
            input_ids: torch.Tensor = self.tokenize_str_input(inputs=inputs)
            input_ids.to(self.device)
            model_output: SequenceClassifierOutputWithPast = self.model(
                input_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                attention_mask=attention_mask,
                return_dict=True,
                **infer_kwargs,
            )
            output = PolicyOutput()
            if output_logits:
                output["logits"] = model_output["logits"]
            if output_attentions:
                output["attentions"] = model_output["attentions"]
            if output_hidden_states:
                output["hidden_states"] = model_output["hidden_states"]
            return output

        # OPT. C) model_type == "actor", "reference", ...
        print(f"[{self.__class__.__name__}] self.infer() kwargs: {infer_kwargs}")
        assert isinstance(inputs, torch.Tensor)
        input_ids: torch.Tensor = inputs.to(self.device)
        model_output: CausalLMOutputWithPast = self.model(
            input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            **infer_kwargs,
        )
        output = PolicyOutput()
        if output_logits:
            output["logits"] = model_output["logits"]
        if output_attentions:
            output["attentions"] = model_output["attentions"]
        if output_hidden_states:
            output["hidden_states"] = model_output["hidden_states"]
        if output_logprobs:
            logpy = logprobs_from_logits(
                logits=output["logits"][:, :-1, :], labels=input_ids[:, 1:], gather=True
            )
            logpy_shift_right_byone = torch.zeros_like(input_ids, dtype=output["logits"].dtype)
            logpy_shift_right_byone[:, 1:] = logpy
            output["logprobs"] = logpy_shift_right_byone
        return output

    # Generate
    @torch.no_grad()
    def generate(
        self,
        inputs: Union[torch.Tensor, str, list[str]],
        batch_size: Optional[int] = -1,
        attention_mask=None,
        step=-1,
        output_str=True,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        chat_template=None,
        generate_kwargs: Optional[dict] = {},
        **_ignored,
    ) -> PolicyOutput:
        print(f"[{self.__class__.__name__}] self.generate() kwargs: {generate_kwargs}")
        input_ids: torch.Tensor = self.tokenize_str_input(
            inputs=inputs, chat_template=chat_template
        )
        assert isinstance(input_ids, torch.Tensor)

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model

        max_new_tokens = (
            MAXIMUM_NEW_TOKENS
            if "eos_token_id" in generate_kwargs
            else DEFAULT_NEW_TOKENS
        )
        max_new_tokens = step if step > 0 else max_new_tokens

        # TODO: stop if meeting eos_token_id
        model_output: GenerateDecoderOnlyOutput = model.generate(
            input_ids.to(model.device),
            use_cache=True,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_logits=output_logits,  # transformers >= 4.38.2
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **generate_kwargs,
        )

        output_ids = model_output["sequences"]
        output = PolicyOutput(output_ids=output_ids)
        # masks
        output["question_mask"], output["answer_mask"] = get_question_answer_mask(
            input_ids,
            output_ids,
            tokenizer_pad_token_id=self.tokenizer.pad_token_id,
            generate_pad_token_id=generate_kwargs.get("pad_token_id"),
        )
        output["attention_mask"] = output.question_mask + output.answer_mask

        if output_logits:
            output["logits"] = model_output["logits"]  # tuple(torch.Tensor, )
        if output_attentions:
            output["attentions"] = model_output["attentions"]
        if output_hidden_states:
            output["hidden_states"] = model_output["hidden_states"]
        if output_str:  # customized post processing
            output["output_str"] = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output["output_ans_str"] = self.tokenizer.batch_decode(
                output_ids * output.answer_mask,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        output.to("cpu")
        return output

    def get_model(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.accelerator.unwrap_model(self.model)
        return self.model

    def set_seed(self, seed=None):
        set_seed(seed)

    def tokenize_str_input(
        self,
        inputs: Union[list[str], str],
        chat_template: str = None,
    ) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs
        elif isinstance(inputs, str):
            if chat_template != None:
                inputs = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": inputs}],
                    tokenize=False,
                    chat_template=chat_template,
                    add_generation_prompt=True,
                )
            input_strs = inputs
        elif isinstance(inputs, list):
            topping = inputs[0]
            if isinstance(topping, torch.Tensor):
                print(f"[{self.__class__.__name__}] Cat list[torch.Tensor]: {inputs}")
                return torch.cat(inputs, dim=0)
            # if not isinstance(topping, str):
            #     raise TypeError(f"Unsupported type: type({topping}) inputs({inputs})")
            # if chat_template != None:
            #     inputs = [
            #         self.tokenizer.apply_chat_template(
            #             [{"role": "user", "content": input}],
            #             tokenize=False,
            #             chat_template=chat_template,
            #             add_generation_prompt=True,
            #         )
            #         for input in inputs
            #     ]
            if isinstance(topping, list):
                inputs = [self.tokenizer.apply_chat_template(mes, tokenize=False, add_generation_prompt=True, return_tensors="pt") for mes in inputs]
            input_strs = inputs
        print(f"[{self.__class__.__name__}] encode string input into input_ids ...")
        output = self.tokenizer(input_strs, return_tensors="pt", padding=True)
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

    # Generation
    def generate_async(self, *args, **kwargs):
        return [actor.generate.remote(*args, **kwargs) for actor in self.ray_actors]

    def generate_get(self, object_refs, timeout=None):
        outputs = ray.get(object_refs, timeout=timeout)
        return concat_policy_outputs(outputs)

    def generate(self, *args, **kwargs):
        object_refs = self.generate_async(*args, **kwargs)
        return self.generate_get(object_refs)

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
