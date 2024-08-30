import os
import json
from typing import Optional
import torch
import torch.nn as nn
from loguru import logger

from transformers.modeling_utils import no_init_weights
from transformers import AutoConfig, AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from .internlm.configuration_internlm2 import InternLM2Config, InternLM2RewardConfig
from .internlm.modeling_internlm2 import (
    InternLM2ForCausalLM,
    InternLM2ForRewardModel,
    InternLM2ForCriticModel,
)


def build_critic_model(
    model_path, head_name="v_head", two_linear=False, extra_kwargs={}, exclude_keys=[]
):
    model_type = "unknown"
    if os.path.exists(os.path.join(model_path, "config.json")):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)
            model_type = config.get("model_type", "unknown")
    if model_type == "internlm2":
        logger.info(f"[Critic model] Loading InternLM2 model from {model_path}")
        logger.warning(
            "[Critic model] Using InternLM2ForRewardModel as critic model, the setting of `head_name` and `two_linear` will be ignored."
        )
        config = InternLM2RewardConfig.from_pretrained(model_path)
        config.update(extra_kwargs)
        with no_init_weights():
            model = InternLM2ForCriticModel(config)
        # check head_name and two_linear
        if not hasattr(model, head_name):
            logger.warning(
                f"[Critic model] {head_name} not found in model, although it may not affect the model's behavior, but it is recommended to check whether the model matches the configuration."
            )
        if two_linear and not isinstance(getattr(model, head_name), nn.Sequential):
            logger.warning(
                f"[Critic model] two_linear is set to True, but the head is not a nn.Sequential, although it may not affect the model's behavior, but it is recommended to check the model configuration."
            )
        if not two_linear and not isinstance(getattr(model, head_name), nn.Linear):
            logger.warning(
                f"[Critic model] two_linear is set to False, but the head is not a nn.Linear, although it may not affect the model's behavior, but it is recommended to check the model configuration."
            )

        # load model weights
        model = model.from_pretrained(model_path, **extra_kwargs)
    else:
        logger.info(f"[Critic model] Loading AutoCriticModel from {model_path}")
        model_class = get_critic_model(model_path, head_name, two_linear)
        model = model_class.from_pretrained(model_path, **extra_kwargs)

    state_dict = model.state_dict()
    for key in exclude_keys:
        # state_dict[key] = torch.nn.init.zeros_(state_dict[key])
        state_dict[key] = torch.nn.init.normal_(state_dict[key], mean=0.0, std=0.02)
    model.load_state_dict(state_dict, strict=False)
    logger.warning(f"[Critic model] init {exclude_keys} with zeros ...")
    return model


def build_reward_model(
    model_path, head_name="v_head", two_linear=False, extra_kwargs={}
):
    model_type = "unknown"
    if os.path.exists(os.path.join(model_path, "config.json")):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)
            model_type = config.get("model_type", "unknown")
    if model_type == "internlm2":
        logger.info(f"[Reward model] Loading InternLM2 model from {model_path}")
        logger.warning(
            "[Reward model] Using InternLM2ForRewardModel as reward model, the setting of `head_name` and `two_linear` will be ignored."
        )
        config = InternLM2RewardConfig.from_pretrained(model_path)
        config.update(extra_kwargs)
        with no_init_weights():
            model = InternLM2ForRewardModel(config)
        # check head_name and two_linear
        if not hasattr(model, head_name):
            logger.warning(
                f"[Reward model] {head_name} not found in model, although it may not affect the model's behavior, "
                "but it is recommended to check whether the model matches the configuration."
            )
        if two_linear and not isinstance(getattr(model, head_name), nn.Sequential):
            logger.warning(
                f"[Reward model] two_linear is set to True, but the head is not a nn.Sequential, "
                "although it may not affect the model's behavior, but it is recommended to check the model configuration."
            )
        if not two_linear and not isinstance(getattr(model, head_name), nn.Linear):
            logger.warning(
                f"[Reward model] two_linear is set to False, but the head is not a nn.Linear, "
                "although it may not affect the model's behavior, but it is recommended to check the model configuration."
            )

        # load model weights
        model = model.from_pretrained(model_path, **extra_kwargs)
    else:
        logger.info(f"[Reward model] Loading AutoCriticModel from {model_path}")
        model_class = get_reward_model(model_path, head_name, two_linear)
        model = model_class.from_pretrained(model_path, **extra_kwargs)
    return model


def build_language_model(model_path, extra_kwargs={}):
    model_type = "unknown"
    if os.path.exists(os.path.join(model_path, "config.json")):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)
            model_type = config.get("model_type", "unknown")
    if model_type == "internlm2":
        logger.info(f"Loading InternLM2 model from {model_path}")
        config = InternLM2Config.from_pretrained(model_path)
        with no_init_weights():
            model = InternLM2ForCausalLM(config)
        # load model weights
        model = model.from_pretrained(model_path, **extra_kwargs)
    else:
        logger.info(f"Loading AutoModel from {model_path}")
        model = AutoModel.from_pretrained(model_path, **extra_kwargs)
    return model


def _get_model_class(model_name_or_path: str):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config_class = type(config)
    if config_class in AutoModel._model_mapping:
        model_class = AutoModel._model_mapping[type(config)]
        model_base_class = model_class.__base__
        return model_class, model_base_class

    if "AutoModel" in config.auto_map:
        module_file, causal_model_name = config.auto_map["AutoModel"].split(".")
    elif "AutoModelForCausalLM" in config.auto_map:
        module_file, causal_model_name = config.auto_map["AutoModelForCausalLM"].split(
            "."
        )
    else:
        raise Exception(
            f"config of {model_name_or_path} has no AutoModel or AutoModelForCausalLM in auto_map"  # noqa: E501
        )

    model_class_name = (
        causal_model_name.split("For")[0] + "Model"
    )  # e.g., "InternLM2Model"
    model_class = get_class_from_dynamic_module(
        f"{module_file}.{model_class_name}", model_name_or_path
    )
    model_base_class_name = (
        causal_model_name.split("For")[0] + "PreTrainedModel"
    )  # e.g., "InternLM2PreTrainedModel"
    model_base_class = get_class_from_dynamic_module(
        f"{module_file}.{model_base_class_name}", model_name_or_path
    )
    return model_class, model_base_class


def get_critic_model(model_name_or_path: str, head_name: str, two_linear: bool = False):
    model_class, model_base_class = _get_model_class(model_name_or_path)

    class AutoCriticModel(model_base_class):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            self.model = model_class(config)
            self.head_name = head_name
            if two_linear:
                setattr(
                    self,
                    head_name,
                    nn.Sequential(
                        nn.Linear(config.hidden_size, config.hidden_size, bias=False),
                        nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps),
                        nn.Tanh(),
                        nn.Linear(config.hidden_size, 1, bias=False),
                    ),
                )
            else:
                setattr(
                    self,
                    head_name,
                    nn.Linear(config.hidden_size, 1, bias=False),
                )

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            **_ignored,
        ) -> torch.Tensor:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = outputs[0]
            logits = getattr(self, self.head_name)(hidden_states).squeeze(-1)[:, :-1]

            return SequenceClassifierOutputWithPast(
                logits=logits,
            )

    return AutoCriticModel


def get_reward_model(model_name_or_path: str, head_name: str, two_linear: bool = False):
    model_class, model_base_class = _get_model_class(model_name_or_path)

    class AutoRewardModel(model_base_class):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            self.model = model_class(config)
            self.head_name = head_name
            if two_linear:
                setattr(
                    self,
                    head_name,
                    nn.Sequential(
                        nn.Linear(config.hidden_size, config.hidden_size, bias=False),
                        nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps),
                        nn.Tanh(),
                        nn.Linear(config.hidden_size, 1, bias=False),
                    ),
                )
            else:
                setattr(
                    self,
                    head_name,
                    nn.Linear(config.hidden_size, 1, bias=False),
                )

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            **_ignored,
        ) -> torch.Tensor:
            eos_indices = (
                attention_mask.size(1)
                - 1
                - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            )
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = outputs[0]
            values = getattr(self, self.head_name)(hidden_states).squeeze(-1)
            reward_scores = values.gather(dim=1, index=eos_indices).squeeze(1)

            return SequenceClassifierOutputWithPast(
                logits=reward_scores,
            )

    return AutoRewardModel
