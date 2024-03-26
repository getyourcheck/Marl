import ray
import torch
import marl.utils as marl_util
from transformers import AutoConfig
from typing import Optional
from ..model_backend import HfModelRunnerRayActorGroup
from ..tokenizer.tokenizer_utils import get_tokenizer
from ..config_consts import *


class BaseModelServer:
    # Initialize
    def __init__(self, model_name: str, model_config: dict):
        self.model_name = model_name
        self.model_config = model_config
        self.trainer = None
        self.generator = None
        self.tokenizer = None
        self.model_ref = None
        self.is_initialized = False
        print(
            f"[{self.__class__.__name__}] model_name={model_name}, model_config={model_config}"
        )

    def initialize(self):
        model_path: str = self.model_config["model_path"]  # requisite
        self.model_type: str = self.model_config["model_type"]  # requisite
        if self.model_type == MODEL_TYPE_REWARD:
            temp_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self.reward_token_id = temp_config.reward_token_id
        trainer_config: dict = self.model_config["trainer_config"]  # requisite
        generator_config: dict = self.model_config.get("generator_config")  # optional
        tokenizer_path: str = self.model_config.get("tokenizer_path", model_path)  # opt

        trainer_config["model_path"] = model_path
        trainer_config["model_type"] = self.model_type
        trainer_config["tokenizer_path"] = tokenizer_path
        # Tokenizer is initialized in ModelServer (not ModelTrainer) to avoid remote call
        # FIXME: os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = get_tokenizer(tokenizer_path, trust_remote_code=True)
        trainer_config["tokenizer_pad_token_id"] = self.tokenizer.pad_token_id

        if self.trainer_type == ENGINE_HUGGINGFACE:
            self.trainer = HfModelRunnerRayActorGroup(
                name=f"{self.model_name}_trainer", config=trainer_config
            )
        elif self.trainer_type == ENGINE_INTERNEVO:
            raise NotImplementedError(f"{self.trainer_type}.")
        else:
            raise ValueError(f"No trainer is registered with type: {self.trainer_type}")

        self.generator = self.trainer  # use trainer for self.generate() by default
        if generator_config is not None:  # optional
            generator_config["model_path"] = model_path
            generator_config["tokenizer_path"] = tokenizer_path
            shared_with_trainer = generator_config.get("shared_with_trainer", True)
            if shared_with_trainer:
                self.generator = self.trainer
            else:
                if self.generator_type == ENGINE_HUGGINGFACE:
                    self.generator = HfModelRunnerRayActorGroup(
                        f"{self.model_name}_generator", generator_config
                    )
                elif self.generator_type == ENGINE_VLLM:
                    raise NotImplementedError(f"{self.generator_type}.")
                else:
                    raise ValueError(
                        f"No generator is registered with type '{self.generator_type}'."
                    )

        self.is_initialized = True
        print(
            f"[{self.__class__.__name__}] {self.model_name} has been initialized.  self.generator_eq_trainer: {self.generator_eq_trainer}"
        )

    @property
    def generator_eq_trainer(self):
        return id(self.generator) == id(self.trainer)

    @property
    def trainer_type(self):
        trainer_config = self.model_config["trainer_config"]
        trainer_type: str = trainer_config.get("trainer_type")
        return trainer_type.lower() if trainer_type is not None else trainer_type

    @property
    def generator_type(self):
        generator_config: dict = self.model_config.get("generator_config")
        if generator_config is None:
            return None
        generator_type: str = generator_config.get("generator_type")
        return generator_type.lower() if generator_type is not None else generator_type

    def model_get(self):
        if not self.model_ref:
            self.model_ref = self.trainer.get_model()  # an reference
        return ray.get(self.model_ref, timeout=600.0)  # 10min timeout

    # Training
    def train_async(
        self, input_ids, labels=None, attention_mask=None, *args, **train_kwargs
    ):
        return self.trainer.train_async(
            input_ids, labels, attention_mask, *args, **train_kwargs
        )

    def train_get(self, object_refs, timeout: Optional[float] = None):
        return self.trainer.train_get(object_refs, timeout=timeout)

    def train(self, input_ids, labels=None, attention_mask=None, *args, **train_kwargs):
        object_refs = self.train_async(
            input_ids, labels, attention_mask, *args, **train_kwargs
        )
        return self.train_get(object_refs)

    # Inference
    def infer_async(self, inputs, attention_mask=None, *args, **infer_kwargs):
        if not isinstance(inputs, torch.Tensor):
            input_ids, attention_mask = marl_util.encode(inputs, self.tokenizer)
            if self.model_type == MODEL_TYPE_REWARD:
                input_ids, attention_mask = marl_util.expand_reward_token_id(
                    self.reward_token_id, input_ids, attention_mask
                )
        else:
            input_ids = inputs
        return self.trainer.infer_async(
            input_ids=input_ids, attention_mask=attention_mask, *args, **infer_kwargs
        )

    def infer_get(self, object_refs, timeout: Optional[float] = None):
        return self.trainer.infer_get(object_refs, timeout=timeout)

    def infer(self, inputs, *args, **infer_kwargs):
        object_refs = self.infer_async(inputs, *args, **infer_kwargs)
        return self.infer_get(object_refs)

    # Generation
    def generate_async(self, inputs, attention_mask=None, *args, **generate_kwargs):
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs
        elif isinstance(inputs, list):
            input_ids, attention_mask = marl_util.encode(
                inputs, self.tokenizer, add_generation_prompt=True
            )
        else:
            raise NotImplementedError(f"unknown inputs: {inputs}")
        return self.generator.generate_async(
            input_ids=input_ids, attention_mask=attention_mask, *args, **generate_kwargs
        )

    def generate_get(self, object_refs, timeout: Optional[float] = None):
        return self.generator.generate_get(object_refs, timeout=timeout)

    def generate(self, inputs, *args, **generate_kwargs):
        object_refs = self.generate_async(inputs, *args, **generate_kwargs)
        return self.generate_get(object_refs)

    # Others
    def set_seed(self, seed: int = None):
        self.trainer.set_seed(seed)
        if not self.generator_eq_trainer:
            self.generator.set_seed(seed)

    def clean_up(self):
        self.trainer.release_resources()
        if not self.generator_eq_trainer:
            self.generator.release_resources()
        print(f"[{self.__class__.__name__}] {self.model_name} is destroyed.")

    def _load_chat_template(self, chat_template):
        # adopted from: https://github.com/vllm-project/vllm/blob/0e163fce18594c7e29dc5a143dd6b33d213fcbf3/vllm/entrypoints/openai/serving_chat.py#L245
        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    self.tokenizer.chat_template = f.read()
            except OSError:
                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                pass
            print(
                f"[INFO] Using supplied chat template:\n{self.tokenizer.chat_template}"
            )
        elif self.tokenizer.chat_template is not None:
            print(
                f"[INFO] Using default chat template:\n{self.tokenizer.chat_template}"
            )
        else:
            print("[WARNING] No chat template provided. Chat API will not work.")

    def save_model(self, path):
        self.trainer.save_model(path)
