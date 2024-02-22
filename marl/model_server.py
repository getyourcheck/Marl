import ray
from transformers import AutoTokenizer
from typing import Optional

from .model_backend import (
    HfModelRunnerRayActor
)

class ModelServer():
    # Initialize
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.model_name = model_config.get("model_name")
        self.trainer = None
        self.generator = None
        self.model_ref = None
        self.is_initialized = False
        print(f"[{self.__class__.__name__}] model_config={model_config}")

    def initialize(self):
        model_path = self.model_config.get("model_path")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # TODO: os.environ["TOKENIZERS_PARALLELISM"] = "false"

        default_runner_config: dict = {"model_path": model_path}
        generator_config: dict = self.model_config.get("generator_config", default_runner_config)
        generator_type = generator_config.get("generator_type", "huggingface")
        if generator_type == "huggingface":
            self.generator = HfModelRunnerRayActor.remote(generator_config)
        elif generator_type == "vllm":
            raise NotImplementedError
        else:
            raise ValueError(f"No generator is registered with type '{generator_type}'.")

        trainer_config: dict = self.model_config.get("trainer_config", default_runner_config)
        trainer_type = self.model_config.get("model_trainer", "huggingface")
        if trainer_type == generator_type:  # reuse the same runner
            self.trainer = self.generator
        elif trainer_type == "huggingface":
            self.trainer = HfModelRunnerRayActor.remote(trainer_config)
        elif trainer_type == "internevo":
            raise NotImplementedError
        else:
            raise ValueError(f"No trainer is registered with type '{trainer_type}'.")
        self.model_ref = self.trainer.get_model.remote() # an reference

        self.is_initialized = True
        print(f"[{self.__class__.__name__}] {self.model_name} has been initialized.  self.generator_eq_trainer: {self.generator_eq_trainer}")

    @property
    def generator_eq_trainer(self):
        return id(self.generator) == id(self.trainer)

    def model_get(self):
        return ray.get(self.model_ref, timeout=600.0) # 10min timeout

    # Training
    def train(self, input_ids, labels=None, attention_mask=None, **train_kwargs):
        object_refs = self.train_async(input_ids, labels, attention_mask, **train_kwargs)
        return self.train_get(object_refs)

    def train_async(self, input_ids, labels=None, attention_mask=None, **train_kwargs):
        return self.trainer.train.remote(input_ids, labels, attention_mask, **train_kwargs)

    def train_get(self, object_refs, timeout: Optional[float] = None):
        return ray.get(object_refs, timeout=timeout)

    # Inference
    def infer(self, input_ids, **infer_kwargs):
        object_refs = self.infer_async(input_ids, **infer_kwargs)
        return self.infer_get(object_refs)

    def infer_async(self, input_ids, **infer_kwargs):
        return self.generator.infer.remote(input_ids, infer_kwargs)

    def infer_get(self, object_refs, timeout: Optional[float] = None):
        return ray.get(object_refs, timeout=timeout)

    # Others
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
            print(f"[INFO] Using supplied chat template:\n{self.tokenizer.chat_template}")
        elif self.tokenizer.chat_template is not None:
            print(f"[INFO] Using default chat template:\n{self.tokenizer.chat_template}")
        else:
            print("[WARNING] No chat template provided. Chat API will not work.")
