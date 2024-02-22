import ray
from transformers import AutoTokenizer
from typing import Optional

from .model_backend import HfModelRunnerRayActor

class ModelServer():
    # Initialize
    def __init__(self, model_config):
        self.model_config = model_config
        self.model_name = model_config.get("model_name")
        model_path = model_config.get("model_path")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # TODO: os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.trainer = None
        self.inferer = None
        self.model_ref = None
        self.is_initialized = False
        print(f"[{self.__class__.__name__}] model_config={model_config}")

    def initialize(self):
        model_trainer_type = self.model_config.get("model_trainer")
        model_inferer_type = self.model_config.get("model_inferer")
        if model_trainer_type is None and model_inferer_type is None:
            self.trainer = HfModelRunnerRayActor.remote(self.model_config)
            self.inferer = self.trainer # reuse the same actor
            self.model_ref = self.trainer.get_model.remote() # an reference
        elif model_trainer_type == "deepspeed" and model_inferer_type == "vllm":
            raise NotImplementedError
        else:
            raise NotImplementedError
        self.is_initialized = True
        print(f"[{self.__class__.__name__}] {self.model_name} has been initialized.")

    def model_get(self):
        return ray.get(self.model_ref, timeout=600.0) # 10min

    # Training
    def train(self, input_ids, labels, attention_mask, **train_kwargs):
        object_refs = self.train_async(input_ids, labels, attention_mask, **train_kwargs)
        return self.train_get(object_refs)

    def train_async(self, input_ids, labels, attention_mask, **train_kwargs):
        model_trainer_type = self.model_config.get("model_trainer")
        if model_trainer_type is None:
            return self.trainer.train.remote(input_ids, labels, attention_mask, **train_kwargs)
        elif model_trainer_type == "internevo":
            raise NotImplementedError
        elif model_trainer_type == "deepspeed":
            raise NotImplementedError
        else:
            raise NotImplementedError(f"{model_trainer_type} is unexpected")

    def train_get(self, object_refs, timeout: Optional[float] = None):
        return ray.get(object_refs, timeout=timeout)

    # Inference
    def infer(self, input_ids = None, **infer_kwargs):
        object_refs = self.infer_async(input_ids, **infer_kwargs)
        return self.infer_get(object_refs)

    def infer_async(self, input_ids = None, **infer_kwargs):
        model_inferer_type = self.model_config.get("model_inferer")
        if model_inferer_type is None:
            return self.inferer.infer.remote(input_ids, infer_kwargs)
        elif model_inferer_type == "vllm":
            raise NotImplementedError
        else:
            raise NotImplementedError(f"{model_inferer_type} is unexpected")

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
