# keywords for config files

# model type (actor, critic, reward, reference, ...) for `model_type`
MODEL_TYPE_ACTOR: str = "actor"
MODEL_TYPE_REFERENCE: str = "reference"
MODEL_TYPE_REWARD: str = "reward"
MODEL_TYPE_CRITIC: str = "critic"

# training or generation engines for `trainer_type` and `generator_type`
ENGINE_HUGGINGFACE: str = "huggingface"
ENGINE_INTERNEVO: str = "internevo"
ENGINE_VLLM: str = "vllm"
ENGINE_LMDEPLOY: str = "lmdeploy"

# plugins for trainer engine (e.g., huggingface accelerate)
ENGINE_PLUGIN_FSDP: str = "fsdp"
ENGINE_PLUGIN_DEEPSPEED: str = "deepspeed"