# keywords for config files

# model type (actor, critic, reward, reference, ...) for `model_type`
MODEL_TYPE_REWARD: str = "reward"
MODEL_TYPE_CRITIC: str = "critic"

# training or generation engines for `trainer_type` and `generator_type`
ENGINE_HUGGINGFACE: str = "huggingface"
ENGINE_INTERNEVO: str = "internevo"
ENGINE_VLLM: str = "vllm"