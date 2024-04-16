from .hf_model_runner import HfModelRunnerRayActorGroup
from .vllm_model_runner import VllmGeneratorRayActorGroup

SUPPORTED_MODEL_RUNNERS = [
    HfModelRunnerRayActorGroup,
    VllmGeneratorRayActorGroup,
]
