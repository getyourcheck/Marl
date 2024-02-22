import ray
from typing import List
from .model_server import ModelServer
from .config import Config

class Coordinator:
    def __init__(self, model_configs: List[Config]):
        ray.init()
        self.model_configs = model_configs
        self.models = []

    def create_models(self, model_configs=None) -> List[ModelServer]:
        model_configs = self.model_configs if model_configs is None else model_configs
        self.models = [ModelServer(model_config) for model_config in model_configs]
        self._schedule()
        return self.models

    def _schedule(self):
        for model in self.models:  # naive serial initialize
            model.initialize()
            print(f"[{self.__class__.__name__}] {model.model_name} {model.__class__.__name__}.is_initialized: {model.is_initialized}")
