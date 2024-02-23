import ray
from typing import List, Union
from .model_server import ModelServer

class Coordinator:
    def __init__(self, cluster_address: str, model_configs: dict):
        self.cluster_address = cluster_address
        self.model_configs = model_configs
        self.model_dict = dict()

        resources = Coordinator.analyze_resource_requirement(self.model_configs)
        print(f"[{self.__class__.__name__}] Required resources: {resources}")
        try:
            ray.init(address=self.cluster_address, resources=resources)
            print(f"[{self.__class__.__name__}] Connected to an existing ray cluster at {self.cluster_address}")
        except ConnectionError:
            ray.init(resources=resources)  # FIXME
            print(f"[{self.__class__.__name__}] Initialize a ray cluster at {self.cluster_address}")

    def create_models(self) -> dict[str, ModelServer]:
        self.model_dict = { model_name: ModelServer(model_name, model_config)
            for model_name, model_config in self.model_configs.items()
        }
        self._schedule()
        return self.model_dict

    def _schedule(self):
        for model_name, model in self.model_dict.items():  # naive serial initialize
            model.initialize()
            print(f"[{self.__class__.__name__}] {model_name} {model.__class__.__name__}.is_initialized: {model.is_initialized}")

    @staticmethod
    def analyze_resource_requirement(model_configs: dict) -> dict:
        """
        Analyzes resource requirements for a list of model configs and returns
        a dictionary with the total number of GPUs and CPUs required.

        Args:
            model_configs (dict): A dictionary containing model configurations.

        Returns:
            dict: A dictionary with the total number of GPUs and CPUs required.
        """

        def _get_num_gpus_from_trainer_config(trainer_config):
            # Calculates the number of GPUs required for a given trainer configuration.
            num_gpus = 1
            if "parallel" in trainer_config:
                parallel = trainer_config["parallel"]
                zero1 = parallel.get("zero1", {"size": 1})
                tensor = parallel.get("tensor", {"size": 1})
                pipeline = parallel.get("pipeline", {"size": 1})
                num_gpus = zero1['size'] * tensor['size'] * pipeline['size']
            return num_gpus

        resources = {"num_gpus": 0}
        for _, model_config in model_configs.items():
            trainer_config = model_config["trainer_config"]
            num_gpus = _get_num_gpus_from_trainer_config(trainer_config)

            if "generator_config" in model_config:
                generator_config = model_config["generator_config"]
                if not generator_config.get("shared_with_trainer"):  # None or False
                    num_gpus += _get_num_gpus_from_trainer_config(generator_config)

            resources['num_gpus'] += num_gpus

        resources['num_cpus'] = resources['num_gpus'] * 10
        return resources
