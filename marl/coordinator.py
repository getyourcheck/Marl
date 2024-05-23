from pathlib import Path

from loguru import logger
import ray

from .config_utils import get_resource_requirement
from .model_server import *
from .config_consts import *

ROOT_PATH = Path(__file__).parents[1].resolve()


class Coordinator:
    def __init__(self, cluster_address: str, model_configs: dict):
        self.cluster_address = cluster_address
        self.model_configs = model_configs
        self.model_dict = dict()
        self.context_type: str = None  # "client" or "server"
        self.context: ray._private.workers.BaseContext = None

        resources = get_resource_requirement(self.model_configs)
        logger.info(f"Required resources: {resources}")
        runtime_env = {"working_dir": ROOT_PATH}
        logger.info(f"working_dir (root_path): {ROOT_PATH}")

        try:
            client_context = ray.init(
                address=self.cluster_address,
                runtime_env=runtime_env,
                ignore_reinit_error=True,
            )
            logger.info(f"Connected to a running ray cluster at {self.cluster_address}")
            self.context_type = "client"
            self.context = client_context

        except ConnectionError:
            logger.info(
                f"Error connecting to {self.cluster_address}, try initializing a new ray cluster."
            )
            ray_context = ray.init(
                address=None,
                resources=resources,
                runtime_env=runtime_env,
                ignore_reinit_error=True,
            )
            node_ip_address = ray_context.address_info["node_ip_address"]
            logger.info(f"Initialize a ray cluster at {node_ip_address}")
            self.context_type = "server"
            self.context = ray_context

    def create_models(self) -> dict[str, BaseModelServer]:
        self.model_dict = {}
        for model_name, model_config in self.model_configs.items():
            model_type = model_config["model_type"]
            if model_type == MODEL_TYPE_ACTOR:
                self.model_dict[model_name] = ActorModelServer(model_name, model_config)
            elif model_type == MODEL_TYPE_CRITIC:
                self.model_dict[model_name] = CriticModelServer(model_name, model_config)
            elif model_type == MODEL_TYPE_REWARD:
                self.model_dict[model_name] = RewardModelServer(model_name, model_config)
            elif model_type == MODEL_TYPE_REFERENCE:
                self.model_dict[model_name] = RefModelServer(model_name, model_config)
            else:
                raise NotImplementedError(f"Unknown model_type: {model_type}")
        self._schedule()
        return self.model_dict

    def _schedule(self):
        for model_name, model in self.model_dict.items():  # naive serial initialize
            model.initialize_async()
            logger.info(
                f"{model_name} {model.__class__.__name__}.is_initialized: {model.is_initialized}"
            )
        for model_name, model in self.model_dict.items():  # naive serial initialize
            model.initialize_get()
            logger.info(
                f"{model_name} {model.__class__.__name__}.is_initialized: {model.is_initialized}"
            )

    def clean_up(self):
        for _, model_server in self.model_dict.items():
            if model_server.trainer is not None:
                model_server.trainer.release_resources()
            if model_server.generator is not None:
                model_server.generator.release_resources()
