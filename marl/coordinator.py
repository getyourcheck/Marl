import ray
from .model_server import BaseModelServer
from .config_utils import get_resource_requirement
from pathlib import Path
ROOT_PATH = Path(__file__).parents[1].resolve()

class Coordinator:
    def __init__(self, cluster_address: str, model_configs: dict):
        self.cluster_address = cluster_address
        self.model_configs = model_configs
        self.model_dict = dict()
        self.context_type: str = None  # "client" or "server"
        self.context: ray._private.workers.BaseContext = None

        resources = get_resource_requirement(self.model_configs)
        print(f"[{self.__class__.__name__}] Required resources: {resources}")
        runtime_env={"working_dir": ROOT_PATH}
        print(f"[{self.__class__.__name__}] root_path: {ROOT_PATH}")

        try:
            client_context = ray.init(address=self.cluster_address, runtime_env=runtime_env, ignore_reinit_error=True)
            print(f"[{self.__class__.__name__}] Connected to an existing ray cluster at {self.cluster_address}")
            self.context_type = "client"
            self.context = client_context
        except ConnectionError:
            print(f"[{self.__class__.__name__}] Error in connecting to {self.cluster_address}, try initializing a new ray cluster.")
            ray_context = ray.init(address=None, resources=resources, runtime_env=runtime_env, ignore_reinit_error=True)
            node_ip_address = ray_context.address_info['node_ip_address']
            print(f"[{self.__class__.__name__}] Initialize a ray cluster at {node_ip_address}")
            self.context_type = "server"
            self.context = ray_context

    def create_models(self) -> dict[str, BaseModelServer]:
        self.model_dict = { model_name: BaseModelServer(model_name, model_config)
            for model_name, model_config in self.model_configs.items()
        }
        self._schedule()
        return self.model_dict

    def _schedule(self):
        for model_name, model in self.model_dict.items():  # naive serial initialize
            model.initialize()
            print(f"[{self.__class__.__name__}] {model_name} {model.__class__.__name__}.is_initialized: {model.is_initialized}")

    def clean_up(self):
        for _, model_server in self.model_dict.items():
            if model_server.trainer != None:
                model_server.trainer.release_resources()
            if model_server.generator != None:
                model_server.generator.release_resources()