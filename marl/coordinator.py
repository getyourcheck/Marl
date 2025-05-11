from pathlib import Path

import ray # 分布式计算框架
from loguru import logger # 日志记录工具

from .config.config_consts import (MODEL_TYPE_CRITIC, MODEL_TYPE_POLICY,
                                   MODEL_TYPE_REFERENCE, MODEL_TYPE_REWARD)
from .config.config_utils import get_resource_requirement
from .model_server import (BaseModelServer, CriticModelServer,
                           PolicyModelServer, RefModelServer,
                           RewardModelServer)
from .model_backend import sp_util

ROOT_PATH = Path(__file__).parents[1].resolve() # 项目根目录路径:/ssd/zhaohui/baiyao/marl，用于设置Ray的工作目录


class Coordinator:
    """
    协调器类:负责创建和管理RLHF训练中的各种模型服务器
    作为整个训练框架的核心组件，协调模型的初始化、调度和资源管理
    """

    def __init__(self, cluster_address: str, configs: dict):
        """
        初始化协调器
        
        参数:
            cluster_address: Ray集群的地址
            configs: 包含模型配置的字典
        """
        # 是否启用xtuner调度
        configs = sp_util.add_dispatch_config_if_needed(configs)
        # 保存集群地址
        self.cluster_address = cluster_address
        # 获取模型配置
        self.model_configs = configs['model_configs']
        # 获取分词器配置（如果没有则使用空字典）
        self.tokenizer_config = configs.get('tokenizer_config', {})
        # 初始化模型字典，用于存储创建的模型服务器实例
        self.model_dict = dict()
        # 初始化上下文类型（客户端或服务器）
        self.context_type: str = None  # "client" or "server"
        # 初始化Ray上下文
        self.context: ray._private.workers.BaseContext = None

        # 根据模型配置计算所需的计算资源（CPU、GPU等）
        resources = get_resource_requirement(self.model_configs)
        logger.info(f'Required resources: {resources}')
        # 设置运行环境，指定工作目录
        runtime_env = {'working_dir': ROOT_PATH}
        logger.info(f'working_dir (root_path): {ROOT_PATH}')

        try:
            # 尝试连接到现有的Ray集群
            client_context = ray.init(
                address=self.cluster_address,
                runtime_env=runtime_env,
                ignore_reinit_error=True,  # 忽略重新初始化错误
            )
            logger.info(
                f'Connected to a running ray cluster at {self.cluster_address}'
            )
            # 设置上下文类型为客户端
            self.context_type = 'client'
            self.context = client_context

        except ConnectionError:
            # 如果连接失败，则创建一个新的Ray集群
            logger.info(
                f'Error connecting to {self.cluster_address}, try initializing a new ray cluster.'  # noqa: E501
            )
            ray_context = ray.init(
                address=None,  # 不指定地址，创建新集群
                resources=resources,  # 指定资源需求
                runtime_env=runtime_env,
                ignore_reinit_error=True,
            )
            # 获取节点IP地址
            node_ip_address = ray_context.address_info['node_ip_address']
            logger.info(f'Initialize a ray cluster at {node_ip_address}')
            # 设置上下文类型为服务器
            self.context_type = 'server'
            self.context = ray_context

    def create_models(self) -> dict[str, BaseModelServer]:
        """
        创建各种类型的模型服务器
        
        返回:
            包含所有创建的模型服务器的字典
        """
        # 初始化模型字典
        self.model_dict = {}
        # 遍历配置中的所有模型
        for model_name, model_config in self.model_configs.items():
            # 获取模型类型
            model_type = model_config['model_type']
            # 设置分词器配置（如果模型配置中没有，则使用全局分词器配置）
            model_config['tokenizer_config'] = model_config.get('tokenizer_config', self.tokenizer_config)
            
            # 根据模型类型创建相应的模型服务器
            if model_type == MODEL_TYPE_POLICY:
                # 策略模型：用于生成文本，是RLHF中主要的训练对象
                self.model_dict[model_name] = PolicyModelServer(
                    model_name, model_config)
            elif model_type == MODEL_TYPE_CRITIC:
                # 评论家模型：用于评估策略，在某些算法中用于估计价值函数
                self.model_dict[model_name] = CriticModelServer(
                    model_name, model_config)
            elif model_type == MODEL_TYPE_REWARD:
                # 奖励模型：用于计算奖励，评估生成文本的质量
                self.model_dict[model_name] = RewardModelServer(
                    model_name, model_config)
            elif model_type == MODEL_TYPE_REFERENCE:
                # 参考模型：通常是策略模型的初始版本，用于计算KL散度防止策略模型偏离太远
                self.model_dict[model_name] = RefModelServer(
                    model_name, model_config)
            else:
                # 如果模型类型未知，则抛出异常
                raise NotImplementedError(f'Unknown model_type: {model_type}')
        
        # 调度并初始化所有模型
        self._schedule()
        # 返回创建的模型字典
        return self.model_dict

    def _schedule(self):
        """
        调度并初始化所有模型
        采用两阶段初始化方式：先异步初始化所有模型，然后等待所有模型初始化完成
        """
        # 第一阶段：异步初始化所有模型
        for model_name, model in self.model_dict.items():  # naive serial initialize
            model.initialize_async()
        
        # 第二阶段：等待所有模型初始化完成
        for model_name, model in self.model_dict.items():  # naive serial initialize
            model.initialize_get()
            # 记录每个模型的初始化状态
            logger.info(
                f'{model_name} {model.__class__.__name__}.is_initialized: {model.is_initialized}'  # noqa: E501
            )

    def clean_up(self):
        """
        清理资源：释放所有模型占用的资源
        在训练结束或中断时调用，确保资源正确释放
        """
        # 遍历所有模型服务器
        for _, model_server in self.model_dict.items():
            # 如果训练器存在，释放其资源
            if model_server.trainer is not None:
                model_server.trainer.release_resources()
            # 如果生成器存在，释放其资源
            if model_server.generator is not None:
                model_server.generator.release_resources()
