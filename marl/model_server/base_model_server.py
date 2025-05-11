from typing import Optional

import ray
import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from ..config.config_consts import ENGINE_HUGGINGFACE, ENGINE_INTERNEVO
from ..model_backend.hf_model_runner import HfModelRunnerRayActorGroup
from ..tokenizer import encode_inputs, get_tokenizer

DEFAULT_GET_TIMEOUT = 600.0  # 10 min


class BaseModelServer:
    # Initialize
    def __init__(self, model_name: str, model_config: dict):
        """
        初始化模型服务器
        Args:
            model_name: 模型名称
            model_config: 模型配置字典
        """
        self.model_name = model_name
        self.model_config = model_config
        self.tokenizer = None
        self.tokenizer_config = None
        self.trainer = None
        self.trainer_config = None
        self.model_ref = None
        self.is_initialized = False
        # 是否显示CUDA内存统计信息
        self.show_cuda_mem_stats = self.model_config.get(
            'show_cuda_mem_stats', False)
        logger.info(f'model_name={model_name}, model_config={model_config}')

    def init_tokenizer_and_config(self, model_config):
        """
        初始化分词器和相关配置
        Args:
            model_config: 模型配置字典
        """
        tokenizer_config = model_config.get('tokenizer_config', {})
        # 获取分词器路径，优先级：tokenizer_config > model_config
        if 'tokenizer_path' in tokenizer_config:
            tokenizer_path = tokenizer_config['tokenizer_path']
        elif 'tokenizer_path' in model_config:
            tokenizer_path = model_config['tokenizer_path']
        else:
            tokenizer_path = model_config['model_path']

        # 初始化分词器
        self.tokenizer = get_tokenizer(
            tokenizer_path, trust_remote_code=True, **tokenizer_config)

        # 更新分词器配置
        tokenizer_config['tokenizer_path'] = tokenizer_path
        tokenizer_config['pad_token_id'] = self.tokenizer.pad_token_id
        self.tokenizer_config = tokenizer_config

    def init_trainer_config(self, model_config):
        """
        初始化训练器配置
        Args:
            model_config: 模型配置字典
        """
        model_path = model_config['model_path']
        trainer_config: dict = model_config['trainer_config']  # requisite
        trainer_config['tokenizer_config'] = self.tokenizer_config
        trainer_config['tokenizer_path'] = self.tokenizer_config['tokenizer_path']
        trainer_config['model_path'] = model_path
        trainer_config['model_type'] = model_config['model_type']
        self.trainer_config = trainer_config

    def initialize_async(self):
        """异步初始化模型服务器"""
        self.init_tokenizer_and_config(self.model_config)
        self.init_trainer_config(self.model_config)

        # 根据训练器类型初始化相应的训练器
        trainer_type = self.trainer_config.get('trainer_type',
                                               ENGINE_HUGGINGFACE).lower()
        if trainer_type == ENGINE_HUGGINGFACE:
            self.trainer = HfModelRunnerRayActorGroup(
                name=f'{self.model_name}_trainer', config=self.trainer_config)
        elif trainer_type == ENGINE_INTERNEVO:
            raise NotImplementedError(f'{trainer_type}.')
        else:
            raise ValueError(
                f'No trainer is registered with type: {trainer_type}')

    def initialize_get(self):
        """获取初始化结果"""
        self.trainer.initialize_get()
        self.is_initialized = True
        logger.info(f'{self.model_name} has been initialized.')

    # Inference
    def infer_async(self, inputs, attention_mask=None, *args, **infer_kwargs):
        """
        异步推理方法
        Args:
            inputs: 输入数据
            attention_mask: 注意力掩码
            *args, **infer_kwargs: 其他推理参数
        """
        if not isinstance(inputs, torch.Tensor):
            input_ids, attention_mask = encode_inputs(inputs, self.tokenizer)
        else:
            input_ids = inputs
        return self.trainer.infer_async(
            input_ids=input_ids,
            attention_mask=attention_mask,
            *args,
            **infer_kwargs)

    def infer_get(self, object_refs, timeout: Optional[float] = None):
        """
        获取推理结果
        Args:
            object_refs: Ray对象引用
            timeout: 超时时间
        """
        return self.trainer.infer_get(object_refs, timeout=timeout)

    def infer(self, inputs, *args, **infer_kwargs):
        """
        同步推理方法
        Args:
            inputs: 输入数据
            *args, **infer_kwargs: 其他推理参数
        """
        object_refs = self.infer_async(inputs, *args, **infer_kwargs)
        results = self.infer_get(object_refs)
        self.log_cuda_mem_stats(remark='[infer] ')
        return results

    # Training
    def train_async(self,
                    input_ids,
                    labels=None,
                    attention_mask=None,
                    position_ids=None,
                    *args,
                    **train_kwargs):
        """
        异步训练方法
        Args:
            input_ids: 输入ID
            labels: 标签
            attention_mask: 注意力掩码
            position_ids: 位置ID
            *args, **train_kwargs: 其他训练参数
        """
        return self.trainer.train_async(input_ids, labels, attention_mask,
                                        position_ids, *args, **train_kwargs)

    def train_get(self, object_refs, timeout: Optional[float] = None):
        """
        获取训练结果
        Args:
            object_refs: Ray对象引用
            timeout: 超时时间
        """
        return self.trainer.train_get(object_refs, timeout=timeout)

    def train(self,
              input_ids,
              labels=None,
              attention_mask=None,
              position_ids=None,
              *args,
              **train_kwargs):
        """
        同步训练方法
        Args:
            input_ids: 输入ID
            labels: 标签
            attention_mask: 注意力掩码
            position_ids: 位置ID
            *args, **train_kwargs: 其他训练参数
        """
        object_refs = self.train_async(input_ids, labels, attention_mask,
                                       position_ids, *args, **train_kwargs)
        loss = self.train_get(object_refs)
        self.log_cuda_mem_stats(remark='[train] ')
        return loss

    # Generation
    def generate_async(self,
                       inputs,
                       attention_mask=None,
                       *args,
                       **generate_kwargs):
        """异步生成方法（待实现）"""
        raise NotImplementedError

    def generate_get(self, object_refs, timeout: Optional[float] = None):
        """获取生成结果（待实现）"""
        raise NotImplementedError

    def generate(self, inputs, *args, **generate_kwargs):
        """同步生成方法（待实现）"""
        raise NotImplementedError

    # Model
    def model_get(self):
        """获取模型实例"""
        if not self.model_ref:
            self.model_ref = self.trainer.get_model()  # an reference
        return ray.get(self.model_ref, timeout=DEFAULT_GET_TIMEOUT)

    def state_dict_get(self):
        """获取模型状态字典"""
        return ray.get(
            self.trainer.get_state_dict(), timeout=DEFAULT_GET_TIMEOUT)

    def save(self, path):
        """
        保存模型和分词器
        Args:
            path: 保存路径
        """
        self.trainer.save(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

    # 其他工具方法
    def set_seed(self, seed: int = None):
        """
        设置随机种子
        Args:
            seed: 随机种子值
        """
        self.trainer.set_seed(seed)

    def log_cuda_mem_stats(self, remark=''):
        """
        记录CUDA内存使用统计
        Args:
            remark: 备注信息
        """
        if self.show_cuda_mem_stats:
            trainer_mem = self.trainer.get_cuda_mem_stats()
            logger.info(
                f'{remark}{self.model_name} trainer allocated GPU memory: {trainer_mem.total_current_mb} MiB'  # noqa: E501
            )

    def clean_up(self):
        """清理资源"""
        self.trainer.release_resources()
        logger.info(f'{self.model_name} is destroyed.')
