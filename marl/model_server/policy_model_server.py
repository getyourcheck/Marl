from typing import Optional

import torch
from loguru import logger

from ..config.config_consts import ENGINE_VLLM
from ..tokenizer import encode_inputs
from .base_model_server import BaseModelServer


class PolicyModelServer(BaseModelServer):
    # Initialize
    def initialize_async(self):
        """
        异步初始化策略模型服务器
        
        除了基类的初始化外，还处理生成器的配置和初始化
        策略模型有两个关键组件：训练器(trainer)和生成器(generator)
        训练器用于模型训练，生成器用于文本生成
        它们可以是同一个模型实例(默认)，也可以是两个分离的实例
        """
        # 调用父类的初始化方法，设置训练器
        super().initialize_async()

        # 默认设置生成器等于训练器
        self.generator_eq_trainer = True
        # use trainer for self.generate() by default
        self.generator = self.trainer
        
        # 如果配置中没有指定生成器配置，则直接使用训练器作为生成器
        if 'generator_config' not in self.model_config:
            logger.warning(f"[Generator] No generator config, `generator=trainer` by default.")
            return  # self.generator = self.trainer

        generator_config = self.model_config['generator_config']  # optional
        if generator_config.get('shared_with_trainer', True):
            logger.info(f"[Generator] `shared_with_trainer=True`, generator=trainer.")
            return  # self.generator = self.trainer

        # 如果需要单独的生成器，则设置生成器配置
        # 复用模型路径和分词器配置
        generator_config['model_path'] = self.model_config['model_path']
        generator_config['tokenizer_config'] = self.tokenizer_config
        generator_config['tokenizer_path'] = self.tokenizer_config[
            'tokenizer_path']
            
        # 获取生成器类型
        generator_type = generator_config.get('generator_type', None)
        self.generator_type = generator_type
        
        # 根据生成器类型创建对应的生成器实例
        if generator_type == ENGINE_VLLM:
            # 如果使用VLLM引擎，动态导入VLLM模型运行器
            from ..model_backend.vllm_model_runner import VllmGeneratorRayActorGroup
            # 创建VLLM生成器实例
            self.generator = VllmGeneratorRayActorGroup(
                f'{self.model_name}_generator', generator_config)
            logger.info(f"[Generator] using VLLM generator.")
        else:
            # 不支持的生成器类型
            raise ValueError(
                f"No generator is registered with type '{generator_type}'")
        # 标记生成器与训练器不同
        self.generator_eq_trainer = False

    def initialize_get(self):
        """
        完成初始化过程并获取结果
        
        在异步初始化后调用此方法等待初始化完成
        如果使用VLLM生成器，需要特殊处理训练器和生成器的同步
        """
        if self.generator_type == ENGINE_VLLM:
            # to sync model among trainer and generator
            self.trainer.initialize_get()
            self.trainer.init_process_group(self.generator)
        # 等待生成器初始化完成
        self.generator.initialize_get()
        # 标记初始化完成
        self.is_initialized = True
        logger.info(f'{self.model_name} has been initialized. ')

    # Generation
    def generate_async(self,
                       inputs,
                       attention_mask=None,
                       *args,
                       **generate_kwargs):
        """
        异步生成文本
        
        参数:
            inputs: 输入数据(张量或文本列表)
            attention_mask: 注意力掩码
            *args, **generate_kwargs: 其他生成参数
            
        返回:
            Ray对象引用，指向生成任务的结果
        """
        # 处理不同类型的输入
        if isinstance(inputs, torch.Tensor):
            # 如果输入是张量，直接使用
            input_ids = inputs
        elif isinstance(inputs, list):
            # 如果输入是列表(通常是文本列表)
            if not self.generator_eq_trainer:
                # 如果生成器和训练器不同，需要特殊处理输入编码
                input_ids, attention_mask = encode_inputs(
                    inputs,
                    self.tokenizer,
                    return_tensors=None,
                    padding=False,
                    add_generation_prompt=True)
            else:
                # 对于共享模型，使用标准编码方式
                input_ids, attention_mask = encode_inputs(
                    inputs, self.tokenizer, add_generation_prompt=True)
        else:
            # 不支持的输入类型
            raise NotImplementedError(f'unknown inputs: {inputs}')

        # 调用生成器的异步生成方法
        return self.generator.generate_async(
            input_ids=input_ids,
            attention_mask=attention_mask,
            *args,
            **generate_kwargs)

    def generate_get(self, object_refs, timeout: Optional[float] = None):
        """
        获取生成结果
        
        参数:
            object_refs: Ray对象引用
            timeout: 超时时间
            
        返回:
            生成的文本结果
        """
        return self.generator.generate_get(object_refs, timeout=timeout)

    def generate(self, inputs, *args, **generate_kwargs):
        """
        同步生成文本（组合异步生成和获取结果）
        
        参数:
            inputs: 输入数据
            *args, **generate_kwargs: 生成参数
            
        返回:
            生成的文本结果
        """
        # 异步提交生成任务
        object_refs = self.generate_async(inputs, *args, **generate_kwargs)
        # 等待并获取生成结果
        policy_output = self.generate_get(object_refs)
        # 记录内存使用情况
        self.log_cuda_mem_stats(remark='[generate] ')
        return policy_output

    # Sync
    def sync_model(self, *args, **kwargs):
        """
        同步训练器和生成器之间的模型参数
        
        当训练器更新模型参数后，需要将更新同步到生成器
        只有当生成器和训练器不同时才需要同步
        """
        if not self.generator_eq_trainer:
            self.trainer.broadcast_model_to_generator(self.generator)

    # 其他工具方法
    def log_cuda_mem_stats(self, remark=''):
        """
        记录CUDA内存使用统计
        
        记录训练器和生成器的内存使用情况
        参数:
            remark: 日志备注信息
        """
        if self.show_cuda_mem_stats:
            # 获取训练器和生成器的内存使用情况
            trainer_mem = self.trainer.get_cuda_mem_stats()
            generator_mem = self.generator.get_cuda_mem_stats()
            # 记录详细的内存使用日志
            logger.info(
                f'{remark}{self.model_name} trainer allocated GPU memory: {trainer_mem.total_current_mb} MiB, '  # noqa: E501
                f'generator allocated GPU memory: {generator_mem.total_current_mb} MiB, '  # noqa: E501
                f'generator_eq_trainer: {self.generator_eq_trainer}')
