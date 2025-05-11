import glob
import os
import socket
from typing import Optional, Union

from copy import deepcopy
import ray
import torch
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin
from loguru import logger
from ray.util.placement_group import placement_group as create_placement_group
from ray.util.placement_group import remove_placement_group
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers import get_scheduler as transformers_get_scheduler
from transformers.dynamic_module_utils import init_hf_modules
from transformers.generation.utils import GenerateDecoderOnlyOutput

from marl.config.config_consts import (ENGINE_PLUGIN_DDP, ENGINE_PLUGIN_DEEPSPEED,
                                    ENGINE_PLUGIN_FSDP)
from marl.config.config_utils import get_dp_size, get_gpu_requirement
from marl.modeling.builder import (build_critic_model, build_reward_model, 
                                   build_language_model)
from marl.policy_output import (PolicyOutput, concat_policy_outputs,
                             logprobs_from_logits)
from marl.tokenizer import get_tokenizer
from marl.utils import set_seed
from .dist_utils import init_process_group
from .generate_utils import (get_answer_str, get_question_answer_mask,
                             partition_by_micro_batch_size,
                             partition_list_by_micro_batch_size)
from .ray_actor_group import RayActorGroup
from .ray_actor_mixin import RayActorMixin
from .ray_utils import DEFAULT_NUM_CPUS, DEFAULT_NUM_GPUS, create_ray_actors
from ..config.config_utils import get_sp_size
from . import sp_util
from xtuner.model.modules import dispatch_modules
from xtuner.parallel.sequence import init_sequence_parallel
from .sp_loss_util import reduce_sequence_parallel_loss

from mmengine import MessageHub

DEFAULT_NEW_TOKENS = 64 # 默认新token数量
MAXIMUM_NEW_TOKENS = 1024 # 最大新token数量
"""
HfModelRunner可以单独被其他进程调用
HfModelRunnerRayActor被ModelServer通过.remote()调用
"""

# 基于HuggingFace的模型运行器基类
class HfModelRunner:
    """HfModelRunner is capable of training, inference, and generation."""

    def __init__(self, model_config):
        self.model_config: dict = model_config

    def initialize(self):
        # 环境设置
        envs = self.model_config.get('envs', {})
        for key, value in envs.items():
            os.environ[key] = value

        # Parallel Settings 并行设置
        parallel: dict = self.model_config['parallel']
        assert parallel['tensor']['size'] == 1  # TODO: support TP
        assert parallel['pipeline']['size'] == 1  # TODO: support PP
        self.update_step = 0
        self.zero_stage = 1
        mixed_precision = self.model_config.get('mixed_precision', None)
        if parallel['data'].get('mode') == ENGINE_PLUGIN_FSDP:
            self.accelerator = Accelerator(
                fsdp_plugin=FullyShardedDataParallelPlugin())
            self.zero_stage = 3
        elif parallel['data'].get('mode') == ENGINE_PLUGIN_DEEPSPEED:
            from accelerate import DeepSpeedPlugin

            ds_config = self.model_config['deepspeed_config']  # requisite——必要配置
            self.accelerator = Accelerator(
                deepspeed_plugin=DeepSpeedPlugin(ds_config))
            self.zero_stage = ds_config['zero_optimization']['stage']
        else:
            self.accelerator = Accelerator(mixed_precision=mixed_precision)
            self.zero_stage = 0
        # 序列并行设置
        self.sp_size = get_sp_size(self.model_config)
        logger.info(f'self.sp_size: {self.sp_size}')
        if self.sp_size > 1:
            init_sequence_parallel(self.sp_size)

        # 1. Model
        model_path = self.model_config.get('model_path')
        self.model_type = self.model_config.get('model_type', '').lower()
        torch_dtype = self.model_config.get('torch_dtype', 'auto')
        use_flash_attn = self.model_config.get('use_flash_attn', None)
        extra_kwargs = dict(
            device_map=None if self.zero_stage == 3 else 'auto',
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation='flash_attention_2'
            if use_flash_attn else None,
        )
        
        # 根据模型类型加载不同的模型
        if self.model_type == "critic":
            self.model = build_critic_model(
                model_path, 
                head_name=self.model_config.get('head_name', 'v_head'),
                two_linear=self.model_config.get('two_linear', False),
                extra_kwargs=extra_kwargs,
                exclude_keys=self.model_config.get('exclude_keys', []),
            )
        elif self.model_type == "reward":
            self.model = build_reward_model(
                model_path, 
                head_name=self.model_config.get('head_name', 'v_head'),
                two_linear=self.model_config.get('two_linear', False),
                extra_kwargs=extra_kwargs,
            )
        elif self.model_type == "reference" or self.model_type == "policy":
            self.model = build_language_model(
                model_path, 
                extra_kwargs=extra_kwargs,
            )
        else:
            raise ValueError(f'Unsupported model_type: {self.model_type}')

        # 如果zero_stage不是3，则将模型移动到GPU上
        if not self.zero_stage == 3:
            self.model.to("cuda")

        enable_xtuner_dispatch = self.model_config.get('enable_xtuner_dispatch', False)
        if enable_xtuner_dispatch:
            logger.info(f"invoking xtuner dispatch_modules for {self.model_type} model")
            dispatch_modules(self.model.model)

        # Graident checkpointing 梯度检查点
        gradient_checkpointing = self.model_config.get(
            'gradient_checkpointing', False)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.vocab_size = self.model.config.vocab_size

        # 2. Tokenizer加载
        tokenizer_path = self.model_config.get('tokenizer_path', model_path)
        tokenizer_config = self.model_config.get('tokenizer_config', {})
        self.tokenizer = get_tokenizer(
            tokenizer_path, trust_remote_code=True, **tokenizer_config)

        # 3. Trainer训练器设置
        train_kwargs = self.model_config.get('train_kwargs')
        if train_kwargs is None:  # requires no training
            self.model = self.accelerator.prepare(
                self.model) if self.zero_stage == 3 else self.model
            self.device = self.accelerator.device
            logger.info(
                f'[{self.model_type}] __init__() done without train_kwargs.')
            return
        optimizer_type = train_kwargs.get('optimizer', torch.optim.AdamW)
        learning_rate = train_kwargs.get('lr', 1e-5)
        self.clip_grad_norm = train_kwargs.get('clip_grad_norm', 1.0)
        self.optimizer: torch.optim.Optimizer = optimizer_type(
            params=self.model.parameters(),
            lr=learning_rate,
        )

        # 学习率调度器设置
        lr_scheduler_type = train_kwargs.get('lr_scheduler', 'linear')
        lr_scheduler_kwargs = train_kwargs.get(
            'lr_scheduler_kwargs',
            {
                'num_warmup_steps': 0,
                'num_training_steps': 10000000000
            },
        )
        self.lr_scheduler: _LRScheduler = transformers_get_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            **lr_scheduler_kwargs,
        )
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(  # noqa: E501
            self.model, self.optimizer, self.lr_scheduler)

        # resume optimizer, lr_scheduler
        if bool(len(glob.glob(os.path.join(model_path, '*.step')))):
            self._resume_load_pretrained(model_path=model_path)

        # Others
        self.device = self.accelerator.device
        set_seed(self.model_config.get('seed'))  # 设置随机种子以确保可重现性
        if mixed_precision is not None:
            logger.info(
                f'[{self.model_type}]: Enable mixed_precision = {mixed_precision}'  # noqa: E501
            )
        if gradient_checkpointing:
            logger.info(
                f'[{self.model_type}]: Enable gradient_checkpointing')
        logger.info(
            f'[{self.model_type}] __init__() done with optimizer {self.optimizer.optimizer}.'  # noqa: E501
        )

    def _resume_load_pretrained(self, model_path):
        _, step_pt = os.path.split(
            glob.glob(os.path.join(model_path, '*.step'))[0])
        self.update_step = int(step_pt.split('.step')[0])
        logger.info(f'Resume train step {self.update_step} from {model_path}')
        assert os.path.exists(os.path.join(model_path, 'saved_state'))
        self.accelerator.load_state(os.path.join(model_path, 'saved_state'))

    def compute_loss(  # 计算损失函数
        self,
        input_ids: torch.Tensor,
        labels: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        criterion: Optional[_Loss] = None,
        loss_weight: Optional[float] = None,
        loss_factor: Optional[float] = None,
        **_ignored,
    ) -> torch.Tensor:
        """计算模型的损失函数
        
        支持三种损失计算方式：
        1. 默认设置：使用模型内置的损失计算
        2. 使用预设损失函数:如torch.nn.CrossEntropyLoss
        3. 使用自定义损失函数:通过字典形式的标签和自定义criterion
        
        Args:
            input_ids: 输入的token IDs
            labels: 标签，可以是张量或字典
            attention_mask: 注意力掩码,指示哪些token应被关注
            position_ids: 位置编码
            criterion: 损失函数
            loss_weight: 损失权重，用于多任务学习
            loss_factor: 损失因子，用于缩放损失
            
        Returns:
            计算得到的损失值
        """
        input_ids = input_ids.to(self.device)
        labels = input_ids.clone() if labels is None else labels
        if attention_mask is not None:
            if position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
        batch = {
            'input_ids':
            input_ids,
            'attention_mask':
            attention_mask.to(self.device)
            if attention_mask is not None else None,
            'position_ids':
            position_ids.to(self.device) if position_ids is not None else None
        }
        self.model.train()

        if self.sp_size > 1:
            padding_value_dict = {'input_ids': self.tokenizer.pad_token_id, 'tensor_label': -100, 'default': 0}
            if isinstance(labels, dict):
                # origin_mask = deepcopy(labels['mask'])
                batch,  labels = sp_util.remove_paddings(batch, labels)
                labels = sp_util.labels_add_paddings(batch['input_ids'], labels)
                batch,  labels = sp_util.split_for_sp(batch, labels, padding_value_dict)
                labels = sp_util.labels_remove_paddings(labels)
            else:
                batch,  labels = sp_util.split_for_sp(batch, labels, padding_value_dict)

        if criterion is None:
            # OPT. A) Default settings
            assert isinstance(
                labels, torch.Tensor
            ), 'Please pass in `criterion` for non-tensor labels'
            labels = labels.to(self.device)
            batch['labels'] = labels
            fwd_output = self.model(**batch, use_cache=False)
            loss = fwd_output.loss
        elif isinstance(labels, torch.Tensor):
            # OPT. B) Use preset loss functions, e.g., torch.nn.CrossEntropyLoss()  # noqa: E501
            # Adopted from: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L1199  # noqa: E501
            logits: torch.Tensor = self.model(**batch, use_cache=False).logits
            labels = labels.to(self.device)
            loss = criterion(logits, labels, loss_factor)
        elif isinstance(labels, dict):
            # OPT. C) Use customized loss function, see loss/policy_loss.py
            logits: torch.Tensor = self.model(
                **batch, use_cache=False, return_dict=True).logits
            for k, v in labels.items():
                labels[k] = v.to(self.device)
            loss = criterion(logits, labels, loss_factor)
        else:
            raise ValueError(f'labels of unsupported type: {type(labels)}')
        
        loss_type = criterion.loss_type
        if self.sp_size > 1:
            if isinstance(labels, dict):
                loss_scale = labels['mask'].sum()
            else:
                loss_scale = (labels != -100).sum()

            if loss_type == 'per_token':
                loss_scale = torch.tensor(1).to(self.device)
            loss = reduce_sequence_parallel_loss(loss, loss_scale)

        if loss_weight is not None:
            loss *= loss_weight
        return loss

    def parameter_update(self, step_interval=1):
        """更新模型参数
        
        根据累积的梯度更新模型参数，包括梯度裁剪、优化器步进和学习率调整。
        
        Args:
            step_interval: 参数更新的间隔步数，用于梯度累积
        """
        logger.info(f'[{self.model_type}] self.parameter_update()')
        self.update_step += 1
        if self.update_step % step_interval == 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(),
                                             self.clip_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

    def train(
        self,
        input_ids: Union[list[torch.Tensor], torch.Tensor],
        labels: Optional[Union[list[torch.Tensor], torch.Tensor,
                               dict[str, torch.Tensor]]] = None,
        attention_mask: Optional[Union[list[torch.Tensor],
                                       torch.Tensor]] = None,
        position_ids: Optional[Union[list[torch.Tensor], torch.Tensor]] = None,
        criterion: Optional[Union[list[_Loss], _Loss]] = None,
        loss_weights: Optional[Union[list[float], float]] = None,
        loss_factor: Optional[Union[list[float], float]] = None,
        step_interval: int = 1,
        # None means using the entire input as one batch
        micro_batch_size: Optional[Union[list[int], int]] = None,
        cumulative_len: Optional[list[list[torch.Tensor]]] = None,
        max_seqlen: Optional[list[list[int]]] = None,
        debug=False,
        use_varlen_attn=False,
        **_ignored,
    ):
        """训练模型
        
        支持单批次和多批次训练，以及微批次处理。可以处理不同类型的输入和标签。
        
        Args:
            input_ids: 输入的token IDs,可以是单个张量或张量列表
            labels: 标签，可以是张量、张量列表或字典
            attention_mask: 注意力掩码
            position_ids: 位置编码
            criterion: 损失函数
            loss_weights: 损失权重
            loss_factor: 损失因子
            step_interval: 参数更新间隔
            micro_batch_size: 微批次大小,None表示使用整个输入作为一个批次
            cumulative_len: 累积长度，用于变长注意力
            max_seqlen: 最大序列长度
            debug: 是否启用调试模式
            use_varlen_attn: 是否使用变长注意力
            
        Returns:
            计算得到的损失值或损失值列表
        """
        if isinstance(input_ids, torch.Tensor):
            input_ids = [input_ids]
            labels = [labels]
            attention_mask = [attention_mask]
            position_ids = [position_ids]
            criterion = [criterion]
            loss_weights = [loss_weights]
            loss_factor = [loss_factor]
            micro_batch_size = [micro_batch_size]
            cumulative_len = [cumulative_len]
            max_seqlen = [max_seqlen]
        else:
            if attention_mask is None:
                attention_mask = [None for _ in range(len(input_ids))]
            if position_ids is None:
                position_ids = [None for _ in range(len(input_ids))]
            if criterion is None:
                criterion = [None for _ in range(len(input_ids))]
            if loss_weights is None:
                loss_weights = [None for _ in range(len(input_ids))]
            if loss_factor is None:
                loss_factor = [None for _ in range(len(input_ids))]
            if micro_batch_size is None:
                micro_batch_size = [None for _ in range(len(input_ids))]
            if cumulative_len is None:
                cumulative_len = [None for _ in range(len(input_ids))]
            if max_seqlen is None:
                max_seqlen = [None for _ in range(len(input_ids))]

        assert isinstance(input_ids, list)

        loss_list = [[] for _ in range(len(input_ids))]
        for index in range(len(input_ids)):
            varlen_attn_enable = (cumulative_len[index] is not None) and use_varlen_attn
            if varlen_attn_enable:
                dispatch_modules(self.model.model, use_varlen_attn=True)

            mb_size_entry = micro_batch_size[index]
            if mb_size_entry is None:
                micro_batches: list[dict[str, torch.Tensor]] = []
                micro_batches.append({
                    'input_ids': input_ids[index],
                    'attention_mask': attention_mask[index],
                    'position_ids': position_ids[index],
                    'labels': labels[index]
                })
            else:
                micro_batches = partition_by_micro_batch_size(
                    input_ids=input_ids[index],
                    micro_batch_size=micro_batch_size[index],
                    attention_mask=attention_mask[index],
                    position_ids=position_ids[index],
                    labels=labels[index],
                    cumulative_len = cumulative_len[index],
                    max_seqlen=max_seqlen[index],
                )
            loss_entry = []
            for mb_index, micro_batch in enumerate(micro_batches):
                logger.info(
                    f"[{self.model_type}] will train input_ids[{mb_index}] shape[{micro_batch['input_ids'].shape}]"  # noqa: E501
                )

                if varlen_attn_enable:
                    cumulative_len = micro_batch['cumulative_len']
                    max_seqlen = micro_batch['max_seqlen']
                    assert len(cumulative_len) == 1
                    rank = os.getenv('RANK', '0')
                    message_hub = MessageHub.get_instance('varlen_attn_args')
                    cumulative_len = cumulative_len[0].to(self.device)
                    message_hub.update_info(f'cumulative_len_rank_{rank}', cumulative_len)
                    message_hub.update_info(f'max_seqlen_rank_{rank}', max_seqlen[0])
                    logger.warning(f"verlen rank: {rank}, input_ids[{mb_index}] shape[{micro_batch['input_ids'].shape}], labels shape[{micro_batch['labels'].shape}]")

                # compute loss and backward
                # 计算损失并进行反向传播
                # 1. 使用compute_loss方法计算当前微批次的损失
                # 2. 损失计算考虑了自定义的损失函数(criterion)、损失权重(loss_weight)和损失因子(loss_factor)
                # 3. 支持多种损失计算方式，如交叉熵损失、KL散度等
                loss = self.compute_loss(
                    input_ids=micro_batch['input_ids'],
                    labels=micro_batch['labels'],
                    attention_mask=micro_batch['attention_mask'],
                    position_ids=micro_batch['position_ids'],
                    criterion=criterion[index],
                    loss_weight=loss_weights[index],
                    loss_factor=loss_factor[index]
                )
                self.accelerator.backward(loss)
                loss_entry.append(loss)
                if debug:
                    set_seed(1234)
                if varlen_attn_enable:
                    rank = os.getenv('RANK', '0')
                    message_hub = MessageHub.get_instance('varlen_attn_args')
                    message_hub.update_info(f'cumulative_len_rank_{rank}', None)
                    message_hub.update_info(f'max_seqlen_rank_{rank}', None)

            loss_list[index] = sum(loss_entry) / len(loss_entry)
            if varlen_attn_enable:
                dispatch_modules(self.model.model, use_varlen_attn=False)

        self.parameter_update(step_interval)
        return loss_list if len(loss_list) > 1 else loss_list[0]

    # Inference 单批次推理
    @torch.no_grad() # 不计算梯度
    def _infer(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        output_logprobs=True,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        infer_kwargs: Optional[dict] = {},
        **_ignored,
    ) -> PolicyOutput:
        assert isinstance(input_ids, torch.Tensor) # input_ids must be a Tensor
        position_ids = attention_mask.long().cumsum(-1) - 1 # 生成位置编码
        position_ids.masked_fill_(attention_mask == 0, 1) # 将attention_mask为0的位置填充为1
        # 执行模型前向推理
        model_output = self.model(
            input_ids.to(self.device),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids.to(self.device),
            return_dict=True,
            **infer_kwargs,
        )
        # deal with the output
        output = PolicyOutput()
        if output_logits:
            output['logits'] = model_output['logits']
        if output_attentions:
            output['attentions'] = model_output['attentions']
        if output_hidden_states:
            output['hidden_states'] = model_output['hidden_states']
        if output_logprobs:
            log_probs = logprobs_from_logits(
                logits=model_output['logits'][:, :-1, :],
                labels=input_ids[:, 1:],
                gather=True,
            )
            output['logprobs'] = log_probs
        output.to('cpu') # output change to cpu
        return output

    # 微批次推理接口
    @torch.no_grad()
    def infer(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: Optional[
            int] = -1,  # -1: use the entire input as one batch
        tokenizer=None,  # Only used for reward models
        attention_mask=None,
        output_logprobs=False,
        output_logits=True,
        output_attentions=False,
        output_hidden_states=False,
        infer_kwargs: Optional[dict] = {},
        debug=False,
        **_ignored,
    ) -> PolicyOutput:
        self.model.eval() # 设置模型为评估模式
        logger.info(
            f'[{self.model_type}] self.infer() kwargs: {infer_kwargs}')
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        # returns entire-input-as-one-batch inference results
        if micro_batch_size < 0: # 如果micro_batch_size小于0,则使用整个输入作为一批次进行推理
            logger.info(
                f'[{self.model_type}] infer() input_ids.shape: {input_ids.shape}'  # noqa: E501
            )
            return self._infer(
                input_ids,
                attention_mask,
                output_logprobs,
                output_logits,
                output_attentions,
                output_hidden_states,
                infer_kwargs,
            )

        # Otherwise, partition the input into micro batches and run inference on each micro batch separately  # 将输入分割成微批次,分别对每个微批次进行推理
        micro_batches = partition_by_micro_batch_size(input_ids,
                                                      micro_batch_size,
                                                      attention_mask)
        policy_outputs = []
        for index, micro_batch in enumerate(micro_batches): # 处理每个微批次
            input_ids_mb = micro_batch['input_ids']
            attention_mask_mb = micro_batch['attention_mask']
            if index == 0:
                logger.info(
                    f'[{self.model_type}] will infer() input_ids_mb.shape: {input_ids_mb.shape} * {len(micro_batches)} times'  # noqa: E501
                )
            policy_output_mb = self._infer(
                input_ids_mb,
                attention_mask_mb,
                output_logprobs,
                output_logits,
                output_attentions,
                output_hidden_states,
                infer_kwargs,
            )
            policy_outputs.append(policy_output_mb)
            if debug:
                self.set_seed(1234)
        # Concatenate the policy outputs from each micro batch and return the result  # 合并每个微批次的推理结果并返回
        return concat_policy_outputs(policy_outputs)

    # Generate 内部生成方法，处理单个批次的生成任务
    @torch.no_grad()
    def _generate(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        step=-1,
        output_str=True,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        output_logprobs=False,
        generate_kwargs: Optional[dict] = {},
    ) -> PolicyOutput:
        """
        Args:
            input_ids: 输入token IDs张量
            attention_mask: 注意力掩码,指定哪些token需要被关注
            step: 生成的最大token数,-1表示使用默认值
            output_str: 是否输出解码后的字符串
            output_logits: 是否输出logits
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            output_logprobs: 是否输出log概率
            generate_kwargs: 传递给模型generate方法的额外参数
            
        Returns:
            PolicyOutput对象,包含生成结果及相关信息
        """
        assert isinstance(input_ids, torch.Tensor)
        # 如果模型是DistributedDataParallel类型，需要解包获取原始模型
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model

        # 确定最大生成token数
        max_new_tokens = (
            MAXIMUM_NEW_TOKENS
            if 'eos_token_id' in generate_kwargs else DEFAULT_NEW_TOKENS)
        # 如果step > 0，则使用step作为最大生成token数
        max_new_tokens = step if step > 0 else max_new_tokens

        # TODO: stop if meeting eos_token_id 如果生成过程中遇到eos_token_id，则停止生成
        model_output: GenerateDecoderOnlyOutput = model.generate(
            input_ids.to(model.device),
            use_cache=True,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_logits=(output_logits or output_logprobs),  # transformers >= 4.38.2
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        output_ids = model_output['sequences']
        logger.info(
            f'generate input_ids shape:[{input_ids.shape}], output_ids shape:[{output_ids.shape}]'  # noqa: E501
        )
        
        # 创建PolicyOutput对象存储生成结果
        output = PolicyOutput(output_ids=output_ids)
        
        # 计算问题和回答的掩码
        # question_mask: 输入部分的掩码
        # answer_mask: 生成部分的掩码
        output['question_mask'], output[
            'answer_mask'] = get_question_answer_mask(
                input_ids,
                output_ids,
                tokenizer_pad_token_id=self.tokenizer.pad_token_id,
                generate_pad_token_id=generate_kwargs.get('pad_token_id'),
            )
        output['attention_mask'] = output.question_mask + output.answer_mask
        output['action_mask'] = output['attention_mask'][:,
                                                         input_ids.size(1):]
        output['input_ids'] = input_ids

        if output_logits:
            output['logits'] = model_output['logits']  # tuple(torch.Tensor, )
        if output_attentions:
            output['attentions'] = model_output['attentions']
        if output_hidden_states:
            output['hidden_states'] = model_output['hidden_states']
        # 计算生成token的log概率(用于RL训练)
        if output_logprobs:
            action_start = input_ids.shape[1]
            logprobs = []
            for i in range(len(model_output['logits'])):
                logits = model_output['logits'][i]
                ids = model_output['sequences'][:, action_start+i]
                logp = torch.nn.functional.log_softmax(logits, dim=1)
                logp = logp.gather(1, ids.view(-1, 1))
                logprobs.append(logp)
            # 合并所有步骤的log概率，并用0填充输入部分
            output['logprobs'] = torch.nn.functional.pad(
                torch.cat(logprobs, dim=1),
                (action_start, 0),
                value=0.0)
        if output_str:  # customized post processing 自定义后处理
            output['output_str'] = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            # 仅解码生成的回答部分
            output['output_ans_str'] = get_answer_str(
                tokenizer=self.tokenizer,
                output_ids=output_ids,
                answer_mask=output.answer_mask,
            )

        output.to('cpu')
        return output

    # Generate 支持微批次处理
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: Optional[
            int] = -1,  # -1: use the entire input as one batch
        attention_mask=None,
        step=-1,
        output_str=True,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        output_logprobs=False,
        chat_template=None,
        generate_kwargs: Optional[dict] = {},
        debug=False,
        **_ignored,
    ) -> PolicyOutput:
        """
        Args:
            input_ids: 输入token IDs张量
            micro_batch_size: 微批次大小，-1表示使用整个输入作为一个批次
            attention_mask: 注意力掩码
            step: 生成的最大token数
            output_str: 是否输出解码后的字符串
            output_logits: 是否输出logits
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            output_logprobs: 是否输出log概率
            chat_template: 聊天模板(保留参数)
            generate_kwargs: 传递给模型generate方法的额外参数
            debug: 是否启用调试模式
            
        Returns:
            PolicyOutput对象,包含生成结果及相关信息
        """
        logger.info(
            f'[{self.model_type}] self.generate() kwargs: {generate_kwargs}')
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            assert isinstance(attention_mask, torch.Tensor)
            attention_mask = attention_mask.to(self.device)

        # 如果micro_batch_size < 0，直接使用整个输入批次
        if micro_batch_size < 0:
            return self._generate(
                input_ids,
                attention_mask=attention_mask,
                step=step,
                output_str=output_str,
                output_logits=output_logits,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_logprobs=output_logprobs,
                generate_kwargs=generate_kwargs,
            )

        # 按照微批次大小分割输入数据
        micro_batches = partition_by_micro_batch_size(input_ids,
                                                      micro_batch_size,
                                                      attention_mask)
        policy_outputs = []
        
        # 处理每个微批次
        for micro_batch in micro_batches:
            input_ids_mb = micro_batch['input_ids']
            attention_mask_mb = micro_batch['attention_mask']
            policy_output_mb = self._generate(
                input_ids_mb,
                attention_mask=attention_mask_mb,
                step=step,
                output_str=output_str,
                output_logits=output_logits,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_logprobs=output_logprobs,
                generate_kwargs=generate_kwargs,
            )
            policy_outputs.append(policy_output_mb)
            if debug:
                self.set_seed(1234)

        # 合并所有微批次的输出结果
        # 指定padding token以正确填充output_ids
        padding_token_map = {'output_ids': self.tokenizer.pad_token_id}
        return concat_policy_outputs(policy_outputs, padding_token_map)

    def get_model(self): # 获取模型实例
        parallel: dict = self.model_config['parallel']
        dp = parallel['data'].get('size')
        dp_mode = parallel['data'].get('mode')
        if dp > 1 and dp_mode != ENGINE_PLUGIN_DDP:
            raise ('please use get_state_dict instead when using parallel')
        _model = self.accelerator.unwrap_model(self.model)
        return _model

    def get_state_dict(self): # 获取模型状态字典
        state_dict = self.accelerator.get_state_dict(self.model)
        if not self.accelerator.is_main_process:
            return None
        return state_dict

    def set_seed(self, seed=None): # 设置随机种子
        set_seed(seed)

    def save(self, path):
        # # for resume
        # self.accelerator.wait_for_everyone()
        # self.accelerator.save_state(os.path.join(path, 'saved_state'))

        # save model, tokenizer, step
        if not self.accelerator.is_main_process: # 如果不是主进程，则获取模型状态字典
            self.accelerator.get_state_dict(self.model)
            return
        else:
            path = os.path.normpath(path)
            logger.info(f'[Train step {self.update_step}] '
                        f'Saving {self.model_type} to {path} ...')
            # save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                path,
                is_main_process=True,
                save_function=self.accelerator.save,
                state_dict=self.accelerator.get_state_dict(self.model),
            )
            # save tokenizer
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(path)
            torch.save(self.update_step,
                       os.path.join(path, f'{self.update_step}.step')) # 保存更新步骤
            logger.info(f'{self.model_type} saved.')

    def info_rank0(self, content): # 在主进程打印信息
        if self.accelerator.is_main_process:
            logger.info(content)


# Adapted from https://github.com/OpenLLMAI/OpenRLHF/blob/v0.2.5/openrlhf/trainer/ray/ppo_actor.py  # noqa: E501
class HfModelRunnerRayActor(HfModelRunner, RayActorMixin):
    """A ray.remote Actor Class initialized by HfModelRunnerRayActorGroup,
    extending HfModelRunner with ray related method via RayActorMixin. 用于分布式训练的RayActor类,使用RayActorMixin扩展了与Ray相关的功能"""

    def init_process_group(self, generator):
        """初始化分布式训练的进程组
        
        Args:
            generator: 生成器对象，包含数据并行(DP)和张量并行(TP)的配置信息
        """
        if self.accelerator.is_main_process: # 获取主节点地址和空闲端口号用于进程间通信
            # init process groups for vllm engine 初始化vllm引擎的进程组
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(('', 0))
                master_port = sock.getsockname()[1]

            # 计算总进程数：数据并行数 * 张量并行数 + 1(主进程)
            world_size = generator.dp_size * generator.tp_size + 1
            # 为每个Ray Actor初始化进程组
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * generator.tp_size + 1,  # 计算每个进程的rank
                    world_size,
                    'vllm',
                ) for i, engine in enumerate(generator.ray_actors)
            ]
            # 初始化主进程的进程组，用于模型更新
            self._model_update_group = init_process_group(
                backend='nccl',
                init_method=f'tcp://{master_address}:{master_port}',
                world_size=world_size,
                rank=0,
                group_name='vllm',
            )
            ray.get(refs) # 等待所有进程组初始化完成

    def broadcast_model_to_generator(self, generator):
        """将模型权重广播到所有生成器进程
        
        Args:
            generator: 生成器对象，包含数据并行(DP)和张量并行(TP)的配置信息
        """
        # TODO: Support Pytorch FSDP.
        if self.model_config['parallel']['data'].get(
                'mode') == ENGINE_PLUGIN_FSDP:
            raise NotImplementedError('FSDP is not supported yet.')
        logger.info('Broadcast BEGIN')
        # 获取原始模型（移除加速器包装）
        model = self.accelerator.unwrap_model(self.model)
        for name, param in model.named_parameters():
            if self.accelerator.is_main_process: # 主进程更新模型权重
                shape = param.shape if self.zero_stage != 3 else param.ds_shape
                # 通知所有生成器进程更新权重信息
                for engine in generator.ray_actors:
                    engine.update_weight.remote(
                        name, dtype=param.dtype, shape=shape)

            # 根据不同的ZeRO优化阶段采用不同的广播策略
            if self.zero_stage != 3:
                if self.accelerator.is_main_process:
                    torch.distributed.broadcast(
                        param.data, 0, group=self._model_update_group)
            else:
                from deepspeed.runtime.zero.partition_parameters import \
                    GatheredParameters

                with GatheredParameters([param]):
                    if self.accelerator.is_main_process:
                        torch.distributed.broadcast(
                            param.data, 0, group=self._model_update_group)

        logger.info('Broadcast END')


class HfModelRunnerRayActorGroup(RayActorGroup):
    """HfModelRunnerRayActorGroup manages a list of HfModelRunnerRayActor
    create ray actors. 管理一个HfModelRunnerRayActor列表,创建Ray Actor"""

    # avoid ModuleNotFoundError: No module named 'transformers_modules'
    # refer to https://github.com/vllm-project/vllm/pull/871
    init_hf_modules()

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.released = True
        # 获取所需的GPU数量和并行配置
        num_gpus = get_gpu_requirement(config)
        self.dp_size = get_dp_size(config)
        self.sp_size = get_sp_size(config)
        bundles = [{
            'CPU': DEFAULT_NUM_CPUS,
            'GPU': DEFAULT_NUM_GPUS
        } for _ in range(num_gpus)]
        self.placement_group = create_placement_group(bundles)
        # 创建Ray Actor实例列表
        self.ray_actors: list[HfModelRunnerRayActor] = create_ray_actors(
            name_prefix=name,
            config=config,
            placement_group=self.placement_group,
            trainer_class=ray.remote(
                num_cpus=DEFAULT_NUM_CPUS,
                num_gpus=DEFAULT_NUM_GPUS)(HfModelRunnerRayActor),
        )
        self.released = False
        # 获取主节点地址和空闲端口号用于进程间通信
        master_ip = ray.get(self.ray_actors[0].get_metadata.remote()).node_ip
        master_port = ray.get(self.ray_actors[0].get_free_port.remote())
        # 为每个Ray Actor注入分布式环境变量
        ray.get([
            actor.inject_distribute_env.remote(
                master_ip=master_ip,
                master_port=master_port,
                rank_id=rank,
                world_size=len(self.ray_actors),
            ) for rank, actor in enumerate(self.ray_actors)
        ])
        self.initialize_ref = [
            actor.initialize.remote() for actor in self.ray_actors
        ]

    def initialize_get(self):
        """等待所有Actor完成初始化"""
        if self.initialize_ref is not None:
            ray.get(self.initialize_ref)
        else:
            logger.info(
                'self.initialize_get None, maybe self.generator==self.trainer')
        self.initialize_ref = None

    # Training
    def train_async(self, input_ids, labels, attention_mask, position_ids, micro_batch_size,
                    *args, **kwargs):
        """异步训练方法，支持单个张量和张量列表两种输入形式
        
        Args:
            input_ids: 输入token IDs
            labels: 标签数据
            attention_mask: 注意力掩码
            position_ids: 位置编码
            micro_batch_size: 微批次大小
        """
        if isinstance(input_ids, torch.Tensor): # 单个张量输入
            # loss_factor for per_token loss
            batch_size, seqlen = input_ids.shape
            valid_tokens = labels["mask"].sum().cuda()
            num_micro_bs = batch_size // micro_batch_size
            loss_factor = torch.tensor(num_micro_bs * self.dp_size * self.sp_size / valid_tokens.item())

            micro_dp_batch_size = input_ids.shape[0] // self.dp_size + (
                input_ids.shape[0] % self.dp_size > 0
            )  # round up division, i.e., math.ceil(a / b)
            micro_batches = partition_by_micro_batch_size(
                input_ids=input_ids,
                micro_batch_size=micro_dp_batch_size,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels)
            assert len(micro_batches) == self.dp_size
            object_refs = []
            num_dp_actors = len(self.ray_actors) // self.dp_size
            for dp_index, micro_batch in enumerate(micro_batches):
                for i in range(num_dp_actors):
                    rank = dp_index * num_dp_actors + i
                    object_ref = self.ray_actors[rank].train.remote(
                        input_ids=micro_batch['input_ids'],
                        attention_mask=micro_batch['attention_mask'],
                        position_ids=micro_batch['position_ids'],
                        labels=micro_batch['labels'],
                        loss_factor=loss_factor,
                        micro_batch_size=micro_batch_size,
                        *args,
                        **kwargs,
                    )
                    object_refs.append(object_ref) 
            return object_refs
        elif isinstance(input_ids, list): # 张量列表输入
            cumulative_len = kwargs.pop('cumulative_len')
            max_seqlen = kwargs.pop('max_seqlen')
            """a list of tensors whose training loss will be taken average."""
            assert isinstance(input_ids[0], torch.Tensor)
            loss_factors = [None for _ in range(len(input_ids))]
            for i in range(len(input_ids)):
                batch_size, seqlen = input_ids[i].shape
                if isinstance(labels[i], dict):
                    valid_tokens = labels[i]["mask"].sum().cuda()
                elif isinstance(labels[i], torch.Tensor):
                    valid_tokens = (labels[i] != -100).sum()
                num_micro_bs = batch_size // micro_batch_size[i]
                loss_factors[i] = torch.tensor(num_micro_bs * self.dp_size * self.sp_size / valid_tokens.item())

            micro_dp_batch_size = [i for i in range(len(input_ids))]
            for index, input_id in enumerate(input_ids):
                micro_dp_batch_size[index] = input_id.shape[0] // self.dp_size + (
                    input_id.shape[0] % self.dp_size > 0
                )  # round up division, i.e., math.ceil(a / b)
            micro_batches = partition_list_by_micro_batch_size(
                input_ids=input_ids,
                micro_batch_size=micro_dp_batch_size,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cumulative_len = cumulative_len,
                max_seqlen = max_seqlen,
            )
            assert len(micro_batches) == self.dp_size
            object_refs = []
            num_dp_actors = len(self.ray_actors) // self.dp_size
            for dp_index, micro_batch in enumerate(micro_batches):
                for i in range(num_dp_actors):
                    rank = dp_index * num_dp_actors + i
                    input_ids_mb = []
                    attention_mask_mb = []
                    position_ids_mb = []
                    labels_mb = []
                    cumulative_len_mb = []
                    max_seqlen_mb = []
                    for j in range(len(micro_batch)):
                        input_ids_mb.append(micro_batch[j]['input_ids'])
                        attention_mask_mb.append(micro_batch[j]['attention_mask'])
                        position_ids_mb.append(micro_batch[j]['position_ids'])
                        labels_mb.append(micro_batch[j]['labels'])
                        cumulative_len_mb.append(micro_batch[j]['cumulative_len'])
                        max_seqlen_mb.append(micro_batch[j]['max_seqlen'])
                    object_ref = self.ray_actors[rank].train.remote(
                        input_ids=input_ids_mb,
                        attention_mask=attention_mask_mb,
                        position_ids=position_ids_mb,
                        labels=labels_mb,
                        loss_factor=loss_factors,
                        cumulative_len=cumulative_len_mb,
                        max_seqlen=max_seqlen_mb,
                        micro_batch_size=micro_batch_size,
                        *args,
                        **kwargs,
                    )
                    object_refs.append(object_ref) 
            return object_refs

    def train_get(self, object_refs, timeout=None): # 获取训练结果
        losses = ray.get(object_refs, timeout=timeout)
        if isinstance(losses[0], list):
            p_loss = [sub_loss[0] for sub_loss in losses]
            pt_loss = [sub_loss[1] for sub_loss in losses]
            return [sum(p_loss) / len(p_loss), sum(pt_loss) / len(pt_loss)]
        else:
            return sum(losses) / len(losses)

    def train(self, *args, **kwargs):
        object_refs = self.train_async(*args, **kwargs)
        return self.train_get(object_refs)

    # Inference
    def infer_async(self, input_ids, attention_mask, *args, **kwargs):
        """异步推理方法，支持单个张量和张量列表两种输入形式
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
        """
        # temp add sp_size to dp_size when infer
        self.origin_dp_size = self.dp_size
        self.origin_sp_size = self.sp_size
        self.dp_size = self.dp_size * self.sp_size
        self.sp_size = 1

        micro_batch_size = input_ids.shape[0] // self.dp_size + (
            input_ids.shape[0] % self.dp_size > 0
        )  # round up division, i.e., math.ceil(a / b)
        micro_batches = partition_by_micro_batch_size(input_ids,
                                                      micro_batch_size,
                                                      attention_mask)
        assert len(micro_batches) == self.dp_size
        object_refs = []
        num_dp_actors = len(self.ray_actors) // self.dp_size
        for dp_index, micro_batch in enumerate(micro_batches):
            for i in range(num_dp_actors):
                rank = dp_index * num_dp_actors + i
                object_ref = self.ray_actors[rank].infer.remote(
                    input_ids=micro_batch['input_ids'],
                    attention_mask=micro_batch['attention_mask'],
                    *args,
                    **kwargs,
                )
                if i == 0:
                   object_refs.append(object_ref) 
        return object_refs

    def infer_get(self, object_refs, timeout=None):
        # Modify back dp_size & sp_size
        self.dp_size = self.origin_dp_size
        self.sp_size = self.origin_sp_size

        outputs = ray.get(object_refs, timeout=timeout)
        return concat_policy_outputs(outputs)

    def infer(self, *args, **kwargs):
        object_refs = self.infer_async(*args, **kwargs)
        return self.infer_get(object_refs)

    # Generation
    def generate_async(self, input_ids, attention_mask, *args, **kwargs):
        """异步生成方法，支持单个张量和张量列表两种输入形式
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
        """
        micro_batch_size = input_ids.shape[0] // self.dp_size + (
            input_ids.shape[0] % self.dp_size > 0
        )  # round up division, i.e., math.ceil(a / b)
        micro_batches = partition_by_micro_batch_size(input_ids,
                                                      micro_batch_size,
                                                      attention_mask)
        assert len(micro_batches) == self.dp_size
        return [
            self.ray_actors[index].generate.remote(
                input_ids=micro_batch['input_ids'],
                attention_mask=micro_batch['attention_mask'],
                *args,
                **kwargs,
            ) for index, micro_batch in enumerate(micro_batches)
        ]

    def generate_get(self, object_refs, timeout=None):
        outputs = ray.get(object_refs, timeout=timeout)
        padding_token_map = {
            'output_ids': self.config.tokenizer_config.pad_token_id
        }
        return concat_policy_outputs(outputs, padding_token_map)

    def generate(self, *args, **kwargs):
        object_refs = self.generate_async(*args, **kwargs)
        return self.generate_get(object_refs)

    # Others
    def get_model(self):
        """获取模型实例"""
        return self.ray_actors[0].get_model.remote()

    def get_state_dict(self):
        """获取模型状态字典"""
        state_dicts = [
            actor.get_state_dict.remote() for actor in self.ray_actors
        ]
        return state_dicts[0]

    def set_seed(self, seed=None):
        """设置随机种子"""
        ray.get([actor.set_seed.remote(seed) for actor in self.ray_actors])

    def release_resources(self):
        """release ray resources. 释放ray资源"""
        if self.released:
            return
        for actor in self.ray_actors:
            try:
                ray.kill(actor=actor, no_restart=True)
            except BaseException as exp:
                logger.error(f'failed to kill ray actor {actor}. {exp}')
        remove_placement_group(self.placement_group)
        self.released = True

    def save(self, path):
        ray.get([actor.save.remote(path) for actor in self.ray_actors])

    def init_process_group(self, generator): # 初始化进程组
        refs = [
            hfm.init_process_group.remote(generator)
            for i, hfm in enumerate(self.ray_actors)
        ]
        ray.get(refs)

    def broadcast_model_to_generator(self, generator: None): # 广播模型到生成器
        refs = [
            hfm.broadcast_model_to_generator.remote(generator)
            for i, hfm in enumerate(self.ray_actors)
        ]
        ray.get(refs)
