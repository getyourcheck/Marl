import os
from typing import Optional, Union

import ray
import torch
from loguru import logger
from ray.util.placement_group import placement_group as create_placement_group
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
from vllm.sampling_params import _SAMPLING_EPS

from ..config.config_utils import get_dp_size, get_tp_size
from ..policy_output import PolicyOutput, concat_policy_outputs
from .generate_utils import (get_question_answer_mask,
                             partition_by_micro_batch_size)
from .ray_actor_group import RayActorGroup
from .ray_actor_mixin import RayActorMixin
from .ray_utils import DEFAULT_NUM_CPUS, DEFAULT_NUM_GPUS, set_runtime_env

VLLM_DEFAULT_DEVICE = 'cuda'
import vllm
vllm_version = vllm.__version__


class VllmGenerator:

    def __init__(self, model_config) -> None:
        self.model_config: dict = model_config

        self.model_path = self.model_config.get('model_path')
        self.torch_dtype = self.model_config.get('torch_dtype', 'auto')
        self.tokenizer_path = self.model_config.get('tokenizer_path', self.model_path)
        parallel: dict = self.model_config.get('parallel')
        self.tensor_parallel_size = 1 if parallel is None else parallel['tensor'][
            'size']

        self.distributed_executor_backend = None
        if (vllm_version >= '0.6.1') and (self.tensor_parallel_size != 1):
            self.distributed_executor_backend = 'ray'

    # Adapted from https://github.com/OpenLLMAI/OpenRLHF/blob/v0.2.5/openrlhf/trainer/ray/vllm_engine.py  # noqa: E501
    def initialize(self, *args, **kwargs) -> None:

        if '0.2.7' <= vllm_version <= '0.3.3' and self.tensor_parallel_size != 1:  # noqa: E501
            # NOTE: In 0.2.7, vLLM made a major change to its architecture which move one worker into the driver process.  # noqa: E501
            # Driver process will manually set CUDA_VISIBLE_DEVICES before worker init. To avoid importing torch before  # noqa: E501
            # set CUDA_VISIBLE_DEVICES, we must defer monkey patch.
            # For more detail, see: https://github.com/vllm-project/vllm/pull/2221  # noqa: E501
            def _set_cuda_visible_devices(device_ids: list[int]):
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
                    map(str, device_ids))
                from vllm.worker import worker
                from .vllm_worker_wrap import VllmWorkerWrap

                worker.Worker = VllmWorkerWrap

            vllm.engine.llm_engine.set_cuda_visible_devices = _set_cuda_visible_devices  # noqa: E501
        elif (vllm_version >= '0.6.1') and (self.tensor_parallel_size != 1):
            self.distributed_executor_backend = 'ray'

            from vllm.worker import worker
            from .vllm_worker_wrap import VllmWorkerWrap

            worker.Worker = VllmWorkerWrap
            
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            RayWorkerWrapperPath = vllm.executor.ray_utils
            class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                def __init__(self, *args, **kwargs) -> None:
                    kwargs["worker_module_name"] = "marl.model_backend.vllm_worker_wrap"
                    kwargs["worker_class_name"] = "VllmWorkerWrap"
                    super().__init__(*args, **kwargs)

            RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

            # raise NotImplementedError
        else:
            from vllm.worker import worker
            from .vllm_worker_wrap import VllmWorkerWrap
            worker.Worker = VllmWorkerWrap

        self.llm: LLM = vllm.LLM(
            model=self.model_path,
            tokenizer=self.tokenizer_path,
            trust_remote_code=True,
            dtype=self.torch_dtype,
            swap_space=0,
            tensor_parallel_size=self.tensor_parallel_size,
            device=VLLM_DEFAULT_DEVICE,
            distributed_executor_backend=self.distributed_executor_backend,
        )
        self.tokenizer = self.llm.get_tokenizer()
        tokenizer_config = self.model_config.get('tokenizer_config', {})
        for key, value in tokenizer_config.items():
            setattr(self.tokenizer, key, value)
        self.tokenizer.add_special_tokens({'pad_token' : self.tokenizer.pad_token})

    @staticmethod
    def get_sampling_params_from_dict(generate_kwargs: dict) -> SamplingParams:
        sp_kwargs = {}
        tmp_sp = SamplingParams()
        for k, v in generate_kwargs.items():
            if k in dir(tmp_sp):
                sp_kwargs[k] = v
            elif k == 'num_beams' and v > 1:
                sp_kwargs['use_beam_search'] = True
            elif k == 'eos_token_id':
                if isinstance(v, int):
                    sp_kwargs['stop_token_ids'] = [v]
                elif isinstance(v, list):
                    sp_kwargs['stop_token_ids'] = v
                else:
                    raise NotImplementedError
        sp_kwargs['skip_special_tokens'] = True
        sp = SamplingParams(**sp_kwargs)

        sp.top_k = -1 if sp.top_k <= 1 else sp.top_k
        sp._verify_args()

        if sp.use_beam_search:
            sp._verify_beam_search()
        else:
            sp.early_stopping = False
            sp._verify_non_beam_search()
            if sp.temperature < _SAMPLING_EPS:
                # Zero temperature means greedy sampling.
                sp.top_p = 1.0
                sp.top_k = -1
                sp.min_p = 0.0
                sp._verify_greedy_sampling()
        return sp

    def generate(
        self,
        inputs: Union[torch.Tensor, str, list[str]],
        max_inputs_length: int,
        step=-1,
        output_str=True,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        output_logprobs=False,
        generate_kwargs: Optional[dict] = {},
        **_ignored,
    ) -> list[tuple[list[int], str]]:
        sp = VllmGenerator.get_sampling_params_from_dict(generate_kwargs)
        sp.max_tokens = step if step > 0 else None
        if output_logprobs:
            sp.logprobs = max(0, sp.logprobs or 0)
        logger.info(
            f'[{self.__class__.__name__}] self.generate() SamplingParams: {sp}'
        )

        if isinstance(inputs, torch.Tensor):
            if len(inputs.shape) == 2:  # e.g., [batch_size, seq_len]
                prompt = self.tokenizer.batch_decode(
                    inputs,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            elif len(inputs.shape) == 1:  # e.g., [seq_len]
                prompt = self.tokenizer.decode(
                    inputs,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            else:
                raise ValueError(
                    f'Unsupported tensor inputs of shape({inputs.shape})')

        elif isinstance(inputs, str):
            prompt = inputs  # str
        elif isinstance(inputs, list):
            if isinstance(inputs[0], list):
                prompt = inputs  # list[int]
            else:
                raise ValueError(
                    f'Unsupported inputs[0] with type({type(inputs[0])})')
        else:
            raise ValueError(f'Unsupported inputs with type({type(inputs)})')

        # Calling vllm's generate
        req_outputs = self.llm.generate(
            prompt_token_ids=prompt, sampling_params=sp)

        def pad_list_with_pad_token(int_list, max_length, pad_token_id):
            if len(int_list) < max_length:
                num_pad_token_to_add = max_length - len(int_list)
                padded_list = [pad_token_id] * num_pad_token_to_add + int_list
                return padded_list
            else:
                return int_list

        policy_outputs = []
        for _, req_output in enumerate(req_outputs):
            output = PolicyOutput()
            input_ids = [item for item in req_output.prompt_token_ids]
            input_ids = pad_list_with_pad_token(input_ids, max_inputs_length,
                                                self.tokenizer.pad_token_id)
            output_token_ids = [
                item for item in req_output.outputs[0].token_ids
            ]
            finish_reasons = [req_output.outputs[0].finish_reason]
            output['finish_reasons'] = finish_reasons
            output_ids = input_ids + output_token_ids  # concat
            output['input_ids'] = torch.Tensor(input_ids).to(
                torch.long).unsqueeze(0)
            output['output_ids'] = torch.tensor(output_ids).to(
                torch.long).unsqueeze(0)

            if self.tokenizer.pad_token_id != generate_kwargs.get('pad_token_id', self.tokenizer.pad_token_id):
                logger.warning(f"tokenizer.pad_token_id != generate_kwargs['pad_token_id'], using tokenizer.pad_token_id to pad")
            output['question_mask'], output[
                'answer_mask'] = get_question_answer_mask(
                    output['input_ids'],
                    output['output_ids'],
                    tokenizer_pad_token_id=self.tokenizer.pad_token_id,
                    # generate_pad_token_id=generate_kwargs.get('pad_token_id'),
                )
            output[
                'attention_mask'] = output.question_mask + output.answer_mask  # noqa: E501
            output['action_mask'] = output[
                'attention_mask'][:, max_inputs_length:]
            if output_logits:
                raise NotImplementedError('TODO: output_logits')
            if output_attentions:
                raise NotImplementedError('TODO: output_attentions')
            if output_hidden_states:
                raise NotImplementedError('TODO: output_hidden_states')
            if output_logprobs:
                logprobs = [0.0] * max_inputs_length
                for output_logprob in req_output.outputs[0].logprobs:
                    # Typically, the first element in dict is the chosen token
                    value = list(output_logprob.values())[0]
                    if type(value).__name__ == 'Logprob':
                        logprob = value.logprob
                    elif isinstance(value, float):
                        logprob = value
                    logprobs.append(logprob)
                output['logprobs'] = torch.tensor(logprobs).unsqueeze(0)
            if output_str:  # return list[str]
                output['output_ans_str'] = [req_output.outputs[0].text]
                output_str = self.tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                output['output_str'] = [output_str]
            output.to('cpu')

            policy_outputs.append(output)

        padding_token_map = {'output_ids': self.tokenizer.pad_token_id}
        concated_policy_out = concat_policy_outputs(policy_outputs,
                                                    padding_token_map)
        return concated_policy_out


class VllmGeneratorRayActor(VllmGenerator, RayActorMixin):

    # Adapted from https://github.com/OpenLLMAI/OpenRLHF/blob/v0.2.5/openrlhf/trainer/ray/vllm_engine.py  # noqa: E501
    def init_process_group(self, master_address, master_port, rank_offset,
                           world_size, group_name):

        if vllm_version >= '0.6.1':
            if self.distributed_executor_backend is None:
                return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                    master_address, master_port, rank_offset, world_size, group_name
                )
            elif self.distributed_executor_backend == 'ray':
                return self.llm.llm_engine.model_executor._run_workers(
                    "init_process_group", master_address, master_port, rank_offset, world_size, group_name
                )
            else:
                raise NotImplementedError
        elif '0.2.7' <= vllm_version <= '0.3.3':
            return self.llm.llm_engine._run_workers(
                'init_process_group',
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
            )
        else:
            raise NotImplementedError

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if vllm_version > "0.6.1":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.stop_remote_worker_execution_loop()

        if vllm_version >= '0.6.1':
            if self.distributed_executor_backend is None:
                return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
            elif self.distributed_executor_backend == 'ray':
                return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)
            else:
                raise NotImplementedError
        elif '0.2.7' <= vllm_version <= '0.3.3':
            return self.llm.llm_engine._run_workers('update_weight', name, dtype, shape, empty_cache)
        else:
            raise NotImplementedError


class VllmGeneratorRayActorGroup(RayActorGroup):

    def __init__(self, name: str, config: dict):
        import uuid
        self.released = True
        self.config = config
        self.tp_size = get_tp_size(config)  # tensor parallelism
        self.dp_size = get_dp_size(config)  # num of vllm_engines
        self.ray_actors: list[VllmGeneratorRayActor] = []  # i.e., vllm_engines

        # Adapted from https://github.com/OpenLLMAI/OpenRLHF/blob/v0.2.5/openrlhf/trainer/ray/vllm_engine.py  # noqa: E501
        for dp_i in range(self.dp_size):
            ray_actor_num_gpus = int(self.tp_size == 1)
            scheduling_strategy = None

            if self.tp_size > 1:
                bundles = [{
                    'CPU': DEFAULT_NUM_CPUS,
                    'GPU': DEFAULT_NUM_GPUS
                }] * self.tp_size
                self.placement_group = create_placement_group(bundles)
                ray.get(self.placement_group.ready())

                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=self.placement_group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                )

            namespace = f'{uuid.uuid4()}_{VllmGeneratorRayActor.__class__.__name__}'  # noqa: E501
            self.ray_actors.append(
                ray.remote(VllmGeneratorRayActor).options(
                    name=f'{name}_rank_{dp_i}',
                    namespace=namespace,
                    num_cpus=1,
                    num_gpus=ray_actor_num_gpus,
                    scheduling_strategy=scheduling_strategy,
                    runtime_env=set_runtime_env(),
                ).remote(config))

        self.released = False
        self.initialize_ref = [
            actor.initialize.remote() for actor in self.ray_actors
        ]

    def initialize_get(self):
        shared_with_trainer = self.config.get('shared_with_trainer', False)
        if shared_with_trainer:
            assert self.initialize_ref is None
            return  # assuming trainer.initialize_get() has been called
        if self.initialize_ref is not None:
            ray.get(self.initialize_ref)
        else:
            logger.warning(
                'self.initialize_ref is None when calling initialize_get()')
        self.initialize_ref = None

    # Generation
    def generate_async(self, input_ids, attention_mask, *args, **kwargs):
        assert (
            len(input_ids) >= self.dp_size
        ), f'The length of input_ids({len(input_ids)}) must not be less than dp_size({self.dp_size}).'  # noqa: E501
        micro_batch_size = len(input_ids) // self.dp_size + (
            len(input_ids) % self.dp_size > 0
        )  # round up division, i.e., math.ceil(a / b)
        micro_batches = partition_by_micro_batch_size(input_ids,
                                                      micro_batch_size,
                                                      attention_mask)
        assert len(micro_batches
                   ) == self.dp_size, f'{len(micro_batches)}, :{self.dp_size}'
        return [
            self.ray_actors[index].generate.remote(
                inputs=micro_batch['input_ids'],
                max_inputs_length=micro_batch['max_inputs_length'],
                attention_mask=micro_batch['attention_mask'],
                *args,
                **kwargs,
            ) for index, micro_batch in enumerate(micro_batches)
        ]

    def generate_get(self, object_refs, timeout=None):
        outputs = ray.get(object_refs, timeout=timeout)
        padding_token_map = {
            'output_ids': self.config.tokenizer_config['pad_token_id']
        }
        return concat_policy_outputs(outputs, padding_token_map)

    def generate(self, *args, **kwargs):
        object_refs = self.generate_async(*args, **kwargs)
        return self.generate_get(object_refs)

    # Others
    def get_model(self):
        return self.ray_actors[0].get_model.remote()

    def set_seed(self, seed=None):
        ray.get([actor.set_seed.remote(seed) for actor in self.ray_actors])

    def release_resources(self):
        """release ray resources."""
        if self.released:
            return
        for actor in self.ray_actors:
            try:
                ray.kill(actor=actor, no_restart=True)
            except BaseException as exp:
                logger.error(f'failed to kill ray actor {actor}. {exp}')
        remove_placement_group(self.placement_group)
        self.released = True
