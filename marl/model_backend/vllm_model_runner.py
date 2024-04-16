import torch
from typing import Optional, Union

import vllm
from vllm import LLM, SamplingParams
from vllm.sampling_params import _SAMPLING_EPS

from marl.logger import init_logger
from marl import utils as marl_util
from marl.policy_output import PolicyOutput, concat_policy_outputs
from marl.model_backend.generate_utils import get_question_answer_mask
from .dist_utils import init_process_group
import deepspeed

logger = init_logger(__name__)


class VllmGenerator:

    def __init__(self, model_config) -> None:
        self.model_config: dict = model_config

    def initialize(self) -> None:
        model_path = self.model_config.get("model_path")
        torch_dtype = self.model_config.get("torch_dtype", "auto")
        tokenizer_path = self.model_config.get("tokenizer_path", model_path)
        parallel: dict = self.model_config.get("parallel")
        tensor_parallel_size = 1 if parallel is None else parallel["tensor"]["size"]

        # NOTE: In 0.2.7, vLLM made a major change to its architecture which move one worker into the driver process.
        # Driver process will manually set CUDA_VISIBLE_DEVICES before worker init. To avoid importing torch before
        # set CUDA_VISIBLE_DEVICES, we must defer monkey patch.
        # For more detail, see: https://github.com/vllm-project/vllm/pull/2221
        import vllm
        from vllm.worker import worker
        from marl.model_backend.vllm_worker_wrap import VllmWorkerWrap

        def _set_cuda_visible_devices(device_ids: list[int]):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
            from vllm.worker import worker
            from marl.model_backend.vllm_worker_wrap import VllmWorkerWrap

            worker.Worker = VllmWorkerWrap

        vllm.engine.llm_engine.set_cuda_visible_devices = _set_cuda_visible_devices
        worker.Worker = VllmWorkerWrap
        self.llm: LLM = vllm.LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            trust_remote_code=True,
            dtype=torch_dtype,
            swap_space=0,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.tokenizer = self.llm.get_tokenizer()

    @staticmethod
    def get_sampling_params_from_dict(generate_kwargs: dict) -> SamplingParams:
        sp = SamplingParams()
        for k, v in generate_kwargs.items():
            if k in sp.__dict__:
                sp.__dict__[k] = v
            elif k == "num_beams" and v > 1:
                sp.__dict__["use_beam_search"] = True

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
        step=-1,
        output_str=True,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        generate_kwargs: Optional[dict] = {},
        **_ignored,
    ) -> list[tuple[list[int], str]]:
        sp = VllmGenerator.get_sampling_params_from_dict(generate_kwargs)
        sp.max_tokens = step if step > 0 else None
        logger.info(f"[{self.__class__.__name__}] self.generate() SamplingParams: {sp}")

        if isinstance(inputs, torch.Tensor):
            if len(inputs.shape) == 2:  # e.g., [batch_size, seq_len]
                prompt = self.tokenizer.batch_decode(inputs)
            elif len(inputs.shape) == 1:  # e.g., [seq_len]
                prompt = self.tokenizer.decode(inputs)
            else:
                raise ValueError(f"Unsupported tensor inputs of shape({inputs.shape})")
            # TODO: use prompt_token_ids to accept torch.Tensor input

        elif isinstance(inputs, str):
            prompt = inputs  # str
        elif isinstance(inputs, list):
            if isinstance(inputs[0], str):
                prompt = inputs  # list[str]
            else:
                raise ValueError(f"Unsupported inputs[0] with type({type(inputs[0])})")
            # TODO: support list[dict] input, e.g., [{"role": "user", "content": prompt}]
        else:
            raise ValueError(f"Unsupported inputs with type({type(inputs)})")

        # Calling vllm's generate
        req_outputs = self.llm.generate(prompt, sampling_params=sp)

        policy_outputs = []
        for req_output in req_outputs:
            output = PolicyOutput()
            input_ids = req_output.prompt_token_ids
            output_ids = input_ids + req_output.outputs[0].token_ids  # concat
            output["question_mask"], output["answer_mask"] = get_question_answer_mask(
                torch.Tensor(input_ids).reshape(1, -1),  # [batch_size=1, max_seq_len]
                torch.Tensor(output_ids).reshape(1, -1),
                tokenizer_pad_token_id=self.tokenizer.pad_token_id,
                generate_pad_token_id=generate_kwargs.get("pad_token_id"),
            )
            output["attention_mask"] = output.question_mask + output.answer_mask  # OR
            output["output_ids"] = torch.tensor(output_ids).unsqueeze(0)
            if output_logits:
                raise NotImplementedError("TODO: output_logits")  # TODO
            if output_attentions:
                raise NotImplementedError("TODO: output_attentions")  # TODO
            if output_hidden_states:
                raise NotImplementedError("TODO: output_hidden_states")  # TODO
            if output_str:  # return list[str]
                output["output_ans_str"] = [req_output.outputs[0].text]
                output["output_str"] = [req_output.prompt + req_output.outputs[0].text]
            output.to("cpu")

            policy_outputs.append(output)

        padding_token_map = {"output_ids": self.tokenizer.pad_token_id}
        concated_policy_out = concat_policy_outputs(policy_outputs, padding_token_map)
        return concated_policy_out


########################################################################################
########################################################################################

import os
import ray
from ray.util.placement_group import (
    remove_placement_group,
    placement_group as create_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from .ray_utils import create_ray_actors
from .ray_actor_mixin import RayActorMixin
from .ray_utils import DEFAULT_NUM_CPUS, DEFAULT_NUM_GPUS
from ..config_utils import get_gpu_requirement, get_tp_size, get_dp_size
from ..policy_output import concat_policy_outputs


class VllmGeneratorRayActor(VllmGenerator, RayActorMixin):
    def __init__(self, model_config) -> None:
        self.model_config: dict = model_config

    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name
    ):
        return self.llm.llm_engine._run_workers(
            "init_process_group",
            master_address,
            master_port,
            rank_offset,
            world_size,
            group_name,
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.llm_engine._run_workers(
            "update_weight", name, dtype, shape, empty_cache
        )


class VllmGeneratorRayActorGroup:
    def __init__(self, name: str, config: dict):
        self.released = True
        self.tp_size = get_tp_size(config)  # tensor parallelism
        self.dp_size = get_dp_size(config)  # num of vllm_engines
        # assert dp_size == 1  # TODO: multiple vllm engines
        assert self.tp_size == 1  # TODO: tp
        self.tokenizer_pad_token_id = config.get("tokenizer_pad_token_id", 0)

        self.ray_actors: list[VllmGeneratorRayActor] = []  # i.e., vllm_engines
        for dp_i in range(self.dp_size):
            ray_actor_num_gpus = int(self.tp_size == 1)

            bundles = [
                {"CPU": DEFAULT_NUM_CPUS, "GPU": DEFAULT_NUM_GPUS}
            ] * self.tp_size
            self.placement_group = create_placement_group(bundles)

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            )

            self.ray_actors.append(
                ray.remote(VllmGeneratorRayActor)
                .options(
                    name=f"{name}_rank_{dp_i}",
                    namespace=f"{VllmGeneratorRayActor.__class__.__name__}",
                    num_cpus=1,
                    num_gpus=ray_actor_num_gpus,
                    scheduling_strategy=scheduling_strategy,
                )
                .remote(config)
            )

        self.released = False
        self.initialize_ref = [actor.initialize.remote() for actor in self.ray_actors]

    def init_get(self):
        shared_with_trainer = self.config.get("shared_with_trainer", False)
        if shared_with_trainer:
            assert self.initialize_ref is None
            return  # assuming trainer.init_get() has been called
        if self.initialize_ref is not None:
            ray.get(self.initialize_ref)
        else:
            logger.warning("self.initialize_ref is None when calling init_get()")
        self.initialize_ref = None

    # Generation
    def generate_async(self, input_ids, attention_mask, *args, **kwargs):
        return [
            actor.generate.remote(
                input_ids, attention_mask=attention_mask, *args, **kwargs
            )
            for actor in self.ray_actors
        ]

    def generate_get(self, object_refs, timeout=None):
        outputs = ray.get(object_refs, timeout=timeout)
        padding_token_map = {"output_ids": self.tokenizer_pad_token_id}
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
        """
        release ray resources.

        """
        if self.released:
            return
        for actor in self.ray_actors:
            try:
                ray.kill(actor=actor, no_restart=True)
            except BaseException as exp:
                logger.error(f"failed to kill ray actor {actor}. {exp}")
        remove_placement_group(self.placement_group)
        self.released = True
