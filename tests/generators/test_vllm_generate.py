import pytest
from marl.timer import Timer
from ..conftest import *

MODELS = [
    # "facebook/opt-350m",
    "internlm/internlm2-chat-1_8b-sft",
]


def example_role_prompts(batch_size=8) -> list[list[dict]]:
    prompts = example_prompts()
    if len(prompts) > batch_size:
        multiply = batch_size // len(prompts) + 1
        prompts = prompts * multiply
    prompts = prompts[:batch_size]

    return prompts


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [10])
# @pytest.mark.skip()
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    example_prompts_64 = example_prompts * 1
    with Timer("hf_model generate") as hf_t:
        hf_outputs = hf_model.generate_greedy(example_prompts_64, max_tokens)
    # print("hf_outputs:", hf_outputs)
    del hf_model

    with Timer("vllm_model generate") as vllm_t:
        vllm_model = vllm_runner(model, dtype=dtype)
        vllm_duration = t.duration
    vllm_outputs = vllm_model.generate_greedy(example_prompts_64, max_tokens)
    # print("vllm_outputs:", vllm_outputs)
    del vllm_model

    print(f"hf_duration = {hf_t.duration} sec; vllm_duration = {vllm_t.duration} sec.")
    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert (
            hf_output_str == vllm_output_str
        ), f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}"
        assert (
            hf_output_ids == vllm_output_ids
        ), f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [2048])
@pytest.mark.skip()
def test_generate(vllm_runner, model: str, dtype: str, max_tokens: int):
    max_answer_len = max_tokens
    random_prompt = "请输出 10000 遍 hello"
    actor_model = vllm_runner(model, dtype=dtype)
    # TODO
