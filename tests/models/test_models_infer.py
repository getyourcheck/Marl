"""Compare the outputs of HF and vLLM when using greedy sampling.

# Run `pytest tests/models/test_models.py --forked`.
Run `pytest tests/models/test_models_infer.py`.
"""

import pytest
import torch
from marl.config import Config
from marl.config_consts import ENGINE_HUGGINGFACE
from marl.tokenizer.tokenizer_utils import get_tokenizer
from marl.model_backend.hf_model_runner import HfModelRunner
from marl.utils import set_seed

MODELS = [
    "facebook/opt-350m",
    # "facebook/opt-125m", "meta-llama/Llama-2-7b-hf",
    # "mistralai/Mistral-7B-v0.1", "Deci/DeciLM-7b", "tiiuae/falcon-7b", "gpt2",
    # "bigcode/tiny_starcoder_py", "EleutherAI/gpt-j-6b",
    # "EleutherAI/pythia-70m", "bigscience/bloom-560m", "mosaicml/mpt-7b",
    # "microsoft/phi-2", "stabilityai/stablelm-3b-4e1t"
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(model, dtype=dtype)
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert (
            hf_output_str == vllm_output_str
        ), f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}"
        assert (
            hf_output_ids == vllm_output_ids
        ), f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}"


def test_generate():
    set_seed(1234)
    trainer_config = Config(
        dict(
            model_path="internlm/internlm2-chat-1_8b-sft",
            torch_dtype=torch.bfloat16,
            model_type="actor",
            trainer_type=ENGINE_HUGGINGFACE,
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_token",
            ),
            parallel=dict(
                data=dict(size=1),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
            ),
        ),
    )
    model_path = trainer_config.get("model_path")
    tokenizer_path = trainer_config.get("tokenizer_path", model_path)
    tokenizer = get_tokenizer(tokenizer_path, trust_remote_code=True)
    runner = HfModelRunner(model_config=trainer_config)
    runner.initialize()
    input_strs = ["你好", "请提供三个管理时间的建议。"]
    input_strs = [[{"role": "user", "content": input_str}] for input_str in input_strs]
    generate_kwargs = {
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.9,
        "min_new_tokens": 1,
        "num_beams": 1,
        "early_stopping": True,
        "eos_token_id": 92542,
        "pad_token_id": 0,
    }

    policy_output = runner.generate(
        input_strs,
        step=256,
        output_logits=False,
        output_str=True,
        generate_kwargs=generate_kwargs,
    )
    output_ids = policy_output.output_ids
    question_mask = policy_output.question_mask
    answer_mask = policy_output.answer_mask

    expected_hello = "非常好，有什么需要帮助的吗？"
    if "V100" in torch.cuda.get_device_name(0):
        expected_advice = """以下是三个管理时间的建议：

1. 制定计划和时间表：在开始一天之前，制定一个计划和时间表，列出您要完成的任务和优先级，以便您可以更好地组织您的时间和资源，并在一天结束时回顾您的计划。

2. 学会说“不”：学会拒绝一些不必要的任务和活动，以确保您的时间得到更好地利用。这不仅可以帮助您减少压力，还可以确保您有足够的时间来完成最重要的任务。

3. 减少分心：避免分心，专注于您正在做的事情，这将帮助您更快地完成任务。关闭电子邮件通知、社交媒体和其他不必要的通知，以便您可以更好地集中注意力。"""
    else:
        expected_advice = """好的，以下是三个关于管理时间的建议：

1. 制定计划和时间表： 计划是管理时间的重要组成部分，可以帮助您确定每项任务的重要性和优先级，并帮助您更好地安排时间。制定计划和时间表可以帮助您更好地掌控时间，确保您有足够的时间来完成任务，同时避免浪费时间和任务。

2. 设定优先级： 在制定计划和时间表中，应该根据任务的重要性和紧急程度设定优先级。将任务分为紧急且重要、紧急但不重要、重要但不紧急和不重要不紧急四个类别，并根据重要性确定优先级。这有助于您更好地利用时间，确保最重要和最紧急的任务得到优先处理。

3. 学会拒绝： 有时候，我们会被别人请求帮助或任务安排所迫，导致我们不能完成任务。要学会拒绝别人的请求，以便您有更多的时间来完成自己的任务。学会说“不”是一种很重要的技能，可以帮助您更好地掌控自己的时间，确保您有足够的时间来完成任务，并避免将时间浪费在不必要的事情上。"""
    expected = [expected_hello, expected_advice]
    
    input_strs = [
        tokenizer.apply_chat_template(
            input_str,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        for input_str in input_strs
    ]
    input_ids = tokenizer(input_strs, return_tensors="pt", padding=True).input_ids
    question_len = input_ids.shape[-1]
    answer_1_len = tokenizer([expected[0]], return_tensors="pt", padding=True).input_ids.shape[-1]
    output_len = output_ids.shape[-1]
    assert (
        tokenizer.decode(output_ids[0][question_len : question_len + answer_1_len - 1]) == expected[0]
    ), f"expected: {expected[0]}\noutput: {tokenizer.decode(output_ids[0][question_len : question_len + answer_1_len])}"
    assert (
        tokenizer.decode(output_ids[1][question_len : -1]) == expected[1]
    ), f"expected: {expected[1]}\noutput: {tokenizer.decode(output_ids[1][question_len : -1])}"

    # check question_mask
    assert torch.equal(
        question_mask[0][:6], torch.zeros(6, dtype=int, device=question_mask.device)
    ), f"expected: {torch.zeros(6,dtype=int)}\noutput: {question_mask[0][:6]}"  # padding
    assert torch.equal(
        question_mask[0][6 : question_len],
        torch.ones(question_len - 6, dtype=int, device=question_mask.device),
    ), f"expected: {torch.ones(question_len - 6, dtype=int)}\noutput: {question_mask[0][6 : question_len]}"  # question
    assert torch.equal(
        question_mask[0][question_len :],
        torch.zeros(output_len - question_len, dtype=int, device=question_mask.device),
    ), f"expected: {torch.zeros(output_len - question_len, dtype=int)}\noutput: {question_mask[0][question_len :]}"  # answer

    assert torch.equal(
        question_mask[1][: question_len],
        torch.ones(question_len, dtype=int, device=question_mask.device),
    ), f"expected: {torch.ones(question_len, dtype=int)}\noutput: {question_mask[1][: question_len]}"  # question
    assert torch.equal(
        question_mask[1][question_len :],
        torch.zeros(output_len - question_len, dtype=int, device=question_mask.device),
    ), f"expected: {torch.zeros(output_len - question_len, dtype=int)}\noutput: {question_mask[1][question_len :]}"  # answer

    # check answer_mask
    assert torch.equal(
        answer_mask[0][: question_len],
        torch.zeros(question_len, dtype=int, device=answer_mask.device),
    ), f"expected: {torch.zeros(question_len, dtype=int)}\noutput: {answer_mask[0][: question_len]}"  # question
    assert torch.equal(
        answer_mask[0][question_len : question_len + answer_1_len],
        torch.ones(answer_1_len, dtype=int, device=answer_mask.device),
    ), f"expected: {torch.ones(answer_1_len, dtype=int)}\noutput: {answer_mask[0][question_len : question_len + answer_1_len]}"  # answer
    assert torch.equal(
        answer_mask[0][question_len + answer_1_len:], torch.zeros(output_len - question_len - answer_1_len, dtype=int, device=answer_mask.device)
    ), f"expected: { torch.zeros(output_len - question_len - answer_1_len, dtype=int)}\noutput: {answer_mask[0][question_len + answer_1_len:]}"  # padding

    assert torch.equal(
        answer_mask[1][: question_len],
        torch.zeros(question_len, dtype=int, device=answer_mask.device),
    ), f"expected: {torch.zeros(question_len, dtype=int)}\noutput: {answer_mask[1][: question_len]}"  # question
    assert torch.equal(
        answer_mask[1][question_len :],
        torch.ones(output_len - question_len, dtype=int, device=answer_mask.device),
    ), f"expected: {torch.ones(output_len - question_len, dtype=int)}\noutput: {answer_mask[1][question_len :]}"  # answer
