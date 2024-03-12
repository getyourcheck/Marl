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
            torch_dtype=torch.float16,
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
    meta_instruction = """
        You are an AI assistant whose name is InternLM (书生·浦语).
        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文
        """.replace(" ", "")

    chat_template = ""
    chat_template += "{{ bos_token }}"
    chat_template += f"{{'<|im_start|>system\n{meta_instruction}<|im_end|>\n'}}"
    chat_template += "{% for message in messages %}"
    chat_template += "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    chat_template += "{% endfor %}"
    chat_template += "{% if add_generation_prompt %}"
    chat_template += "{{ '<|im_start|>assistant\n' }}{% endif %}"
    policy_output = runner.generate(
        input_strs,
        step=256,
        output_logits=False,
        output_str=True,
        chat_template=chat_template,
        generate_kwargs=generate_kwargs,
    )
    output_ids = policy_output.output_ids
    question_mask = policy_output.question_mask
    answer_mask = policy_output.answer_mask
    expected = [
        "非常好，有什么需要我帮助的吗？",
        """好的，以下是三个关于管理时间的建议：

1. 制定计划和时间表： 计划是管理时间的关键。制定一个合理的计划和时间表可以帮助您更好地组织自己的时间，并确保您有足够的时间来完成任务。在计划和时间表上，您应该包括每天、每周、每月的任务和目标。确保您的计划和时间表清晰、具体、可行，并根据需要进行调整。

2. 优先级管理： 将任务按照优先级排序，以确保您首先完成最重要的任务。将时间分配给每个任务，并尽力保证每个任务都有一个足够的时间。如果有多个任务需要完成，请确保优先处理最紧急和最重要的任务，以确保您的生产力最大化。

3. 学会说“不”： 有时候，我们可能被要求做很多不必要的事情或任务，这可能会浪费我们的时间。学会说“不”可以避免这种情况发生。当您接受不必要的任务时，请确保您有足够的时间来完成其他重要的任务，并为它们腾出时间。""",
    ]
    input_strs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": input}],
            tokenize=False,
            chat_template=chat_template,
            add_generation_prompt=True,
        )
        for input in input_strs
    ]
    input_ids = tokenizer(input_strs, return_tensors="pt", padding=True).input_ids

    assert (
        tokenizer.decode(output_ids[0][input_ids.shape[-1] : 134]) == expected[0]
    ), f"expected: {expected[0]!r}\noutput: {tokenizer.decode(output_ids[0][input_ids.shape[-1]:135])!r}"
    assert (
        tokenizer.decode(output_ids[1][input_ids.shape[-1] : -1]) == expected[1]
    ), f"expected: {expected[1]!r}\noutput: {tokenizer.decode(output_ids[1][input_ids.shape[-1]:-1])!r}"

    # check question_mask
    assert torch.equal(
        question_mask[0][:6], torch.zeros(6, dtype=int)
    ), f"expected: {torch.zeros(6,dtype=int)}\noutput: {question_mask[0][:6]}"  # padding
    assert torch.equal(
        question_mask[0][6 : input_ids.shape[-1]],
        torch.ones(input_ids.shape[-1] - 6, dtype=int),
    ), f"expected: {torch.ones(input_ids.shape[-1] - 6,dtype=int)}\noutput: {question_mask[0][6:input_ids.shape[-1]]}"  # question
    assert torch.equal(
        question_mask[0][input_ids.shape[-1] :],
        torch.zeros(question_mask.shape[-1] - input_ids.shape[-1], dtype=int),
    ), f"expected: {torch.zeros(question_mask.shape[-1] - input_ids.shape[-1],dtype=int)}\noutput: {question_mask[0][input_ids.shape[-1]:]}"  # answer

    assert torch.equal(
        question_mask[1][: input_ids.shape[-1]],
        torch.ones(input_ids.shape[-1], dtype=int),
    ), f"expected: {torch.ones(input_ids.shape[-1],dtype=int)}\noutput: {question_mask[1][:input_ids.shape[-1]]}"  # question
    assert torch.equal(
        question_mask[1][input_ids.shape[-1] :],
        torch.zeros(question_mask.shape[-1] - input_ids.shape[-1], dtype=int),
    ), f"expected: {torch.zeros(question_mask.shape[-1] - input_ids.shape[-1],dtype=int)}\noutput: {question_mask[1][input_ids.shape[-1]:]}"  # answer

    # check answer_mask
    assert torch.equal(
        answer_mask[0][: input_ids.shape[-1]],
        torch.zeros(input_ids.shape[-1], dtype=int),
    ), f"expected: {torch.zeros(input_ids.shape[-1],dtype=int)}\noutput: {answer_mask[0][:input_ids.shape[-1]]}"  # question
    assert torch.equal(
        answer_mask[0][input_ids.shape[-1] : 135],
        torch.ones(135 - input_ids.shape[-1], dtype=int),
    ), f"expected: {torch.ones(135 - input_ids.shape[-1],dtype=int)}\noutput: {answer_mask[0][input_ids.shape[-1]:135]}"  # answer
    assert torch.equal(
        answer_mask[0][135:], torch.zeros(answer_mask.shape[-1] - 135, dtype=int)
    ), f"expected: {torch.zeros(answer_mask.shape[-1] - 135,dtype=int)}\noutput: {answer_mask[0][135:]}"  # padding

    assert torch.equal(
        answer_mask[1][: input_ids.shape[-1]],
        torch.zeros(input_ids.shape[-1], dtype=int),
    ), f"expected: {torch.zeros(input_ids.shape[-1],dtype=int)}\noutput: {answer_mask[1][:input_ids.shape[-1]]}"  # question
    assert torch.equal(
        answer_mask[1][input_ids.shape[-1] :],
        torch.ones(answer_mask.shape[-1] - input_ids.shape[-1], dtype=int),
    ), f"expected: {torch.ones(answer_mask.shape[-1] - input_ids.shape[-1],dtype=int)}\noutput: {answer_mask[1][input_ids.shape[-1]:]}"  # answer
