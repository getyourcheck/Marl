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
    meta_instruction = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文"""

    chat_template = ""
    chat_template += "{{ bos_token }}"
    chat_template += "{{'<|im_start|>system\n" + meta_instruction + "<|im_end|>\n'}}"
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
        "你好，有什么可以帮助你的吗？",
        """好的，以下是三条关于管理时间的建议：

1. 制定计划：每天开始之前，列出您需要完成的任务和目标，并根据重要性和优先级安排优先级。根据截止日期的截止时间来安排时间表，并遵循计划。

2. 确定优先事项：将任务分为紧急和重要、紧急且重要、重要但不紧急、紧急但不重要、既不紧急也不重要的五个类别。然后按照重要性对任务进行排序，优先完成重要且紧急或重要但不紧急的任务。

3. 管理时间：不要在一项任务上花费过多时间，避免浪费太多时间在低优先级任务上。尝试避免多任务处理，而是专注于一项任务直到完成。另外，利用工具和技术，如番茄钟、待办事项应用程序等，可以帮助您更好地管理时间并提高生产力。""",
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
        question_mask[0][:6], torch.zeros(6, dtype=int)
    ), f"expected: {torch.zeros(6,dtype=int)}\noutput: {question_mask[0][:6]}"  # padding
    assert torch.equal(
        question_mask[0][6 : question_len],
        torch.ones(question_len - 6, dtype=int),
    ), f"expected: {torch.ones(question_len - 6, dtype=int)}\noutput: {question_mask[0][6 : question_len]}"  # question
    assert torch.equal(
        question_mask[0][question_len :],
        torch.zeros(output_len - question_len, dtype=int),
    ), f"expected: {torch.zeros(output_len - question_len, dtype=int)}\noutput: {question_mask[0][question_len :]}"  # answer

    assert torch.equal(
        question_mask[1][: question_len],
        torch.ones(question_len, dtype=int),
    ), f"expected: {torch.ones(question_len, dtype=int)}\noutput: {question_mask[1][: question_len]}"  # question
    assert torch.equal(
        question_mask[1][question_len :],
        torch.zeros(output_len - question_len, dtype=int),
    ), f"expected: {torch.zeros(output_len - question_len, dtype=int)}\noutput: {question_mask[1][question_len :]}"  # answer

    # check answer_mask
    assert torch.equal(
        answer_mask[0][: question_len],
        torch.zeros(question_len, dtype=int),
    ), f"expected: {torch.zeros(question_len, dtype=int)}\noutput: {answer_mask[0][: question_len]}"  # question
    assert torch.equal(
        answer_mask[0][question_len : question_len + answer_1_len],
        torch.ones(answer_1_len, dtype=int),
    ), f"expected: {torch.ones(answer_1_len, dtype=int)}\noutput: {answer_mask[0][question_len : question_len + answer_1_len]}"  # answer
    assert torch.equal(
        answer_mask[0][question_len + answer_1_len:], torch.zeros(output_len - question_len - answer_1_len, dtype=int)
    ), f"expected: { torch.zeros(output_len - question_len - answer_1_len, dtype=int)}\noutput: {answer_mask[0][question_len + answer_1_len:]}"  # padding

    assert torch.equal(
        answer_mask[1][: question_len],
        torch.zeros(question_len, dtype=int),
    ), f"expected: {torch.zeros(question_len, dtype=int)}\noutput: {answer_mask[1][: question_len]}"  # question
    assert torch.equal(
        answer_mask[1][question_len :],
        torch.ones(output_len - question_len, dtype=int),
    ), f"expected: {torch.ones(output_len - question_len, dtype=int)}\noutput: {answer_mask[1][question_len :]}"  # answer
