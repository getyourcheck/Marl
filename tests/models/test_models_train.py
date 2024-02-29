"""Compare the outputs of HF and Torch when training.

Run `pytest tests/models/test_models_train.py`.
"""
import pytest
import torch

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
@pytest.mark.parametrize("optimizer", ["AdamW"])
def test_models(
    hf_runner,
    torch_runner,
    model: str,
    dtype: str,
    optimizer: str,
    example_prompts,
) -> None:

    hf_model = hf_runner(model, dtype=dtype, optimizer=optimizer)
    input_ids = hf_model.tokenizer(
        example_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True,
    ).input_ids.cuda()
    attention_mask = input_ids.not_equal(1).long()
    labels = torch.zeros(input_ids.shape, dtype=torch.long).cuda()
    hf_model.train(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    torch_model = torch_runner(model, dtype=dtype, optimizer=optimizer)
    torch_model.train(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert set(hf_model.model.state_dict().keys()) == set(
        torch_model.model.state_dict().keys()
    )
    for key in hf_model.model.state_dict():
        assert torch.equal(
            hf_model.model.state_dict()[key], torch_model.model.state_dict()[key]
        )
