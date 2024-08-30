import torch

from marl.modeling.internlm.modeling_internlm2 import (
    InternLM2ForRewardModel,
    InternLM2ForCriticModel,
)
from marl.modeling.builder import (
    build_critic_model,
    build_reward_model,
)


def test_build_critic_model_internlm_reward_two_linear():
    model_path = "tests/modeling/dummy_model/dummy_internlm_reward_two_linear/"
    extra_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = build_critic_model(model_path, extra_kwargs=extra_kwargs)
    assert model is not None, "Model should be initialized"
    assert isinstance(
        model, torch.nn.Module
    ), "Model should be an instance of torch.nn.Module"
    assert isinstance(
        model, InternLM2ForCriticModel
    ), "Model should be an instance of InternLM2ForCriticModel"
    assert isinstance(
        model.v_head, torch.nn.Sequential
    ), "Model should have a sequential v_head attribute"
    assert (
        len(model.v_head) == 4
    ), "Model should have two linear layers, norm, and act in v_head"
    assert (
        model.v_head[-1].weight.dtype == torch.bfloat16
    ), f"Model should have float16 dtype, but got {model.v_head[-1].weight.dtype}"

    input_ids = torch.randint(0, 8, (2, 10), device=model.device)
    output = model(input_ids)
    logits = output.logits
    assert logits.shape == (2, 9), f"Expected logits shape (2, 9), got {logits.shape}"


def test_build_critic_model_internlm_reward_one_linear():
    model_path = "tests/modeling/dummy_model/dummy_internlm_reward_one_linear/"
    extra_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    model = build_critic_model(
        model_path, extra_kwargs=extra_kwargs, exclude_keys=["v_head.weight"]
    )
    assert model is not None, "Model should be initialized"
    assert isinstance(
        model, torch.nn.Module
    ), "Model should be an instance of torch.nn.Module"
    assert isinstance(
        model, InternLM2ForCriticModel
    ), "Model should be an instance of InternLM2ForCriticModel"
    assert isinstance(
        model.v_head, torch.nn.Linear
    ), "Model should have a linear v_head attribute"
    assert (
        model.v_head.weight.dtype == torch.float16
    ), f"Model should have float16 dtype, but got {model.v_head.weight.dtype}"

    input_ids = torch.randint(0, 8, (3, 13), device=model.device)
    output = model(input_ids)
    logits = output.logits
    assert logits.shape == (3, 12), f"Expected logits shape (3, 12), got {logits.shape}"


def test_build_critic_model_auto():
    model_path = "tests/modeling/dummy_model/dummy_llama_reward"
    extra_kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    model = build_critic_model(
        model_path, head_name="v_head_test", two_linear=False, extra_kwargs=extra_kwargs
    )
    assert isinstance(
        model, torch.nn.Module
    ), "Model should be an instance of torch.nn.Module"
    assert (
        model.__class__.__name__ == "AutoCriticModel"
    ), "Model should be an instance of AutoCriticModel"
    assert (
        model.model.__class__.__name__ == "LlamaModel"
    ), "Model should be an instance of LlamaModel"
    assert hasattr(model, "v_head_test"), "Model should have a v_head_test attribute"
    assert isinstance(
        model.v_head_test, torch.nn.Linear
    ), "Model should have a linear v_head_test attribute"

    model_path = "tests/modeling/dummy_model/dummy_llama_reward"
    extra_kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    model = build_critic_model(
        model_path, head_name="v_head", two_linear=True, extra_kwargs=extra_kwargs
    )
    assert isinstance(
        model, torch.nn.Module
    ), "Model should be an instance of torch.nn.Module"
    assert (
        model.__class__.__name__ == "AutoCriticModel"
    ), "Model should be an instance of AutoCriticModel"
    assert (
        model.model.__class__.__name__ == "LlamaModel"
    ), "Model should be an instance of LlamaModel"
    assert hasattr(model, "v_head"), "Model should have a v_head attribute"
    assert isinstance(
        model.v_head, torch.nn.Sequential
    ), "Model should have a sequential v_head attribute"
    assert (
        len(model.v_head) == 4
    ), "Model should have two linear layers, norm, and act in v_head"

    input_ids = torch.randint(0, 8, (1, 5), device=model.device)
    output = model(input_ids)
    logits = output.logits
    assert logits.shape == (1, 4), f"Expected logits shape (1, 4), got {logits.shape}"


def test_build_reward_model_internlm_reward_two_linear():
    model_path = "tests/modeling/dummy_model/dummy_internlm_reward_two_linear/"
    extra_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = build_reward_model(model_path, extra_kwargs=extra_kwargs)
    assert model is not None, "Model should be initialized"
    assert isinstance(
        model, torch.nn.Module
    ), "Model should be an instance of torch.nn.Module"
    assert isinstance(
        model, InternLM2ForRewardModel
    ), "Model should be an instance of InternLM2ForRewardModel"
    assert isinstance(
        model.v_head, torch.nn.Sequential
    ), "Model should have a sequential v_head attribute"
    assert (
        len(model.v_head) == 4
    ), "Model should have two linear layers, norm, and act in v_head"
    assert (
        model.v_head[-1].weight.dtype == torch.bfloat16
    ), f"Model should have float16 dtype, but got {model.v_head[-1].weight.dtype}"

    input_ids = torch.randint(0, 8, (4, 5), device=model.device)
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids, attention_mask=attention_mask)
    logits = output.logits
    assert logits.shape == (4,), f"Expected logits shape (4), got {logits.shape}"


def test_build_reward_model_internlm_reward_one_linear():
    model_path = "tests/modeling/dummy_model/dummy_internlm_reward_one_linear/"
    extra_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    model = build_reward_model(model_path, extra_kwargs=extra_kwargs)
    assert model is not None, "Model should be initialized"
    assert isinstance(
        model, torch.nn.Module
    ), "Model should be an instance of torch.nn.Module"
    assert isinstance(
        model, InternLM2ForRewardModel
    ), "Model should be an instance of InternLM2ForRewardModel"
    assert isinstance(
        model.v_head, torch.nn.Linear
    ), "Model should have a linear v_head attribute"
    assert (
        model.v_head.weight.dtype == torch.float16
    ), f"Model should have float16 dtype, but got {model.v_head.weight.dtype}"

    input_ids = torch.randint(0, 8, (1, 9), device=model.device)
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids, attention_mask=attention_mask)
    logits = output.logits
    assert logits.shape == (1,), f"Expected logits shape (1), got {logits.shape}"


def test_build_reward_model_auto():
    model_path = "tests/modeling/dummy_model/dummy_llama_reward"
    extra_kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    model = build_reward_model(
        model_path, head_name="v_head_test", two_linear=False, extra_kwargs=extra_kwargs
    )
    assert isinstance(
        model, torch.nn.Module
    ), "Model should be an instance of torch.nn.Module"
    assert (
        model.__class__.__name__ == "AutoRewardModel"
    ), f"Model should be an instance of AutoRewardModel, got {model.__class__.__name__}"
    assert (
        model.model.__class__.__name__ == "LlamaModel"
    ), "Model should be an instance of LlamaModel"
    assert hasattr(model, "v_head_test"), "Model should have a v_head_test attribute"
    assert isinstance(
        model.v_head_test, torch.nn.Linear
    ), "Model should have a linear v_head_test attribute"

    model_path = "tests/modeling/dummy_model/dummy_llama_reward"
    extra_kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    model = build_reward_model(
        model_path, head_name="v_head", two_linear=True, extra_kwargs=extra_kwargs
    )
    assert isinstance(
        model, torch.nn.Module
    ), "Model should be an instance of torch.nn.Module"
    assert (
        model.__class__.__name__ == "AutoRewardModel"
    ), f"Model should be an instance of AutoRewardModel, got {model.__class__.__name__}"
    assert (
        model.model.__class__.__name__ == "LlamaModel"
    ), "Model should be an instance of LlamaModel"
    assert hasattr(model, "v_head"), "Model should have a v_head attribute"
    assert isinstance(
        model.v_head, torch.nn.Sequential
    ), "Model should have a sequential v_head attribute"
    assert (
        len(model.v_head) == 4
    ), "Model should have two linear layers, norm, and act in v_head"

    input_ids = torch.randint(0, 8, (10, 3), device=model.device)
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids, attention_mask=attention_mask)
    logits = output.logits
    assert logits.shape == (10,), f"Expected logits shape (10), got {logits.shape}"
