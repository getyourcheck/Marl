import torch
from marl.config import Config
from marl.config_consts import ENGINE_HUGGINGFACE, ENGINE_VLLM
from marl.model_backend.hf_model_runner import HfModelRunnerRayActorGroup
from marl.model_backend.vllm_model_runner import VllmGeneratorRayActorGroup
from marl.tokenizer.tokenizer_utils import get_tokenizer
from marl.policy_output import PolicyOutput
from transformers import AutoModelForCausalLM

def test_get_model_from():
    actor = dict(
        model_path="internlm/internlm2-chat-1_8b-sft",
        model_type="actor",
        torch_dtype=torch.bfloat16,
        trainer_config=dict(
            trainer_type=ENGINE_HUGGINGFACE,
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_seq",
            ),
            parallel=dict(
                data=dict(size=2, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
        generator_config=dict(
            shared_with_trainer=False,
            generator_type=ENGINE_VLLM,
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            generate_kwargs={
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 0.9,
                "min_new_tokens": 1,
                "num_beams": 1,
                "early_stopping": True,
                "eos_token_id": 92542,
                "pad_token_id": 0,
            },
        ),
    )

    model_config = Config(actor)
    model_path: str = model_config["model_path"]  # requisite
    model_type: str = model_config["model_type"]
    tokenizer_path: str = model_config.get("tokenizer_path", model_path)  # opt
    tokenizer = get_tokenizer(tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token

    trainer_config: dict = model_config["trainer_config"]  # requisite
    trainer_config["model_path"] = model_path
    trainer_config["model_type"] = model_type
    trainer_config["tokenizer_path"] = tokenizer_path
    trainer_config["tokenizer_pad_token_id"] = tokenizer.pad_token_id

    hfRayActorGroup = HfModelRunnerRayActorGroup(name="model", config=trainer_config)

    # modify model
    modified_model_path = "/tmp/internlm2-chat-1_8b-sft-modify"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cpu()
    with torch.no_grad():
        for _, param in model.named_parameters():
            param.mul_(2)
    model.save_pretrained(modified_model_path)

    generator_config: dict = model_config.get("generator_config")  # optional
    generator_config["model_path"] = modified_model_path
    generator_config["tokenizer_path"] = tokenizer_path
    vllmGeneratorRayActorGroup = VllmGeneratorRayActorGroup(
        name="vllm_engine", config=generator_config
    )
    
    hfRayActorGroup.init_process_group(vllmGeneratorRayActorGroup)

    expected_output = """The capital of France is, just tell me the city name.
Paris
Paris, the capital of France is known as the world's most beautiful and iconic city as well as the most expensive and exciting city."""
    
    # generation
    step = 32
    input_str = [
        "The capital of France is, just tell me the city name.",
    ]
    policy_output: PolicyOutput = vllmGeneratorRayActorGroup.generate(
        input_str,
        step=step,
        output_logits=False,
        output_str=True,
        attention_mask=None,
        **generator_config['generate_kwargs'],
    )
    print(policy_output.output_str[0])
    print(expected_output)
    assert (
        policy_output.output_str[0] != expected_output
    ), f"expected: modified model output unreadable text but got: {policy_output.output_str[0]}"

    # broadcast model to vllm
    hfRayActorGroup.broadcast_model_to_generator(vllmGeneratorRayActorGroup)
    print("Update Model DONE")
    
    # generation again
    policy_output: PolicyOutput = vllmGeneratorRayActorGroup.generate(
        input_str,
        step=step,
        output_logits=False,
        output_str=True,
        attention_mask=None,
        **generator_config['generate_kwargs'],
    )
    print(policy_output.output_str[0])
    print(expected_output)
    assert (
        policy_output.output_str[0] == expected_output
    ), f"expected: updated model output {expected_output} but got: {policy_output.output_str[0]}"
    print("TEST DONE")