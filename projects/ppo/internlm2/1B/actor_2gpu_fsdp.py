import torch

actor = dict(
    model_path="internlm/internlm2-chat-1_8b-sft",
    torch_dtype=torch.float16,
    model_type="actor",
    trainer_config=dict(
        trainer_type="huggingface",
        train_kwargs=dict(
            micro_bsz=1,
            lr=1e-6,
            total_steps=1e9,
            lr_decay_rate=1,
            loss_type="per_token",
        ),
        parallel=dict(
            data=dict(size=2, mode="fsdp"),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence=False,
        ),
        envs={
            "FSDP_AUTO_WRAP_POLICY":"SIZE_BASED_WRAP",
            "FSDP_MIN_NUM_PARAMS":"1",
        },
    ),
    generator_config=dict(
        shared_with_trainer=True,
        generate_kwargs=dict(
            max_new_tokens=64,
        ),
    ),
)
