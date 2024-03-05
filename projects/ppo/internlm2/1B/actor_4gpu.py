import torch

actor = dict(
    model_path="internlm/internlm2-chat-1_8b-sft",
    model_type="actor",
    torch_dtype=torch.float16,
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
            data=dict(size=4, mode="ddp"),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence=False,
        ),
    ),
    generator_config=dict(
        shared_with_trainer=True,
        generate_kwargs=dict(
            max_new_tokens=64,
        ),
    ),
)
