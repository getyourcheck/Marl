import torch

actor = dict(
    model_path="internlm/internlm2-chat-1_8b-sft",
    model_type="actor",
    torch_dtype=torch.bfloat16,
    trainer_config=dict(
        trainer_type="huggingface",
        train_kwargs=dict(
            micro_bsz=1,
            lr=1e-6,
            total_steps=1e9,
            lr_decay_rate=1,
            loss_type="per_seq",
        ),
        parallel=dict(
            data=dict(size=1, mode="ddp"),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence=False,
        ),
    ),
    generator_config=dict(
        shared_with_trainer=True,
        generate_kwargs=dict(
            max_new_tokens=2048,
        ),
    ),
)

reference = dict(
    model_path="internlm/internlm2-chat-1_8b-sft",
    model_type="reference",
    torch_dtype=torch.bfloat16,
    trainer_config=dict(
        trainer_type="huggingface",
        parallel=dict(
            data=dict(size=1, mode="ddp"),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence=False,
        ),
    ),
)

critic = dict(
    model_path="/cpfs01/shared/public/llm_model/ckpt/Luyou_1B/R-Luyou-1B-8k-D20240130-v1-hf/",
    model_type="critic",
    torch_dtype=torch.bfloat16,
    trainer_config=dict(
        trainer_type="huggingface",
        train_kwargs=dict(
            micro_bsz=1,
            lr=1e-6,
            total_steps=1e9,
            lr_decay_rate=1,
            loss_type="per_seq",
        ),
        parallel=dict(
            data=dict(size=1, mode="ddp"),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence=False,
        ),
    ),
)

reward = dict(
    model_path="/cpfs01/shared/public/llm_model/ckpt/Luyou_1B/R-Luyou-1B-8k-D20240130-v1-hf/",
    model_type="reward",
    torch_dtype=torch.bfloat16,
    trainer_config=dict(
        trainer_type="huggingface",
        parallel=dict(
            data=dict(size=1, mode="ddp"),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence=False,
        ),
    ),
)
