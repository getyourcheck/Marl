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
            loss_type="per_token",
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
    ),
)
