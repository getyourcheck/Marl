import torch

actor=dict(
    model_path="/fs-computility/llm/shared/marl/models/internlm2/1.8B/hf/Luyou_1B_FT_0.19_130_avg5/",
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
            data=dict(size=2, mode="deepspeed"),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence=False,
        ),
        deepspeed_config={
            "zero_optimization": {
                "stage": 1,
            },
            "gradient_accumulation_steps": 1,
            "train_micro_batch_size_per_gpu": 1,
        },
    ),
    generator_config=dict(
        shared_with_trainer=False,
        # shared_with_trainer=True,
        generator_type="vllm",
        parallel=dict(
            data=dict(size=1, mode="ddp"),
            tensor=dict(size=4, mode="1d"),
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
            "skip_special_tokens": True,
        },
    ),
)