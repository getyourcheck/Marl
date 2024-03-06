import torch

actor = dict(
    # RLHF
    # model_path="/cpfs01/shared/public/llm_model/ckpt/Luyou_1B/3920_hf",
    # tokenizer_path="/cpfs01/shared/public/llm_model/tokenizer/v13.model",
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
            data=dict(size=1, mode="ddp"),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence=False,
        ),
    ),
    generator_config=dict(
        shared_with_trainer=True,
        # shared_with_trainer = False,
        #
        # generator_type = "huggingface",
        #
        # parallel = dict(
        #     pipeline = 1,
        #     tensor = dict(size = 1, mode = "1d"),
        # ),
        generate_kwargs=dict(
            max_new_tokens=64,
        ),
    ),
)

critic = dict(
    model_path="/cpfs01/shared/public/llm_model/ckpt/Luyou_1B/R-Luyou-1B-8k-D20240130-v1-hf/",
    model_type="critic",
    trainer_config=dict(  # required for each model, even training is not required.
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
)

# reward = dict(
# )

# ref = dict(
# )
