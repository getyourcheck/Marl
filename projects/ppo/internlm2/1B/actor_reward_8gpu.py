import torch

model_configs=dict(
    actor = dict(
        model_path="/cpfs01/shared/public/public_hdd/lishuaibin/models/1.8B_baseline/sft/Luyou_1B_FT_0.19_130_avg5/",
        model_type="actor",
        trainer_config=dict(
            torch_dtype=torch.bfloat16,
            trainer_type="huggingface",
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_seq",
            ),
            parallel=dict(
                data=dict(size=4, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            )
        ),
        generator_config=dict(
            shared_with_trainer=True,
        ),
    ),
    reward = dict(
        model_path="/cpfs01/shared/public/public_hdd/lvchengqi/ckpts/reward_model/R-Luyou-1B-8k-D20240130-v1/819_hf_bf16/",
        model_type="reward",
        use_flash_attn=False,
        trainer_config=dict(
            trainer_type="huggingface",
            torch_dtype=torch.bfloat16,
            parallel=dict(
                data=dict(size=4, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),
)
