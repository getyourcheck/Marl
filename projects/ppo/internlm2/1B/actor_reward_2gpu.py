
actor = dict(
    # RLHF
    # model_path = "/cpfs01/shared/public/public_hdd/wangyudong/ckpt/exps/20231227/aliyun_Luyou_1B_FT_0.19/3920_hf",
    model_path = "facebook/opt-1.3b",

    model_class = "actor",

    trainer_config = dict(
        trainer_type = "huggingface",

        train_kwargs = dict(
            micro_bsz = 1,
            lr = 1e-6,
            total_steps = 1e9,
            lr_decay_rate = 1,
            loss_type = "per_token",
        ),

        parallel = dict(
            data = dict(size = 1, mode = "ddp"),
            tensor = dict(size = 1, mode = "1d"),
            pipeline = dict(size = 1, interleaved_overlap = False),
            sequence = False,
        ),
    ),

    generator_config = dict(
        shared_with_trainer = True,

        # shared_with_trainer = False,
        #
        # generator_type = "huggingface",
        #
        # parallel = dict(
        #     pipeline = 1,
        #     tensor = dict(size = 1, mode = "1d"),
        # ),

        generate_kwargs = dict(
            max_new_tokens = 64,
        ),
    ),
)

# critic = dict(
# )

# ref = dict(
# )

reward = dict(
    model_path = actor['model_path'],

    model_class = "reward",

    trainer_config = dict(  # required for each model, even training is not required.
        trainer_type = "huggingface",

        parallel = dict(
            data = dict(size = 1, mode = "ddp"),
            tensor = dict(size = 1, mode = "1d"),
            pipeline = dict(size = 1, interleaved_overlap = False),
            sequence = False,
        ),
    ),
)
