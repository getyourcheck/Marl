import torch

model_configs=dict(
    actor = dict(
        model_path="/fs-computility/llm/shared/marl/models/internlm2/1.8B/hf/Luyou_1B_FT_0.19_130_avg5/",
        model_type="actor",
        trainer_config=dict(
            torch_dtype="auto",
            trainer_type="huggingface",
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_seq",
            ),
            parallel=dict(
                data=dict(size=4, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                "zero_optimization": {
                    "stage": 2, 
                    "offload_param": {
                        "device": "none"
                    },
                    "reduce_bucket_size": "auto", 
                    "zero_hpz_partition_size": 1, 
                    "zero_quantized_weights": False, 
                    "zero_quantized_gradients": False,
                    "stage3_gather_16bit_weights_on_model_save": True,
                }, 
                "bf16": {
                    "enabled": True
                }, 
                "gradient_clipping": 1.0, 
                "prescale_gradients": False, 
                "wall_clock_breakdown": False, 
                "data_types": {
                    "grad_accum_dtype": "fp32"
                }, 
                "train_micro_batch_size_per_gpu": 2, 
                "gradient_accumulation_steps": 64,
                "train_batch_size": 512
            }
        ),
        generator_config=dict(
            shared_with_trainer=True,
        ),
    ),
    reward = dict(
        model_path="/fs-computility/llm/shared/marl/models/internlm2/1.8B/hf/R-Luyou-1B-8k-D20240130-v1/819_hf_bf16/",
        model_type="reward",
        use_flash_attn=False,
        trainer_config=dict(
            trainer_type="huggingface",
            torch_dtype="auto",
            parallel=dict(
                data=dict(size=4, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                "zero_optimization": {
                    "stage": 2, 
                    "offload_param": {
                        "device": "none"
                    },
                    "reduce_bucket_size": "auto", 
                    "zero_hpz_partition_size": 1, 
                    "zero_quantized_weights": False, 
                    "zero_quantized_gradients": False
                }, 
                "bf16": {
                    "enabled": True
                }, 
                "gradient_clipping": 1.0, 
                "prescale_gradients": False, 
                "wall_clock_breakdown": False, 
                "data_types": {
                    "grad_accum_dtype": "fp32"
                }, 
                "train_micro_batch_size_per_gpu": 2, 
                "gradient_accumulation_steps": 64,
                "train_batch_size": 512
            }
        ),
    ),
)
