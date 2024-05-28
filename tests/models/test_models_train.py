import torch
import pytest
from loguru import logger
from marl.config.config import Config
from marl.model_backend.hf_model_runner import HfModelRunnerRayActorGroup
from marl.loss.actor_loss import ActorLoss
from marl.tokenizer.tokenizer_utils import get_tokenizer

actor=dict(
    model_path="internlm/internlm2-chat-20b-sft",
    model_type="actor",
    torch_dtype=torch.float32,
    trainer_config=dict(
        trainer_type="huggingface",
        use_flash_attn=None,
        train_kwargs=dict(
            micro_bsz=1,
            lr=1e-6,
            total_steps=1e9,
            lr_decay_rate=1,
            loss_type="per_seq",
        ),
        parallel=dict(
            data=dict(size=1, mode="deepspeed"),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence=False,
        ),
        deepspeed_config={
            "bf16": {"enable": False},
            "fp16": {"enable": False},
            "zero_optimization": {
                "stage": 3,
            },
            "gradient_accumulation_steps": 1,
            "train_micro_batch_size_per_gpu": 1,
        },
    ),
    generator_config=dict(
        shared_with_trainer=True,
    ),
)

configs_list = [
    (660,  False, -434.61, 16),
    (660,  True,  -433.67, 16),
    (1122, True,  -441.30, 16),
]
@pytest.mark.parametrize("configs", configs_list)
@pytest.mark.skip()
def test_model_train(configs):
    max_seq_len, use_flash_attn, target_loss, dp_size = configs
    SEQ_LEN_MAGNIFY=max_seq_len // 33  # input_ids.shape[1] = 33
    BATCH_SIZE=dp_size * 1

    actor["trainer_config"]["use_flash_attn"] = use_flash_attn
    actor["trainer_config"]["parallel"]["data"]["size"] = dp_size
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
    hfRayActorGroup.initialize_get()
    logger.success(f"After Init : {hfRayActorGroup.get_cuda_mem_stats()}")

    ####################################################################################
    input_ids = torch.tensor(
        [[7558,  1244,  4795,  9551,   446,   395, 48929,   262,  1238,  1184,  2460,
          579,   940,  6022,   454, 31449,   328,   607,   784,   629,   1896,   697,
          725,  2320,  1263,   884,  5426,   333,   352, 23845,   352, 27232,   489]]
        ).cuda().repeat(BATCH_SIZE, SEQ_LEN_MAGNIFY)

    log_probs = torch.tensor(
        [[0, -5.9688,  -7.6875,  -4.5000,  -0.0232,  -2.4375,  -4.7500,  -4.2812,
        -1.6406,  -0.1846,  -0.0232,  -5.3438,  -4.7500,  -6.2188,  -1.4219,
        -2.5625,  -1.9062,  -4.3125,  -4.7500,  -0.0605,  -0.7461,  -1.2734,
        -2.2031,  -2.3438,  -2.0312,  -2.4062,  -2.6562,  -7.9375,  -6.3125,
        -13.2500,  -2.0312,  -5.0625,  -1.2031]], dtype=torch.bfloat16
        ).cuda().repeat(BATCH_SIZE, SEQ_LEN_MAGNIFY)

    advantages = torch.tensor(
        [[0, 224.,  752.,  366.,  492.,  596.,  340.,  416.,  652.,  832., 1012.,  624., 
        -21., -152.,  880.,  196.,  856.,  628.,  860.,  704., 1112., 410.,  524.,
        912., 1056.,  480., 1040.,  584.,  362., -204.,  124., -138.,  376.]],
        dtype=torch.bfloat16).cuda().repeat(BATCH_SIZE, SEQ_LEN_MAGNIFY)

    response_mask = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1 ]]).cuda().repeat(BATCH_SIZE, SEQ_LEN_MAGNIFY)

    valid_tokens = response_mask.sum().cuda()
    loss_factor = 1 / valid_tokens.item()
    labels = dict(
        input_ids=input_ids,
        old_logprobs=log_probs,
        advantages=advantages,
        mask=response_mask,
        loss_factor=torch.tensor(loss_factor),
    )
    ####################################################################################

    logger.success(f"Train inputs: input_ids.shape={input_ids.shape}")
    result_loss = hfRayActorGroup.train(input_ids, labels, attention_mask=None, criterion=ActorLoss(), step_interval=999).item()
    logger.success(f"After Train: {hfRayActorGroup.get_cuda_mem_stats()}")
    logger.success(f"Output loss: {result_loss}")
    assert round(result_loss, 2) == target_loss, f"{result_loss} != {target_loss}"
    hfRayActorGroup.release_resources()
    return
