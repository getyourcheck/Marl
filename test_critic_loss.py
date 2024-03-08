import torch
from marl.config import Config
from marl.config_consts import ENGINE_HUGGINGFACE, MODEL_TYPE_CRITIC
from marl.model_backend.hf_model_runner import HfModelRunner
from marl.loss.critic_loss import CriticLoss
from marl.ppo_learner_utils import compute_rewards, get_advantages_and_returns
from marl.tokenizer.tokenizer_utils import get_tokenizer
from marl.dataset.txt_loader import TxtSequenceDataset
from marl.envs.txt_env import TxtEnv


def test_critic_loss():
    actor_trainer_config = Config(
        dict(
            model_path="internlm/internlm2-chat-1_8b-sft",
            torch_dtype=torch.float16,
            model_type="actor",
            trainer_type=ENGINE_HUGGINGFACE,
            parallel=dict(
                data=dict(size=1),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
            ),
        ),
    )

    critic_trainer_config = Config(
        dict(
            model_path="/cpfs01/shared/public/llm_model/ckpt/Luyou_1B/R-Luyou-1B-8k-D20240130-v1-hf/",
            torch_dtype=torch.float16,
            model_type=MODEL_TYPE_CRITIC,
            trainer_type=ENGINE_HUGGINGFACE,
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_token",
            ),
            parallel=dict(
                data=dict(size=1),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
            ),
        ),
    )

    """ppo reader test here"""
    tokenizer = get_tokenizer(
        "internlm/internlm2-chat-1_8b-sft", trust_remote_code=True
    )
    dataset_config = {
        "task_group_filename": "data/config/task_ppo.json",
        "tokenizer": tokenizer,
        "max_seq_len": 1024,
        "num_samples_each_epoch": 7,
        "random_seed": 1,
        "start_token": "[UNUSED_TOKEN_146]user\n",
        "end_token": "[UNUSED_TOKEN_145]\n",
    }

    """Create txt env for PPO """
    txt_loader = TxtSequenceDataset(**dataset_config)
    txt_env = TxtEnv(dataloader=txt_loader, reward_function=None)

    actor_model = HfModelRunner(model_config=actor_trainer_config)
    actor_model.initialize()

    trajectories = txt_env.rollout(
        policy=actor_model, generate_kwargs={"max_new_tokens": 2048}
    )

    print((trajectories.output_ids))
    for i, s in enumerate(trajectories.output_str):
        print(f"[REPLY {i} BGN] {'#' * 20}\n{s}\n[REPLY {i} END] {'#' * 20}\n")

    input_ids: torch.Tensor = trajectories["output_ids"]  # shape: (bsz, seqlen)
    old_values: torch.Tensor = trajectories.get(
        "values", input_ids.new_ones(input_ids.shape, dtype=torch.float16) * 0.01
    )  # shape: (bsz, seqlen - 1). NOTE: call trajectories.values is wrong because `PolicyOutput.values` is a built-in method.
    returns = trajectories.get(
        "returns", old_values.new_zeros(input_ids.shape, dtype=torch.float16) * 0.01
    )  # shape: (bsz, seqlen - 1)
    mask = trajectories.get(
        "answer_mask", old_values.new_zeros(input_ids.shape, dtype=torch.float16) * 0.01
    )  # shape: (bsz, seqlen - 1)

    labels = dict(
        old_values=old_values,
        returns=returns,
        mask=mask,
        loss_factor=torch.Tensor([0.1]),
    )
    for k, v in labels.items():
        print(k, v.shape)

    critic_model = HfModelRunner(model_config=critic_trainer_config)
    critic_model.initialize()

    chat_1_str = "<s><|im_start|>user\nHello! What's your name?<|im_end|>\n<|im_start|>assistant\nMy name is InternLM2!<|im_end|>\n"
    infer_str_outputs = critic_model.infer(inputs=chat_1_str)  # get_score
    print("critic logits of str input:", infer_str_outputs.logits)

    infer_id_outputs = critic_model.infer(inputs=input_ids)  # get_score
    print("critic logits of input_ids input:", infer_id_outputs.logits)


    train_outputs = critic_model.train(
        input_ids=input_ids, labels=labels, criterion=CriticLoss
    )
    print(train_outputs)

test_critic_loss()
