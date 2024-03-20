from abc import abstractmethod
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset
from copy import deepcopy


class TxtEnv(object):
    """
    A generic RL environment to generate textual sequences.
    """

    def __init__(self, dataloader: IterableDataset, max_step=2048, reward_function=None, reward_model_config=None):
        """
        Args:
            dataloader (IterableDataset): generate rl data iteratively
            reward_function: reward function that computes scalar reward for each episode
        """
        self.dataloader = iter(dataloader)
        self.reward_function = reward_function

        self._cur_messagess = []
        self.max_step = max_step
        self.max_infer_batchsize = 16
        self.clip_reward_min = - 1.5
        self.clip_reward_max = 1.5
        self.generate_config = {'do_sample': True, 
                                'temperature': 1.0, 
                                'top_k': 0, 
                                'top_p': 0.9, 
                                'min_new_tokens': 1,
                                'num_beams': 1, 
                                'eos_token_id': 92542, 
                                'pad_token_id': 0}

    def rollout(self, policy_model, display=False):
        s_t = time.time()
        sample_data = next(self.dataloader)

        ppo_input_messages = []
        sft_input_messages = []
        for data in sample_data:
            message = data.message
            if data.mes_type == "ppo":
                ppo_input_messages.append(message)
            elif data.mes_type == "sft":
                sft_input_messages.append(message)
            else:
                raise TypeError(f"Wrong message type {data.mes_type}")

        # ppo data
        trajectories = policy_model.generate(ppo_input_messages, step=self.max_step, output_logits=True, output_str=True, generate_kwargs=self.generate_config)
        print(f"[TxtEnv & {policy_model.__class__.__name__}] {round(time.time() - s_t, 2)}s generate {len(ppo_input_messages)} ppo episodes.")
        rewards = self._get_reward(ppo_input_messages, trajectories)
        clipped_rewards = torch.clamp(rewards, min=self.clip_reward_min, max=self.clip_reward_max)
        trajectories["rewards"] = rewards
        trajectories["clipped_rewards"] = clipped_rewards

        # sft data
        if len(sft_input_messages) > 0:
            sft_inputs = [policy_model.tokenizer.apply_chat_template(mes, tokenize=False, add_generation_prompt=False, return_tensors="pt") for mes in sft_input_messages]
            trajectories.sft_data = policy_model.tokenizer(sft_inputs, return_tensors="pt", padding=True)
            print(f"[TxtEnv & {policy_model.__class__.__name__}] gets {len(sft_input_messages)} sft episodes.")

        return trajectories
    
    def _get_reward(self, input_messages, policyout):
        input_messages = deepcopy(input_messages)
        if self.reward_function is not None:
            for i in range(len(range(len(policyout.output_ans_str)))):
                input_messages[i].append({"role": "assistant", "content": policyout.output_ans_str[i]})
            rm_out = self.reward_function.infer(input_messages)
            rewards = rm_out.logits.cpu().squeeze(-1)
            return rewards
        print(f"[TxtEnv] No reward funtion, no reward provided.")
        return None


if __name__ == "__main__":
    import sys

    sys.path.extend(["./", "marl/dataset"])
    from collections import defaultdict
    from marl.dataset.txt_loader import TxtMessageDataset
    from marl.tokenizer.tokenizer_utils import get_tokenizer
    # from marl.envs.txt_env import TxtEnv
    import torch
    """txt env test here"""
    model_path = "internlm/internlm2-chat-1_8b-sft"
    tokenizer_path = model_path
    tokenizer = get_tokenizer(tokenizer_path, trust_remote_code=True)

    dataset_config = {
        "ppo_data_filename": "data/config/1.8B_ppo.json",
        "sft_data_filename": "data/config/1.8B_sft.json",
        "num_samples_each_epoch": 10,
        "sft_data_samples": 2,
        "tokenizer": tokenizer,
        "max_seq_len": 4096,
        "random_seed": 1024,
    }

    # actor model
    from marl.config import Config
    trainer_config = Config(
        dict(
            model_path=model_path,
            torch_dtype=torch.bfloat16,
            trainer_type="huggingface",
            parallel=dict(
                data=dict(size=1),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
            ),
        ),
    )
    from marl.model_backend.hf_model_runner import HfModelRunner
    actor_model = HfModelRunner(model_config=trainer_config)
    actor_model.initialize()
    # rm model
    from marl.config_consts import MODEL_TYPE_REWARD, ENGINE_HUGGINGFACE
    reward_trainer_config = Config(
        dict(
            model_path="/cpfs01/shared/public/llm_model/ckpt/Luyou_1B/R-Luyou-1B-8k-D20240130-v1-hf/",
            model_type=MODEL_TYPE_REWARD,
            trainer_type=ENGINE_HUGGINGFACE,
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    )
    reward_model = HfModelRunner(model_config=reward_trainer_config)
    reward_model.initialize()

    """Create txt env for PPO """
    txt_loader = TxtMessageDataset(**dataset_config)
    txt_env = TxtEnv(dataloader=txt_loader, reward_function=reward_model)
    trajectories = txt_env.rollout(policy_model=actor_model)
    
    print(dir(trajectories))
    print(trajectories.sft_data.input_ids.shape)
    print(trajectories.sft_data.attention_mask.shape)
    # for i, s in enumerate(trajectories.output_str):
    #     print(f"[REPLY {i} BGN] {'#' * 20}\n{s}\n[REPLY {i} END] {'#' * 20}\n")
