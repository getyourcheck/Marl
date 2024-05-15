import time
import torch
from loguru import logger
from torch.utils.data import IterableDataset
from copy import deepcopy
from marl.model_server.base_model_server import BaseModelServer
import re

def get_data(path):
    file = open(path, "r")
    line = file.readline()
    data = {}
    array = []
    while line:
        if line == '\n':
            line = file.readline()
            continue
        if line == '#######################\n':
            GLOBAL_STEPS = file.readline() ## GLOBAL_STEPS = 0000 #
            step = int(GLOBAL_STEPS[17:21])
            if step > 0:
                data[step - 1] = array
                array = []
            line = file.readline() ########################
            line = file.readline()
            continue
        if line == '******** Prompt ********\n':
            while line != '******** For Reward ********\n':
                line = file.readline()
            # print(line) #******** For Reward ********
            line = file.readline()
            tmp = ""
            while not line.startswith("reward_score"):
                tmp += line
                line = file.readline()
            tmp=tmp[:-1]
            # print(tmp)
            array.append(tmp)
            continue
        line = file.readline()
    # for k,v in data.items():
    #     print(f"[{k}][{len(v)}]")
    file.close()

    def convert_to_json_array(raw_string):
        pattern = r"(\[UNUSED_TOKEN_146\]user|\[UNUSED_TOKEN_146\]assistant)(.+?)(?=(\[UNUSED_TOKEN_146\]user|\[UNUSED_TOKEN_146\]assistant|$))"
        matches = re.findall(pattern, raw_string, re.DOTALL)
        json_array = []
        for match in matches:
            role, content = match[0], match[1].strip()
            role = role.replace('[UNUSED_TOKEN_146]', '')
            json_array.append({"role": role, "content": content})
        return json_array

    new_data={}
    for k,v in data.items():
        new_v = []
        for strs in v:
            # strs = strs.replace('[UNUSED_TOKEN_146]', '')
            strs = strs.replace('[UNUSED_TOKEN_145]', '')
            strs = strs.replace('[UNUSED_TOKEN_130]', '')
            strs=strs[:-1]
            json_str=convert_to_json_array(strs)
            new_v.append(json_str)
        new_data[k]=new_v
    return new_data


class TxtEnv(object):
    """
    A generic RL environment to generate textual sequences.
    """

    def __init__(
            self, 
            dataloader: IterableDataset, 
            max_new_tokens:int=1024, 
            actor_micro_bs:int=32,
            reward_micro_bs:int=32,
            clip_reward_min:int=-1.5,
            clip_reward_max:int=1.5,
            reward_function:BaseModelServer=None, 
            generate_kwargs:dict=None,
            **kwargs,
        ):
        """
        Args:
            dataloader (IterableDataset): generate rl data iteratively
            reward_function: reward function that computes scalar reward for each episode
        """
        self.dataloader:IterableDataset = iter(dataloader)
        self.reward_function:BaseModelServer = reward_function
        self._cur_messagess = []
        self.max_new_tokens = max_new_tokens
        self.actor_micro_bs = actor_micro_bs
        self.reward_micro_bs = reward_micro_bs
        self.clip_reward_min = clip_reward_min
        self.clip_reward_max = clip_reward_max
        self.generate_kwargs:dict = generate_kwargs
        self.async_reward:bool = True
        self.data = get_data("/fs-computility/llm/shared/marl/datasets/tmp/1.8B-baseline-c2kcl5.log.txt")

    def rollout(self, policy_model:BaseModelServer, round_=-1, display=False):
        sample_data = next(self.dataloader)
        ppo_input_messages = []
        pt_input_messages = []
        if round_ == -1:
            for data in sample_data:
                message = data.message
                if data.mes_type == "ppo":
                    ppo_input_messages.append(message)
                elif data.mes_type == "pt":
                    pt_input_messages.append(message)
                else:
                    raise TypeError(f"Wrong message type {data.mes_type}")
        else:
            for i in range(len(self.data[round_])):
                ppo_input_messages.append(self.data[round_][i][:-1])
            print(f"========get {len(self.data[round_])} data from rl3m=============", ppo_input_messages[0])
        # ppo data
        s_t = time.time()
        trajectories = policy_model.generate(
            inputs=ppo_input_messages, 
            micro_batch_size=self.actor_micro_bs, 
            step=self.max_new_tokens,
            output_str=True, 
            generate_kwargs=self.generate_kwargs
        )
        sequences, attention_mask, action_mask = self.process_sequences(trajectories.output_ids, trajectories.input_ids.size(1), self.generate_kwargs['eos_token_id'], self.generate_kwargs['pad_token_id'])
        trajectories['output_ids'] = sequences
        trajectories['attention_mask'] = attention_mask
        trajectories['action_mask'] = action_mask
        logger.info(f"[actor generate] duration: {round(time.time() - s_t, 2)} s, len(inputs): {len(ppo_input_messages)} ")

        if self.async_reward:
            reward_output_ref = self.get_reward_async(trajectories)
            trajectories["reward_output_ref"] = reward_output_ref
        else:
            rewards = self.get_reward(trajectories)
            clipped_rewards = torch.clamp(rewards, min=self.clip_reward_min, max=self.clip_reward_max)
            trajectories["rewards"] = rewards
            trajectories["clipped_rewards"] = clipped_rewards

        # pretrain data
        if len(pt_input_messages) > 0:
            pt_inputs = [policy_model.tokenizer.apply_chat_template(mes, tokenize=False, add_generation_prompt=False, return_tensors="pt") for mes in pt_input_messages]
            trajectories.pt_data = policy_model.tokenizer(pt_inputs, return_tensors="pt", padding=True)
            print(f"[TxtEnv & {policy_model.__class__.__name__}] gets {len(pt_input_messages)} pretrain episodes.")

        return trajectories
    

    # default get_reward() is blocking. get_reward_async() needs to call get_reward_collect()
    def get_reward_async(self, policyout):
        s_t = time.time()
        reward_output_ref = self.reward_function.infer_async(
            policyout.output_ids,
            output_logprobs=False,
            attention_mask=policyout.attention_mask,
            micro_batch_size=self.reward_micro_bs
        )
        logger.info(f"[reward infer] async duration: {round(time.time() - s_t, 2)} s")
        return reward_output_ref

    def get_reward_collect(self, reward_output_ref):
        s_t = time.time()
        rm_out = self.reward_function.infer_get(reward_output_ref)
        logger.info(f"[reward infer] async wait duration: {round(time.time() - s_t, 2)} s")
        rewards = rm_out.logits.squeeze(-1)
        return rewards

    def get_reward(self,policyout):
        s_t = time.time()
        rm_out = self.reward_function.infer(
            policyout.output_ids, 
            output_logprobs=False,
            attention_mask=policyout.attention_mask,
            micro_batch_size=self.reward_micro_bs
        )
        logger.info(f"[reward infer] duration: {round(time.time() - s_t, 2)} s")
        rewards = rm_out.logits.squeeze(-1)
        return rewards
    
    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        #
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        attention_mask.scatter_(dim=1, index=eos_indices, value=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        # we only calculate the loss of state_i != eos | pad
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        return sequences, attention_mask, action_mask



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
        "pt_data_filename": "data/config/1.8B_pt.json",
        "num_samples_each_epoch": 10,
        "pt_data_samples": 2,
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
    print(trajectories.pt_data.input_ids.shape)
    print(trajectories.pt_data.attention_mask.shape)
    # for i, s in enumerate(trajectories.output_str):
    #     print(f"[REPLY {i} BGN] {'#' * 20}\n{s}\n[REPLY {i} END] {'#' * 20}\n")
