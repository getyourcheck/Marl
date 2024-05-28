import sys
sys.path.extend(["./", "marl/dataset"])
from marl.dataset.txt_loader import TxtMessageDataset
from marl.tokenizer.tokenizer_utils import get_tokenizer
from marl.envs.txt_env import TxtEnv
from marl.repeaters.base import BaseRepeater
from marl.trainer.ppo import PPOTrainer
from marl.coordinator import Coordinator
from marl.config.config import Config

if __name__ == "__main__":

    model_path = "internlm/internlm2-chat-1_8b-sft"
    tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"

    dataset_config = {
        "task_group_filename": "data/config/task_ppo.json",
        "tokenizer": tokenizer,
        "max_seq_len": 1024,
        "num_samples_each_epoch": 8,
        "random_seed": 1,
        "start_token": "[UNUSED_TOKEN_146]user\n",
        "end_token": "[UNUSED_TOKEN_145]\n",
    }

    # init model
    cluster_address = "auto"
    print(f"cluster_address={cluster_address}")
    model_configs_path = "projects/ppo/internlm2/1B/four_model_4gpu.py"
    model_configs = Config.from_file(model_configs_path)
    coordinator = Coordinator(cluster_address, model_configs)
    model_dict = coordinator.create_models()
    sft_model = model_dict["reference"]
    actor_model = model_dict["actor"]
    reward_model = model_dict["reward"]
    critic_model = model_dict["critic"]

    # init txt env
    txt_loader = TxtMessageDataset(**dataset_config)
    txt_env = TxtEnv(dataloader=txt_loader, reward_function=reward_model)
    # init repeater
    rl_repeater = BaseRepeater(sft_model=sft_model, reward_scale=False, fine_grained_rm=False, value_ema=False)
    # init trainer
    ppo = PPOTrainer(policy_model=actor_model, value_model=None)
    
    while True:
        for _ in range(2):
            trajectories = txt_env.rollout(policy_model=actor_model)
            trajectories = rl_repeater.process(trajectories, policy_model=actor_model, value_model=critic_model, sft_model=None)
            ppo_loss = ppo.policy_learn(trajectories, actor_model)
            value_loss = ppo.value_learn(trajectories, critic_model)
        break
