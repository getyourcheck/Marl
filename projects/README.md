Projects are for **internal** usage.

Public [examples](../examples) are not available yet.

## Config explain

### Batch Size -> Mini Batch Size -> Micro Batch Size
- Lv0 (exp): dataset_config.num_samples_each_epoch = 512
    - Lv1 (ppo)
    - ppo_minibatch = 512  # answers to generate
    - value_minibatch = 512  # answer to score
        - Lv2 (model server)
        - [xxxx]_micro_bs = 32 # answers to infer/train/generate per round, 512 / 32 = 16 rounds
        - in `rollout_config`: actor_micro_bs is for generation, reward_micro_bs is for scoring 
        - in `repeater_config`: actor/ref/critic_micro_bs is for inference
        - in `train_config`: actor/critic_micro_bs is for training