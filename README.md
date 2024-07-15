# Marl: Models Augmented by Reinforcement Learning

## Usage

```bash

# 跑通基于 Ray 的 PPO（使用example—7B config）
nohup python -u main.py -c projects/internlm2/example_7B_ds_vllm_32gpu.py > nohup_7b.log 2>&1 &
```

## WIP

- [ ] support sp for hugging face
- [ ] support internevo+vllm
- [ ] mincro_batch remove padding while infering & training

## Outline

1. RL Algorithm 部分:

   1. 环境：

      1. 实现 BaseDataset 基类及 TXTDataset 衍生类

      ```python
      def __init__(data_path, ratio=0.5, size=1000)
      def sample(batch_size)
      ```
      2. 实现 BaseEnv 基类及 TXTEnv 衍生类

      ```python
      def __init__(Dataset, models)
      def rollout(nums=512):
      ```
   2. 算法：

      1. 实现 RepeaterBase 基类
      2. 实现 RLLearner基类及 PPOLearner 衍生类
2. Coordinator 部分:

   1. 实现 BaseModel 及其衍生类
   2. 实现 ActorModel 和 CriticModel 的 训练方法，预留接入 InternLM & DeepSpeed Trainer 的可扩展性。
   3. 实现 ActorModel 的 生成方法，预留接入其他开源框架的推理引擎 (e.g., vLLM) 的可扩展性。
   4. 实现 Coordinator 的 创建模型 和 调度方法
3. Trainer 和 Inferer 的对接：

   1. Trainers: HuggingFace Accelerator, InternEvo, DeepSpeed...
   2. Inferers: vLLM, LMDeploy, ...
4. 其他

   1. 启动config projects
   2. 监控、日志、工具
   3. 测试 tests
