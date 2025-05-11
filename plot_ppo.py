import re
import matplotlib.pyplot as plt
import numpy as np

# 读取并解析日志
log_path = 'train_rlhf.log'
steps, critic_losses, rewards = [], [], []
ppo_dict = {}
with open(log_path, 'r') as f:
    for line in f:
        if (m := re.search(r'\[Critic Train\] step:\s*(\d+), critic loss: \[([\d\.E+-]+)\]', line)):
            steps.append(int(m.group(1)))
            critic_losses.append(float(m.group(2)))
        elif (m := re.search(r'rewards:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)):
            rewards.append(float(m.group(1)))
        elif (m := re.search(r'\[Policy Train\] Step:\s*(\d+), ppo loss: \[([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\]', line)):
            ppo_dict[int(m.group(1))] = float(m.group(2))

# 对齐数据长度
min_len = min(len(steps), len(rewards))
steps = steps[:min_len]
rewards = rewards[:min_len]
critic_losses = critic_losses[:min_len]
max_step = max(steps)

# 构建PPO数据
ppo_steps_full = np.arange(max_step + 1)
ppo_losses_full = [ppo_dict.get(s, 0.0) for s in ppo_steps_full]

# 绘图
fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

def plot_dense_markers(x, y, ax, color):
    ax.plot(x, y, color=color, linewidth=1.2)
    ax.plot(x[::2], y[::2], linestyle='None', marker='o', markersize=3, color=color)

# Reward Mean
plot_dense_markers(steps, rewards, axs[0], 'tab:blue')
axs[0].set_title('Reward Mean over Steps')
axs[0].set_ylabel('Reward Mean')
axs[0].grid(alpha=0.3)
axs[0].xaxis.set_tick_params(labelbottom=True)  # 确保显示刻度标签

# PPO Loss
plot_dense_markers(ppo_steps_full, ppo_losses_full, axs[1], 'tab:orange')
axs[1].set_title('PPO Loss over Steps')
axs[1].set_ylabel('PPO Loss')
axs[1].grid(alpha=0.3)
axs[1].xaxis.set_tick_params(labelbottom=True)  # 确保显示刻度标签

# Critic Loss
plot_dense_markers(steps, critic_losses, axs[2], 'tab:green')
axs[2].set_title('Critic Loss over Steps')
axs[2].set_ylabel('Critic Loss')
axs[2].set_xlabel('Step')
axs[2].grid(alpha=0.3)
axs[2].xaxis.set_tick_params(labelbottom=True)

# 设置横坐标刻度，每20步
xticks = np.arange(0, max_step + 1, 20)
for ax in axs:
    ax.set_xticks(xticks)

plt.tight_layout()
plt.show()
