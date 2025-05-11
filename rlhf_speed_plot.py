# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
# 调整图表比例，使其更高一些
plt.figure(figsize=(10, 7), dpi=100)

# 性能指标数据
xt_scores = [54.3, 102.4, 249.4]
ds_scores = [61.8, 131.2, 316.6]
settings = ['Batch Size: 128', 'Batch Size: 256', 'Batch Size: 512']

# 设置条形图的位置
x = np.arange(len(settings))
bar_width = 0.35

# 绘制两组条形图
xt_bars = plt.bar(x - bar_width/2, xt_scores, bar_width, label='Marl-RLHF', color='#4c72b0')
ds_bars = plt.bar(x + bar_width/2, ds_scores, bar_width, label='DeepSpeed-Chat', color='#dd8452')

# 在条形图上添加数值标签
for bar in xt_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{height:.1f}', ha='center', va='bottom')

for bar in ds_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{height:.1f}', ha='center', va='bottom')

# 添加百分比变化标签（DeepSpeed相对于XTuner的变化）
for i in range(len(settings)):
    # 计算百分比变化 (DeepSpeed与XTuner的比较)
    pct = (ds_scores[i] - xt_scores[i]) / xt_scores[i] * 100
    # 判断百分比的符号，确保显示正确的增减方向
    sign = '+' if pct > 0 else ''
    # 放置在两个条形图之间，高度适中
    plt.text(x[i], min(xt_scores[i], ds_scores[i]) + abs(ds_scores[i] - xt_scores[i])/2,
             f'{sign}{pct:.1f}%', ha='center', va='center')

# 设置X轴刻度位置和标签
plt.xticks(x, settings)

# 将图例移到右上角
plt.legend(loc='upper right')

# 添加标题和标签
plt.title('PPO Batch Latency: Marl vs. DeepSpeed-Chat on Internlm 1.8B', fontsize=16)
plt.xlabel('Settings: (1) Dataset: Anthropic/hh-rlhf; (2) Generation: 1024--2048; (3) 8 x 4090 (24G)', fontsize=12)
plt.ylabel('Duration Per Iteration (sec)', fontsize=14)
plt.ylim(0, 400)
plt.grid(alpha=0.3, linestyle='-.')

# 调整布局，确保所有元素可见
plt.tight_layout()

# 显示图表
plt.show()
