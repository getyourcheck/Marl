import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']  # 首选Arial Unicode MS，然后是黑体和微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置整体风格
plt.style.use('seaborn-v0_8-pastel')

# 使用自定义的数据点
selected_indices = [40, 80, 120, 160, 200]
selected_rewards = [5.17, 5.23, 5.36, 5.55, 5.62]  # 更新的自定义值

# 创建一个图形和子图
fig, ax = plt.subplots(figsize=(12, 8))  # 增加图形高度

# 使用渐变红色方案
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.8, len(selected_rewards)))

# 画柱状图，增加宽度，使其更突出
bars = ax.bar(range(len(selected_rewards)), selected_rewards, width=0.7, color=colors, edgecolor='gray', linewidth=0.7, alpha=0.9)

# 添加标签和标题
ax.set_xlabel('Fine Tuning Checkpoints', fontsize=14, fontweight='bold')
ax.set_ylabel('Scores', fontsize=14, fontweight='bold')
ax.set_title('OpenCompass Benchmark', fontsize=18, fontweight='bold', pad=20)

# 设置x轴刻度 - 格式改为"40step"等
x_labels = [f"{idx}step" for idx in selected_indices]
ax.set_xticks(range(len(selected_rewards)))
ax.set_xticklabels(x_labels, fontsize=13)

# 为每个柱添加数值标签，使其更突出
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 增加垂直偏移
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=13, fontweight='bold')

# 添加网格线使图表更易读
ax.grid(True, linestyle='--', alpha=0.2, axis='y')

# 为柱状图添加渐变效果
for i, bar in enumerate(bars):
    bar.set_zorder(1)  # 确保柱子在网格之上
    
# 设置y轴范围，使柱状图更高
min_value = 5.0  # 固定最小值为5.0，使柱子看起来更高
max_value = 5.8  # 设置较大的最大值，留出足够空间
ax.set_ylim(min_value, max_value)

# 细化背景和边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# 添加3D效果的阴影
for i, bar in enumerate(bars):
    # 主柱体
    bar.set_edgecolor('white')
    bar.set_linewidth(0.8)
    
    # 底部添加稍深色的矩形作为3D效果
    bottom_color = colors[i] * 0.8  # 稍微深一点的颜色
    ax.add_patch(plt.Rectangle((bar.get_x()-0.015, min_value), 
                               bar.get_width()+0.03, 
                               0.02,
                               color=bottom_color,
                               alpha=0.7,
                               zorder=0))

# 保存图片，增加DPI提高质量
plt.tight_layout()
plt.savefig('reward_mean_selected_bar_chart.png', dpi=300, bbox_inches='tight')
plt.show() 