import re
import matplotlib.pyplot as plt
import numpy as np

# 读取日志文件
steps = []
durations = []

with open('train_rlhf.log', 'r') as f:
    for line in f:
        # 使用正则表达式匹配duration值
        match = re.search(r'\[end to end\] duration: ([\d.]+) s', line)
        if match:
            duration = float(match.group(1))
            durations.append(duration)
            steps.append(len(steps))  # 使用索引作为步骤数

# 创建图表
plt.figure(figsize=(12, 6))
plt.plot(steps, durations, marker='o', markersize=3, alpha=0.7)

# 设置每20步显示一个刻度
plt.xticks(np.arange(0, max(steps)+1, 20))

# 添加标题和标签
plt.title('Training Duration', fontsize=14)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Duration (seconds)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 优化布局
plt.tight_layout()

# 保存图表
plt.savefig('training_duration.png', dpi=300, bbox_inches='tight')
plt.close() 