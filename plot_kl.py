import json
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = []
with open('train_rlhf.log.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# 提取数据
steps = [d['step'] for d in data]
kl = [d['kl'] for d in data]
kl_gen_train = [d['kl(gen, train)'] for d in data]

# 创建图表
plt.figure(figsize=(12, 6))
plt.plot(steps, kl, label='KL Divergence', marker='o', markersize=3, alpha=0.7)
plt.plot(steps, kl_gen_train, label='KL(gen, train)', marker='o', markersize=3, alpha=0.7)

# 设置每20步显示一个刻度
plt.xticks(np.arange(0, max(steps)+1, 20))

# 添加标题和标签
plt.title('KL Divergence during training', fontsize=14)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('KL Divergence Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# 优化布局
plt.tight_layout()

# 保存图表
plt.savefig('kl_divergence.png', dpi=300, bbox_inches='tight')
plt.close() 