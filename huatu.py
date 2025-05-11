import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 输出文件名
save_path = "./marl_framework_beautified_code.png"

# 画布设置
fig, ax = plt.subplots(figsize=(8, 10))
ax.axis("off")                        # 关闭坐标轴

# ——— 辅助函数：绘制某一层（带子单元） ———
def draw_layer(y_top, height, sub_titles, layer_title,
               face="#FFFFFF", edge="#333"):
    """y_top: 顶部 y；height: 高度；sub_titles: 子格标题列表"""
    total_w, x0 = 7.5, 0.5           # 总宽度 & 左起点
    # 外框
    ax.add_patch(
        patches.FancyBboxPatch(
            (x0, y_top - height), total_w, height,
            boxstyle="round,pad=0.015",
            edgecolor=edge, facecolor=face, linewidth=1.2
        )
    )
    # 层标题（居中）
    ax.text(x0 + total_w / 2, y_top - 0.25 * height,
            layer_title, ha="center", va="center",
            fontsize=11, fontweight="semibold")

    # 子单元
    sub_h = height * 0.55
    sub_y = y_top - height + 0.10 * height
    sub_w = total_w / len(sub_titles)
    for i, t in enumerate(sub_titles):
        xi = x0 + i * sub_w
        if i:                        # 竖线分隔
            ax.plot([xi, xi], [sub_y, sub_y + sub_h],
                    color=edge, lw=1)
        ax.text(xi + sub_w / 2, sub_y + sub_h / 2,
                t, ha="center", va="center",
                fontsize=9, linespacing=1.3)

# ——— 逐层绘制 ———
gap, H = 1.3, 1.6
y = 9.3

draw_layer(
    y, H,
    ["Policy\nModel", "Reference\nModel", "Reward\nModel",
     "Critic\nModel", "资源调度\n与同步机制"],
    "Coordinator（协调器）")
y -= H + gap

draw_layer(
    y, H,
    ["BaseModel\nServer", "PolicyModel\nServer",
     "RewardModel\nServer", "ReferenceModel\nServer"],
    "Model Server（模型服务器）")
y -= H + gap

draw_layer(
    y, H,
    ["HuggingFace模型", "vLLM高性能推理", "InternEvo后端"],
    "Model Backend（模型后端）")
y -= H + gap

draw_layer(
    y, H,
    ["TxtEnv（环境）", "LOORepeater", "RLOOTrainer",
     "PPOLoss /\nRLOOLoss"],
    "RL Components（强化学习组件）")

# ——— 连接箭头（垂直主链路） ———
center_x = 4.25
ys = [9.3 - H, 9.3 - (H + gap) - H,
      9.3 - 2 * (H + gap) - H,
      9.3 - 3 * (H + gap) - H]
for yy in ys:
    ax.annotate("", xy=(center_x, yy - 0.1),
                xytext=(center_x, yy - gap + 0.1),
                arrowprops=dict(arrowstyle='-|>', lw=1.2, color="#333"))

plt.tight_layout()
plt.savefig(save_path, dpi=300)
print("Saved ->", save_path)
