import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['Noto Serif CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 整理数据（来自实验报告的10个Epoch指标，两个模型数据完全一致）
epochs = np.arange(1, 11)  # 1-10个Epoch
train_loss = [0.1591, 0.0405, 0.0367, 0.0347, 0.0333, 0.0322, 0.0314, 0.0308, 0.0304, 0.0301]
val_loss = [0.0356, 0.0329, 0.0319, 0.0315, 0.0311, 0.0304, 0.0301, 0.0299, 0.0298, 0.0297]

# 创建画布与子图
plt.figure(figsize=(10, 6))
plt.grid(True, alpha=0.3, linestyle='--')  # 添加网格（增强可读性）

# 绘制折线图
plt.plot(epochs, train_loss, label='Train Loss',
         color='#2E86AB', linewidth=2.5, marker='o', markersize=6)
plt.plot(epochs, val_loss, label='Val Loss',
         color='#A23B72', linewidth=2.5, marker='s', markersize=6)

# 设置标题与坐标轴标签
plt.title('Transformer Cross-Entropy Loss', fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)

# 调整坐标轴范围（让趋势更清晰）
plt.xlim(0.8, 10.2)
plt.ylim(0.028, 0.17)

# 添加图例（位置在右上角）
plt.legend(loc='upper right', fontsize=11)

# 标注最终损失值（文件中记录的最终指标）
plt.annotate(f'Final Train Loss: 0.0301', xy=(10, 0.0301), xytext=(9, 0.04),
             arrowprops=dict(arrowstyle='->', color='#2E86AB', alpha=0.7), fontsize=10)
plt.annotate(f'Final Val Loss: 0.0297', xy=(10, 0.0297), xytext=(9, 0.035),
             arrowprops=dict(arrowstyle='->', color='#A23B72', alpha=0.7), fontsize=10)

# 保存图片（分辨率300dpi，适配报告插入）
plt.tight_layout()
plt.savefig('results/loss_curve_final.png', dpi=300, bbox_inches='tight')
plt.show()