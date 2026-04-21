import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

df = pd.read_csv("category_accuracy_by_config.csv")
category_names = {'number': '数字/数量', 'text_ocr': '文本OCR', 'brand': '品牌/商标'}
df['category_cn'] = df['category'].map(category_names)

# 请替换为第一步实际输出的 baseline 各类别准确率
baseline_acc = {
    'number': 0.6104,
    'text_ocr': 0.5673,
    'brand': 0.5913
    
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = {0.3: '#1f77b4', 0.5: '#ff7f0e', 0.7: '#2ca02c'}

for ax, (cat, cat_cn) in zip(axes, category_names.items()):
    sub = df[df['category'] == cat].copy()
    tokens = [16, 32, 64]
    alphas = [0.3, 0.5, 0.7]
    x = np.arange(len(tokens))
    width = 0.25
    multiplier = 0

    for alpha in alphas:
        data = sub[sub['alpha'] == alpha].sort_values('tokens')
        if data.empty:
            continue
        acc = data['accuracy'].values
        offset = width * multiplier
        rects = ax.bar(x + offset, acc, width, label=f'α={alpha}', color=colors[alpha])
        ax.bar_label(rects, fmt='%.3f', padding=3, fontsize=9)
        multiplier += 1

    # 添加 Baseline 水平虚线
    ax.axhline(y=baseline_acc[cat], color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc[cat]:.3f})')

    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{t} tokens' for t in tokens])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel('准确率')
    ax.set_title(cat_cn, pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', framealpha=0.9)

fig.suptitle('MMTok 分类准确率对比（含 Baseline）', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('category_accuracy_with_baseline.png', dpi=300, bbox_inches='tight', facecolor='white')
print("图片已保存至 category_accuracy_with_baseline.png")