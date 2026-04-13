import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# ------------------------
# Data
# ------------------------
ratios = ['0%', '25%', '50%', '75%', '100%']
accuracy = [0.87489, 0.86475, 0.86306, 0.86137, 0.86052]
f1_score = [0.88972, 0.88060, 0.87892, 0.87743, 0.87659]

x = np.arange(len(ratios))
width = 0.29

# ------------------------
# Global style
# ------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 20  # 图例放大

# ------------------------
# Figure and axis
# ------------------------
fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Colors similar to HCGBot figure
blue = '#1f77b4'
orange = '#ff7f0e'

# ------------------------
# Bars
# ------------------------
bars1 = ax.bar(
    x - width/2, accuracy, width,
    label='Accuracy',
    color=blue,
    edgecolor='black',
    linewidth=0.8,
    hatch='/',
    zorder=3
)

bars2 = ax.bar(
    x + width/2, f1_score, width,
    label='F1-score',
    color=orange,
    edgecolor='black',
    linewidth=0.8,
    hatch='\\',
    zorder=3
)

# ------------------------
# Axes settings
# ------------------------
ax.set_xlabel('Edge Perturbation Ratio')
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(ratios)

# Keep decimal range here; formatter will display percentages
ax.set_ylim(0.84, 0.897)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

# Gridlines
ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.28, zorder=0)
ax.set_axisbelow(True)

# Thinner border lines
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

# Legend at top center
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.0),
    ncol=2,
    frameon=False,
    handlelength=1.6,
    columnspacing=1.5
)

# ------------------------
# Value labels (percentage format)
# ------------------------
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.00045,
            f'{h*100:.1f}',
            ha='center',
            va='bottom',
            fontsize=18
        )

# ------------------------
# Layout and save
# ------------------------
plt.tight_layout()

# 保存为 PDF
plt.savefig('edge_perturbation_bar_chart.pdf', format='pdf', bbox_inches='tight')

plt.show()