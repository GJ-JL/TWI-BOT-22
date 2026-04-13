import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-tmp'

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# =========================
# Times New Roman 字体注入（服务器版）
# =========================
system_font_dir = "/usr/share/fonts/truetype/msttcorefonts"
times_fonts = ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]

for font_file in times_fonts:
    font_path = os.path.join(system_font_dir, font_file)
    if os.path.exists(font_path):
        try:
            font_manager.fontManager.addfont(font_path)
        except Exception as e:
            print(f"⚠️ Warning: Could not add {font_file}: {e}")

# =========================
# 全局画图风格（保持你论文统一字号）
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 75
plt.rcParams['xtick.labelsize'] = 70
plt.rcParams['ytick.labelsize'] = 70
plt.rcParams['legend.fontsize'] = 60
plt.rcParams['lines.linewidth'] = 10
plt.rcParams['lines.markersize'] = 26

# =========================
# Edge Perturbation 数据
# =========================
# interval_month = [2, 4, 8, 12, 16, 20, 24]
# accuracy = [0.86898, 0.87236, 0.873, 0.87489, 0.8732, 0.8732, 0.86898]
# f1_score = [0.8851, 0.8879, 0.88806, 0.88972, 0.88823, 0.88806, 0.88441]
ratios = ['1:1', '1:2', '1:3', '1:5', '1:8', '1:10']
accuracy = [0.87151, 0.87236, 0.87405, 0.87489, 0.8732, 0.87151]
f1_score = [0.8864, 0.88723, 0.88889, 0.88972, 0.88823, 0.88657]


acc_error = 0.002
f1_error = 0.002

# =========================
# 创建画布
# =========================
plt.figure(figsize=(20, 16))
ax = plt.gca()

# 黑色边框（论文风格）
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(4)

# =========================
# Accuracy
# =========================
plt.plot(
    ratios, accuracy,
    marker='o',
    linestyle='-',
    color='#2c7fb8',
    label='Accuracy'
)
#
# plt.fill_between(
#     ratios,
#     [v - acc_error for v in accuracy],
#     [v + acc_error for v in accuracy],
#     color='gold',
#     alpha=0.2
# )

# =========================
# F1-score
# =========================
plt.plot(
    ratios, f1_score,
    marker='s',
    linestyle='--',
    color='#e41a1c',
    label='F1-Score'
)

# plt.fill_between(
#     ratios,
#     [v - f1_error for v in f1_score],
#     [v + f1_error for v in f1_score],
#     color='#e41a1c',
#     alpha=0.2
# )

# =========================
# 坐标轴
# =========================
plt.xlabel('Positive–Negative Ratio')
plt.ylabel('Score (%)')

plt.xticks(ratios , ['1:1', '1:2', '1:3', '1:5', '1:8', '1:10'])

def to_percent(y, _):
    return f"{y * 100:.1f}"

ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.ylim(0.858, 0.892)

# =========================
# 图例
# =========================
plt.legend(
    markerscale=1.2,
    handlelength=3,
    loc='lower right',
    frameon=True,
    framealpha=0.9,
    edgecolor='lightgray'
)

# =========================
# 导出
# =========================
plt.tight_layout()
plt.savefig(
    'Positive–Negative Ratio.pdf',
    dpi=300,
    bbox_inches='tight',
    format='pdf'
)
plt.show()
