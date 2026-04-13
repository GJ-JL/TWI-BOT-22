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
# 全局画图风格（完全沿用第一个代码）
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 100
plt.rcParams['xtick.labelsize'] = 80
plt.rcParams['ytick.labelsize'] = 80
plt.rcParams['legend.fontsize'] = 65
plt.rcParams['lines.linewidth'] = 10
plt.rcParams['lines.markersize'] = 26

# 建议：避免某些 PDF 查看器把字体渲染得发虚
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# =========================
# Interval 数据（来自你第二段代码）
# =========================
interval_month = [2, 4, 8, 12, 16, 20, 24]
accuracy = [0.86898, 0.87236, 0.87300, 0.87489, 0.87320, 0.87320, 0.86898]
f1_score  = [0.88510, 0.88790, 0.88806, 0.88972, 0.88823, 0.88806, 0.88441]

acc_error = 0.002
f1_error  = 0.002

# =========================
# 创建画布（沿用第一个代码）
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
    interval_month, accuracy,
    marker='o',
    linestyle='-',
    color='gold',
    label='Accuracy'
)
plt.fill_between(
    interval_month,
    [v - acc_error for v in accuracy],
    [v + acc_error for v in accuracy],
    color='gold',
    alpha=0.2
)

# =========================
# F1-score
# =========================
plt.plot(
    interval_month, f1_score,
    marker='s',
    linestyle='--',
    color='#e41a1c',
    label='F1-Score'
)
plt.fill_between(
    interval_month,
    [v - f1_error for v in f1_score],
    [v + f1_error for v in f1_score],
    color='#e41a1c',
    alpha=0.2
)

# =========================
# 坐标轴
# =========================
plt.xlabel('Interval (month)')
plt.ylabel('Score (%)')

plt.xticks(interval_month, [str(x) for x in interval_month])

def to_percent(y, _):
    return f"{y * 100:.1f}"

ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.ylim(0.860, 0.895)

# =========================
# 图例（格式沿用第一个）
# =========================
plt.legend(
    markerscale=1.2,
    handlelength=3,
    loc='lower right',   # 如果想完全跟第一个一样可改成 'upper right'
    frameon=True,
    framealpha=0.9,
    edgecolor='lightgray'
)

# =========================
# 导出
# =========================
plt.tight_layout()
plt.savefig(
    'interval_month2.pdf',
    dpi=300,
    bbox_inches='tight',
    format='pdf'
)
plt.show()
