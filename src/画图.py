# import os
# import matplotlib
# # 设置临时配置目录，避免权限问题
# os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-tmp'
#
# from matplotlib import font_manager
# import matplotlib.pyplot as plt
# import numpy as np
#
# import matplotlib as mpl
# system_font_dir = "/usr/share/fonts/truetype/msttcorefonts"
# times_fonts = ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]
#
# for font_file in times_fonts:
#     font_path = os.path.join(system_font_dir, font_file)
#     if os.path.exists(font_path):
#         try:
#             font_manager.fontManager.addfont(font_path)
#         except Exception as e:
#             print(f"⚠️ Warning: Could not add {font_file}: {e}")
import os
import matplotlib

# =========================
# 避免服务器权限问题
# =========================
# os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-tmp'
#
# from matplotlib import font_manager
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import FuncFormatter
#
# # =========================
# # Times New Roman 字体注入（服务器版）
# # =========================
# system_font_dir = "/usr/share/fonts/truetype/msttcorefonts"
# times_fonts = ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]
#
# for font_file in times_fonts:
#     font_path = os.path.join(system_font_dir, font_file)
#     if os.path.exists(font_path):
#         try:
#             font_manager.fontManager.addfont(font_path)
#         except Exception as e:
#             print(f"⚠️ Warning: Could not add {font_file}: {e}")
#
# # =========================
# # 全局画图风格（保持你论文统一字号）
# # =========================
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.labelsize'] = 100
# plt.rcParams['xtick.labelsize'] = 80
# plt.rcParams['ytick.labelsize'] = 80
# plt.rcParams['legend.fontsize'] = 65
# plt.rcParams['lines.linewidth'] = 10
# plt.rcParams['lines.markersize'] = 26
#
# # =========================
# # Edge Perturbation 数据
# # =========================
# perturb_ratio = [0, 0.25, 0.5, 0.75, 1.0]
#
# accuracy = [0.87574, 0.86475, 0.86306, 0.86137, 0.86137]
# f1_score = [0.89038, 0.88060, 0.87892, 0.87743, 0.87743]
#
# acc_error = 0.002
# f1_error = 0.002
#
# # =========================
# # 创建画布
# # =========================
# plt.figure(figsize=(20, 16))
# ax = plt.gca()
#
# # 黑色边框（论文风格）
# for spine in ax.spines.values():
#     spine.set_color('black')
#     spine.set_linewidth(4)
#
# # =========================
# # Accuracy
# # =========================
# plt.plot(
#     perturb_ratio, accuracy,
#     marker='o',
#     linestyle='-',
#     color='gold',
#     label='Accuracy'
# )
#
# plt.fill_between(
#     perturb_ratio,
#     [v - acc_error for v in accuracy],
#     [v + acc_error for v in accuracy],
#     color='gold',
#     alpha=0.2
# )
#
# # =========================
# # F1-score
# # =========================
# plt.plot(
#     perturb_ratio, f1_score,
#     marker='s',
#     linestyle='--',
#     color='#e41a1c',
#     label='F1-Score'
# )
#
# plt.fill_between(
#     perturb_ratio,
#     [v - f1_error for v in f1_score],
#     [v + f1_error for v in f1_score],
#     color='#e41a1c',
#     alpha=0.2
# )
#
# # =========================
# # 坐标轴
# # =========================
# plt.xlabel('Perturbation Ratio')
# plt.ylabel('Score (%)')
#
# plt.xticks(perturb_ratio, ['0', '0.25', '0.5', '0.75', '1.0'])
#
# def to_percent(y, _):
#     return f"{y * 100:.1f}"
#
# ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
# plt.ylim(0.858, 0.892)
#
# # =========================
# # 图例
# # =========================
# plt.legend(
#     markerscale=1.2,
#     handlelength=3,
#     loc='upper right',
#     frameon=True,
#     framealpha=0.9,
#     edgecolor='lightgray'
# )
#
# # =========================
# # 导出
# # =========================
# plt.tight_layout()
# plt.savefig(
#     'edge_perturbation_analysis_times_new_1.pdf',
#     dpi=300,
#     bbox_inches='tight',
#     format='pdf'
# )
# plt.show()


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
interval_month = [4, 8, 12, 16, 20, 24]
accuracy = [0.86898, 0.873, 0.87489, 0.8732, 0.8732, 0.86898]
f1_score = [0.8851, 0.88806, 0.88972, 0.88823, 0.88806, 0.88441]


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

plt.xticks(interval_month, ['4', '8', '12', '16', '20', '24'])

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
    'interval_month4.pdf',
    dpi=300,
    bbox_inches='tight',
    format='pdf'
)
plt.show()
