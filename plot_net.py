import matplotlib.pyplot as plt
import numpy as np

# 新数据
warmup = np.array([8, 16, 24, 32, 40, 48, 56, 64])
d_lambda = np.array([0.0194, 0.0204, 0.0208, 0.0209, 0.0208, 0.0215, 0.0215, 0.0217])
ds = np.array([0.0228, 0.0161, 0.0142, 0.0138, 0.0142, 0.0146, 0.0144, 0.0144])
hqnr = np.array([0.9583, 0.9639, 0.9653, 0.9657, 0.9653, 0.9642, 0.9644, 0.9643])

fig, ax1 = plt.subplots(figsize=(10, 6))

# 使用 RGB 转换为十六进制颜色
hqnr_color = "#5b9bd5"    # 用于条形图和左侧文字
d_lambda_color = "#c55a11"  # 用于 Dλ 折线及右侧文字
d_lambdaword_color = "#ff7900"
ds_color = "#ff6d00"      # 用于 Ds 折线

# 绘制 HQNR 条形图，去除边框
bar_width = 3
bars = ax1.bar(warmup, hqnr, width=bar_width, color=hqnr_color, alpha=0.85,
               edgecolor=None, label='HQNR ↑')
ax1.set_xlabel(r'$d$', fontsize=14, fontweight='bold',
               color='black', labelpad=15)
ax1.set_ylabel('HQNR', fontsize=14, fontweight='bold',
               color=hqnr_color, labelpad=15)
ax1.tick_params(axis='y', which='major', labelsize=12, colors=hqnr_color)
ax1.set_xticks(np.arange(0, 70, 8))
ax1.set_ylim(0.94, 0.97)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 设置五角星颜色（这里选择金色）
star_color = "#fa87a1"

# 在 bar 图中添加注释：最佳效果在 warmup=32
best_index = np.where(warmup == 32)[0][0]
ax1.annotate('best point',
             xy=(32, hqnr[best_index]),
             xytext=(33, 0.966),
             textcoords='data',
             fontsize=14, fontweight='bold', color=star_color)

# 在最佳点处绘制一个五角星标记
ax1.scatter(32, hqnr[best_index], marker=(5, 1), s=200, color=star_color, zorder=10)

# 添加更粗且透明的虚线标注 warmup=32 的位置
ax1.axvline(x=32, color='black', linestyle='--', linewidth=2, alpha=0.5)

# 右侧坐标轴绘制 Dλ 与 Ds 折线图
ax2 = ax1.twinx()
ax2.plot(warmup, d_lambda, marker='o', color=d_lambda_color, linewidth=2,
         markersize=6, markerfacecolor='white', label=r'$D_{\lambda} \downarrow$')
ax2.plot(warmup, ds, marker='s', color=ds_color, linewidth=2,
         markersize=6, markerfacecolor='white', label=r'$D_{s} \downarrow$', linestyle='--')
ax2.set_ylabel(r'$D_{\lambda}$ & $D_{s}$', fontsize=14, fontweight='bold',
               color=d_lambdaword_color, labelpad=15)
ax2.tick_params(axis='y', which='major', labelsize=12, colors=d_lambdaword_color)
ax2.set_ylim(0.01, 0.0325)  # 修改右侧纵轴最大值为 0.0325
ax2.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# 合并图例并放置在左上角，启用边框，缩小图例标记
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
           fontsize=12, frameon=True, markerscale=0.75)

plt.tight_layout()
plt.show()
