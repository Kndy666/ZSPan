import matplotlib.pyplot as plt
import numpy as np

# 数据
warmup = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
d_lambda = np.array([0.0266, 0.0221, 0.0221, 0.0215, 0.0209, 0.0210, 0.0211, 0.0210, 0.0209])
ds = np.array([0.0263, 0.0132, 0.0150, 0.0147, 0.0138, 0.0138, 0.0137, 0.0139, 0.0140])
hqnr = np.array([0.9478, 0.9650, 0.9633, 0.9641, 0.9657, 0.9656, 0.9655, 0.9655, 0.9654])

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
ax1.set_xlabel(r'$m$', fontsize=14, fontweight='bold',
               color='black', labelpad=15)
ax1.set_ylabel('HQNR', fontsize=14, fontweight='bold',
               color=hqnr_color, labelpad=15)
ax1.tick_params(axis='y', which='major', labelsize=12, colors=hqnr_color)
ax1.set_xticks(np.arange(0, 45, 5))
ax1.set_ylim(0.94, 0.97)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 设置五角星颜色（这里选择金色）
star_color = "#fa87a1"

# 在 bar 图中添加注释：最佳效果在 warmup=20
best_index = np.where(warmup == 20)[0][0]
ax1.annotate('best point',
             xy=(20, hqnr[best_index]),
             xytext=(21, 0.966),
             textcoords='data',
             fontsize=14, fontweight='bold', color=star_color)

# 在最佳点处绘制一个五角星标记
ax1.scatter(20, hqnr[best_index], marker=(5, 1), s=200, color=star_color, zorder=10)

# 添加更粗且透明的虚线标注 warmup=20 的位置
ax1.axvline(x=20, color='black', linestyle='--', linewidth=2, alpha=0.5)

# 右侧坐标轴绘制 Dλ 与 Ds 折线图
ax2 = ax1.twinx()
ax2.plot(warmup, d_lambda, marker='o', color=d_lambda_color, linewidth=2,
         markersize=6, markerfacecolor='white', label=r'$D_{\lambda} \downarrow$')
ax2.plot(warmup, ds, marker='s', color=ds_color, linewidth=2,
         markersize=6, markerfacecolor='white', label=r'$D_{s} \downarrow$',linestyle='--')
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
