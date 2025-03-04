import matplotlib.pyplot as plt
import numpy as np

# Updated data
wv3_ratio = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
d_lambda = np.array([0.0230, 0.0229, 0.0218, 0.0219, 0.0213, 0.0217, 0.0213, 0.0211, 0.0212])
ds = np.array([0.0346, 0.0238, 0.0212, 0.0179, 0.0176, 0.0141, 0.0144, 0.0137, 0.0141])
hqnr = np.array([0.9433, 0.9539, 0.9575, 0.9607, 0.9616, 0.9646, 0.9645, 0.9655, 0.9651])

fig, ax1 = plt.subplots(figsize=(10, 6))

# Color setup
hqnr_color = "#5b9bd5"    # For bar chart and left-side text
d_lambda_color = "#c55a11"  # For Dλ line and right-side text
d_lambdaword_color = "#ff7900"
ds_color = "#ff6d00"      # For Ds line
highlight_color = "#fa87a1"  # Special color for best point

# Plot HQNR as a bar chart
bar_width = 0.05  # Default bar width
bars = ax1.bar(wv3_ratio, hqnr, width=bar_width, color=hqnr_color, alpha=0.85,
               edgecolor=None, label='HQNR ↑')

# Remove the special handling for the bar at 0.8, making it consistent with others
highlight_index = np.where(wv3_ratio == 0.8)[0][0]
bars[highlight_index].set_width(bar_width)  # Set the same width as other bars
bars[highlight_index].set_color(hqnr_color)  # Set the color back to the original

ax1.set_xlabel(r'$p$', fontsize=14, fontweight='bold',
               color='black', labelpad=15)
ax1.set_ylabel('HQNR', fontsize=14, fontweight='bold',
               color=hqnr_color, labelpad=15)
ax1.tick_params(axis='y', which='major', labelsize=12, colors=hqnr_color)
ax1.set_xticks(np.arange(0, 1.1, 0.1))
ax1.set_ylim(0.94, 0.97)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Remove the five-point star annotation and just add a label for the best point
ax1.annotate('best point ',
             xy=(wv3_ratio[highlight_index], hqnr[highlight_index]),
             xytext=(wv3_ratio[highlight_index] , hqnr[highlight_index] + 0.001),
             textcoords='data',
             fontsize=14, fontweight='bold', color=highlight_color)

# Adjust the position of the star at the center of the bar at 0.8
star_y = hqnr[highlight_index]  # Place it at the center of the bar's height
ax1.plot(wv3_ratio[highlight_index], star_y, marker='*', markersize=15, color=highlight_color)  # No legend for this star

# Plot Dλ and Ds as lines on the second y-axis
ax2 = ax1.twinx()
ax2.plot(wv3_ratio, d_lambda, marker='o', color=d_lambda_color, linewidth=2,
         markersize=6, markerfacecolor='white', label=r'$D_{\lambda} \downarrow$')
ax2.plot(wv3_ratio, ds, marker='s', color=ds_color, linewidth=2,
         markersize=6, markerfacecolor='white', label=r'$D_{s} \downarrow$', linestyle='--')
ax2.set_ylabel(r'$D_{\lambda}$ & $D_{s}$', fontsize=14, fontweight='bold',
               color=d_lambdaword_color, labelpad=15)
ax2.tick_params(axis='y', which='major', labelsize=12, colors=d_lambdaword_color)
ax2.set_ylim(0.01, 0.045)  # Adjust the right y-axis max value to 0.045
ax2.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Merge legends and place them at the top left corner (excluding the star legend)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
           fontsize=12, frameon=True, markerscale=0.75)

plt.tight_layout()
plt.show()
