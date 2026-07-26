"""Generate the four comparison plots for the revamped deck, from numbers
verified and logged in docs/V3DEV_LOG.md. No new evals run here -- this is
pure presentation of already-measured results.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'results/plots'
os.makedirs(OUT, exist_ok=True)

INK = '#1a1a1a'
GRAY = '#888888'
LIGHT = '#cccccc'
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'text.color': INK,
    'axes.edgecolor': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'axes.grid': True,
    'grid.color': LIGHT, 'grid.linewidth': 0.6, 'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

V2_EPE = 2.324

# ---------------------------------------------------------------- Plot A
# Training curriculum: EPE across every stage, in chronological/logical order
stages = [
    ('Untrained\n(bilinear init)', 2.476, 'local'),
    ('VKITTI2\nonly', 2.388, 'local'),
    ('FlyingChairs\nonly', 2.275, 'local'),
    ('Chairs -> VKITTI2\n(sequential)', 2.499, 'local'),
    ('Chairs + VKITTI2\n(mixed, local)', 2.183, 'local'),
    ('grandmix\n(HPC, +Sintel)', 2.166, 'hpc'),
    ('big18\n(HPC, scaled up)', 2.072, 'hpc'),
    ('uncG\n(HPC, +uncertainty)', 2.082, 'hpc'),
    ('spring\n(HPC, 70% done)', 2.080, 'hpc_partial'),
]
labels = [s[0] for s in stages]
vals = [s[1] for s in stages]
color_map = {'local': '#999999', 'hpc': INK, 'hpc_partial': '#555555'}
colors = [color_map[s[2]] for s in stages]
hatches = ['//' if s[2] == 'hpc_partial' else None for s in stages]

fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
bars = ax.bar(labels, vals, color=colors, width=0.6, zorder=3)
for bar, h in zip(bars, hatches):
    if h:
        bar.set_hatch(h)
        bar.set_edgecolor('white')
ax.axhline(V2_EPE, color='#b03030', linestyle='--', linewidth=1.5, zorder=2)
ax.text(len(labels) - 0.4, V2_EPE + 0.025, f'NeuFlow v2 reference: {V2_EPE:.3f} px',
        color='#b03030', fontsize=10, ha='right', va='bottom')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v:.3f}',
            ha='center', fontsize=9.5, color=INK)
ax.set_ylabel('Mean end-point error, px (lower is better)')
ax.set_title('The full training curriculum, in order', loc='left', fontsize=13, fontweight='bold')
ax.set_ylim(2.0, 2.6)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#999999', label='Local (RTX 4060, batch 4)'),
                   Patch(color=INK, label='HPC (Explorer cluster, batch 16, 100K steps)'),
                   Patch(facecolor='#555555', hatch='//', edgecolor='white', label='HPC, truncated at 70K/100K (8h wall limit)')],
          loc='upper right', frameon=False, fontsize=9)
plt.xticks(rotation=0, fontsize=9.5)
plt.tight_layout()
plt.savefig(f'{OUT}/curriculum_epe.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/curriculum_epe.png')

# ---------------------------------------------------------------- Plot B
# 1px / 3px accuracy: v2 vs the three HPC checkpoints
names = ['NeuFlow v2', 'grandmix', 'big18', 'uncG']
acc1 = [77.6, 76.25, 77.02, 77.51]
acc3 = [89.8, 89.48, 89.91, 90.02]
x = np.arange(len(names))
w = 0.32

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
b1 = ax.bar(x - w/2, acc1, width=w, color=INK, label='1px accuracy', zorder=3)
b3 = ax.bar(x + w/2, acc3, width=w, color='#999999', label='3px accuracy', zorder=3)
for bars in (b1, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, f'{h:.2f}',
                ha='center', fontsize=9, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel('% of pixels within threshold')
ax.set_ylim(74, 93)
ax.set_title('Precision: closing the sub-pixel gap', loc='left', fontsize=13, fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT}/precision_bars.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/precision_bars.png')

# ---------------------------------------------------------------- Plot C
# Speed: v2 dense vs v3 first-query vs v3 repeat-query, same V100
speed_labels = ['NeuFlow v2\n(every call)', 'NeuFlow v3\n(first query,\nnew frame)', 'NeuFlow v3\n(repeat query,\nsame frame)']
speed_ms = [19.6, 19.1, 2.65]
speed_colors = ['#b03030', INK, INK]

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
bars = ax.bar(speed_labels, speed_ms, color=speed_colors, width=0.55, zorder=3)
for bar, v in zip(bars, speed_ms):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.4, f'{v:.1f} ms',
            ha='center', fontsize=10.5, color=INK, fontweight='bold')
ax.annotate('~7x cheaper\n(no cached state in v2)',
            xy=(2, 2.65), xytext=(1.55, 9),
            fontsize=9.5, color=INK, ha='center',
            arrowprops=dict(arrowstyle='->', color=INK, lw=1))
ax.set_ylabel('Latency, ms (RTX-class V100, 384x1248)')
ax.set_title('Same hardware, same input: cost per query', loc='left', fontsize=13, fontweight='bold')
ax.set_ylim(0, 23)
plt.tight_layout()
plt.savefig(f'{OUT}/speed_bars.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/speed_bars.png')

# ---------------------------------------------------------------- Plot D
# Calibration: predicted b bins vs real error
bin_labels = ['[0.01, 0.14]', '[0.14, 0.27]', '[0.27, 0.54]', '[0.54, 1.48]', '[1.48, 172.2]']
bin_err = [0.221, 0.334, 0.600, 1.410, 7.377]

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
bars = ax.bar(bin_labels, bin_err, color=INK, width=0.6, zorder=3)
for bar, v in zip(bars, bin_err):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.12, f'{v:.2f} px',
            ha='center', fontsize=10, color=INK)
ax.set_xlabel('Predicted error scale b, binned (5 equal-count bins)')
ax.set_ylabel('Mean REAL error, px')
ax.set_title('Uncertainty calibration: predicted vs. actual error', loc='left', fontsize=13, fontweight='bold')
ax.text(0.02, 0.92, 'Pearson r = 0.38 (n = 2.35M points)\nMonotonic across all 5 bins',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=LIGHT))
plt.tight_layout()
plt.savefig(f'{OUT}/calibration_bars.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/calibration_bars.png')

# ---------------------------------------------------------------- Plot E
# Distillation: baseline-3iter vs distilled-3iter vs baseline-8iter (coarse-only, decoder-independent)
dist_labels = ['Baseline\n3 iterations', 'Distilled\n3 iterations', 'Baseline\n8 iterations\n(target)']
dist_epe = [2.899, 2.528, 2.475]
dist_colors = ['#b03030', INK, '#999999']

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
bars = ax.bar(dist_labels, dist_epe, color=dist_colors, width=0.55, zorder=3)
for bar, v in zip(bars, dist_epe):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10.5, color=INK, fontweight='bold')
ax.annotate('closes 87.5% of the\n3-vs-8 iteration gap\nat 3-iteration speed',
            xy=(1, 2.528), xytext=(1.05, 2.72),
            fontsize=9.5, color=INK, ha='left',
            arrowprops=dict(arrowstyle='->', color=INK, lw=1))
ax.set_ylabel('Mean end-point error, px (coarse flow only)')
ax.set_title('Refinement self-distillation: same speed, most of the accuracy', loc='left', fontsize=12.5, fontweight='bold')
ax.set_ylim(2.3, 3.0)
plt.tight_layout()
plt.savefig(f'{OUT}/distillation_bars.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/distillation_bars.png')

print('\nAll five plots generated.')
