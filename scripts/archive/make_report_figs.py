"""Figures for docs/report/NeuFlow_v3_report.tex.

Only numbers recorded in docs/V3DEV_LOG.md are used. Results affected by the
2026-07-26 train/val leak are drawn hatched and labelled as invalid, never
silently mixed with the clean holdout results.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = 'results/report'
os.makedirs(OUT, exist_ok=True)

INK, ACCENT, BAD, MUTED = '#1a1a1a', '#186f6a', '#b03030', '#8a8a8a'
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.edgecolor': '#cccccc', 'axes.grid': True,
    'grid.color': '#e8e8e8', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
})

V2 = 2.324

# ---------------------------------------------------------------- Fig 1
# Accuracy: clean holdout vs leak-affected, drawn distinctly.
clean = [('NeuFlow v2\n(FlyingThings)', 2.324),
         ('v3 untrained\n(no training)', 2.476),
         ('v3 chairs-only\n(FlyingChairs)', 2.275),
         ('v3 chairs+PE\n(FlyingChairs)', 2.288)]
leaked = [('v3 grand-mix', 2.166), ('v3 spring-mix\n(70K)', 2.080),
          ('v3 uncertainty', 2.082), ('v3 chairs+vk', 2.072)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160,
                             gridspec_kw={'width_ratios': [1, 1]})

names = [n for n, _ in clean]
vals = [v for _, v in clean]
cols = [BAD, MUTED, ACCENT, ACCENT]
b = a1.bar(names, vals, color=cols, width=0.6)
for bar, v in zip(b, vals):
    a1.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f'{v:.3f}',
            ha='center', fontsize=9.5, fontweight='bold', color=INK)
a1.axhline(V2, color=BAD, ls='--', lw=1.1)
a1.set_ylim(2.15, 2.55)
a1.set_ylabel('Mean EPE, px (lower is better)')
a1.set_title('(a) Valid comparison: no model saw the eval scenes',
             loc='left', fontsize=11, fontweight='bold')
a1.tick_params(axis='x', labelsize=8.5)

names2 = [n for n, _ in leaked]
vals2 = [v for _, v in leaked]
b2 = a2.bar(names2, vals2, color='#d8d8d8', width=0.6,
            hatch='//', edgecolor=BAD, linewidth=1.0)
for bar, v in zip(b2, vals2):
    a2.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f'{v:.3f}',
            ha='center', fontsize=9.5, fontweight='bold', color=MUTED)
a2.axhline(V2, color=BAD, ls='--', lw=1.1)
a2.text(3.45, V2 + 0.006, 'v2 = 2.324', color=BAD, fontsize=8.5, ha='right')
a2.set_ylim(2.15, 2.55)
a2.set_title('(b) INVALID: these models trained on the eval scenes',
             loc='left', fontsize=11, fontweight='bold', color=BAD)
a2.tick_params(axis='x', labelsize=8.5)
a2.legend(handles=[Patch(facecolor='#d8d8d8', hatch='//', edgecolor=BAD,
                         label='train-on-test, not comparable')],
          loc='upper right', fontsize=8.5, frameon=False)

plt.tight_layout()
plt.savefig(f'{OUT}/accuracy.pdf', bbox_inches='tight')
plt.close()
print(f'{OUT}/accuracy.pdf')

# ---------------------------------------------------------------- Fig 2
# Speed: unaffected by the leak (timing is timing).
fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=160)
labels = ['v2\ndense\n(full frame)', 'v3\nfirst query\n(coarse+decode)',
          'v3 dense\nstride-2', 'v3 repeat query\n(same frame)']
ms = [19.6, 19.1, 21.9, 2.6]
cols = [BAD, ACCENT, MUTED, ACCENT]
bars = ax.bar(labels, ms, color=cols, width=0.58)
for bar, v in zip(bars, ms):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
            f'{v:.1f} ms\n({1000/v:.0f} FPS)', ha='center',
            fontsize=9, fontweight='bold', color=INK)
ax.annotate('7.5x cheaper than v2,\nwhich must recompute\nthe whole frame',
            xy=(3, 2.6), xytext=(2.25, 12),
            fontsize=9, color=INK,
            arrowprops=dict(arrowstyle='->', color=INK, lw=1))
ax.set_ylabel('Latency, ms (V100, fp16, 384x1248)')
ax.set_ylim(0, 26)
ax.set_title('Per-call cost: v3 caches one coarse pass, v2 cannot',
             loc='left', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', labelsize=8.5)
plt.tight_layout()
plt.savefig(f'{OUT}/speed.pdf', bbox_inches='tight')
plt.close()
print(f'{OUT}/speed.pdf')

# ---------------------------------------------------------------- Fig 3
# Uncertainty calibration: monotonic bins.
fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=160)
bins = ['[0.01,\n0.14]', '[0.14,\n0.27]', '[0.27,\n0.54]', '[0.54,\n1.48]', '[1.48,\n172]']
err = [0.221, 0.334, 0.600, 1.410, 7.377]
bars = ax.bar(bins, err, color=ACCENT, width=0.6)
for bar, v in zip(bars, err):
    ax.text(bar.get_x() + bar.get_width() / 2, v * 1.12, f'{v:.2f}',
            ha='center', fontsize=9, fontweight='bold', color=INK)
ax.set_yscale('log')
ax.set_ylabel('Actual mean EPE, px (log)')
ax.set_xlabel('Predicted uncertainty $b$, binned')
ax.set_title('Predicted confidence tracks real error (Pearson $r$ = 0.38)',
             loc='left', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/calibration.pdf', bbox_inches='tight')
plt.close()
print(f'{OUT}/calibration.pdf')

print('\nreport figures done')
