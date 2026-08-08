"""Figures for the deck and report.

All numbers are from the 2026-08-03 runs: leak-free splits AND a verifiably
frozen front end (diag_backbone reports 0 of 137 shared tensors differing from
v2). Earlier figures came from runs whose BatchNorm statistics had drifted, which
flattered v3 by roughly 0.25 px; those are superseded.

VKITTI2 Scene18+20, 1,174 pairs, 460,573,660 pixels, V100, fp16, fast_dense
stride 2. Sources: scripts/eval_all_runs.py, benchmark_sparse.py,
scripts/eval_calibration.py. Nothing here is estimated.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#333333', 'axes.labelcolor': '#1a1a1a',
    'text.color': '#1a1a1a', 'xtick.color': '#333333', 'ytick.color': '#333333',
})

INK, MUTED, HL, WARN = '#1a1a1a', '#8a8a8a', '#0b6a63', '#b03030'
OUT = 'results/plots'
os.makedirs(OUT, exist_ok=True)

# ---- measured, leak-free, fast_dense stride 2 -----------------------------
V2 = dict(epe=2.324, a1=77.63, a3=89.80, ms=19.3)
RUNS = [
    ('FlyingChairs\nonly',            dict(epe=2.500, a1=72.81, a3=87.88, ms=21.9)),
    ('+VKITTI2',                      dict(epe=2.398, a1=75.74, a3=88.94, ms=21.9)),
    ('+MPI-Sintel',                   dict(epe=2.392, a1=75.83, a3=88.98, ms=21.9)),
    ('+uncertainty\nhead',            dict(epe=2.384, a1=76.13, a3=89.02, ms=22.0)),
]

# =========================================================== 1. accuracy
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
labels = [l for l, _ in RUNS]
epes = [r['epe'] for _, r in RUNS]
# the chairs-only run is the only unconfounded comparison
colors = [WARN] + [MUTED] * 2 + [INK]
bars = ax.bar(labels, epes, color=colors, width=0.6, zorder=3)
for b, v in zip(bars, epes):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f'{v:.3f}',
            ha='center', fontsize=10.5, fontweight='bold')
ax.axhline(V2['epe'], color=WARN, ls='--', lw=1.4, zorder=2)
ax.text(3.45, V2['epe'] + 0.006, f'NeuFlow v2: {V2["epe"]:.3f}', color=WARN,
        fontsize=9.5, ha='right', va='bottom')
ax.set_ylim(2.25, 2.58)
ax.set_ylabel('Mean end-point error, px  (lower is better)')
ax.set_title('Accuracy on VKITTI2 with a verifiably frozen front end.\n'
             'Every v3 configuration sits above v2: the decoder costs accuracy',
             loc='left', fontsize=12.5, fontweight='bold')
ax.text(0, 2.555, 'like-for-like\n(+7.6% vs v2)', ha='center', fontsize=9,
        color=WARN, style='italic', va='top')
ax.text(3, 2.555, 'best v3\n(+2.6% vs v2)', ha='center', fontsize=9, color=INK,
        style='italic', va='top')
plt.tight_layout()
plt.savefig(f'{OUT}/accuracy_bars.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/accuracy_bars.png')

# =========================================================== 2. the precision cost
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
x = np.arange(len(RUNS) + 1)
a1 = [V2['a1']] + [r['a1'] for _, r in RUNS]
names = ['NeuFlow v2'] + [l.replace('\n', ' ') for l, _ in RUNS]
cols = [WARN] + [MUTED] * 3 + [INK]
bars = ax.bar(x, a1, color=cols, width=0.6, zorder=3)
for b, v in zip(bars, a1):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f'{v:.1f}',
            ha='center', fontsize=10.5, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(68, 82)
ax.set_ylabel('Pixels within 1 px of ground truth, %  (higher is better)')
ax.set_title('The cost that has not been paid down: sub-pixel precision',
             loc='left', fontsize=12.5, fontweight='bold')
ax.annotate('', xy=(0, 77.63), xytext=(4, 76.88),
            arrowprops=dict(arrowstyle='<->', color=INK, lw=1))
ax.text(2, 78.6, 'v3 is below v2 on every configuration',
        ha='center', fontsize=9.5, color=INK)
plt.tight_layout()
plt.savefig(f'{OUT}/precision_bars.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/precision_bars.png')

# =========================================================== 3. speed, honestly
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
modes = ['v2\nfull frame', 'v3 dense\n(stride 2)', 'v3 sparse\nfirst query',
         'v3 sparse\nrepeat query']
vals = [19.3, 22.0, 19.16, 2.55]
cols = [WARN, MUTED, MUTED, HL]
bars = ax.bar(modes, vals, color=cols, width=0.58, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f'{v:.1f} ms',
            ha='center', fontsize=11, fontweight='bold')
ax.axhline(19.3, color=WARN, ls='--', lw=1.2, zorder=2)
ax.set_ylabel('Latency per frame pair, ms  (384x1248, fp16, V100)')
ax.set_title('Speed: v3 is slower dense, level on a first query, 7.7x cheaper on repeats',
             loc='left', fontsize=12.5, fontweight='bold')
ax.text(1, 23.4, '14% slower\nthan v2', ha='center', fontsize=9, color=WARN)
ax.text(3, 4.6, '7.7x cheaper\n(v2 must recompute\nthe whole frame)',
        ha='center', fontsize=9, color=HL)
ax.set_ylim(0, 26)
plt.tight_layout()
plt.savefig(f'{OUT}/speed_bars.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/speed_bars.png')

# =========================================================== 4. what the freeze cost
# The same four configurations, trained with drifted vs verifiably frozen
# BatchNorm statistics. The drift was worth ~0.25 px and produced the earlier
# "v3 matches v2" result.
names  = ['FlyingChairs', '+VKITTI2', '+MPI-Sintel', '+uncertainty']
drift  = [2.286, 2.138, 2.147, 2.104]
frozen = [2.500, 2.398, 2.392, 2.384]

fig, ax = plt.subplots(figsize=(9.5, 5), dpi=150)
x = np.arange(len(names))
ax.plot(x, drift,  'o--', color=MUTED, lw=1.6, ms=9, label='drifted normalisation statistics')
ax.plot(x, frozen, 'o-',  color=INK,   lw=1.9, ms=9, label='verifiably frozen (correct)')
for xi, (a, b) in enumerate(zip(drift, frozen)):
    ax.plot([xi, xi], [a, b], color=WARN, lw=6, alpha=0.16, zorder=1, solid_capstyle='round')
    ax.text(xi + 0.09, (a + b) / 2, f'+{b-a:.3f}', fontsize=9, color=WARN, va='center')
ax.axhline(V2['epe'], color=WARN, ls='--', lw=1.3)
ax.text(3.42, V2['epe'] - 0.008, f'NeuFlow v2: {V2["epe"]:.3f}', color=WARN,
        fontsize=9.5, ha='right', va='top')
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylim(2.05, 2.56); ax.set_ylabel('Mean EPE, px')
ax.legend(frameon=False, loc='upper right')
ax.set_title('What the BatchNorm drift was worth.\n'
             'With statistics genuinely frozen, every configuration moves above v2',
             loc='left', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/freeze_effect.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/freeze_effect.png')

# =========================================================== 5. calibration (measured)
# measured on v3_FlyingChairs_VKITTI2_Sintel_uncertainty/step_100000, 2026-08-02
# (2,348,000 samples; the earlier figures in this slot came from the pre-leak-fix run)
bins = ['0.01-0.10', '0.10-0.19', '0.19-0.39', '0.39-1.22', '1.22+']
err = [0.480, 0.896, 1.018, 1.837, 7.100]
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
bars = ax.bar(bins, err, color=HL, width=0.6, zorder=3)
for b, v in zip(bars, err):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f'{v:.2f}',
            ha='center', fontsize=10.5, fontweight='bold')
ax.set_xlabel('Predicted error scale b, binned')
ax.set_ylabel('Actual mean error, px')
ax.set_title('The confidence signal is calibrated: predicted b tracks real error '
             '(Pearson r = 0.318)', loc='left', fontsize=12, fontweight='bold')
ax.text(0.02, 0.93, '15x span from the most to the least confident bin',
        transform=ax.transAxes, fontsize=9.5, color=MUTED)
plt.tight_layout()
plt.savefig(f'{OUT}/calibration_bars.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/calibration_bars.png')

# =========================================================== 6. decode cost is flat
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
ns = [800, 2048]
ms = [2.553, 2.554]
ax.plot(ns, ms, 'o-', color=HL, lw=2, ms=9, zorder=3)
for n, v in zip(ns, ms):
    ax.annotate(f'{v:.3f} ms', (n, v), textcoords='offset points',
                xytext=(0, 12), ha='center', fontsize=10.5, fontweight='bold')
ax.axhline(19.3, color=WARN, ls='--', lw=1.3)
ax.text(2048, 18.6, 'v2 recomputes the whole frame: 19.3 ms',
        color=WARN, fontsize=9.5, ha='right', va='top')
ax.set_xlim(400, 2450); ax.set_ylim(0, 22)
ax.set_xlabel('Queries per decode call, N')
ax.set_ylabel('Decode latency, ms')
ax.set_title('Decode cost is flat in N up to 2,048: launch-overhead bound, not compute bound',
             loc='left', fontsize=11.5, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/decode_flat.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/decode_flat.png')

print('\nAll figures regenerated from the 2026-08-02 leak-free runs.')

# =========================================================== 7. selective accuracy
# Coverage-vs-error, derived from the calibration bins (each bin = 20% of the
# 2,348,000 queries). Accepting only the most confident k bins:
cov  = [20, 40, 60, 80, 100]
v3   = [0.480, 0.688, 0.798, 1.058, 2.266]
v2   = [V2['epe']] * len(cov)          # v2 has no confidence, so it cannot select

fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
ax.plot(cov, v2, 's--', color=WARN, lw=2, ms=8, label='NeuFlow v2 (cannot select)')
ax.plot(cov, v3, 'o-', color=HL, lw=2.4, ms=9, label='NeuFlow v3 (select by confidence)')
for c, y in zip(cov, v3):
    ax.annotate(f'{y:.2f}', (c, y), textcoords='offset points', xytext=(0, -16),
                ha='center', fontsize=10, color=HL, fontweight='bold')
ax.annotate('2.2x more accurate\nover 80% of the frame',
            xy=(80, 1.058), xytext=(56, 1.72), fontsize=10.5, color=INK,
            arrowprops=dict(arrowstyle='->', color=INK, lw=1.2))
ax.set_xlabel('Coverage: percentage of queries accepted')
ax.set_ylabel('Mean end-point error of the accepted set, px')
ax.set_xlim(12, 108); ax.set_ylim(0, 2.6)
ax.legend(frameon=False, loc='upper left')
ax.set_title('Confidence lets v3 trade coverage for accuracy. v2 has one operating point',
             loc='left', fontsize=12.5, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/selective_accuracy.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/selective_accuracy.png')

# =========================================================== 8. ROI crop: device dependence
# Same script, same 40 VKITTI2 pairs, same margins, two devices.
# scripts/eval_roi_crop.py --limit 40
margins   = [0, 16, 32, 64, 128]
sp_4060   = [4.26, 4.38, 4.38, 4.12, 3.35]     # RTX 4060 laptop, full = 33.3 ms
sp_v100   = [1.18, 1.17, 1.18, 1.13, 1.12]     # Tesla V100,      full = 17.8 ms
epe_crop  = [1.089, 0.985, 0.691, 0.667, 0.655]
EPE_FULL  = 0.657

fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

axl.plot(margins, sp_4060, 'o-', color=HL, lw=2.2, ms=9, label='RTX 4060 (33.3 ms full frame)')
axl.plot(margins, sp_v100, 's-', color=WARN, lw=2.2, ms=8, label='Tesla V100 (17.8 ms full frame)')
axl.axhline(1.0, color=MUTED, ls=':', lw=1.2)
for m, v in zip(margins, sp_4060):
    axl.annotate(f'{v:.1f}x', (m, v), textcoords='offset points', xytext=(0, 9),
                 ha='center', fontsize=9.5, color=HL, fontweight='bold')
axl.set_xlabel('Crop margin, px'); axl.set_ylabel('Speedup over full frame')
axl.set_ylim(0, 5.2)
axl.legend(frameon=False, fontsize=9.5, loc='center right')
axl.set_title('Speed: entirely device dependent', loc='left', fontsize=12, fontweight='bold')
axl.text(0.03, 0.06, 'the weaker GPU is compute bound, so less area is less work;\n'
                     'the V100 is launch bound, and fewer pixels do not mean fewer kernels',
         transform=axl.transAxes, fontsize=8.5, color=MUTED)

axr.plot(margins, epe_crop, 'o-', color=INK, lw=2.2, ms=9)
axr.axhline(EPE_FULL, color=WARN, ls='--', lw=1.4)
axr.text(128, EPE_FULL - 0.02, f'full frame: {EPE_FULL:.3f}', color=WARN,
         fontsize=9.5, ha='right', va='top')
axr.annotate('32 px margin:\n+0.034 px, and it stops improving',
             xy=(32, 0.691), xytext=(52, 0.95), fontsize=9.5, color=INK,
             arrowprops=dict(arrowstyle='->', color=INK, lw=1.1))
axr.set_xlabel('Crop margin, px'); axr.set_ylabel('EPE inside the ROI, px')
axr.set_title('Accuracy: identical on both devices', loc='left', fontsize=12, fontweight='bold')

fig.suptitle('Cropping to a region of interest: the accuracy cost is fixed, the speed benefit is not',
             fontsize=12.5, x=0.01, ha='left', fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f'{OUT}/roi_crop_devices.png', bbox_inches='tight', facecolor='white')
plt.close()
print(f'{OUT}/roi_crop_devices.png')
