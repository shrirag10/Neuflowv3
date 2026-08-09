# Figures

Figures used by `README.md`, `NeuFlow_v3_Report.tex` and `NeuFlow_v3_IEEE.tex`.

`results/` is gitignored, so anything under `results/plots/` or `results/visuals/`
exists only on the machine that generated it. Figures that documents depend on
belong here, tracked.

## Provenance

Most files here were written by `scripts/make_final_plots.py`, which holds the
measured values inline and can regenerate them without the datasets.

Four files have a different history and are recorded here so they are not
mistaken for script output.

| File | Origin |
|---|---|
| `scenarios_two_domain.png` | Driving and aerial scenario montage, both columns. Composed by `scripts/make_domain_comparison.py` from a VKITTI2 and a TartanAir run of `scripts/make_scenario_figure.py`. The intermediates were never committed, so this was recovered from the embedded media of `docs/NeuFlow_v3_Overview.pptx`. |
| `scenarios_aerial.png` | Aerial column alone, from `scripts/make_scenario_figure.py --dataset tartanair`. Recovered from the embedded media of `docs/NeuFlow_v3_Presentation.pptx`. |
| `scenarios_illustrated_untitled.png` | `scenarios_illustrated.png` with the title strip removed, for use where the surrounding document supplies its own heading. |
| `margin_rule.png` | Regenerated from the values in `scripts/make_final_plots.py` (lines 323-324) with the two annotations repositioned. The previous version overlapped the "margin = one frame of motion" label with the explanatory note, leaving both unreadable when projected. The data is unchanged. |

Regenerating the two scenario montages requires TartanAir, which is not on every
machine. It lives at `/scratch/$USER/neuflow_datasets/tartanair` on the cluster;
see `hpc/download_tartanair.sbatch`.
