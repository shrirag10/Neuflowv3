# NeuFlow v3

Built on top of [NeuFlow v2](https://arxiv.org/abs/2408.10161). NeuFlow v2 computes
optical flow densely, at a fixed resolution, always. v3 replaces only its final
upsampling stage with a queryable implicit decoder (in the lineage of
[InfiniDepth](https://arxiv.org/abs/2601.03252) and AnyFlow): instead of always
producing a full H×W map, the model answers flow queries at arbitrary continuous
(x, y) coordinates, at a cost that scales with the number of queries, O(N), not
image area, O(H×W).

Everything upstream of the upsampler — the CNN backbone, cross-attention, matching,
and recurrent refinement — is frozen and byte-for-byte unchanged from v2.

Full history, every experiment (including the failed ones) and every verified
number: [`docs/V3DEV_LOG.md`](docs/V3DEV_LOG.md). Status deck:
[`docs/NeuFlow_v3_status.pptx`](docs/NeuFlow_v3_status.pptx).

---

## Results (verified, full VKITTI2 Scene18+20 validation set, 1,174 pairs)

| Configuration | Trained on | Mean EPE | 1px acc | 3px acc |
|---|---|---|---|---|
| NeuFlow v2 (reference) | FlyingThings | 2.324 px | 77.6% | 89.8% |
| v3, untrained (bilinear init) | — | 2.476 px | 74.7% | 88.2% |
| **v3, big18** (best EPE) | chairs+VKITTI2, 100K steps | **2.072 px** | 77.02% | 89.91% |
| **v3, uncG** (uncertainty head) | chairs+VKITTI2, 100K steps | 2.082 px | **77.51%** | **90.02%** |
| v3, grandmix | chairs+VKITTI2+Sintel, 100K steps | 2.166 px | 76.25% | 89.48% |
| v3, spring (truncated at 70% of training) | +Spring, 70K/100K steps | 2.080 px | 76.94% | 89.88% |

The best checkpoints beat v2 by ~11% on mean EPE, and `uncG` is the first configuration
to also beat v2 on 3px accuracy (90.02 vs 89.8), with the 1px gap nearly closed
(77.51 vs 77.6). All HPC runs used batch size 16, 100K steps, Explorer cluster.

**Speed, identical V100 hardware:** v2 pays 19.6 ms on every call. v3 pays ~19.1 ms
on the first query of a new frame (parity), then ~2.6 ms for every additional query
batch on that same frame — v2 has no equivalent, since it always recomputes its
full dense pass. That ~7x repeat-query speedup, not the EPE delta, is the real
deployment argument.

Two validated side capabilities v2 cannot express at all:

- **Sparse queries match dense output exactly** — decoding N points equals the
  dense value at those points to 0.00 px, verified.
- **Calibrated per-query confidence** (`uncG`, the uncertainty head): predicted
  error scale correlates with real error, Pearson r=0.38, rising monotonically
  from 0.22 to 7.38 px across five bins on 2.35M sample points.

Full details, every rejected idea, and exactly what's still unverified (sparse
speed on every checkpoint, whether the uncertainty head's regularization effect
is real or seed noise, no edge-device measurement yet) are in
[`docs/V3DEV_LOG.md`](docs/V3DEV_LOG.md) and the deck's Limitations slide.

---

## Architecture

The decoder samples four feature sources per query:

- `ctx_s8` — 64d context from img0 at 1/8 resolution
- `feat_s8` — 128d matching features at 1/8 resolution
- `feat_s16` — 128d features at 1/16 resolution
- `feat1_s8` (warped) — img1 features at the coarse-flow-predicted correspondence,
  giving the decoder explicit cross-frame information (InfiniDepth doesn't need
  this since it's single-image depth, not two-frame flow)

These fuse hierarchically (shallow → deep, gated residual) into a 260-d vector fed
to a head that outputs **softmax weights over a 3×3 coarse-flow neighborhood plus
a bilinear candidate** — a bounded convex blend, so it cannot hallucinate large
flow. The head is zero-initialized, so an untrained decoder exactly reproduces
plain bilinear upsampling (2.476 px EPE, verified).

An earlier direct-regression head (predicting a flow delta directly, unbounded)
never trained below its own initialization — see the log for the full story. The
convex head above is the fix, and is what every result in this README uses.

Optional: `--uncertainty` adds one extra output channel, the predicted error
scale `b`, trained with a self-calibrating Laplace loss (`|error|/b + 2 log b`).

### Two-pass query API

```python
model = NeuFlow(use_implicit=True, head_mode='convex')
state = model.infer_coarse_state(img0, img1)              # once per frame pair, ~17-33 ms
flow  = model.decode_queries(state, query_coords=q)        # q: [B, N, 2] pixel coords -> [B, N, 2]
flow  = model.decode_queries(state, target_h=H, target_w=W)   # dense grid, any resolution
flow, b = model.decode_queries(state, query_coords=q, return_uncertainty=True)  # needs predict_uncertainty=True
```

---

## Training

Needs `neuflow_mixed.pth` (pretrained v2 checkpoint) in the project root.

```bash
python3 train.py \
  --stage mix_chairs_vkitti2 --implicit --head convex --sparse_loss \
  --num_sparse_points 4096 --adaptive_query_ratio 0.5 --no_query_jitter \
  --batch_size 16 --lr 2e-4 --onecycle --gamma 0.8 \
  --train_iters_s16 1 --train_iters_s8 8 --num_steps 100000 \
  --resume neuflow_mixed.pth --checkpoint_dir checkpoints/my_run
```

`--head convex` is required — the default is now `convex` (the old `regress` head
is kept only for loading pre-2026-07-09 checkpoints). Dataset stages available in
`data_utils/datasets.py`: `chairs`, `vkitti2`, `vkitti2_all`, `mix_chairs_vkitti2`,
`grand_mix` (+Sintel), `spring_mix` (+Spring), `things`, `sintel`, `kitti`, `viper`.

HPC (Northeastern Explorer cluster) setup and job scripts: [`hpc/`](hpc/),
walkthrough in [`hpc/explorer_setup.md`](hpc/explorer_setup.md).

### Evaluation

```bash
python3 scripts/eval_vkitti2.py --head convex \
  --checkpoint checkpoints/my_run/step_100000.pth --dataset_root datasets/vkitti2
```

Add `--fast_dense --stride 2` for the accelerated dense path (folds window
projections into per-image convs, ~3x faster, <0.02 px EPE cost). Add
`--uncertainty` if the checkpoint was trained with `--uncertainty`.

Sparse-query speed (the deployment-relevant number): `scripts/benchmark_sparse.py`.
Uncertainty calibration check: `scripts/eval_calibration.py`.

---

## Interactive tool

```bash
python3 scripts/query_gui.py --img1 <path> --img2 <path> --checkpoint <path>
```

PyQt5 GUI: click any pixel for its flow, adaptive/grid/region query modes, a
model selector to compare against the v2 baseline in place, video and YouTube
playback with live motion detection, and a system-resources tab (FPS, latency,
GPU/CPU/RAM graphs).

---

## Repository layout

```text
NeuFlow/                  model code
  implicit_decoder.py      the queryable decoder (the v3 contribution)
  neuflow.py               wires the decoder into the frozen v2 pipeline
  config.py, backbone_v7.py, transformer.py, matching.py, corr.py, refine.py, ...

data_utils/                dataset loaders (chairs/vkitti2/sintel/spring/...), flow viz, frame utils
utils/                     checkpoint loading, loss functions, DDP utils

scripts/
  eval_vkitti2.py          full-set evaluation against ground truth
  eval_coarse.py           decoder-free coarse-flow eval (for refinement-only changes)
  eval_calibration.py      uncertainty-head calibration check
  benchmark_sparse.py      sparse-query deployment-speed benchmark
  query_gui.py             interactive PyQt5 tool
  train_distill.py         refinement self-distillation (no ground truth)
  make_final_plots.py      regenerates the deck's comparison plots
  build_final_deck.py      regenerates docs/NeuFlow_v3_status.pptx

hpc/                       Explorer cluster setup, sbatch job scripts
docs/
  V3DEV_LOG.md             complete running history — read this for anything historical
  NeuFlow_v3_status.pptx   current status deck
  NeuFlow_v3_Report.md     prose report
  base_parameters.md       paper-derived parameter choices and their justification

train.py                   training entry point
```

---

Based on NeuFlow v2 (Zhao et al., 2024), InfiniDepth (Yu et al., 2025), and AnyFlow (CVPR 2023).
