# NeuFlow v3 — Project Report

**Shriman Raghav Srinivasan · MS Robotics, Northeastern University · Field Robotics Lab**
Rewritten 2026-08-02 against leak-free results. Supersedes all earlier versions.

---

## 0. What changed in this revision, and why

Every number in the previous version of this report was produced under one or more
of three defects. They are described in §3 because they matter more than any
individual result. Two headline claims did not survive their correction:

| Previous claim | Status |
|---|---|
| "v3 dense is faster than v2 (28 vs 36 ms)" | **Withdrawn.** Measured pre-BatchNorm-fix on other hardware. On the corrected pipeline v3 dense is 22.0 ms against v2's 19.6 ms, i.e. 12% *slower*. |
| "v3 beats v2 by 6% on mean EPE" | **Withdrawn.** True only for models trained on VKITTI2 imagery v2 never saw. The like-for-like comparison is a tie. |
| "MPI-Sintel adds nothing" | **Withdrawn.** Reverses sign between checkpoints (§5.3). |
| "The uncertainty head improves EPE by 2.0%" | **Withdrawn as an accuracy claim.** Also reverses sign between checkpoints. Its *calibration* result stands. |

Nothing below is an estimate unless explicitly labelled. Numbers without a label
come from a full-set run recorded in `docs/V3DEV_LOG.md`.

---

## 1. What the project is

NeuFlow v2 (Zhang, Gupta, Jiang, Singh, arXiv:2408.10161) is a real-time optical
flow network for edge devices: a shallow CNN backbone, cross-attention and global
matching at 1/16, recurrent refinement (1 iteration at 1/16, 8 at 1/8) producing
flow at 1/8 resolution, then a learned convex upsampler to full resolution.

v3 keeps all of that, frozen, and replaces only the upsampler with an **implicit
decoder that answers flow queries at arbitrary continuous coordinates**:

- Two-phase API: `infer_coarse_state()` once per pair (16.6 ms), then
  `decode_queries()` at 2.55 ms per batch of up to 2,048 points.
- Convex head: the MLP predicts softmax weights over the 3×3 coarse-flow
  neighbourhood plus a bilinear candidate, so output is a bounded blend and
  cannot hallucinate unsupported motion. Zero-initialised, it reproduces bilinear
  upsampling exactly (verified: 0.011 px max deviation).
- Optional uncertainty channel: a per-query error scale `b` trained under a
  Laplace likelihood.
- 7.83 M parameters against v2's 9.03 M.

The intended consumers are registration and mapping, which need a few hundred
correspondences at points they choose, not half a million they discard.

---

## 2. Evaluation protocol

VKITTI2 Scene18 + Scene20, **1,174 pairs, 460,573,660 valid pixels**, per-pixel
metrics. Scenes 18 and 20 are excluded from all training sets. fp16, V100,
384×1248 input, `--fast_dense --stride 2` unless stated.

All five training runs are generated from a single template
(`hpc/make_sbatch.py`) so they are identical in seed (1234), batch size (16),
learning rate (2e-4 OneCycle), loss weighting (γ=0.8), refinement schedule
(1×s16 + 8×s8, matching evaluation), query count (4,096), and step count
(100,000). Exactly one variable differs between any two.

A 9-check pre-flight suite (`scripts/verify_pipeline.py`) runs on CPU and GPU
before any training job and must pass: BatchNorm frozen, only decoder trainable,
zero-init equals bilinear, sparse equals dense, uncertainty in both decode paths,
stride-2 fidelity, and seed reproducibility.

---

## 3. Three defects that invalidated earlier results

**3.1 Train/test leak.** VKITTI2 Scene18 and Scene20 were in both the training
set and the evaluation set. Models were scored on frames they had trained on.
Fixed by excluding them by default with a guard that raises unless
`allow_val_scenes=True`; verified at both pair and frame level
(`scripts/check_leak.py`, reports `OVERLAP 0`). VKITTI2's contribution to the
training mix drops from 12,726 to 5,682 pairs as a result.

**3.2 BatchNorm was never frozen.** BatchNorm updates `running_mean` and
`running_var` on every forward pass in train mode regardless of
`requires_grad=False`. Over 30K steps the "frozen" backbone drifted 7.4% on
running_mean and 17.4% on running_var, changing the coarse flow by 0.350 px —
roughly seven times the v3-vs-v2 difference being studied. So v3 was not sharing
v2's front end at all. Fixed by `set_frozen_bn_eval()`; verified drift is exactly
zero.

**3.3 Runs differed in three variables at once.** The earlier four runs varied in
dataset, batch size (12 vs 16) *and* refinement schedule (2,4 vs 1,8), while all
were evaluated at (1,8) — so two of them had a train/eval mismatch. No comparison
between them was valid. Fixed by generating all sbatch files from one template.

---

## 4. Accuracy

| Configuration | Training data | EPE | 1px % | 3px % |
|---|---|---|---|---|
| **NeuFlow v2 (reference)** | FlyingThings (authors') | 2.324 | **77.63** | **89.80** |
| v3 FlyingChairs | FlyingChairs | 2.286 | 71.30 | 87.57 |
| v3 +VKITTI2 | + VKITTI2 Scene01/02/06 | 2.138 | 76.38 | 89.46 |
| v3 +MPI-Sintel | + MPI-Sintel | 2.147 | 76.81 | 89.56 |
| v3 +uncertainty head | same, uncertainty on | 2.104 | 76.88 | 89.61 |

**Only the FlyingChairs row is a fair comparison with v2.** It contains no driving
imagery, exactly as v2's training contained none. It scores 2.286 against 2.324 —
a 1.6% difference, which is **a tie, not a win**.

The other three train on VKITTI2 scenes from the same simulator and camera as the
test scenes. Their better numbers measure domain advantage, not a better method,
and are not presented as beating v2.

### 4.1 The precision cost

v3 is below v2 on 1-pixel accuracy in **every** configuration, by 0.8 to 6.3
points. Mean EPE hides this: v3 makes fewer large errors, which pulls the mean
down, while being less exact on the majority of pixels.

Diagnosed cause: the decoder's finest input is at 1/8 resolution, so within an
8×8 cell the evidence it sees barely changes. v2's upsampler reads the
full-resolution frame directly. A Fourier positional encoding of the sub-cell
offset was tried and changed nothing (2.288 vs 2.275 on the pre-fix runs, 1px
identical), which **rules out missing positional signal** as the explanation and
points at missing high-resolution *features*.

---

## 5. What this experiment can and cannot resolve

Evaluating step 90,000 and step 100,000 of the **same four runs**:

| Run | @90k | @100k | spread |
|---|---|---|---|
| FlyingChairs | 2.294 | 2.286 | 0.008 |
| +VKITTI2 | 2.160 | 2.138 | 0.022 |
| +MPI-Sintel | 2.109 | 2.147 | **0.038** |
| +uncertainty | 2.120 | 2.104 | 0.016 |

Between-checkpoint variation reaches 0.038 px, which is **the same size as the
differences between runs**. Two orderings reverse:

- MPI-Sintel helps at 90k (−0.051) and hurts at 100k (+0.009)
- the uncertainty head hurts at 90k (+0.011) and helps at 100k (−0.043)

With one seed and one checkpoint, **neither question is answerable**, and both
claims are withdrawn. Resolving them needs multiple seeds and an average over
late checkpoints.

**What is robust at both checkpoints:** adding driving data to FlyingChairs gains
about 0.15 px of EPE and 5 points of 1px accuracy. Everything finer sits inside
the noise.

---

## 6. Speed

| Mode | Latency | vs v2 |
|---|---|---|
| v2, full frame (its only mode) | **19.6 ms** | — |
| v3 dense, stride 2 | 22.0 ms | **12% slower** |
| v3 sparse, first query on a new pair | 19.16 ms | level |
| **v3 sparse, repeat query on a cached pair** | **2.55 ms** | **7.7× cheaper** |

Breakdown of the sparse path: coarse pass 16.61 ms (87%) + decode 2.55 ms (13%).
The coarse pass is inherited from v2 and cannot be avoided — global matching needs
whole-image context.

Decode costs 2.553 ms at N=800 and 2.554 ms at N=2,048. Identical, so the decode
is **kernel-launch-overhead bound, not compute bound**: 2,048 points cost the same
as 800.

**The only genuine speed win is the repeat query**, and it is structural rather
than a margin: v2 holds no state between calls, so a second question about the
same frame costs it a full recomputation.

### 6.1 Identified but unimplemented speed work

Estimates, not results: CUDA graphs on the decode path (launch-bound, so 2.55 →
~0.5 ms plausible), `torch.compile` (measured 9% on an RTX 4060, zero accuracy
cost), and fusing three of the four `grid_sample` calls (they share coordinates;
exact). Together these would plausibly bring the first query to ~15.5 ms, i.e.
faster than v2 rather than level.

---

## 7. Calibrated uncertainty

Measured on `v3_FlyingChairs_VKITTI2_Sintel_uncertainty/step_100000`, 2,348,000
samples:

| Predicted b | Actual mean error |
|---|---|
| 0.01 – 0.11 | 0.313 px |
| 0.11 – 0.20 | 0.504 px |
| 0.20 – 0.41 | 0.807 px |
| 0.41 – 1.22 | 1.652 px |
| 1.22 + | 6.723 px |

Monotonic across all five bins, a 21× span, Pearson r = 0.345. Not a strong
correlation, but clearly informative and directly usable: weight correspondences
in RANSAC, reject unreliable matches, or steer queries toward uncertain regions.

**v2 emits flow only**, so there is no equivalent quantity to compare against.
This is the clearest capability difference in the project, and unlike the accuracy
claims it does not depend on a margin.

---

## 8. Honest scorecard

| Objective | Verdict | Evidence |
|---|---|---|
| Better accuracy than v2 | **NOT MET** | fair comparison 2.286 vs 2.324, a tie; better numbers need in-domain data |
| Less compute than v2 | **PARTLY** | dense 12% slower; first query level; repeat query 7.7× cheaper |
| Runs on edge devices | **UNPROVEN** | all figures from V100 and RTX 4060; no Jetson measurement exists |

**What the project does deliver, measured:**

1. Flow at any continuous coordinate, with sparse output matching dense to
   0.00057 px.
2. Repeat queries on a cached frame at 2.55 ms against v2's 19.6 ms recomputation.
3. A calibrated per-query confidence signal with no counterpart in v2.
4. A working interactive tool: load a video, drag a region, get flow there, with
   live cost breakdown (`scripts/video_region_gui.py`).

---

## 9. Limitations

- **Sub-pixel precision is worse than v2** by 0.8–6.3 points of 1px accuracy.
  Cause diagnosed, not fixed.
- **No edge-device measurement.** "Edge" in the title is a design target.
- **One evaluation domain.** All accuracy numbers are VKITTI2, a synthetic driving
  benchmark. It says little about field or survey imagery.
- **One seed, and checkpoint noise the size of the effects** (§5). Nothing finer
  than ~0.05 px is resolvable.
- **Spring run did not complete.** Timed out at 8 hours having reached step 50,000
  of 100,000 (data loading on 1080p frames caps throughput at ~1.7 steps/s, so the
  run needs ~16 h). Excluded, not reported as a data point.
- **The fusion-on-grid approximation in `fast_dense` is larger than previously
  documented**: 0.068 px mean and 22 px max against the exact per-query path
  (measured 2026-08-02), not the "+0.02 px" recorded earlier. It does not harm
  aggregate EPE, but it is an approximation and is now labelled as one.

---

## 10. Next steps, in priority order

1. **Full-resolution stem.** Give the decoder a cheap full-resolution feature map
   (~1–2 ms) so evidence varies within an 8×8 cell. Directly attacks the diagnosed
   precision gap, and makes continuous querying substantively meaningful rather
   than nominal.
2. **Spring 4K evaluation** (`scripts/eval_spring_4k.py`, written and geometry-
   verified). Spring provides ground truth at twice the input resolution: v3 can be
   queried there natively, v2 can only be interpolated. A capability comparison
   rather than a margin, and it does **not** require the Spring training run.
3. **Free-FPS stack** (§6.1). Exact, no retraining, would convert "level with v2"
   into "faster than v2".
4. **Jetson measurement.** Converts the edge claim into a result or refutes it.
5. **Field or survey imagery.** A registration demonstration on lab data would
   test the actual intended use case.

---

## Appendix: reproducing any number here

```bash
python3 scripts/verify_pipeline.py                     # 9 pre-flight checks
python3 scripts/check_leak.py --stage FlyingChairs+VKITTI2
python3 scripts/eval_all_runs.py --fast_dense --stride 2
python3 scripts/eval_calibration.py --checkpoint <uncertainty ckpt>
python3 scripts/benchmark_sparse.py --checkpoint <ckpt> --head convex --n 800 2048
python3 scripts/eval_spring_4k.py --check_units        # GT unit convention
```

Full chronological record, including every failed attempt and its cause:
`docs/V3DEV_LOG.md`.
