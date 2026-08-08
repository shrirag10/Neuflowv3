# NeuFlow v3 — Handoff

You are picking up an MS-thesis project (Shriman Raghav Srinivasan, Northeastern,
advisor Hanumant Singh, Field Robotics Lab, deadline August 2026). Read this
before touching anything. It is written so a model with no prior context can
continue without re-deriving what exists or re-breaking what was fixed.

---

## 1. Working rules (user-imposed, repeatedly reinforced)

1. **No false positives.** A number is a result only if measured on the full
   validation set. Subset numbers, expectations and "should improve" are
   labelled PENDING. Never present an expectation as a result.
2. **Negative results are reported as prominently as wins.**
3. **Verify before asserting.** Run the check; do not reason from what the code
   probably does. Three separate bugs in this project produced plausible,
   confidently-stated, wrong numbers.
4. **Presentation materials contain no development history.** The user has
   asked explicitly: no mention of bugs, fixes, leaks, drift, or corrections in
   the deck or report. Those live here and in `V3DEV_LOG.md`.
5. **No device-dependent comparisons in presentation materials.** Report one
   device (RTX 4060 laptop) consistently.
6. Style: terse, no em dashes, no hype. Deck is serif, monochrome, white
   background, no "takeaway" lines, speaker notes on every slide.
7. Git: no Claude co-author trailers. Remote `neuflowv3`
   (github.com/shrirag10/Neuflowv3), branches `v3-dev` and `main`.
8. Long jobs run on the cluster and the user runs them; do not launch training
   automatically. Give one command per paste (the OOD terminal mangles
   multi-line pastes).

---

## 2. What the project is

NeuFlow v2 computes dense flow for every pixel at one fixed resolution. v3
replaces **only its final upsampling stage** with an implicit decoder that
answers flow queries at arbitrary continuous coordinates. Cost scales with the
number of queries, O(N), instead of image area, O(H×W).

Everything upstream (backbone, cross-attention, global matching, recurrent
refinement) is frozen and verified bit-identical to v2.

**The framing that matters** (advisor's, 2026-08-06): a fast platform (drone,
high-speed car) cannot afford full-frame flow, so it flows only the region that
matters. Three scenarios: S1 first encounter, S2 turn with overlapping objects,
S3 a new object appearing in a frame already being processed.

---

## 3. Verified results

All on held-out data. Timings RTX 4060 laptop, fp16, unless stated.

### Accuracy (VKITTI2 Scene18+20, 1,174 pairs, 460M px)

| Model | Training data | EPE | 1px | 3px | Params |
|---|---|---|---|---|---|
| NeuFlow v2 | FlyingThings | **2.324** | **77.63** | **89.80** | 9.03 M |
| v3 | FlyingChairs | 2.500 | 72.81 | 87.88 | **7.83 M** |
| v3 | + VKITTI2 | 2.398 | 75.74 | 88.94 | 7.83 M |
| v3 | + MPI-Sintel | 2.392 | 75.83 | 88.98 | 7.83 M |
| v3 | + uncertainty head | 2.384 | 76.13 | 89.02 | 7.83 M |

**v3 is less accurate than v2 on driving data.** Best is 2.6% worse on the mean,
like-for-like (chairs-only) is 7.6% worse. Cause diagnosed, see §5.

### Compute

| Mode | Latency |
|---|---|
| v2 full frame | 33.3 ms |
| v3 sparse, first query | 34.1 ms |
| **v3 sparse, repeat query on cached frame** | **1.25 ms (27x)** |
| v3 dense, stride 2 | 37.1 ms |

Decode is flat in N up to ~2,048 points (1.254 ms at 800, 1.285 at 2,048): it is
launch-overhead bound there, compute bound beyond. A dense 192x192 ROI is 36,864
points and costs ~4 ms, not 1.25.

### Selective accuracy (the strongest v3 result)

Confidence lets the model abstain. Coverage vs EPE of the accepted set:

| Coverage | v3 | v2 (cannot select) |
|---|---|---|
| 20% | 0.480 | 2.324 |
| 80% | **1.058** | 2.324 |
| 100% | 2.266 | 2.324 |

**2.2x more accurate than v2 over 80% of the frame.** Derived from the
calibration bins, same 2.35M queries.

### Calibrated uncertainty

Monotonic across five bins (0.480, 0.896, 1.018, 1.837, 7.100 px), 15x span,
Pearson r = 0.318, 2,348,000 queries.

### Region-of-interest flowing

| Region | Area | Latency | Speedup | EPE in region |
|---|---|---|---|---|
| Full frame | 100% | 33.3 ms | 1.0x | 0.657 |
| + 32 px margin | 13.8% | 7.6 ms | **4.4x** | 0.691 |

**Margin rule, validated at two motion scales:**

| Domain | Mean motion | Margin needed | Penalty |
|---|---|---|---|
| Driving (VKITTI2) | 26.6 px | ~32 px | +0.034 px |
| Aerial (TartanAir) | 9.26 px | ~8 px | +0.120 px |

margin ≈ expected inter-frame motion ≈ speed / frame rate. Both start at the
same 0.43 px penalty with no margin. Predicts the *scale*, not the exact curve
(residuals beyond ratio 1 differ by scene).

### Scenario 3, a new object mid-frame

| Policy | Cost of new object | EPE |
|---|---|---|
| **v3 sparse (800 pts)** | **1.68 ms** | 2.10 |
| v3 dense ROI | 4.36 ms | 1.77 |
| v2 new crop | 7.29 ms | 3.60 |
| v3 new crop | 8.23 ms | 3.61 |

Two disjoint crops cost more than one full frame: **crop once, or not at all.**

### Aerial domain (the only unconfounded head-to-head)

v3 beats v2 at all 7 margin settings on TartanAir (0.777 vs 0.781 full frame,
margins 0.004–0.029 px). Neither model trained on aerial data, unlike every
VKITTI2 table where v3's training included VKITTI2. Small but consistent; v3 is
still worse on large-motion failures (7.2% vs 6.5%).

---

## 4. Documented negative results — do not retry blindly

| Attempt | Outcome |
|---|---|
| Unbounded regression head | Never trained below its own initialisation. Fixed by the convex head. |
| Fourier positional encoding | Null (2.288 vs 2.275). Rules out missing positional signal. |
| Sequential finetune (chairs → vkitti2) | Catastrophic forgetting, 2.28 → 2.50. Use mixed training. |
| Reduced-resolution coarse + full-res query | Matches bilinear upsampling exactly. Querying below input sampling rate gains nothing. |
| Refinement self-distillation | 87.5% of the gap closed in isolation, only 27% end-to-end. Isolated-component wins do not transfer. |
| Spring training run | Timed out at 50k/100k (data loading bound at 1080p). Excluded. |

**Three independent experiments (PE, sub-pixel querying, reduced-resolution
querying) all returned null for the same reason:** the decoder's finest input is
1/8 resolution, so evidence barely varies within an 8x8 cell, while v2's
upsampler reads the full-resolution frame.

---

## 5. The one diagnosed, unfixed limitation

**The decoder never sees full-resolution features.** This explains the 1px
accuracy gap, the PE null, the sub-pixel querying null, and the
reduced-resolution null — one mechanism, four observations.

**Proposed remedy: a full-resolution stem.** A cheap stride-8 convolution over
the full-resolution image, fed to the decoder. ~1-2 ms. This is the single
highest-value next experiment and the only candidate that is v3-specific (v2's
upsampler already reads full resolution).

Second candidate: **tiled refinement.** Refinement is 60% of runtime and its
operations are all local (correlation radius 4 + context), so it can run on ROI
tiles with a halo. Measured halo decay says 6 cells is effectively exact, so it
should need no retraining. Estimated ~2x. Not built. Note it would benefit v2
equally.

---

## 6. Bugs found and fixed (context for why the checks exist)

Each of these produced plausible, wrong numbers that were stated confidently
before being caught. The corresponding checks now live in
`scripts/verify_pipeline.py` (10 checks) and the `--sanity` mode of
`scripts/bench_scenarios.py`.

1. **Metrics mislabelled** — "1px accuracy" was actually per-frame, not
   per-pixel. All pre-2026-07-08 numbers are invalid.
2. **Train/test leak** — VKITTI2 Scene18/20 were in both training and eval.
   Loader now raises unless `allow_val_scenes=True`.
3. **BatchNorm not frozen** — `model.train()` in the validation block re-enabled
   running-stat updates; 24,765 updates accumulated over a "frozen" run, worth
   ~0.25 px. This was what made v3 appear to match v2.
4. **Padded-frame query offset** — `decode_queries` takes coordinates in the
   padded frame; three call sites passed raw frame coordinates. VKITTI2 pads by
   (6,4), so sub-region queries landed several px off. Worth 0.099 px.
5. **Spring GT scale** — `read_flo5` divided by 2.0 on a wrong assumption,
   halving every Spring target. Caught by a units check, not by reading.
6. **Unequal stride** between v3 policies in the scenario benchmark.
7. **Missing validity mask** in the sparse policy (reported 58 px instead of 0.6).

**Lesson encoded in the tooling:** every dataset loader now gets a convention
gate that determines the format by measurement rather than assumption. See
`data_utils/tartanair.py::check_convention` — it scores v2 against four candidate
readings and requires a clear winner.

---

## 7. Codebase map

```
NeuFlow/
  implicit_decoder.py   THE contribution. Convex head over 3x3 coarse-flow
                        neighbourhood + bilinear candidate; zero-init == bilinear.
  neuflow.py            infer_coarse_state() / decode_queries() / decode_dense_fast()
data_utils/
  datasets.py           stages; STAGE_ALIASES gives real dataset names
  tartanair.py          loader + convention gate
  frame_utils.py        InputPadder (NOTE: centres padding, mode='sintel')
scripts/
  flow_engine.py        Qt-free compute core, shared by GUI and benchmarks
  verify_pipeline.py    10-check pre-flight suite. RUN THIS FIRST.
  check_leak.py         train/eval overlap at pair AND frame level
  eval_vkitti2.py       main eval; --fast_dense --stride 2
  eval_all_runs.py      all checkpoints -> one table
  bench_scenarios.py    S1/S2/S3 x 5 policies; --sanity is the harness gate
  eval_roi_crop.py      margin sweep; --dataset {vkitti2,tartanair}
  benchmark_sparse.py   coarse/decode split timing
  eval_calibration.py   uncertainty calibration
  make_final_plots.py   all figures
  build_final_deck.py   the deck (20 slides)
  query_gui.py          interactive tool
  video_region_gui.py   video + region tool
hpc/
  make_sbatch.py        generates all training jobs from one template
  _template.sbatch      edit this, not the generated files
  download_tartanair.sbatch
docs/
  NeuFlow_v3_Report.tex/.pdf   the report
  NeuFlow_v3_status.pptx       the deck
  V3DEV_LOG.md                 full chronological record
  base_parameters.md           parameter provenance
```

---

## 8. Cluster (Northeastern Explorer)

- `ssh explorer` from the laptop. OOD web terminal as fallback.
- Env: `$HOME/.conda/envs/neuflow/bin/python3`, **always with `PYTHONNOUSERSITE=1`
  and the absolute path** (a stray python3.13 torchvision in `~/.local` breaks
  activation).
- GPU partition `gpu`, 8 h limit, `--gres=gpu:v100-sxm2:1` or `h200`.
- Data: `/scratch/$USER/neuflow_datasets/{vkitti2,FlyingChairs_release,Sintel,spring,tartanair}`
- Checkpoints: `/scratch/$USER/neuflow_ckpts/v3_*`. **Scratch purges monthly.**
- Login nodes kill heavy processes; run downloads/extractions as sbatch on
  partition `short`.
- **Compute nodes are behind a squid proxy.** Port 8080 is blocked (the AirLab
  TartanAir bucket 403s); HuggingFace on 443 works. boto3 fails on the proxy,
  curl works.

---

## 9. Where things stand and what to do next

Everything for the presentation is finished and pushed to `v3-dev` and `main`.

**In priority order:**

1. **Full-resolution stem.** The one v3-specific fix for the one diagnosed
   limitation. Half a day to implement, 5 h to train, 1 h to evaluate.
2. **Repeat the uncertainty run with a different seed.** The uncertainty head
   appears to help (2.392 → 2.384) but that is within checkpoint noise
   (measured at up to 0.038 px between step 90k and 100k of the same run).
3. **Jetson measurement.** "Edge" is in the title and there is no embedded
   measurement anywhere.
4. **Spring 4K evaluation.** Query above the input sampling rate against
   Spring's 2x ground truth — the one test v2 structurally cannot take.
   `scripts/eval_spring_4k.py` exists; Spring GT scale is now fixed.
5. Tiled refinement (~2x, benefits v2 equally).

**Open loose end not mine to close:** the user's GitHub token appeared in
terminal output and is embedded in `.git/config` remotes. Needs rotating.
