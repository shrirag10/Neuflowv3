# NeuFlow v3 — Base Parameters (paper-derived)

Set 2026-07-08 after reading the three source papers. Every choice cites its origin.
Baseline script: `train_baseline_v1.sh`. Eval: `scripts/eval_vkitti2.py` (per-pixel metrics).

## Source papers (local copies in repo root)

| Paper | File | What we take from it |
|---|---|---|
| NeuFlow-V2 (arXiv:2408.10161) | `2408.10161v3.pdf` | Backbone/refinement recipe, iteration counts, RAFT training protocol |
| AnyFlow (CVPR 2023) | `Anyflow.pdf` | Implicit **convex-weight** upsampler, positional encoding, multi-scale training, loss |
| InfiniDepth | `Infidepth.pdf` | Multi-scale gated fusion decoder, random coordinate-value pair training, L1 |

## Base parameters and their justification

| Parameter | Value | Source / reason |
|---|---|---|
| Refinement iters (train) | s16=1, s8=8 | NeuFlow-V2 default AND eval setting. Previous runs trained 4+7 but evaluated 1+8 — distribution mismatch, now fixed. |
| Loss gamma | 0.8 | RAFT/AnyFlow Eq. 6 (was 0.9, undocumented). |
| Optimizer | AdamW, wd 1e-4, clip 1.0 | RAFT recipe (NeuFlow-V2 "same procedure as RAFT"). |
| LR schedule | OneCycle, peak 2e-4, pct_start 0.05 | RAFT/AnyFlow. Replaces ad-hoc ×0.8-every-10-validations decay. |
| Batch size | 4 | VRAM-bound (8 GB RTX 4060). AnyFlow used 16 on cluster GPUs — revisit on HPC. |
| Crop | 256×512 | Unchanged pending baseline; VKITTI2 native height 375 caps at 368. |
| Query points N | 4096/image | InfiniDepth trains on N random coordinate-value pairs, L1 (their Eq. 7). |
| Query jitter | OFF (integer pixels) | GT flow at continuous coords requires bilinear interpolation of GT, which *blends two motions at boundaries* — exactly where 50% of our queries go. Integer sampling gives exact GT. Revisit jitter later with nearest-GT sampling. |
| Adaptive ratio | 0.5 | Unchanged (InfiniDepth §3.3-style importance sampling). |
| Backbone | frozen | Budget constraint; InfiniDepth trains 800K steps on 8 GPUs for joint training. |

## Known architecture gaps vs papers (NOT yet applied — next after baseline)

1. **Residual parameterization.** Our MLP outputs flow delta in *image fractions* (×W≈1248 amplification).
   AnyFlow's implicit upsampler instead predicts **3×3 convex-combination weights** over the coarse-flow
   neighborhood (their Eq. 2) — the continuous-coordinate generalization of v2's convex upsampler.
   Output is bounded, scale-free, and reduces to v2's mechanism exactly. This is the principled fix if the
   baseline still trails v2.
2. **Positional encoding.** AnyFlow applies Fourier encoding ψ(x_q − v*) to the query's offset from the
   nearest coarse cell; we feed raw normalized coords only. Needed for high-frequency detail.
3. **Multi-scale training.** AnyFlow randomly downsamples inputs (prob p) and supervises at original
   resolution — this is what trains genuine arbitrary-scale querying, our thesis claim.

## Measured reference points (per-pixel, VKITTI2 Scene18+20, 460M px, 2026-07-08)

| Model | Mean EPE | 1px acc | 3px acc |
|---|---|---|---|
| v2 convex upsampler (`neuflow_mixed.pth`) | 2.32 | 77.6% | 89.8% |
| v3 window best (`neuflowv3_window_phase2/step_008000`) | 3.50 | 71.9% | 86.6% |
| v3 step_0 (zero-init decoder = bilinear coarse) | 2.48 | 74.7% | 88.2% |

**Key finding (2026-07-08): the untrained decoder beats every trained checkpoint.** All prior
training runs degraded EPE (2.48 → 3.50). Prime suspects are the train/eval iteration mismatch
(4+7 vs 1+8), jittered GT blending, and the ×W residual scale — all corrected in baseline v1.
The baseline run's success criterion is therefore: **beat 2.48.**

Historical caution: numbers before 2026-07-08 used per-frame statistics mislabeled as pixel accuracy;
do not compare against them.

## v2-dev result (2026-07-09): convex head + 6x variant data — first net-positive training

| Model | Mean EPE | 1px acc | 3px acc |
|---|---|---|---|
| v2-dev step_0 (bilinear-prior init) | 2.476 | 74.7% | 88.2% |
| **v2-dev step_15000 (final)** | **2.388** | **74.7%** | **88.9%** |
| v2 reference | 2.324 | 77.6% | 89.8% |

- First checkpoint ever to train below its initialization (2.39 < 2.48).
- Best checkpoint is the FINAL one (OneCycle tail) — no overfitting collapse with 6x data;
  the curve is still descending at 15K. Longer training is the obvious next lever.
- Remaining gap to v2: 0.06 px EPE and 2.9 points of 1px accuracy.
- Run: `train_v2dev.sh` (head=convex, stage=vkitti2_all, 15K steps).
