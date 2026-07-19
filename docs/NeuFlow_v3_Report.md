# NeuFlow v3 — Project Report

**Shriman Raghav Srinivasan · MS Robotics, Northeastern University**
Updated 2026-07-10 · replaces `docs/archive/report.pdf` and `docs/archive/proposal.pdf`

---

## 1. Background: what NeuFlow v2 is

NeuFlow v2 (Zhang, Gupta, Jiang, Singh — arXiv:2408.10161) is a real-time optical flow
network for edge devices. Pipeline: a shallow CNN backbone extracts features at 1/8 and
1/16 scale → cross-attention + global matching at 1/16 gives an initial flow → a
lightweight recurrent refinement (1 iteration at 1/16, 8 at 1/8) produces flow at 1/8
resolution → a **convex upsampler** (learned 3×3 blending weights on a fixed 8× grid)
produces the full-resolution flow map. It runs 10–70× faster than SOTA methods at
comparable accuracy (>20 FPS at 512×384 on a Jetson Orin Nano).

Its one structural limit: the output is a **dense, fixed-resolution map**. Every pixel is
always computed, and only at the input resolution.

## 2. What NeuFlow v3 changes

v3 keeps the entire v2 pipeline through the 1/8-resolution coarse flow (backbone frozen,
weights untouched) and replaces only the upsampler with an **implicit, queryable decoder**
(InfiniDepth/AnyFlow lineage):

- Any continuous (x, y) coordinate can be queried directly: cost is O(N) in the number
  of queries, not O(H×W).
- Two-pass API: `infer_coarse_state()` once per image pair (~33 ms), then
  `decode_queries()` at ~1.6 ms per batch of ≤2k points — repeatable at no extra
  backbone cost.
- Decoder architecture: 3×3 local-window sampling of 4 feature sources (context,
  1/8 features, 1/16 features, flow-warped frame-1 features), gated hierarchical fusion
  (InfiniDepth Eq. 3), then a head.
- **Convex-weight head** (added 2026-07-09, AnyFlow-style): the MLP outputs softmax
  weights over the 3×3 coarse-flow neighborhood + a bilinear candidate. Output is a
  bounded blend — it cannot hallucinate large flow. Initialization is biased to the
  bilinear candidate, so an untrained decoder exactly reproduces bilinear upsampling.
- **Fourier positional encoding** (added 2026-07-10, ablation in progress): encodes the
  query's sub-cell offset so the head can produce sub-pixel-sharp output.
- v3 is *smaller* than v2: 7.83M vs 9.03M parameters.

## 3. Results (per-pixel, VKITTI2 Scene18+20, 1174 pairs, 460M pixels)

All numbers from `scripts/eval_vkitti2.py` after the 2026-07-08 metrics fix.
(Numbers reported before that date used per-frame statistics and are invalid.)

| Configuration | Trained on | Mean EPE | 1px acc | 3px acc |
|---|---|---|---|---|
| NeuFlow v2 (reference) | FlyingThings | 2.324 | **77.6%** | **89.8%** |
| v3, **no training at all** (bilinear init) | — | 2.476 | 74.7% | 88.2% |
| v3 convex head | VKITTI2 (6 variants, 12.7k pairs) | 2.388 | 74.7% | 88.9% |
| v3 convex head | **FlyingChairs only (22.2k pairs)** | **2.275** | 69.7% | 87.8% |
| v3 convex, chairs → vkitti2 finetune | both, sequential | 2.499 | 74.6% | 88.6% |
| v3 convex + Fourier PE | FlyingChairs | 2.288 | 69.7% | 87.8% |
| **v3 convex, MIXED training** | **chairs + VKITTI2 jointly** | **2.183** | **76.4%** | **89.6%** |

Visual comparisons: `results/visuals/compare_*.png` (GT vs v2 vs v3 + error maps),
`results/visuals/sparse_queries.png` (300 corner queries in one 1.6 ms call),
`results/visuals/query_gui_selftest.png` (interactive GUI).

In-domain check (FlyingChairs validation, 640 held-out pairs): v2 2.238 px EPE / 78.7% 1px;
chairs-trained v3 2.399 px / 76.6%. The 1px gap is smaller in-domain than on VKITTI2.

Key findings:

1. **Zero-training operating point.** With no decoder training whatsoever, v3 delivers
   2.48 px EPE (+0.15 vs v2) while adding queryability. The sparse-query mechanism is
   *exact*: decoding N points matches the dense output at those points to 0.00 px.
2. **Chairs-only beats v2 on mean EPE** (2.275 vs 2.324) with no driving data in
   training — evidence the queryable decoder generalizes rather than memorizes.
3. **Failure mode found and fixed.** The original head regressed flow deltas scaled by
   image width; it *never trained below its own initialization* (best 2.77). Bounding
   the output via convex weights fixed this immediately.
4. **Sequential finetuning forgets.** chairs → vkitti2 at lr 1e-4 lost the chairs
   robustness (2.28 → 2.50). Mixed-dataset training is the identified fix.
5. Remaining gap to v2: sub-pixel precision (1px accuracy 69.7–74.7% vs 77.6%).
6. **PE ablation: null result (2026-07-10).** Adding Fourier sub-cell encoding to the
   chairs recipe changed nothing (2.288 vs 2.275 EPE, 1px acc identical at 69.7%).
   Clean falsification: the 1px gap is NOT missing positional signal. Remaining
   hypotheses: (a) the 1/8 coarse flow bounds recoverable detail; (b) chairs' large
   motions never supervise sub-pixel discrimination — test PE with vkitti2/mixed data.

## 4. Compute: v2 vs v3 (RTX 4060 Laptop 8GB, fp16, 384×1248)

| Metric | NeuFlow v2 | NeuFlow v3 |
|---|---|---|
| Parameters | 9.03 M | **7.83 M** |
| Full-frame dense flow | **37.0 ms (27 FPS)** | 326 ms (3 FPS) |
| Coarse pass (once per pair) | — | 32.9 ms |
| Decode 800 queries | not possible | **1.6 ms** |
| Decode 4,096 queries | not possible | 2.2 ms |
| Sparse total (≤2k pts) | — | **~35 ms (~29 FPS)** |
| Extra queries on same pair | full recompute (37 ms) | 1.6–2.2 ms |
| Inference VRAM (sparse) | — | ~2.2 GB |

**Video-pipeline throughput** (60 frames of a 640×360 YouTube stream, end-to-end,
`scripts/benchmark_fps.py`): v3 sparse-800 **63.6 FPS** · v2 dense 60.3 FPS ·
v3 + live motion boxes 47.1 FPS · v3 dense 5.8 FPS. At video resolution the sparse
mode outpaces v2's full map while answering targeted questions.

The operating point that matters for edge robotics: **v3 answers sparse queries at the
same latency v2 needs for a full frame** — and any *additional* queries on an
already-processed pair are ~200× cheaper than v2 recomputing. Dense v3 output is 8.8×
slower than v2 and is not the intended use.

**Objective scorecard** (better EPE at less compute, edge-capable):
- Mean EPE better than v2: ✅ (2.183 vs 2.324, mixed training — 6% better)
- Less compute for the sparse use case: ✅ (~35 ms, O(N), fewer params)
- Sub-pixel precision parity: ~✅ within 1.2 points (76.4% vs 77.6%); 3px at parity (89.6 vs 89.8)
- Edge-device validation (Jetson): pending — next after PE

## 5. The query interface — exact numbers and how to use it

**Query size.** A "query" is one continuous (x, y) coordinate. N is free: 1 (a single
click) to H×W (dense). Reference points at 384×1248 input:

| Mode | N | Share of dense | Decode cost |
|---|---|---|---|
| Single point (GUI click) | 1 | — | ~1.6 ms |
| Registration demo | 800 | 0.17% | 1.6 ms |
| Training supervision | 4,096 per image | 3.1% of a 256×512 crop | — |
| Edge sweet spot | ≤2,048 | 0.4% | 1.6 ms (flat) |
| Dense (full frame) | 479,232 | 100% | ~293 ms |

Queries are continuous — (312.7, 188.2) is as valid as (312, 188); the decoder
interpolates features bilinearly, so sub-pixel positions are first-class.

**How to query (the entire API):**

```python
model = NeuFlow(use_implicit=True, head_mode='convex')          # + use_pe=True for PE checkpoints
state = model.infer_coarse_state(img0, img1)                    # once per pair, ~33 ms
flow  = model.decode_queries(state, query_coords=q)             # q: [B, N, 2] (x, y) pixels -> [B, N, 2]
flow  = model.decode_queries(state, target_h=H, target_w=W)     # dense grid at ANY resolution
flow  = model.decode_queries(state, adaptive_n=1000)            # auto-allocate at motion boundaries
```

**Interactive GUI** (`scripts/query_gui.py`, PyQt5 — all features self-tested offscreen):
- Click-to-query with arrows/values; uniform grid; boundary-adaptive; dense overlay.
- **Region query window**: drag-select a rectangle — flow is computed only inside it
  (per-pixel within the window, auto-strided above 80k points).
- **Video sources**: local files and YouTube URLs (via yt-dlp); N/P frame stepping.
- **Real-time playback with motion detection**: Space plays the video through the
  pipeline continuously; moving regions are boxed live from the coarse flow with the
  median (ego-motion) subtracted — zero additional decode cost. ~39 FPS end-to-end on
  the self-test clip at 1024-width.
- **System resources tab**: live 2 Hz graphs of pipeline FPS, coarse-pass latency,
  GPU utilization, VRAM, CPU, and RAM — VRAM stays flat during interaction, showing
  the cached two-pass design at work.
- CSV export of queries, screenshot save, checkpoint switching.
The two-pass API is what makes all of this possible — v2 would need a full 37 ms
recompute per interaction; v3 answers from cache in ~1.6 ms.

**Training configuration (current):**

| Parameter | Value |
|---|---|
| Batch size | 4 (VRAM-bound, RTX 4060 8 GB) |
| Crop | 256×512 (VKITTI2) / 384×512 (chairs) |
| Queries per image | 4,096 (50% at motion boundaries, 50% uniform, integer pixels) |
| Datasets in use | VKITTI2 clone+5 variants (12,726 pairs) · FlyingChairs (22,232 pairs) |
| Optimizer | AdamW, OneCycle peak 2e-4, wd 1e-4, clip 1.0, gamma 0.8 |
| Steps | 15K (vkitti2_all) / 30K (chairs) · backbone frozen |

## 6. Anticipated questions (first principles)

**Q: Why can flow be queried at continuous coordinates at all?**
The feature maps are discrete, but bilinear interpolation makes them a continuous
function of position, and the decoder is an MLP applied to that function. The
composition is defined at every real-valued (x, y) — integer pixels are just the
special case a dense map hard-codes.

**Q: Why freeze the backbone?**
InfiniDepth stabilizes joint training over 800K steps on 8 GPUs. At our 15–30K-step
budget the decoder chases shifting features and diverges (measured: EPE oscillating
1.9→54 in early experiments). Frozen features = stationary target. It also means v3
inherits v2's matching quality by construction.

**Q: Why did training make things worse for weeks?**
The original head predicted a correction scaled by image width (~1248×). To be
harmless it had to output almost exactly zero; any noise became pixels of error. The
convex head bounds outputs to blends of neighboring coarse-flow values — after this
one change, the same recipe trained below its initialization for the first time.

**Q: Why does FlyingChairs training beat VKITTI2 training on VKITTI2?**
22k diverse pairs with large motions teach robustness; 12.7k frames from five driving
scenes teach memorization. The error *tail* (occlusions, fast regions) dominates mean
EPE, and chairs attacks exactly that. The 2.1k-frame version of this lesson was worse.

**Q: Is sparse output an approximation of dense?**
No — identical function, fewer evaluation points. Verified: sparse queries match the
dense map at the same coordinates to 0.00 px.

**Q: If sparse is fast, why is dense v3 slow?**
Dense means 479k MLP evaluations (~293 ms); v2's convex upsample is one fused conv
(~4 ms). v3 wins when you need *some* points, not *all* points — which is the actual
requirement in registration, tracking, and mapping.

**Q: What limits sub-pixel (1px) accuracy?**
Measured answer: not missing positional signal — injecting Fourier sub-cell encoding
changed nothing (2.288 vs 2.275 EPE, 1px acc identical). The remaining suspects are
the 1/8 coarse flow itself bounding recoverable detail, and training data whose
motions are too large to supervise sub-pixel discrimination (chairs). Testing PE with
fine-motion data (vkitti2/mixed) separates the two.

**Q: Why batch size 4?**
8 GB VRAM. The recipe is otherwise standard RAFT; on cluster GPUs batch 8–16 with the
same settings is the expected scale-up.

**Q: Can it output at higher resolution than the input?**
Yes — query any grid (`target_h/w` are free). The Spring benchmark (GT at 2× input
resolution) is the planned test only queryable decoders can take natively.

## 8. Repository organization

```
NeuFlow_v3/
├── NeuFlow/                  model (implicit_decoder.py = the v3 contribution)
├── data_utils/ utils/        loaders (chairs/things/sintel/kitti/vkitti2[_all]/viper), loss, DDP
├── scripts/                  eval_vkitti2.py · benchmark_edge.py · demo_registration.py
│                             stream_chairs_png.py (zip-less dataset acquisition)
├── train_*.sh                one script per documented experiment
├── checkpoints/<run>/        step_XXXXXX.pth + train_log.csv per run
├── datasets/                 vkitti2 (37 GB, 10 variants) · FlyingChairs_release (46 GB, PNG)
├── docs/
│   ├── NeuFlow_v3_Report.md      ← this file (single source of truth with base_parameters.md)
│   ├── base_parameters.md        parameter derivations + full result log
│   ├── NeuFlow_v3_update.pptx    progress deck (2026-06-27, +baseline slide 07-09)
│   ├── NeuFlow_v3_status.pptx    current status deck (2026-07-10)
│   └── archive/                  superseded reports (report.pdf, proposal.pdf, meeting prep)
└── results/                  charts + demo outputs
```

Branches: `v1-dev` = corrected metrics + paper-aligned recipe · `v2-dev` = convex head,
curriculum, PE (current). Remote: github.com/shrirag10/Neuflowv3.

## 8. Next steps

1. ~~Fourier PE ablation~~ — done, null result; 1px gap is not positional.
2. ~~Mixed chairs + vkitti2 training~~ — done: 2.183 EPE / 76.4% 1px, best result to date;
   hypothesis confirmed (joint sampling prevents forgetting).
3. **Jetson benchmark** — port `benchmark_edge.py`; the O(N) claim is strongest where
   dense flow genuinely cannot run.
4. Thesis framing unchanged: queryable flow for registration/mapping — one backbone
   pass, on-demand correspondences at chosen points, arbitrary resolution.
