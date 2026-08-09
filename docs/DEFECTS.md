# NeuFlow v3 — defect register

Every bug, wrong number and false result encountered building this project, in one
place. `V3DEV_LOG.md` is the chronological record and `HANDOFF.md` is the catch-up
document; this is the consolidated list, organised by status rather than by date.

Written because almost every defect here produced a **plausible, confidently stated,
wrong number** rather than a crash. That is the failure mode this project actually
has, and the checks in `scripts/verify_pipeline.py` exist because of the entries
below, not the other way round.

Last updated 2026-08-08.

---

## A. Fixed, and now guarded

### A1. Metrics mislabelled
"1px accuracy" was computed per frame, not per pixel: the fraction of *frames* whose
mean EPE was below 1 px. Every accuracy number published before 2026-07-08 is
invalid. `eval_vkitti2.py` now prints both weightings side by side and labels the old
one explicitly.

### A2. Train/test leak
VKITTI2 Scene18 and Scene20 appeared in both the training set and the evaluation set.
Fixed structurally rather than by discipline: `VKITTI2.__init__` raises unless
`allow_val_scenes=True`, and `scripts/check_leak.py` verifies overlap at pair *and*
frame level before every run.

### A3. BatchNorm not frozen — the expensive one
`model.train()` in the validation block re-enabled running-statistic updates inside
the supposedly frozen front end. 24,765 updates accumulated over a 100k-step run,
drifting running_mean 7.4% and running_var 17.4%, changing the coarse flow by
0.350 px mean.

Worth about **0.25 px**, which is what made v3 appear to match v2. The corrected
result moved v3 from "matches v2" to "2.6% behind v2". All pre-fix figures were
withdrawn.

Root cause worth generalising: **BatchNorm updates running stats in train mode
regardless of `requires_grad`.** Freezing weights is not freezing a module.
`set_frozen_bn_eval` is now called at three separate points in `train.py`, because
one was not enough.

### A4. Padded-frame query offset
`decode_queries` takes coordinates in the *padded* frame, and `InputPadder` centres
its padding. Three call sites passed raw frame coordinates, so on a 375x1242 VKITTI2
frame every sub-region query landed 3 px off in x and 4 px in y. Worth 0.099 px.
Guarded by check 7b in `verify_pipeline.py`.

### A5. Spring ground-truth scale
`read_flo5` divided by 2.0 on the assumption that a 2x-resolution grid meant
2x-scaled values. It does not; the values are already in input-resolution pixels.
This halved every Spring training target. Caught by a units check, not by reading the
code. `--check_units` in `eval_spring_4k.py` now determines the convention by
measurement.

### A6. Unequal stride between policies
The scenario benchmark ran `v3_full` at stride 1 and `v3_crop` at stride 2, making
v3_full look 4x more expensive than it is. All v3 policies now share one stride.

### A7. Missing validity mask in the sparse policy
VKITTI2 stores unusable values where flow is occluded or leaves the frame. Without
filtering them the sparse policy reported **58 px** EPE instead of 0.6.

### A8. Unbounded regression head
The original decoder head regressed flow directly and never trained below its own
initialisation. Replaced by the convex head, which blends neighbouring coarse-flow
values and therefore cannot leave the locally supported range. Zero-initialised it is
exactly bilinear upsampling, so training starts from a known-good operating point.

### A9. `fusion_on_grid` cost understated
Recorded as "+0.02 px". Actually **0.068 px mean, 22.3 px max**, concentrated at
motion boundaries, with 1.4% of pixels exceeding 1 px. Every `--fast_dense` number
carries this approximation.

### A10. Uncertainty channel missing from the fast path
`forward_dense_fast` was never updated for the extra output channel when the
uncertainty head was added, so `--fast_dense --uncertainty` crashed. The regular
decode path had been patched; the second path was missed.

---

## B. Open, found in the 2026-08-08 audit

Severity is about the effect on published claims, not on code correctness.

### B1. The repeat-query compute claim assumes the baseline discards its own output
**Severity: high. This is the most consequential entry in the file.**

The headline "27x cheaper repeat query" compares v3's 1.25 ms decode against v2's
33.3 ms *full recomputation*. That is only the right comparison if a v2 integrator
throws the dense flow field away between questions. A dense field at 384x1248 is
3.8 MB; any real system keeps it, and answering a second question is then an array
index at effectively zero cost.

The project's own scenario benchmark already accounts for it this way, and the totals
say so plainly (`results/scenarios_4060.json`, median ms, RTX 4060):

| scenario | policy | first | +2nd ROI | total | EPE |
|---|---|---:|---:|---:|---:|
| S3 | v2_crop | 7.82 | 7.29 | **15.11** | 3.600 |
| S3 | v3_crop | 8.46 | 8.23 | 16.69 | 3.610 |
| S3 | v2_full | 32.75 | 0.00 | **32.75** | 1.761 |
| S3 | v3_sparse | 33.52 | 1.68 | 35.21 | 2.099 |
| S3 | v3_full | 36.24 | 4.36 | 40.59 | 1.769 |

**v3 is not the cheapest policy in any of the three scenarios.** v2_full's marginal
cost is zero because it has already computed everything, and its first-pass cost is
*lower* than v3's coarse pass (32.75 against 33.52). The "1.68 ms against 7.29 ms"
figure in the deck compares v3_sparse's marginal against **v2_crop's** marginal, not
against v2_full's zero.

This does not make v3 worthless; it means the advantage is capability, not compute.
See §D for the framing the data does support.

### B2. Selective prediction is a confounded comparison
**Severity: high.** v2 is plotted as a flat line at 2.324 across every coverage level
and is never evaluated on the accepted subset that v3's confidence selects. The
accepted 80% is the easy 80%, so most of the claimed 2.2x is pixel difficulty rather
than the confidence head. The honest comparison scores v2 on the same accepted
pixels. **Left as published by explicit decision, 2026-08-08.**

### B3. Two values published for one quantity
**Severity: high.** The accuracy table gives 2.384 and the selective-prediction table
gives 2.266 for the same model on the same data. Cause confirmed:
`eval_calibration.py:66` samples exactly 2,000 points per image, making its aggregate
an equal-weight-per-image mean, while `eval_vkitti2.py:132` reports per-pixel. Gap
replicated at 0.102 px. Under matched weighting v2 scores 2.2267, so the 100%
coverage row changes sign.

### B4. The accuracy table compares different decode paths
**Severity: medium.** v3 numbers come from `--fast_dense --stride 2`, whose
`fusion_on_grid=True` is the approximation in A9. v2 cannot take that path
(`decode_dense_fast` raises for `use_implicit=False`), so it is measured exactly.
Undisclosed in the paper's Experimental Setup. Exact-path v3 measures **2.3807**
against the published 2.3925.

### B5. Half-pixel misregistration in `query_region`
**Severity: medium.** `flow_engine.py:151` resamples a strided grid with `cv2.resize`,
which assumes the samples span the full extent; they actually sit at `y0, y0+2, ...`.
Verified numerically on a ramp: a uniform 0.5 px shift, 1.0 px at the far edge.
Costs +0.0125 EPE on the `v3_full` policy. `decode_dense_fast` gets this right, so one
path was out of step with the other.

### B6. `verify_pipeline.py` check 4 is worse than vacuous
**Severity: medium.** `build()` always loads `neuflow_mixed.pth`, so the decoder is
Xavier-random and pinned to bilinear by `convex_prior = 9.0`. Checks 3, 4 and 7 then
compare bilinear against bilinear and pass with wide margins. On a trained checkpoint
check 4 **fails at 1.193 px**, because it compares `decode_queries` (exact) against
`decode_dense_fast` (approximate) rather than against the exact forward. Adding a
`--checkpoint` flag without repointing check 4 turns the suite red for the wrong
reason.

### B7. The 36,864-point latency claim is wrong by 6x
**Severity: medium.** `HANDOFF.md:79` and `bench_scenarios.py:20` state that a dense
192x192 ROI, 36,864 points, costs about 4 ms. Measured, `decode_queries` at that N is
**23.2 ms**; the 4 ms belongs to `decode_dense_fast`, a different path. The sparse
path reaches v2's full-frame cost at roughly **55,000 points, about 11% of the
frame**, which is a real ceiling on the O(N) story that the docs currently obscure.

### B8. Three values for the repeat-query latency
**Severity: low, but visible.** `benchmark_sparse.py` reports 1.25 ms,
`bench_scenarios.py` reports 1.68 ms, and its own docstring says about 2.5 ms. The
presentation uses the smallest. They use different protocols; measured
decode-after-coarse is 1.51 ms and the second query batch on a cached state is
1.32 ms.

### B9. `adaptive_query` coordinate offset
**Severity: none for published results.** `adaptive_query.py:179` maps coarse
coordinates with `c*8` where the align_corners=False cell centre is `8c+3.5`, biasing
adaptive query placement 3.5 px toward the top left and never sampling the last 8
rows or columns. It only chooses *where* to query; the returned flow is correct. No
evaluation or benchmark script passes `adaptive_n`.

---

## C. Investigated and cleared

### C1. Truncated ground-truth lookup in the scenario benchmark
`bench_scenarios.py:233` reads GT at `pxy.astype(int)`, which truncates rather than
rounds. Suspected of inflating the sparse policy's EPE. It does not:
`cv2.goodFeaturesToTrack` is called without sub-pixel refinement and returns
integer-valued float32, so decode coordinates and GT indices coincide exactly.
Latent, and would bite the moment `cornerSubPix` or a non-integer detector is added.

### C2. Batched timing hiding launch overhead
Suspected that `benchmark_sparse.py` understated latency by running iterations
back-to-back, letting CPU enqueue overlap with GPU execution. Tested with
per-iteration CUDA-event timing and a synchronise inside the loop: batched and
isolated agree within **3%**. The call is host-dispatch bound, 498 µs of GPU kernel
time across 144 launches against about 1.15 ms of enqueue, which makes 1.25 ms
conservative rather than optimistic.

---

## D. What the compute data actually supports

Stated here because §B1 removes the claim the project had been leaning on.

**Not supported:** that v3 completes flow in less time than v2. On total latency v2 is
cheaper in all three scenarios, and v2_crop is cheapest overall.

**Supported and measured:**
- A first sparse query is level with v2, 34.1 ms against 33.3, so queryability costs
  about 2% on the first pass rather than a multiple.
- Decode is flat from N=64 to N=2,048, so 2,048 points cost what 800 do.
- 13% fewer parameters, 7.83 M against 9.03 M.
- Region-directed inference gives 4.4x for +0.034 px, and the margin needed scales
  with motion. **This applies to v2 unchanged**; it is a platform technique, not a
  property of the decoder, and the paper already says so.

**Supported, structural, not yet measured:** output bandwidth. v2 must materialise
479,232 vectors, 3.8 MB per frame, regardless of how many are consumed. v3 at N=800
writes 6.4 KB. On a bandwidth-limited embedded target that is a real difference and it
is measurable, but no measurement exists today.

**Capability, not compute, and this is where the case actually is:** continuous
coordinates, and a calibrated per-query confidence that v2 cannot express at any
price. Those do not appear in a latency table.

---

## E. Infrastructure and process faults

- **Scratch purges monthly.** Every uncertainty result depended on a checkpoint that
  existed only on the cluster, and no local checkpoint could substitute (all had
  `convex_head` out_dim 10; the uncertainty head needs 11). Retrieved 2026-08-08.
- **TartanAir and Spring are not on the laptop**, so nothing aerial can be
  re-verified or regenerated locally.
- **Laptop disk at 97%**, 8.6 GB free. The exact dense evaluation OOMs on 8 GB VRAM
  without `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Hand-written sbatch files drifted**: two runs used 2x s16 + 4x s8 while others
  used 1x + 8x, and one used batch 12 instead of 16, so three variables moved at once
  and no comparison between them was valid. Fixed by generating every run from one
  template in `hpc/make_sbatch.py`.
- **A GitHub token is embedded in `.git/config` remotes** and has appeared in terminal
  output. Needs rotating.

---

## F. Deck build faults, 2026-08-08

Caught by rendering the decks and looking at them, which had not been done before.
All are now build-time assertions in `scripts/deck_style.py` rather than things to
remember.

| Fault | Effect |
|---|---|
| Table row height a constant, not derived from point size | Header rule drawn through the header text on three slides |
| `word_wrap=False` on table cells | LibreOffice centres text instead of honouring alignment, so every column was mis-aligned |
| Figures placed at hardcoded y with the next element also hardcoded | Two figures overlaid their own captions; a table grew into the panel beside it |
| Key number pinned to a fixed height | Laid over the last bullet of its column |
| Bullet-block height estimated from total width over column width | Undercounted lines; text ran under the citation line |
| Line height taken as the paragraph multiple | DejaVu Serif's own line height compounds with it; measured leading is 1.42x, not 1.14x |
| Figures placed at 0.55 scale on a 10-inch canvas | Plot labels rendered at half the size of the body text beside them |
| Speaker notes never re-checked after the BatchNorm fix | Three notes in the status deck stated withdrawn numbers, including "2.286, that is a tie" |

---

## Lesson that generalises

Three separate defects here (A1, A3, B3) were **weighting or mode errors that produced
a better number than the truth**, and none of them crashed anything. The project's
guard against that is now: measure rather than assume the convention (`check_convention`
in `tartanair.py` scores four candidate readings and requires a clear winner), print
both weightings side by side, and make the checks fail loudly on a *trained*
checkpoint rather than a convenient one.
