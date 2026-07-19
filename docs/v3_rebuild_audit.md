# NeuFlow v3 — Critical Audit and First-Principles Rebuild

2026-07-19. Purpose: separate what is *proven* from what is *approximate, confounded,
subset-only, or pending* before designing the rebuilt algorithm. No expectations stated
as results anywhere in this document.

---

## Part 1 — Claims ledger

### VERIFIED (full evidence, safe to state publicly)

| Claim | Evidence |
|---|---|
| Sparse queries are exact: decoding N points == dense output at those points | 0.00 px difference, verified twice (2026-06-27, 2026-07-10) |
| Conv-folding of window projections is mathematically exact | Derivation (linearity) + measured: full-set-eval EPE identical to 3 decimals on 118 pairs; synthetic-feature test max diff 0.002 on 0.74-magnitude features |
| Untrained decoder = bilinear upsampling of coarse flow, 2.476 px EPE (full 1,174-pair set) | Full-set eval 2026-07-08 |
| v2 reference on our benchmark: 2.324 px / 77.6% 1px / 89.8% 3px (full set) | Full-set eval 2026-07-08 |
| v3 mixed (standard dense): 2.183 px / 76.4% / 89.6% (full set) | Full-set eval 2026-07-10 |
| PE ablation null result | Full-set evals of both arms, identical metrics |
| Chairs-trained v3 loses in-domain to v2 (2.399 vs 2.238 on 640 chairs val pairs) | Full chairs-val eval 2026-07-11 — an honest negative |
| Original regress head never trained below its initialization | 6-checkpoint full-set sweep 2026-07-09 |

### TRUE BUT CONFOUNDED (do not state without the caveat)

| Claim | Confound |
|---|---|
| "v3 mixed beats v2 by 6% EPE" (2.183 vs 2.324) | **Training data is not matched.** v3-mixed trained on VKITTI2 Scene01/02/06 (same simulator and camera as the Scene18/20 eval); v2 never saw any VKITTI2. Part of the 6% is domain familiarity, not architecture. The size of that part is UNKNOWN. |
| "chairs-only v3 beats v2" (2.275 vs 2.324) | Weaker version of the same issue: none of v3's training data is driving-domain (good), but v3's decoder was still *selected* using VKITTI2 evals during development. Selection bias is mild but nonzero. |
| "v3 dense (2,4)+stride2 at equal EPE to v2" | Inherits the domain confound above AND was subset-only until today's full-set run. |

The only fully clean comparison would be: retrain v3's decoder on exactly v2's training
mixture, then evaluate both on a dataset neither saw. This is PENDING (needs HPC).

### APPROXIMATE BY DESIGN (measured, small, must be disclosed)

| Approximation | Measured cost |
|---|---|
| Fusion computed on 1/8 grid instead of per-pixel | +0.02 px EPE (118 pairs) |
| Stride-2 dense decoding + bilinear upsample | none detectable (3 decimals, 118 pairs) |
| (2,4) refinement schedule instead of (1,8) | +0.04 px EPE (118 pairs) |

### FULL-SET RESULTS (2026-07-19, 1,174 pairs — supersede all subset numbers)

| Config | Latency | EPE | 1px | 3px |
|---|---|---|---|---|
| v3 fast dense (2,4) stride2 | 28.0 ms (35.7 FPS) | 2.3248 | 75.70% | 89.03% |
| v3 fast dense (1,8) stride2 | 38.0 ms (26.3 FPS) | 2.2027 | 76.47% | 89.44% |
| v2 reference | ~37 ms (27 FPS) | 2.324 | 77.6% | 89.8% |

The 118-pair subset flattered the (2,4) mode (2.284 subset vs 2.325 full). Corrected
statements: (2,4) = same EPE as v2, 24% faster, 1px 1.9 points worse. (1,8) = 5.2%
better EPE at the same speed, 1px 1.1 points worse. "Faster AND more accurate
simultaneously" did NOT survive full evaluation. Domain confound still applies to both.

Still subset/single-run: video-pipeline FPS (one YouTube video).

### PENDING / NOT DONE (state as such, never as results)

2. **Jetson/edge validation.** Every latency number is from an RTX 4060 laptop GPU.
   "Edge-capable" is currently an extrapolation, not a result.
3. Matched-training-data comparison vs v2 (needs HPC).
4. Decoder retrained against (2,4) coarse flow (needs HPC; until then (2,4) numbers
   use a decoder trained on (1,8) statistics).
5. Spring 4K evaluation (queryable-resolution claim has no quantitative benchmark yet).
6. The 1px-accuracy gap (75.7–76.4% vs v2's 77.6%): no fix demonstrated. PE ruled out
   positional signal; coarse-resolution bound vs training-data hypotheses UNTESTED.
7. Sintel/KITTI standard benchmarks: never run. All accuracy statements are
   VKITTI2 (+ chairs val) only.
8. Training was decoder-only throughout; end-to-end v3 training never attempted.

### MISUNDERSTANDINGS FOUND AND CORRECTED DURING THIS AUDIT

- Pre-2026-07-08 "1px accuracy" numbers were per-frame statistics mislabeled as
  per-pixel; every table since has been rebuilt (this was caught and fixed on 07-08).
- "1.03 px reprojection error" in the June registration demo was RANSAC self-consistency,
  not accuracy against ground truth (caught 06-27; never quote it as accuracy).
- The dense-fast exactness check initially looked broken (0.34 px mean deviation);
  root cause is fp16 + interpolation noise amplified by softmax near motion
  boundaries, not a math error — the synthetic-feature test isolates this. EPE unchanged.

---

## Part 2 — First-principles design of the rebuilt v3

Question the algorithm answers: *given two images, return flow at the coordinates the
consumer asks for, at cost proportional to what is asked.*

Design decisions, each justified only by evidence from Part 1:

1. **Keep v2's pipeline through the 1/8 coarse flow, frozen.** Evidence: every accuracy
   gain and loss we measured came from the decoder and data; the coarse flow was never
   the failure mode. Frozen training is also the only regime proven stable at our compute.
2. **Convex-weight head, bilinear-prior init.** Evidence: the unbounded regress head
   never beat its init; the bounded head trained below init immediately and degrades
   gracefully under coarse-flow truncation (+0.04 vs +0.13 px for v2's upsampler).
3. **No Fourier PE.** Evidence: null result. (Cheap to re-add if fine-motion training
   data ever changes the picture — that is a hypothesis, not a plan-of-record.)
4. **Conv-form decoding as the PRIMARY dense path** (not a retrofit): window projections
   are convolutions by construction; the per-query path exists for sparse/continuous
   queries. Evidence: mathematically exact, 4x latency reduction measured.
5. **Stride-2 default for dense output.** Evidence: no measurable EPE change.
6. **Train at the inference schedule.** The (1,8)-trained decoder is being run at (2,4);
   rebuild trains at (2,4) directly. (Effect size: PENDING, HPC.)
7. **Mixed-dataset single-stage training.** Evidence: sequential finetune forgot
   (2.28 -> 2.50); joint sampling kept both strengths (2.18).
8. **Evaluation protocol fixes baked in:** full-set only; matched-training-data
   comparison as the headline number once HPC training lands; in-domain AND
   cross-domain results reported together; 1px accuracy always reported next to EPE.

What the rebuild does NOT include (rejected for lack of evidence): PE (null), deeper
fusion (never ablated — flagged as an open ablation, not assumed useful), end-to-end
training (diverged at local compute; untested on HPC), half-resolution input (measured
-1.2 px untrained; only revisit with retraining).

## Part 2b — Rebuild results (2026-07-19, full set, VKITTI2 Scene18+20)

`v3-rebuild` branch: unified conv-form decoder, trained 30K steps on the mixed stage AT
the (2,4) schedule (`train_v3rebuild.sh`, checkpoints/neuflowv3_rebuild). Regression
gate passed first (old checkpoint through rebuilt path reproduced 2.2027 exactly).

| Config | Latency | EPE | 1px | 3px |
|---|---|---|---|---|
| rebuild, (2,4)+stride2 (native) | 28.3 ms (35.4 FPS) | 2.2338 | 75.89% | 89.31% |
| rebuild, (1,8)+stride2 | 38.0 ms (26.3 FPS) | 2.0946 | 76.53% | 89.66% |
| pre-rebuild decoder, same configs | 28.0 / 38.0 ms | 2.3248 / 2.2027 | 75.70 / 76.47% | |
| v2 | ~37 ms | 2.324 | 77.6% | 89.8% |

Measured statements:
- Training at the (2,4) schedule improved (2,4) inference by 3.9% EPE; the open audit
  item "effect size pending" is now answered.
- It also improved (1,8) inference by 4.9% (2.0946, best v3 number recorded). Training
  against rougher coarse flow did not hurt the longer schedule; it helped it.
- The 28 ms config is now better than v2 on mean EPE (2.234 vs 2.324) AND 24% faster,
  full-set. The 1px gap REMAINS (75.9 vs 77.6) and the domain confound REMAINS
  (training mixture includes same-simulator scenes; v2's does not).

Still pending, unchanged: matched-data comparison (HPC), Jetson, Sintel/KITTI, Spring,
1px-gap root cause.

## Part 3 — Execution order

1. Full-set evals (running) — replaces every subset number in the docs.
2. Local: `v3_rebuild` branch with the decoder rebuilt per Part 2 (small, mostly
   deletion: PE code out, conv-form primary, (2,4) default).
3. HPC: matched-data training (v2's mixture), train-at-(2,4), then the one clean
   headline comparison this project currently lacks.
4. Jetson benchmark before any "edge" wording survives into the thesis.
5. Sintel/KITTI eval for external validity.
