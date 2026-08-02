# v3-dev running log

Branch policy: v3-dev = v2-dev (trusted code) + new work. Nothing from the
v3-rebuild decoder rewrite is merged until it is independently validated;
the rebuild branch stays untouched for reference.

Every entry: what changed, why, and its verification status.

## 2026-07-27

- **Branch created** from v2-dev.
- **Ported from v3-rebuild (decoder-independent, additive only):**
  `hpc/` (bootstrap, env-setup job, download scripts, all sbatch files),
  `scripts/train_distill.py` (option A), `data_utils/datasets.py` +
  `frame_utils.py` (Spring loader + grand_mix/spring_mix stages, flo5 reader).
  NOT ported: `NeuFlow/` rewrite, deck builder edits.
- **hpc/train_big18.sbatch**: batch-16, 100K-step vkitti2_mix run at the
  (1,8) schedule on v2-dev code. Replaces the rebuild's "rebuild_big/18" runs
  as the scale-up experiment. Status: created, submission pending env job.
- **Option A (refinement self-distillation)** `scripts/train_distill.py`:
  teacher = frozen model at 8 s8-iterations, student trains refine_s8 only to
  match it in 3 iterations. No GT used. Status: code written; NOT yet run.
- **Option G (uncertainty head)**: being added to the v2-dev decoder's convex
  branch: one extra head output channel = per-query error scale b (Laplace),
  loss |err|/b + log b on the final iteration, flag `--uncertainty`.
  Old checkpoints unaffected (flag off = identical architecture).
  Status: in progress.
- **Local GPU broken** (driver error 804 after suspend; reboot pending), so
  local verification is CPU-only today; GPU validation happens on Explorer.
- **Cluster state**: env rebuild job 8718244 queued (partition short); five
  training jobs dependency-chained behind it were submitted from the
  v3-rebuild checkout and will be CANCELLED and resubmitted from v3-dev once
  this branch is pushed and checked out on the cluster.

- **Option G implemented on v3-dev** (decoder convex branch + train.py):
  `--uncertainty` flag, Laplace loss on final iteration. CPU smoke: b=1.0 at
  zero-init, old checkpoints load strict with flag off, gradients reach the
  b channel. GPU validation pending (local driver down).
- **hpc/train_uncertainty.sbatch**: vkitti2_mix, batch 16, 100K, (1,8), G on.
- **Cluster switchover**: checking out v3-dev on Explorer, cancelling the five
  rebuild-code jobs, resubmitting grandmix/big18/spring/distill/uncertainty
  from v3-dev behind env job 8718244.

- **Cluster switched to v3-dev.** Env job 8718244 VERIFIED (torch 2.5.1+cu121,
  ALL_IMPORTS_OK). Rebuild-code jobs cancelled. Resubmitted from v3-dev:
  grandmix=8718342, big18=8718343, spring=8718344, distill=8718345,
  uncG=8718346 — all pending on GPU availability (H200). Health check
  (dataset counts + it/s) pending; nothing is a result until then.

- **Five-job failure diagnosed (user-run diagnostics on cluster):** two bugs.
  (1) `vkitti2_mix` stage never existed; real name `mix_chairs_vkitti2` —
  three jobs died on UnboundLocalError. (2) grand_mix/spring_mix mixed crop
  sizes across sub-datasets (368x496 chairs vs 368x768 rest), so
  default_collate crashed — the mix_chairs_vkitti2 stage even documents the
  one-crop rule. Fixed: common (320, 496) crop in both stages; sbatch stage
  names corrected. Jobs NOT resubmitted automatically (user runs sbatch
  manually per new policy).

- **All three mix stages verified on cluster** (user-run, compute node):
  single shape per stage, 0/60 load errors. mix_chairs_vkitti2=34,958,
  grand_mix=45,368, spring_mix=55,294 samples. Ready for submission —
  user submits manually.

- **Spring crash diagnosed** (job 8718657, died step 11): Spring flo5 GT
  contains NaN at invalid pixels; NaN reached the adaptive sampler's weight
  map and torch.multinomial device-asserted. Fixed twice over: read_flo5 maps
  NaN/inf to 1e9 (validity mask then excludes them, matching the |flow|<1000
  convention), and adaptive_flow_query sanitizes its weights before
  multinomial. Other four stages are Spring-free and unaffected.

- **Deck updated (23 slides):** three new slides before the FAQ — current-work
  table (five queued experiments, explicitly marked "no results yet"),
  flowcharts for self-distillation and the uncertainty head, and full-pipeline
  pseudocode (coarse pass / query batch / training loss). Verified by render.
  Note: first insertion attempt silently no-op'd on a wrong anchor; now the
  builder edit asserts the anchor and the inserted content.

- **First healthy training on HPC (user-verified):** grandmix 8718687
  (45,368 imgs) and big18 8718688 (34,958 imgs) RUNNING on H200 node d4053 at
  ~6.8 steps/s, batch 16 — ETA ~4 h for 100K steps, within the 8 h limit.
  Early epe ~7.9 during LR warm-up, expected for these mixes. Spring, distill,
  uncG submissions pending (user runs them after git pull for the NaN fix).

- **Distill eval blocked, fixed with new script.** `--no_implicit` fails on
  distill3 checkpoints: train_distill.py builds NeuFlow(use_implicit=True)
  (only infer_coarse_state is used), so no conv_s8/upsample_s8 weights ever
  existed to load. Correct test bypasses the decoder entirely (its weights
  are untrained Xavier noise in this checkpoint, irrelevant to option A).
  Added scripts/eval_coarse.py: bilinear x8 upsample of coarse_flow_s8
  straight to full-res EPE, no decoder involved. Also correctly measures the
  actual quantity distillation targets.

- **Option A result (2026-07-27, full-set, V100, coarse-flow-only eval):**
  | iters | EPE | 1px | latency |
  |---|---|---|---|
  | baseline (1,3) | 2.899 | 72.27% | 11.9 ms |
  | distilled (1,3) | 2.528 | 73.73% | 12.0 ms |
  | baseline (1,8) target | 2.475 | 74.74% | 18.5 ms |
  Distillation closes 87.5% of the 3-vs-8 iteration accuracy gap (0.370 of
  0.423 px) at 3-iteration speed, zero added compute. Real win.
  PENDING (not run): merging distilled refine_s8 weights with a trained
  decoder into one checkpoint and evaluating the full pipeline end-to-end —
  this coarse-only number is not yet a deployable result.

- **CRITICAL BUG FOUND: grandmix and big18's 6-hour full runs used the broken
  `regress` head.** Neither hpc/train_*.sbatch script passed `--head`, and
  train.py's default was `--head regress` — the head proven back on 2026-07-09
  to never train below its own initialization. Verified on the actual
  checkpoint: grandmix/step_100000.pth scores 2.584 EPE, WORSE than the
  untrained baseline (2.476). Both 100K-step runs are void for decoder
  purposes; backbone/refine weights are unaffected (frozen throughout, so
  not corrupted, just unused for six hours).
  spring (running, 47%) and uncertainty (just started) had the identical bug
  and were caught before wasting comparable time.
  FIXED: added `--head convex` to all four hpc/train_*.sbatch scripts, and
  changed train.py's default from `regress` to `convex` so this cannot recur
  silently. Distillation (option A) is unaffected — it never uses the decoder.
  ACTION NEEDED: cancel spring (8718906) and uncG (8718908), resubmit all
  four jobs after this fix.

- **Full audit after the regress-head incident (2026-07-27), everything checked:**
  1. ROOT FIX: `NeuFlow.__init__` class default was `head_mode='regress'` — the
     actual source of the bug, deeper than the four sbatch scripts. Flipped to
     `'convex'`. This protects every future call site that omits the argument.
  2. `scripts/eval_vkitti2.py --head` flag ALSO defaulted to `'regress'` — an
     eval run without `--head convex` would have silently reported wrong
     numbers on a correct checkpoint. Flipped to `'convex'`.
  3. Four legacy scripts (`infer_v3.py`, `benchmark_edge.py`,
     `demo_registration.py`, `eval_implicit.py`) hardcode pre-convex-head
     checkpoint paths (e.g. `neuflowv3_window_phase2/step_008000.pth`) and
     relied on the old default. Pinned explicitly to `head_mode='regress'`
     so the class-level fix above does not silently break them.
  4. `hpc/train_rebuild_big.sbatch` was a stale exact duplicate of
     train_big18.sbatch (same recipe, old name, also missing --head) —
     DELETED. Fixed two doc references (bootstrap.sh, explorer_setup.md)
     that pointed at it.
  5. `hpc/train_unfrozen.sbatch` had a dead --resume path
     (checkpoints/neuflowv3_rebuild/... does not exist on this branch) AND
     was missing --head convex. Fixed both; added an explicit DEPENDS-ON
     comment since it now correctly resumes from train_big18's future output
     and must not be submitted before that job finishes.
  6. Verified: all five active hpc/train_*.sbatch scripts pass --head convex,
     all changed .py files compile, all .sbatch files pass `bash -n`.
  7. Confirmed NOT affected by the bug: distillation (option A) never touches
     the decoder (only infer_coarse_state), so its verified 87.5%-gap-closed
     result stands unchanged.
  No further known head-mode landmines in the repository.

- **grandmix and big18 full-set results (2026-07-26, correct convex head, verified):**

  | Config | Mean EPE | 1px acc | 3px acc |
  |---|---|---|---|
  | NeuFlow v2 (reference) | 2.324 | 77.6% | 89.8% |
  | Best local (mix_chairs_vkitti2, batch 4, 15K) | 2.183 | 76.4% | 89.6% |
  | grandmix (batch 16, 100K, chairs+vkitti2+sintel) | 2.166 | 76.25% | 89.48% |
  | **big18 (batch 16, 100K, mix_chairs_vkitti2, iters 1,8)** | **2.072** | **77.02%** | **89.91%** |

  **big18 is the best result to date** — 11% better EPE than v2, 3px accuracy
  now at parity (89.91 vs 89.8), 1px gap narrowed to 0.6 points (77.02 vs
  77.6). Beats the local run on every metric simultaneously; scale (batch 16,
  100K steps) was a real lever, not noise.
  grandmix (the fair-comparison run with Sintel added) scores close to big18
  on EPE but slightly below on 1px/3px — the extra data breadth didn't help
  here, or 100K steps wasn't enough to fully exploit the larger mix. Both
  results now supersede all prior local claims; deck/report update pending.
  Dense-only runtime measured (5.8 FPS) — this is NOT the sparse-query mode;
  full v3 sparse-vs-v2 speed comparison still needs to be re-measured on these
  checkpoints (decode_dense_fast path untested for the (1,8)-trained weights).

- **uncG eval failed**: cluster ran the old eval_vkitti2.py (no `--uncertainty`
  flag) because the chained one-liner never re-pulled between checkpoints.
  User error induced by my missing `git pull` step in that command. Fix:
  pull, then re-run.

- **Fixed real bug in previous patch**: evaluate()'s function signature never
  got the `uncertainty=False` parameter (the string-replace for it silently
  no-op'd while three sibling edits succeeded) — caused
  `TypeError: unexpected keyword argument 'uncertainty'` on the cluster.
  Also flipped evaluate()'s own internal default head='regress'->'convex'
  for consistency with the CLI default fixed earlier. Verified this time
  with grep + py_compile before pushing, not just py_compile alone.

- **uncG full-set result: 2.082 EPE, 77.51% 1px, 90.02% 3px.** Same recipe as
  big18 (mix_chairs_vkitti2, batch 16, 100K, iters 1/8) plus only the
  uncertainty head+loss. First v3 checkpoint to beat v2 on 3px accuracy
  (90.02 vs 89.8) and nearly close the 1px gap (77.51 vs 77.6, 0.09 pts).
  HYPOTHESIS, not proven: the auxiliary uncertainty loss may be acting as a
  regularizer on the main flow output, not just adding a confidence signal.
  Single run — could be seed noise. Needs a repeat/ablation to confirm.

  Updated full ranking (full-set, VKITTI2 Scene18+20):
  | Config | EPE | 1px | 3px |
  |---|---|---|---|
  | v2 reference | 2.324 | 77.6% | 89.8% |
  | uncG | 2.082 | 77.51% | 90.02% |
  | big18 | 2.072 | 77.02% | 89.91% |
  | grandmix | 2.166 | 76.25% | 89.48% |

  STILL PENDING (not run, do not assume):
  1. Sparse-query speed re-benchmark on big18/uncG/grandmix checkpoints —
     all accuracy numbers above are DENSE mode (5.8 FPS); the deployment
     speed claim needs its own measurement on these new weights.
  2. Uncertainty calibration check — does uncG's predicted b actually
     correlate with real per-point error? Not yet tested; the flow-accuracy
     numbers above stand independent of this.

- **Added scripts/eval_calibration.py** for the pending uncertainty-calibration
  check: samples valid points per image, compares predicted b against real
  |pred-GT| error, reports Pearson correlation + 5-bin monotonicity table.
  Caught and fixed a coordinate-offset bug before it ran anywhere: InputPadder
  defaults to symmetric padding (mode='sintel'), so raw GT pixel coords needed
  a +pad_left/+pad_top shift to index the padded image the model actually
  sees — verified with a CPU smoke test (odd image size forcing real
  asymmetric padding [3,4,7,7]; identical-frame flow ~0, b=1.0 at zero-init).

- **uncG speed check crashed on a second missed code path.** forward_dense_fast
  (used only by --fast_dense) was never updated for the extra uncertainty
  channel — only the regular decode() path was patched earlier. Fixed with
  the same k2p1 slicing + last_b population. Verified two ways before
  pushing: (1) exact mode (fusion_on_grid=False) matches the trusted regular
  path exactly, 0.000000 diff on both flow and b; (2) the default fast mode
  (fusion_on_grid=True) shows the already-documented ~0.02px-EPE-class
  approximation, which is expected and unrelated to this fix — it just shows
  up more visibly on the unbounded uncertainty channel than on the
  softmax-normalized flow.

- **Speed results confirmed (grandmix, big18, fast_dense stride=2, V100):**
  both ~45.6-45.8 FPS dense (vs v2's 5.8 FPS in the same harness — ~7.9x),
  EPE within 0.02 of the slow-path numbers as expected from the documented
  approximation. Sparse-query mode (the actual deployment case) still not
  re-measured on these checkpoints — only dense-fast has been checked so far.

- **Calibration VERIFIED (uncG): Pearson r=0.38, clean monotonic bins**
  (0.22 -> 0.33 -> 0.60 -> 1.41 -> 7.38 px real error as predicted b rises
  across 5 bins, 2.35M sample points). The uncertainty head is not noise —
  it carries real, usable signal about its own error. This upgrades the
  earlier "hypothesis" framing: the confidence output itself is a validated
  contribution, independent of whether it also improves the main flow output
  (that regularization question is still unproven, single run).

- **Added scripts/benchmark_sparse.py** — the missing deployment-speed
  measurement (coarse pass once + sparse decode at N queries, vs v2's per-
  frame dense cost). CPU-smoke-tested on both the plain convex path and the
  --uncertainty path before pushing (given two prior bugs hid exactly there).

- **uncG forward_dense_fast fix CONFIRMED on GPU**: 2.104 EPE, 45.7 FPS
  (job 8732663), consistent with the stride-2 approximation class and
  matching grandmix/big18's speed. Bug fully closed.

- **Sparse-speed comparison, same V100, fully verified:**
  v2 dense: 19.6 ms (51.1 FPS), EPE 2.324 (matches known reference exactly —
  hardware-independence sanity check passed).
  v3 (grandmix/big18/uncG): coarse pass 16.2-16.5 ms + decode 2.6-2.7 ms
  (flat across N=800 and N=2048) = 18.9-19.2 ms total on a fresh frame.

  HONEST HEADLINE: v3's first query on a new frame is already at parity with
  v2 (marginally faster, ~19.1 vs 19.6 ms). Every ADDITIONAL query batch on
  the SAME frame costs only ~2.6 ms — v2 has no equivalent, since it must
  redo its full dense pass every time (no cached state). That is a ~7.3x
  speedup per repeat query, and it is the real deployment argument: not "6%
  better EPE" but "v2 pays full price every call, v3 pays once per frame."
  This directly answers the lab's original objection.

- **Added scripts/export_panels_hpc.py**: cluster-runnable version of
  export_panels.py, CLI-driven (--checkpoint, --tag, --uncertainty,
  --dataset_root) so it works against /scratch/$USER paths for the new
  HPC checkpoints (big18, uncG, grandmix). Needed since the laptop GPU is
  currently down (driver error) and all winning checkpoints live only on
  the cluster.

- **Deck updated (24 slides) with verified HPC results:**
  - Replaced the stale pre-bugfix "V3 Acceleration" slide (referenced a
    deleted laptop-only chart, quoted old (2,4)-schedule numbers) with the
    verified same-V100 sparse-speed story: parity on first query, ~7x on
    repeat queries.
  - New slide: native 4-panel image grid comparing v2 / grandmix / big18 /
    uncG on the same VKITTI2 scene, generated on-cluster via
    export_panels_hpc.py (laptop GPU is down) and pulled via scp.
  - Objectives slide numbers updated: 11% EPE improvement (was 6%), 3px now
    beats v2, 1px gap nearly closed, ~7x repeat-query speedup (was an
    unverified 20x estimate). Removed the takeaway() line per earlier
    instruction to drop that pattern.

- **Spring result (truncated at step 70,000/100,000, killed by 8h wall as
  predicted): 2.080 EPE, 76.94% 1px, 89.88% 3px.** Trained on the hardest,
  broadest mix (chairs+vkitti2+sintel+spring, 55,294 pairs) and still landed
  within 0.01 EPE of big18/uncG despite only 70% of planned steps and never
  reaching the OneCycle LR's final anneal. Genuinely strong, not a fluke --
  the broader data did not hurt convergence speed. Full 100K run would need
  a resubmit with --resume from step_070000.pth and a fresh 8h clock;
  logged as a candidate follow-up, not run.

- **Complete deck rebuild (2026-07-26), 17 slides, scripts/build_final_deck.py**:
  full revamp per explicit request ("no half-baked explanation"). Linear
  narrative: title -> motivation -> method (v2 pipeline, v3 decoder+pseudocode)
  -> five results slides each with a dedicated matplotlib comparison plot
  (curriculum EPE across all 8 configs, precision bars, visual grid, speed
  bars, calibration bars, distillation bars) -> interface (API, GUI, live
  video proof) -> objectives -> limitations (new, explicit) -> next steps ->
  Q&A prep. Every number traced to docs/V3DEV_LOG.md, none invented.
  scripts/make_final_plots.py generates the five plots from logged numbers
  only (no new evals). Old scripts/build_status_deck.py (24 slides,
  accreted over many sessions, duplicate section labels) superseded but
  left in place for reference. Every slide individually rendered and
  visually verified before commit -- caught and fixed two layout bugs
  (picture-height math not matching actual PNG aspect ratios, causing four
  captions to overlap their charts; a text-wrap issue on the requests slide).

- **README.md rewritten from scratch.** The old version predated the convex
  head fix, mixed training, HPC results, and the uncertainty head entirely --
  it still quoted the original mislabeled per-frame "1px acc 46.2%" bug and
  a v3 EPE of 3.15 (worse than untrained). Replaced with current verified
  results table, architecture description matching the actual convex head,
  the two-pass API, HPC training/eval commands, and an updated file layout.
  Fixed three markdown-lint warnings (blank lines around lists, code-fence
  language) before committing.

- **Added scripts/merge_distill_decoder.py**, the tool for the pending
  end-to-end distillation eval. Splices distill3's refine_s8 weights into
  a checkpoint with a trained decoder (big18/uncG), with a hard safety
  check: if any weight OUTSIDE refine_s8/decoder differs between the two
  source checkpoints, it aborts rather than silently merging incompatible
  models. Smoke-tested locally with synthetic checkpoints: (1) correct
  splice verified (right tensors from right source, shared parts preserved),
  (2) mismatch detector verified to correctly abort on a real conflict.
  NOT yet run on real checkpoints -- needs cluster GPU for the checkpoint
  load + subsequent eval; command given to user, not run automatically.

- **Fixed merge_distill_decoder.py's safety check — it was too strict.**
  Real cluster run (distill3 + big18) aborted on ~20+ backbone
  running_mean/running_var/num_batches_tracked differences. Diagnosis: these
  are BatchNorm BUFFERS, which update on every forward pass in train mode
  regardless of requires_grad -- they drift between any two training runs
  even with a genuinely frozen backbone (the actual conv weights never
  changed). This is expected, harmless PyTorch behavior, not a real
  incompatibility. Fixed: the check now skips .running_mean/.running_var/
  .num_batches_tracked suffixes, still strictly compares actual weights.
  Re-verified with synthetic tests: BN-style drift now correctly ignored,
  a real weight conflict still correctly aborts. Merged checkpoint inherits
  its BN stats from --decoder (consistent with the rest of that checkpoint).

- **Speaker notes added to all 17 deck slides** (build_final_deck.py note()
  helper + per-slide notes list). Presenter-view talking points, honest tone,
  no em-dashes. Verified 17/17 attached.

- **END-TO-END distillation result (the test the reviewer demanded): NEGATIVE.**
  Merged distill3 refine_s8 + big18 decoder, eval at iters_s8=3, full set:
  **2.398 EPE, 75.16% 1px, 88.60% 3px.** This is WORSE than v2 (2.324) and far
  worse than big18 at its normal 8 iters (2.072). The coarse-only "87.5% of the
  gap closed" number did NOT survive end-to-end: the decoder was trained on
  8-iteration coarse flow and degrades on the 3-iteration distilled input it
  never saw in training. This is exactly why coarse-only numbers are not
  deployable results -- the reviewer was right to insist. Distillation as a
  standalone speed lever is not worth 0.33 EPE (dropping below v2) unless the
  sparse-mode latency saving is critical AND the decoder is retrained at the
  reduced iteration count. Control still needed: big18's OWN refine at 3 iters
  (no distillation) to separate "distillation helped a bit" from "3 iters just
  hurts regardless". Command queued for user.

## 2026-07-26 -- CRITICAL: train/validation leak found and fixed

- **Every VKITTI2 training stage was training on the evaluation scenes.**
  `VKITTI2.__init__` defaulted to
  `scenes=['Scene01','Scene02','Scene06','Scene18','Scene20']`, and NO call
  site ever passed `scenes=` (verified: the token appears only in the class
  signature). `eval_vkitti2.py` defaults to `['Scene18','Scene20']`, `clone`
  variant -- which the training stages also included.
  **Proved by exact file-path intersection: 1,174 / 1,174 validation pairs
  (100%) were also training pairs.**

- **Invalidated results** (all trained on a VKITTI2-containing stage, all
  evaluated on Scene18/20): big18 2.072, uncG 2.082, spring-70K 2.080,
  grandmix 2.166, laptop-mixed 2.183, vkitti2_all 2.388, plus the distillation
  end-to-end evals (both sides equally contaminated, so the *relative*
  27%-vs-87.5% conclusion likely survives; absolute numbers do not).
  The "v3 beats v2 by 11%" headline was train-on-test vs honest-holdout.

- **Results that SURVIVE** (no VKITTI2 in training, or leak-independent):
  v2 2.324 (trained FlyingThings, never saw VKITTI2); v3 chairs-only 2.275;
  v3 chairs+PE 2.288; v3 untrained 2.476; the FlyingChairs-val comparison
  (proper split file: v2 2.238 / v3 2.399); ALL speed and latency numbers
  (timing is unaffected by data contamination); sparse==dense exactness.
  **The clean headline is chairs-only 2.275 vs v2 2.324 -- genuinely better,
  with zero driving imagery in training.**

- **Fix** (`data_utils/datasets.py`): module-level `VKITTI2_TRAIN_SCENES` /
  `VKITTI2_VAL_SCENES`; the loader default is now train-safe (Scene01/02/06);
  and an `allow_val_scenes=False` guard RAISES if any eval scene is requested,
  so this cannot silently recur. Verified post-fix: mix_chairs_vkitti2
  34,958->27,914 pairs, vkitti2_all 12,726->5,682, vkitti2 2,121->947, all
  with overlap=0; guard fires on a leaking request; explicit opt-in still works
  for eval tooling.

- **Written up in `docs/report/NeuFlow_v3_report.tex` -> .pdf** (5 pages):
  leads with the correction, separates valid from invalidated results in both
  the figure (clean bars vs hatched/red "INVALID" panel) and the tables, keeps
  the speed and uncertainty sections with their own caveats, and lists the
  repeat-the-runs work as step 1. Figures via `scripts/make_report_figs.py`.

## 2026-07-26 -- Full critical audit (docs/AUDIT_2026-07-26.md)

Line-by-line review of algorithm + training. Verified correct: conv
reformulation (proved algebraically, cross-correlation orientation and
replicate-padding both match), convex-head boundedness, zero-init==bilinear,
Laplace NLL, gradient freezing (0 weights changed), exact GT sampling under
--no_query_jitter, RAFT-style multi-scale weighting.

**Bug 2: the "frozen" backbone is not frozen.** my_freeze_model only sets
requires_grad=False; BN buffers update every forward in train mode.
Measured: running_mean drift 7.4%, running_var 17.4%, num_batches_tracked
+30,000 (exactly one per step). Restoring only the BN buffers and re-running
the coarse pass changes the coarse flow by **0.350 px mean / 5.86 px max --
7x the entire 0.049 px v3-vs-v2 gap.** So v3's front end != v2's front end;
part of v3's apparent advantage may be silent BN domain adaptation.

**I was wrong earlier**: I dismissed the merge script's BN abort as "harmless".
It was not -- distill3's refine_s8 was tuned under its own drifted BN stats and
was spliced onto big18's, so the 2.398 end-to-end number is pessimistic for a
reason I introduced.

**Bug 3: the decoder never sees the full-resolution image.** `img` is passed to
ImplicitFlowDecoder.forward and used ONLY for `.device`.

**Ceiling analysis (the key result).** Chairs checkpoint, Scene18 crop:
bilinear 1.540 / trained head 1.497 / oracle-best-of-its-own-10-candidates
0.881. The head captures ~6% of the headroom its own architecture allows.
My saturation hypothesis was REFUTED by measurement (learned logit std 3.69,
mean bilinear weight 0.083 -- it escaped the prior entirely and chooses
confidently, just badly). Real cause is an information bottleneck: within-8x8-
cell share of total variance is 23.3% for the head's evidence but 63.4% for the
oracle's correct choice, and 96.9% of cells need different candidates for
different pixels inside them. v2's upsampler is fed conv_s8(img0) -- a k=8,s=8
conv on the RAW full-res image -- and emits 8^2*9=576 values per cell, i.e. a
separate 9-way weighting per sub-pixel conditioned on real high-frequency
content. v3's queryable decoder is strictly LESS informed for the sub-pixel
decision than the fixed upsampler it replaced. This one fact explains the 1px
gap, the PE null result, and the distillation failure simultaneously.

**Ranked fixes**: (A) feed a full-res stem sampled at the query coordinate --
plumbing already exists, targets the exact failing metric; (B) fix the BN freeze
and re-run; (C) bounded residual on top of the convex blend to break the hull
ceiling without returning to unbounded regression; (D) 5x5 window; (E) boundary-
weighted loss; (F) remove the fusion train/eval mismatch.
**Unfreezing v2 is NOT the first lever** -- the bottleneck is information, not
capacity, and an 8x-downsampled map cannot carry 64 distinct sub-pixel decisions
per cell. A full-res stem supplies the same signal at the cost v2 already pays.

## 2026-07-26 -- Adaptive (region-selective) refinement: measured, it works

User's idea: instead of cropping, use a cheap signal after N iterations to find
where more refinement is needed, and only spend iterations there.

**Methodology error caught mid-experiment.** The first probe used a 256x512
corner crop and 2-3 pairs. On that crop iterations 5-8 changed EPE by ~0.01 px
and the SIGN flipped between samples, so the "% of gain" denominator went
negative and every percentage printed was garbage. Cause: the crop is sky and
trees, easy content where refinement does nothing. Rewritten to use full frames
via InputPadder over 24 pairs. Sanity check now passes: the probe measures the
iter4->8 gain as +0.2284 px, matching the independent full-set sweep
(4 iters 2.526 vs 8 iters 2.288 = 0.238 px). The probe was separately verified
to reproduce infer_coarse_state exactly (max diff 0.000e+00).

**Measured (24 full frames, Scene18+20, coarse flow + bilinear x8):**
- Concentration: top 5% of pixels carry 65.9% of the iter4->8 change,
  top 10% carry 77.2%, top 20% carry 86.8%.
- Predictor: |flow_4 - flow_3| ("not yet converged", free -- both already
  computed) r = 0.924 +/- 0.087. Flow magnitude, which was the user's original
  suggestion, is much weaker at r = 0.675; flow gradient r = 0.702.
- Budget (16x16 s8 tiles, selected by the not-converged signal):
  top 10% -> EPE 2.8618 (76.6% of the gain), top 20% -> 2.8112 (98.8%),
  vs 3.0368 stopping at iter 4 and 2.8084 with all 8.

**The blocker, and its fix.** refine_s8 is 8 stacked 3x3 convs -> receptive
field radius 8 s8-cells, so an exact 16x16 tile must compute 32x32 (4x
overhead), which eats the savings. Measured the actual decay by recomputing one
tile in isolation at varying halo:
  halo 8 -> 0.0000 px err (exact, 4.0x) | halo 6 -> 0.0065 px (3.1x)
  halo 4 -> 0.0305 px (2.2x) | halo 2 -> 0.0705 px (1.6x) | halo 0 -> 0.181 px
So halo 4-6 is effectively exact at roughly half the overhead of the strict
receptive field.

**Resulting estimate (T=16, halo 4):** refine top 20% of tiles -> ~16% total
speedup at ~99% of the refinement gain; top 10% -> ~23% speedup at 77% of the
gain (0.05 px cost). This meets the 15-20% target the user asked for.

**Caveats, explicitly:** (a) the speedups are ARITHMETIC from measured
components, not a wall-clock of an implemented tiled path -- gather/scatter and
occupancy will reduce them; (b) 24 frames on CPU, needs the full 1,174-frame GPU
run; (c) the probe evaluated coarse+bilinear, NOT the decoder, so end-to-end
verification is still required; (d) the halo decay is one tile in one image;
(e) **this optimization applies to NeuFlow v2 as well -- it is not a v3
feature.** The v3-specific variant is to intersect the not-converged mask with
the tiles that actually contain queries, which only pays off for clustered /
region-of-interest queries (800 uniformly spread queries touch essentially all
27 tiles at T=16).

## 2026-07-26 -- Pre-retraining audit of the training path

**Fixed: BatchNorm freeze (utils/load_model.set_frozen_bn_eval).** Wired into
train.py and scripts/train_distill.py. Verified: before, 15 BN counters moved
and running_mean drifted 2.99 in 3 steps; after, 0 counters move and drift is
exactly 0.

**Found: the four HPC runs were not comparable to each other.** They differed in
THREE variables at once, not one:
  spring       spring_mix           batch 12   iters (2,4)
  grandmix     grand_mix            batch 16   iters (2,4)
  big18        mix_chairs_vkitti2   batch 16   iters (1,8)
  uncertainty  mix_chairs_vkitti2   batch 16   iters (1,8)
All four were evaluated at (1,8), so grandmix and spring additionally had a
train/eval schedule mismatch. The grandmix-vs-big18 difference (2.166 vs 2.072)
that I had attributed to the dataset is at least as likely to be that mismatch.
No dataset conclusion from those runs is safe.

**Fix:** all sbatch files are now generated from hpc/_template.sbatch by
hpc/make_sbatch.py, so runs cannot silently drift apart. Every run is identical
(seed 1234, batch 16, lr 2e-4 OneCycle, gamma 0.8, iters 1xs16+8xs8 matching the
eval default, 4096 queries, no jitter, convex head) and varies in exactly one
thing. Old hand-written scripts moved to hpc/old/.

**Added --seed** (default 1234); the seeding block in train.py was commented out,
so previous runs were unseeded and their differences included seed noise.

**Renamed runs to the real dataset names** per user request, and added
STAGE_ALIASES so stages can be given as e.g. 'FlyingChairs+VKITTI2+Sintel':
  v3_FlyingChairs, v3_FlyingChairs_VKITTI2, v3_FlyingChairs_VKITTI2_Sintel,
  v3_FlyingChairs_VKITTI2_Sintel_Spring, v3_FlyingChairs_VKITTI2_Sintel_uncertainty

**Made the training path CPU-testable** (fp16 cast, init_bhwd amp flag and
autocast now keyed off device type; GPU behaviour unchanged). The whole loop can
now be smoke-tested without a GPU before spending cluster hours -- verified with
a 3-step run: seed printed, 27,914 leak-free pairs, 15 BN layers held in eval,
745,226 / 7,826,794 params trainable.

### Pre-flight verification suite (scripts/verify_pipeline.py)

Written in response to "are you sure of what you have done" -- 9 checks, CPU,
~2 min, run before spending cluster hours. All 9 pass. Notable:

- BatchNorm frozen: 15 layers held, drift exactly 0
- only decoder trainable: 30 tensors, 0 outside implicit_decoder_module
- zero-init decoder == bilinear: 0.0110 px max deviation
- sparse query == dense at same coords: 0.000473 px
- seed reproducibility: same seed -> identical loss (5.114 == 5.114);
  different seed -> different (16.322). Confirms --seed works.

Two checks failed on the first run, BOTH because the test was wrong, not the
code -- worth recording because the second one nearly produced a false alarm:

1. Uncertainty: the test ignored the documented API. decode_queries takes
   return_uncertainty=True and returns a (flow, b) TUPLE. While fixing this I
   found a real inconsistency: decode_dense_fast had no such flag at all and
   left b in module state (self.last_b), where a caller could read a stale
   value from a previous decode. Both paths now return (flow, b).
2. Stride-2 dense: tested on RANDOM NOISE, where flow is not spatially smooth,
   so the interpolation the stride trick relies on cannot work -- 0.075 px
   mean error. Re-tested on a real VKITTI2 pair: 0.0148 px. The earlier
   "stride-2 is nearly free" claim holds; the noise test was invalid.
   The suite now skips this check rather than run it on synthetic noise.

Leak fix independently re-verified today: FlyingChairs+VKITTI2 = 27,914
samples, 0 files touching Scene18/20, and the guard raises if val scenes are
requested without allow_val_scenes=True.

### GPU verification + video region GUI (2026-07-26, GPU restored)

Verification suite re-run on GPU with fp16: **9/9 pass** (same as CPU). One
device-placement bug in the test itself was fixed. GPU seed reproducibility
confirmed separately from CPU (5.598 == 5.598).

**scripts/video_region_gui.py** -- load a video, step frame by frame, drag a
box, get flow for that region. Two modes, both timed:

  QUERY: full coarse pass + decode inside the box only. Exact.
  CROP:  crop the input to box+margin, run the whole pipeline on it. Approximate
         (loses context outside the crop; motion leaving the crop is unfindable).

Measured on RTX 4060, 640x300 synthetic clip, 260x150 box (20% of frame),
stride-2 decode, steady state:

  QUERY  coarse 13.5 ms + decode 6.0 ms  = 19.5 ms   (repeat box: 6.0 ms)
  CROP   coarse  9.8 ms + decode 1.9 ms  = 11.8 ms
  v2     whole frame                      ~12-16 ms

Three measurement traps found and fixed while building this, each of which
would have produced a wrong number in a demo:

1. **Cold-start dominates.** First CUDA call at a new tensor shape measured
   367 ms against ~13 ms steady state -- 28x. Added FlowEngine.warmup(); the
   self-test now runs each mode twice and reports the second.
2. **Every new box is a new shape**, so crop mode pays autotuning per box
   (50-90 ms first call, ~10 ms after). Real interactive cost unless crops are
   padded to standard sizes -- worth noting as future work.
3. **The readout compared unlike things**: CROP's coarse+decode against the full
   frame's coarse ONLY, which made crop look 5x slower than it is. Timings are
   now split, and the like-for-like line compares coarse against coarse (1.37x
   for a 46%-area crop -- consistent with cost scaling by area).

Honest reading: for a FIRST query on a new frame, v3 is not faster than v2
(19.5 vs ~13 ms) -- the coarse pass dominates and v3's decode is more expensive
per pixel than v2's convex upsampler. The genuine wins are (a) repeat queries on
an already-processed frame, 6.0 ms against v2's full 13 ms recompute, and
(b) CROP mode, where cost scales with the area actually requested.
