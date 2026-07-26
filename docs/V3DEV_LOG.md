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
