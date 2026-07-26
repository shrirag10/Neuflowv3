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
