# NeuFlow v3 — Handoff Prompt

You are picking up an MS-thesis research project (Shriman Raghav Srinivasan,
Northeastern, advisor Hanumant Singh, deadline Aug 2026). Read this whole file
before touching anything. It is written so a model with no prior context can
continue without re-deriving or, worse, re-breaking what exists.

## Non-negotiable working rules (user-imposed, repeatedly reinforced)

1. **No false positives.** A number is a result only if measured on the FULL
   validation set (1,174 pairs, VKITTI2 Scene18+20 clone, per-pixel metrics).
   Subset numbers, expectations, and "should improve" are labeled PENDING.
2. **Negative results are reported as prominently as wins.**
3. Every claim carries its caveat inline (see Confounds below).
4. Communication: terse, plain language, no em dashes, no hype. The user asks
   "explain simply" often; be ready to explain any design from first principles.
5. Never wait synchronously on long evals/training; run in background, report
   when done (user preference; tqdm always on).
6. Git: no Claude co-author trailers. Push to github.com/shrirag10/Neuflowv3.
7. Ask before sudo or deleting user files. The deck must not look AI-written
   (serif, monochrome, no takeaway lines, no meta-labels).

## What the project is

NeuFlow v2 (arXiv:2408.10161) = real-time dense optical flow for edge devices,
frozen upstream. v3 replaces only its final upsampler with a **queryable
implicit decoder**: flow at arbitrary continuous (x, y), cost O(N) in queries.
Two-pass API: `infer_coarse_state()` once (~23-33 ms), `decode_queries()`
(~1.6 ms per ≤2k points). Thesis pitch: a new accuracy/compute/resolution
operating point (registration/mapping consumers), NOT a leaderboard delta.

## State of the code

- Repo: `/home/shrirag10/Projects/NEUFLOW/NeuFlow_v3` (laptop),
  `~/NeuFlow_v3` on Explorer HPC. Remote `neuflowv3`.
- Branch **v3-rebuild** (current): unified decoder after the first-principles
  rewrite. `NeuFlow/implicit_decoder.py` has ONE compute path: window
  projections applied as 3x3 convs on the 1/8 grid (mathematically exact
  equivalent of per-query window sampling; regression-verified identical to 4
  decimals), gated fusion computed once on the grid (approximation, +0.02 px,
  measured), convex-weight head over 3x3 coarse-flow candidates + bilinear
  candidate with zero-init = exact bilinear start (full-set 2.476).
- Branch v2-dev: pre-rebuild history (three head experiments, PE, GUI, deck).
- Key entry points: `train.py` (flags matter: `--head convex` is implicit in
  rebuild; `--train_iters_s16/s8`, `--no_query_jitter`, `--onecycle`,
  `--gamma 0.8`), `scripts/eval_vkitti2.py` (`--fast_dense --stride 2
  --iters_s16 2 --iters_s8 4`), `scripts/query_gui.py` (PyQt5 tool: click/grid/
  adaptive/region/video/YouTube/motion/resources tabs), `scripts/
  benchmark_fps.py`, `scripts/build_status_deck.py` (deck generator; deck =
  `docs/NeuFlow_v3_status.pptx`, 22 slides).

## Verified results (full set, RTX 4060 laptop, fp16, 384x1248)

| Config | EPE | 1px | Latency |
|---|---|---|---|
| NeuFlow v2 | 2.324 | 77.6% | 37 ms |
| v3 untrained (bilinear init) | 2.476 | 74.7% | — |
| v3 rebuild trained@(2,4), eval (2,4), stride2 | 2.234 | ~75.7% | 28 ms |
| v3 rebuild trained@(2,4), eval (1,8), stride2 | **2.095** | 76.5% | 38 ms |

Sparse: 800 queries in 23.9 ms end-to-end at (2,4) = 41.8 FPS. Sparse values
match dense exactly (verified 0.00 px). Video 640x360: v3 sparse 63.6 FPS,
v2 60.3, v3+motion boxes 47.1 (single YouTube video, one run).

Negative results (documented, do not retry blindly): unbounded regression head
never trained below its init (fixed by convex head); Fourier PE null on chairs
(2.288 vs 2.275, 1px identical); sequential finetune = catastrophic forgetting
(2.28 -> 2.50); v3 loses in-domain on chairs val (2.40 vs v2 2.24).

## Confounds and open caveats (state them with every relevant claim)

- **Training-data confound**: v3 training mixes include VKITTI2-domain data;
  v2's training did not. Accuracy wins vs v2 are confounded until the
  grand-mix (chairs+vkitti2+sintel) run is evaluated. Fair-claims today:
  chairs-only transfer (2.275, no driving data) and speed-at-equal-EPE.
- 1px accuracy still trails v2 (76.5 vs 77.6). PE ruled OUT positional signal
  as the cause; remaining hypotheses: 1/8 coarse resolution bound, and
  training-motion statistics.
- All latency numbers are RTX 4060; **Jetson validation pending** — do not say
  "edge-proven".
- Audit + claims ledger: `docs/v3_rebuild_audit.md`. Parameter provenance +
  result log: `docs/base_parameters.md`. Prose report: `docs/NeuFlow_v3_Report.md`.

## HPC (Northeastern Explorer) — operational

- Access: user-interactive only (SSO+Duo). From laptop: `ssh explorer`
  (config + enrolled key). OOD web terminal mangles multi-line pastes; give
  the user ONE command per paste.
- Env: `~/.conda/envs/neuflow` (py3.10, torch cu121 wheels). **Always invoke
  by absolute path** `$HOME/.conda/envs/neuflow/bin/python3` with
  `PYTHONNOUSERSITE=1`; batch-shell activation is unreliable and a stray
  ancient torchvision lives in `~/.local` (python3.13). Module (interactive
  only): `miniconda3/25.9.1`. Login nodes kill heavy processes: run
  downloads/extractions as sbatch on partition `short`.
- GPUs: partition `gpu`, 8 h limit, `--gres=gpu:h200:1` (also a100,
  v100-sxm2/pcie). Datasets on `/scratch/srinivasan.shrim/neuflow_datasets`
  (chairs, vkitti2 all 5 scenes, Sintel, Spring train left FW 61 GB —
  layouts verified). **Scratch purges monthly; copy good checkpoints off.**
- Jobs submitted 2026-07-19 (batch 16, 100K steps, trained at (2,4)):
  grand-mix, rebuild-big, spring (IDs in squeue; sbatch files in `hpc/`).
  As of handoff: resubmission after env rebuild was in flight — VERIFY with
  `ssh explorer squeue -u srinivasan.shrim` and check `~/NeuFlow_v3/nf3-*.log`
  for 'Number of training images' (grand_mix ~45k, rebuild ~35k, spring ~50k)
  and it/s before believing anything trains.

## What each pending run answers

1. **grand-mix**: removes the domain confound; the fair v2 comparison.
2. **rebuild-big**: does batch 16 + 100K alone beat laptop 2.095?
3. **spring**: 1080p training; prerequisite for the Spring 4K-GT evaluation,
   the only test where v3 can natively answer at above-input resolution and
   v2 structurally cannot (the thesis's unique-capability demo).
4. Then: `hpc/train_unfrozen.sbatch` (backbone unfrozen, RISKY — diverged at
   batch 4 locally, unverified at 16); Jetson benchmark; advisor's field
   data (lab share `/projects/fieldroboticslab` — user has access).

## Next-action priority

1. Verify the three HPC jobs train; babysit failures (history: three
   env-related crash rounds — absolute interpreter fixed it).
2. Full-set eval each checkpoint ON the cluster; copy winners off scratch.
3. Update deck slide 19 (HPC pending) with real numbers; keep the confound
   language until grand-mix reads out.
4. Spring 4K evaluation script (eval at target_h/w = 2160x3840 vs 4K GT,
   compare v2+bilinear-upscale baseline).
5. Thesis writing support: the audit doc's claims ledger is the skeleton of
   the results chapter.
