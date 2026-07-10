# Professor Meeting Prep: NeuFlow v3

## 30-second pitch

I am working on NeuFlow v3, an optical-flow project where I replaced NeuFlow v2's convex upsampler with an implicit neural decoder inspired by InfiniDepth. Instead of only blending coarse flow values, the model can query a coordinate-conditioned MLP at any pixel and predict a residual flow correction using multi-scale features.

The implementation works and trains stably with a frozen backbone. On VKITTI2, it does not beat NeuFlow v2 yet: v2 gets 2.23 px mean EPE, while my best v3 checkpoint gets 3.15 px. The important result is that I have isolated the bottleneck: training only on about 2.1k VKITTI2 pairs causes quick overfitting. The next step is the standard optical-flow curriculum, FlyingChairs -> FlyingThings3D -> VKITTI2, which likely needs university compute access.

## What to show

1. `NeuFlow_v3/docs/proposal.pdf`
   - Use this as the main polished document.
   - It already includes the architecture, training setup, results table, checkpoint sweep, and compute request.

2. `NeuFlow_v3/README.md`
   - Good quick overview for the professor if they want to scan the repo.
   - Shows the core idea, result table, visualizations, and next steps.

3. Visual results in `NeuFlow_v3/results/readme_vis/`
   - `vkitti_scene18_0.png`
   - `vkitti_scene18_1.png`
   - `vkitti_scene18_2.png`
   - `outdoor_0.png`
   - `outdoor_1.png`

4. Best checkpoint
   - `NeuFlow_v3/checkpoints/neuflowv3/step_010000.pth`
   - This is the best validation checkpoint from the sweep.

## Key numbers

| Model | Mean EPE | Median EPE | 1px Acc | 3px Acc |
| --- | ---: | ---: | ---: | ---: |
| NeuFlow v2 | 2.23 px | 1.08 px | 46.2% | 78.4% |
| NeuFlow v3, step 10k | 3.15 px | 1.87 px | 7.2% | 71.1% |

Checkpoint sweep:

| Step | Mean EPE |
| ---: | ---: |
| 5k | 3.29 px |
| 10k | 3.15 px |
| 15k | 3.47 px |
| 20k | 3.16 px |
| 25k | 3.56 px |
| 30k | 3.69 px |

## Honest framing

Do not pitch this as "I beat the baseline." Pitch it as:

> I implemented the architectural change, got it training stably, reproduced a meaningful evaluation loop, and found that data/compute is now the blocker rather than code. I want guidance on whether this direction is research-worthy and whether I can get access to compute for proper pretraining.

That framing is stronger because it is technically honest and shows research maturity.

## What to ask the professor

1. Is adapting implicit neural representations to optical flow a direction worth pushing further?
2. Can I get access to university GPU/HPC resources to train on FlyingChairs and FlyingThings3D?
3. Should I target a course project, thesis project, workshop paper, or research assistant direction?
4. Are there better baselines or datasets the professor would expect beyond VKITTI2?
5. Would they recommend changing the architecture, or first scaling training data?

## Suggested meeting flow

1. Start with the motivation:
   - NeuFlow v2 is fast, but convex upsampling limits sharp motion boundaries.
   - InfiniDepth showed implicit decoders can improve fine detail for depth.
   - I tried transferring that idea to optical flow.

2. Show what you changed:
   - Frozen NeuFlow v2 backbone.
   - Replaced upsampler with coordinate-conditioned MLP.
   - Multi-scale features plus warped frame-1 features.
   - Zero-initialized output layer so training starts from v2 behavior.

3. Show results:
   - Best v3 checkpoint: 3.15 px EPE on VKITTI2 Scenes 18 and 20.
   - v2 baseline: 2.23 px.
   - Overfitting starts after 10k steps.

4. Explain the bottleneck:
   - VKITTI2-only training is too small.
   - Standard optical-flow models train on FlyingChairs and FlyingThings3D first.
   - End-to-end training diverged on one GPU, so compute is the limiting factor.

5. Make the ask:
   - Feedback on research direction.
   - Compute access or advice.
   - Whether to continue toward a thesis/publication-style project.

## One-minute spoken version

"I have been working on NeuFlow v3, where I replace NeuFlow v2's convex upsampler with an implicit neural decoder inspired by InfiniDepth. The idea is that instead of only blending a local grid of coarse flow vectors, the model queries an MLP at continuous pixel coordinates and predicts a residual flow correction using multi-scale features.

I froze the NeuFlow v2 backbone and trained only the decoder, because end-to-end training diverged at my compute scale. The model trains stably and produces reasonable flow visualizations. Quantitatively, it is not beating v2 yet: on VKITTI2 Scenes 18 and 20, v2 gets 2.23 px EPE and my best v3 checkpoint gets 3.15 px. The checkpoint sweep shows it overfits after about 10k steps, which makes sense because I only trained on around 2.1k VKITTI2 pairs.

My read is that the architecture is implemented and the next blocker is proper data and compute. I would like your feedback on whether this direction is worth pursuing, and whether there is a way to access university GPU resources so I can train on FlyingChairs and FlyingThings3D before fine-tuning on VKITTI2."

## Tiny slide outline

Slide 1: Problem
- NeuFlow v2 is fast, but convex upsampling can blur motion boundaries.
- Goal: test implicit coordinate decoding for optical flow.

Slide 2: Method
- Frozen NeuFlow v2 backbone.
- New implicit decoder with multi-scale feature sampling.
- Warped frame-1 features for correspondence.

Slide 3: Results
- v2: 2.23 px EPE.
- v3: 3.15 px EPE at 10k.
- Overfits with VKITTI2-only training.

Slide 4: Visuals
- Show `vkitti_scene18_0.png`, `vkitti_scene18_1.png`, and one outdoor example.

Slide 5: Ask
- Feedback on research direction.
- GPU/HPC access.
- Advice on datasets, baselines, and publication potential.
