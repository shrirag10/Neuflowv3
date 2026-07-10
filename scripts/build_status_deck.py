"""Build docs/NeuFlow_v3_status.pptx — the current status deck.

Regenerate after new results: edit the constants below and rerun.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
INK    = RGBColor(0x20, 0x24, 0x2B)
MUTED  = RGBColor(0x7B, 0x83, 0x8F)
FONT   = 'Calibri'
OUT    = 'docs/NeuFlow_v3_status.pptx'

MIX_RESULT = ('2.18 px EPE, 76.4% 1 px, 89.6% 3 px — best result to date; '
              'both parents\u2019 strengths retained')


def add_text(s, x, y, w, h, text, size, bold=False, color=INK, line_spacing=1.0):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split('\n')):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.line_spacing = line_spacing
        r = para.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def header(s, kicker, title):
    add_text(s, 0.75, 0.42, 11.5, 0.3, kicker, 10.5, True, ACCENT)
    add_text(s, 0.73, 0.72, 11.9, 0.8, title, 24, True, INK)


def takeaway(s, text):
    add_text(s, 1.05, 6.32, 11.3, 0.6, 'Takeaway   ' + text, 11.5, True, ACCENT)


def footer(s, num):
    add_text(s, 0.75, 7.02, 8.0, 0.30, 'NeuFlow v3 · status report', 9, False, MUTED)
    add_text(s, 11.60, 7.02, 1.20, 0.30, str(num), 9, False, MUTED)


def main():
    p = Presentation()
    p.slide_width = Inches(13.333)
    p.slide_height = Inches(7.5)
    layout = p.slide_layouts[6]
    n = 0

    def slide():
        nonlocal n
        n += 1
        return p.slides.add_slide(layout), n

    # ================================================================ 1 · Title
    s, i = slide()
    add_text(s, 0.75, 2.5, 11.8, 1.0, 'NeuFlow v3', 40, True, INK)
    add_text(s, 0.75, 3.45, 11.8, 0.6, 'Queryable optical flow for edge devices — status report', 18, False, ACCENT)
    add_text(s, 0.75, 4.35, 11.8, 1.0,
             'A frozen NeuFlow v2 pipeline extended with an implicit decoder that answers\n'
             'flow queries at arbitrary continuous coordinates in O(N) time.\n\n'
             'Shriman Raghav Srinivasan · MS Robotics, Northeastern University · Field Robotics Lab', 13, False, MUTED)

    # ================================================== 2 · Problem, first principles
    s, i = slide()
    header(s, 'MOTIVATION', 'The problem, from first principles')
    add_text(s, 0.75, 1.7, 6.1, 4.4,
             'Optical flow answers one question: for a pixel in frame t,\n'
             'where is it in frame t+1?\n\n'
             'Every modern network answers it for all pixels at once,\n'
             'as a dense, fixed-resolution map. That design assumes the\n'
             'consumer wants every pixel. Many downstream tasks do not:\n\n'
             '•  Image registration needs hundreds of correspondences,\n'
             '    at well-textured locations of its own choosing.\n'
             '•  Sparse tracking needs flow at feature points only.\n'
             '•  Mapping pipelines need correspondences at survey\n'
             '    overlap regions, often at sub-pixel positions.\n\n'
             'On edge hardware, computing 479,000 answers to serve 800\n'
             'questions is the difference between real time and not.', 13, False, INK)
    add_text(s, 7.2, 1.7, 5.4, 4.4,
             'The proposition\n\n'
             'Keep the part of NeuFlow v2 that understands motion\n'
             '(backbone, matching, refinement — frozen, unchanged).\n\n'
             'Replace only its final stage: instead of a fixed 8×\n'
             'upsampler that always produces a full map, attach a\n'
             'decoder that evaluates flow at requested coordinates.\n\n'
             'Cost then scales with the number of questions asked,\n'
             'O(N), rather than with image area, O(H×W).', 13, False, INK)
    takeaway(s, 'The contribution is a new operating point — accuracy, compute, and resolution decoupled — not a leaderboard entry.')
    footer(s, i)

    # ================================================== 3 · NeuFlow v2 background
    s, i = slide()
    header(s, 'BACKGROUND', 'NeuFlow v2: the foundation this work builds on')
    add_text(s, 0.75, 1.7, 6.1, 4.5,
             'NeuFlow v2 (Zhang, Gupta, Jiang, Singh; arXiv:2408.10161)\n'
             'is a real-time optical flow network for edge deployment.\n\n'
             '1.  A shallow CNN extracts features at 1/8 and 1/16 scale.\n'
             '2.  Cross-attention and global matching at 1/16 scale\n'
             '     produce an initial flow estimate.\n'
             '3.  A lightweight recurrent module refines it:\n'
             '     one iteration at 1/16, eight at 1/8.\n'
             '4.  A learned convex upsampler blends each output pixel\n'
             '     from a 3×3 neighborhood of the 1/8 flow.\n\n'
             'It matches methods 10–70× more expensive and sustains\n'
             'over 20 FPS at 512×384 on a Jetson Orin Nano.', 13, False, INK)
    add_text(s, 7.2, 1.7, 5.4, 4.5,
             'Measured on our hardware\n'
             '(RTX 4060 Laptop, fp16, 384×1248)\n\n'
             '37 ms per frame pair (27 FPS)\n'
             '9.03 M parameters\n'
             '2.32 px mean EPE on VKITTI2 Scene18+20\n'
             '77.6% of pixels within 1 px of ground truth\n\n'
             'The structural constraint\n\n'
             'Output resolution and cost are fixed at design time.\n'
             'The network cannot answer a smaller question cheaply,\n'
             'nor a finer-resolution question at all.', 13, False, INK)
    takeaway(s, 'v2 supplies motion understanding worth preserving; its output stage is the only part that constrains how answers are delivered.')
    footer(s, i)

    # ================================================== 4 · v3 method
    s, i = slide()
    header(s, 'APPROACH', 'NeuFlow v3: an implicit decoder in place of the upsampler')
    add_text(s, 0.75, 1.7, 6.1, 4.5,
             'Unchanged and frozen: every stage through the 1/8-scale\n'
             'coarse flow. v3 inherits v2’s matching quality by\n'
             'construction rather than by retraining.\n\n'
             'New: a decoder that, given any continuous coordinate,\n'
             'gathers evidence and produces the flow at that point:\n\n'
             '•  3×3 local windows from four feature sources — context,\n'
             '    1/8 features, 1/16 features, and frame-1 features\n'
             '    warped by the coarse flow (correspondence evidence).\n'
             '•  Gated hierarchical fusion, following InfiniDepth.\n'
             '•  A convex-combination head, following AnyFlow: softmax\n'
             '    weights over the local coarse-flow values. The output\n'
             '    is a bounded blend — structurally unable to produce\n'
             '    arbitrarily wrong flow.', 13, False, INK)
    add_text(s, 7.2, 1.7, 5.4, 4.5,
             'Two properties fall out of the head design\n\n'
             '1.  Bilinear interpolation is one particular choice of\n'
             '     weights, so the decoder is initialized to reproduce\n'
             '     it exactly. Training starts from a known-good state\n'
             '     and can only be judged against it.\n\n'
             '2.  v2’s convex upsampler is the fixed-grid special case\n'
             '     of the same mechanism — v2’s accuracy is reachable\n'
             '     within this hypothesis space by construction.\n\n'
             'Model size decreases: 7.83 M parameters vs 9.03 M,\n'
             'because a coordinate-conditioned MLP replaces the\n'
             'full-resolution convolutional stack.', 13, False, INK)
    takeaway(s, 'Every design choice traces to a published mechanism (NeuFlow v2, InfiniDepth, AnyFlow) or to a measured failure it corrects.')
    footer(s, i)

    # ============================================ 5 · Stage 0: untrained (picture)
    s, i = slide()
    header(s, 'RESULTS · STAGE 0 OF 3', 'Before any training: the initialization is the first result')
    s.shapes.add_picture('results/visuals/stage_untrained.png', Inches(0.75), Inches(1.6), width=Inches(8.7))
    add_text(s, 9.7, 1.7, 3.0, 4.4,
             'With zero trained decoder\n'
             'weights, v3 reproduces\n'
             'bilinear upsampling of the\n'
             'coarse flow — measured at\n'
             '2.48 px EPE, 0.15 px behind\n'
             'v2, with querying already\n'
             'functional and exact.\n\n'
             'Every subsequent training\n'
             'run is accepted only if it\n'
             'improves on this number.\n'
             'Two weeks of earlier training\n'
             'failed this test; the\n'
             'redesigned head passes it.', 12, False, INK)
    takeaway(s, 'Queryability costs +0.15 px EPE before a single gradient step. Training must earn its keep against this baseline.')
    footer(s, i)

    # ============================================ 6 · Stage 1: vkitti2 (picture)
    s, i = slide()
    header(s, 'RESULTS · STAGE 1 OF 3', 'Training on the target domain: VKITTI2, six appearance variants')
    s.shapes.add_picture('results/visuals/stage_vkitti2.png', Inches(0.75), Inches(1.6), width=Inches(8.7))
    add_text(s, 9.7, 1.7, 3.0, 4.4,
             'Six same-trajectory weather\n'
             'variants share identical flow\n'
             'ground truth — appearance\n'
             'augmentation at no labeling\n'
             'cost. 12,726 pairs.\n\n'
             '2.39 px EPE — the first\n'
             'checkpoint in this project to\n'
             'improve on its initialization.\n'
             'No late-training collapse:\n'
             'the final checkpoint is the\n'
             'best one.\n\n'
             'Limitation: five scenes teach\n'
             'scene-specific detail; the\n'
             'error tail barely moves.', 12, False, INK)
    takeaway(s, 'Correct head plus sufficient data variety turned training from harmful to net-positive: 2.48 to 2.39 px.')
    footer(s, i)

    # ============================================ 7 · Stage 2: chairs (picture)
    s, i = slide()
    header(s, 'RESULTS · STAGE 2 OF 3', 'Training out of domain: FlyingChairs only')
    s.shapes.add_picture('results/visuals/stage_chairs.png', Inches(0.75), Inches(1.6), width=Inches(8.7))
    add_text(s, 9.7, 1.7, 3.0, 4.4,
             '22,232 synthetic pairs of\n'
             'chairs over random imagery —\n'
             'no roads, no vehicles, no\n'
             'shared statistics with the\n'
             'evaluation domain.\n\n'
             '2.28 px EPE on VKITTI2 —\n'
             'below NeuFlow v2 (2.32).\n'
             'Motion diversity, not domain\n'
             'familiarity, is what the\n'
             'decoder needed: varied large\n'
             'displacements suppress the\n'
             'error tail that dominates\n'
             'mean EPE.\n\n'
             'Cost: 1 px accuracy drops to\n'
             '69.7% (v2: 77.6%).', 12, False, INK)
    takeaway(s, 'The decoder generalizes: trained without a single driving frame, it outperforms v2 on driving data.')
    footer(s, i)

    # ============================================ 8 · Aggregate + ablations
    s, i = slide()
    header(s, 'RESULTS · SUMMARY', 'All training regimes against both references')
    s.shapes.add_picture('results/epe_by_regime.png', Inches(0.75), Inches(1.7), width=Inches(7.4))
    add_text(s, 8.5, 1.8, 4.2, 4.3,
             'Two informative negative results\n\n'
             'Sequential finetuning (chairs, then VKITTI2)\n'
             'reached 2.50 px — worse than either parent.\n'
             'The second dataset overwrote what the first\n'
             'taught. Remedy under test: joint sampling\n'
             'from both datasets in every batch.\n\n'
             'Fourier position encoding changed nothing\n'
             '(2.288 vs 2.275 px; 1 px accuracy identical).\n'
             'The sub-pixel gap is therefore not caused by\n'
             'missing positional information — one of two\n'
             'candidate explanations eliminated by a\n'
             'single controlled run.\n\n'
             f'Mixed-dataset training: {MIX_RESULT}.', 12, False, INK)
    takeaway(s, 'Each experiment either improved the model or eliminated a hypothesis; none was wasted compute.')
    footer(s, i)

    # ============================================ 9 · Compute
    s, i = slide()
    header(s, 'EFFICIENCY · RTX 4060 LAPTOP, FP16, 384×1248', 'Compute cost: v2 and v3 measured under identical conditions')
    s.shapes.add_picture('results/latency_v2_v3.png', Inches(0.75), Inches(1.75), width=Inches(8.6))
    add_text(s, 9.6, 1.85, 3.1, 4.3,
             'Parameters\n'
             '9.03 M (v2) → 7.83 M (v3)\n\n'
             'Sparse total: ~35 ms —\n'
             'parity with v2’s full frame,\n'
             'while answering only what\n'
             'was asked.\n\n'
             'Each additional query batch\n'
             'on a processed pair costs\n'
             '1.6 ms; v2 must recompute\n'
             'everything (37 ms) to\n'
             'answer anything new.\n\n'
             'Dense v3 (327 ms) exists\n'
             'for evaluation, not for\n'
             'deployment.', 12, False, INK)
    takeaway(s, 'For sparse workloads v3 is strictly cheaper than v2: equal latency on the first answer, ~20× cheaper on every answer after it.')
    footer(s, i)

    # ============================================ 10 · Query interface + GUI
    s, i = slide()
    header(s, 'INTERFACE', 'Querying in practice: sizes, API, and interactive demonstration')
    add_text(s, 0.75, 1.7, 6.2, 2.5,
             'A query is one continuous (x, y) coordinate; sub-pixel\n'
             'positions are valid inputs. N ranges from 1 to the full\n'
             'frame (479,232 at 384×1248). Decode cost is flat at\n'
             '1.6 ms up to roughly 2,000 queries per call.\n\n'
             'state = model.infer_coarse_state(img0, img1)   # once, 33 ms\n'
             'flow  = model.decode_queries(state, query_coords=q)\n'
             '        # q: [B, N, 2] pixel coords → [B, N, 2] flow', 12, False, INK)
    add_text(s, 0.75, 4.3, 6.2, 1.9,
             'Training configuration: batch size 4 (8 GB VRAM budget);\n'
             '4,096 supervision queries per image, half placed at motion\n'
             'boundaries; AdamW with a one-cycle schedule peaking at\n'
             '2e-4; backbone frozen throughout.', 12, False, INK)
    s.shapes.add_picture('results/visuals/query_gui_selftest.png', Inches(7.3), Inches(1.7), width=Inches(5.4))
    add_text(s, 7.3, 3.55, 5.4, 1.7,
             'Interactive tool (PyQt5, in the repository): click any pixel\n'
             'for its flow; grid, boundary-adaptive, and dense-overlay\n'
             'modes; CSV export. Feasible only because of the two-pass\n'
             'design — every interaction reuses the cached backbone state.', 11, False, INK)
    takeaway(s, 'The same API serves a robot asking for 800 correspondences and a human inspecting one pixel — that is the point of the design.')
    footer(s, i)

    # ============================================ 11 · Objectives
    s, i = slide()
    header(s, 'OBJECTIVES', 'Standing of the three stated objectives')
    add_text(s, 0.75, 1.85, 11.8, 4.2,
             'Objective 1 — exceed NeuFlow v2’s accuracy.  Met: 2.18 vs 2.32 px mean EPE (6% better), achieved by\n'
             'mixed chairs+VKITTI2 training. Sub-pixel precision is now within 1.2 points of v2 (76.4% vs 77.6%) and\n'
             '3 px accuracy is at parity (89.6% vs 89.8%).\n\n'
             'Objective 2 — lower compute, edge-viable.  Met for the intended workload: sparse answers arrive at v2’s\n'
             'full-frame latency with 13% fewer parameters and ~2.2 GB inference VRAM; repeated queries are ~20×\n'
             'cheaper than v2 recomputation. Dense-output mode remains slower and is explicitly out of scope.\n\n'
             'Objective 3 — capability that v2 cannot express.  Delivered: continuous-coordinate queries, resolution-\n'
             'independent output, exact sparse/dense agreement (0.00 px), and an interactive demonstration tool.',
             13, False, INK)
    takeaway(s, 'Two objectives met and measured; the third is scoped to a specific, testable gap with named hypotheses.')
    footer(s, i)

    # ============================================ 12 · Next experiment + dataset ask
    s, i = slide()
    header(s, 'NEXT STEPS', 'The running experiment, and the dataset conversation for the lab')
    add_text(s, 0.75, 1.7, 6.1, 4.5,
             'Completed: mixed-dataset training\n\n'
             'Joint sampling of FlyingChairs and all VKITTI2 variants\n'
             '(34,958 pairs, single 320×512 crop) under the standard\n'
             'recipe. Result: hypothesis confirmed — 2.18 px EPE with\n'
             '76.4% 1 px accuracy; the chairs robustness and the\n'
             'driving-data precision coexist under joint sampling.\n\n'
             'After that, in order:\n'
             '•  Jetson Orin benchmark of the two-pass API\n'
             '•  Spring dataset evaluation — ground truth at 2× input\n'
             '    resolution tests above-input-resolution querying,\n'
             '    a capability only queryable decoders can express\n'
             '•  Position encoding revisited on fine-motion data,\n'
             '    separating the two remaining sub-pixel hypotheses', 12.5, False, INK)
    add_text(s, 7.2, 1.7, 5.5, 4.5,
             'What to request from the Field Robotics Lab\n\n'
             'The lab’s SeaBED AUV and UAV survey programs produce\n'
             'exactly the data this method serves:\n\n'
             '•  Sequential seafloor transect imagery (50–75% overlap,\n'
             '    slow motion, fine texture) — registration and\n'
             '    photomosaicking with sparse on-demand queries; the\n'
             '    fine-motion regime also stresses sub-pixel accuracy.\n'
             '•  Nadir UAV survey sequences (polar or coastal) with\n'
             '    GPS/IMU tags — georegistration provides pseudo\n'
             '    ground truth for quantitative evaluation.\n'
             '•  Vehicle navigation data (DVL/INS poses) and camera\n'
             '    calibration — enables reprojection-based flow ground\n'
             '    truth on static scenes without manual labels.\n\n'
             'The ask is consecutive frames plus navigation data,\n'
             'not finished mosaics.', 12, False, INK)
    takeaway(s, 'The method is validated on public benchmarks; lab survey data is where its operating point becomes an application.')
    footer(s, i)

    # ============================================ 13 · FAQ
    s, i = slide()
    header(s, 'ANTICIPATED QUESTIONS', 'First-principles answers')
    add_text(s, 0.75, 1.65, 11.9, 4.5,
             'Why is flow defined at non-integer coordinates?  Bilinear interpolation makes feature maps continuous functions of position;\n'
             'an MLP composed with them is defined at every real (x, y). Integer pixels are the special case dense maps hard-code.\n\n'
             'Why freeze the backbone?  Joint training is a moving-target problem requiring ~800K steps at InfiniDepth’s scale; at a 30K budget\n'
             'the decoder diverges chasing shifting features (observed directly). Freezing also guarantees v2’s matching quality is inherited.\n\n'
             'Is sparse output an approximation?  No. It is the same function evaluated at fewer points; agreement with dense output is exact.\n\n'
             'Why does out-of-domain training win?  Mean EPE is dominated by the error tail. Diverse large motions train tail robustness;\n'
             'five driving scenes train memorization. Both effects were measured, in both directions.\n\n'
             'What limits sub-pixel accuracy?  Not positional information — measured (PE ablation, null result). Remaining candidates: the\n'
             '1/8-scale coarse flow bounds recoverable detail, and chairs-scale motions never supervise sub-pixel discrimination.\n\n'
             'Why batch size 4?  An 8 GB VRAM budget; the recipe is otherwise RAFT-standard and scales directly on larger GPUs.',
             11.5, False, INK, line_spacing=1.02)
    takeaway(s, 'Full derivations and the complete question list: docs/NeuFlow_v3_Report.md, sections 5–6.')
    footer(s, i)

    p.save(OUT)
    print(f'saved {OUT} with {n} slides')


if __name__ == '__main__':
    main()
