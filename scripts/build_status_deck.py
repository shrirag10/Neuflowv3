"""Build docs/NeuFlow_v3_status.pptx — current status deck (v2 vs v3, results, compute).

Rerun after new results land; edits go in the SLIDES data below.
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

TEMPLATE = 'docs/NeuFlow_v3_update.pptx'   # borrow slide size + blank layout
OUT      = 'docs/NeuFlow_v3_status.pptx'

PE_RESULT = 'training in progress'  # replaced when the PE eval lands


def add_text(s, x, y, w, h, text, size, bold=False, color=INK):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split('\n')):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = para.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def footer(s, num):
    add_text(s, 0.75, 7.02, 8.0, 0.30, 'NeuFlow v3  ·  status', 9, False, MUTED)
    add_text(s, 11.60, 7.02, 1.20, 0.30, str(num), 9, False, MUTED)


def main():
    p = Presentation()
    p.slide_width = Inches(13.333)
    p.slide_height = Inches(7.5)
    layout = p.slide_layouts[6]   # blank

    # ---- 1 · Title ----
    s = p.slides.add_slide(layout)
    add_text(s, 0.75, 2.6, 11.8, 1.0, 'NeuFlow v3', 40, True, INK)
    add_text(s, 0.75, 3.5, 11.8, 0.6, 'Queryable optical flow for edge devices — status report', 18, False, ACCENT)
    add_text(s, 0.75, 4.4, 11.8, 0.9,
             'Frozen NeuFlow v2 pipeline + implicit convex-weight decoder\n'
             'Shriman Raghav Srinivasan · MS Robotics, Northeastern University', 13, False, MUTED)

    # ---- 2 · NeuFlow v2 background ----
    s = p.slides.add_slide(layout)
    add_text(s, 0.75, 0.42, 11.0, 0.3, 'BACKGROUND', 10.5, True, ACCENT)
    add_text(s, 0.73, 0.72, 11.8, 0.8, 'What NeuFlow v2 does', 24, True, INK)
    add_text(s, 0.75, 1.75, 6.2, 4.4,
             'Real-time optical flow for edge devices (Zhang, Gupta, Jiang, Singh).\n\n'
             '1. Shallow CNN backbone: features at 1/8 and 1/16 scale\n'
             '2. Cross-attention + global matching at 1/16: initial flow\n'
             '3. Lightweight recurrent refinement (1 + 8 iterations)\n'
             '4. Convex upsampler: learned 3x3 blending on a fixed 8x grid\n\n'
             '10-70x faster than SOTA at comparable accuracy;\n'
             '>20 FPS at 512x384 on Jetson Orin Nano.\n\n'
             'Measured on our setup (RTX 4060, 384x1248, fp16):\n'
             '37 ms per frame pair · 9.03M params · 2.32 px EPE on VKITTI2', 13, False, INK)
    add_text(s, 7.3, 1.75, 5.2, 4.0,
             'The structural limit\n\n'
             'Output is a dense, fixed-resolution map.\n'
             'Every pixel is always computed, whether\n'
             'needed or not, and only at input resolution.\n\n'
             'Registration, sparse tracking, and mapping\n'
             'often need flow at a few hundred chosen\n'
             'points — not 479,000 pixels.', 13, False, INK)
    add_text(s, 1.05, 6.3, 11.3, 0.6,
             'Takeaway   v2 is the right backbone for edge flow; the upsampler is the part worth replacing.',
             11.5, True, ACCENT)
    footer(s, 2)

    # ---- 3 · What v3 changes ----
    s = p.slides.add_slide(layout)
    add_text(s, 0.75, 0.42, 11.0, 0.3, 'APPROACH', 10.5, True, ACCENT)
    add_text(s, 0.73, 0.72, 11.8, 0.8, 'v3: replace the upsampler with a queryable implicit decoder', 24, True, INK)
    add_text(s, 0.75, 1.75, 6.2, 4.4,
             'Kept (frozen, untouched): backbone, attention, matching,\n'
             'refinement — everything through the 1/8 coarse flow.\n\n'
             'Replaced: the convex upsampler becomes an MLP decoder\n'
             'answering "what is the flow at this exact (x, y)?"\n\n'
             'Decoder inputs per query: 3x3 local windows of 4 feature\n'
             'sources (context, 1/8, 1/16, flow-warped frame-1 features),\n'
             'gated hierarchical fusion (InfiniDepth).\n\n'
             'Head (AnyFlow-style): softmax weights over the 3x3 coarse-\n'
             'flow neighborhood + bilinear candidate. Bounded output —\n'
             'cannot hallucinate flow. Init reproduces bilinear exactly.\n\n'
             'New: Fourier encoding of the sub-cell offset (sub-pixel\n'
             f'awareness) — {PE_RESULT}.', 12.5, False, INK)
    add_text(s, 7.3, 1.75, 5.2, 4.2,
             'Why it fits edge devices\n\n'
             'Cost is O(N queries), not O(H x W).\n\n'
             'Two-pass API:\n'
             'infer_coarse_state(): 33 ms, once per pair\n'
             'decode_queries(): 1.6 ms per <=2k points,\n'
             'repeatable without re-running the backbone\n\n'
             '7.83M params — smaller than v2 (9.03M).\n'
             '~2.2 GB VRAM at inference.', 13, False, INK)
    add_text(s, 1.05, 6.3, 11.3, 0.6,
             'Takeaway   Same trusted front-end; only the output stage changes — from fixed map to on-demand queries.',
             11.5, True, ACCENT)
    footer(s, 3)

    # ---- 4 · Zero-training result ----
    s = p.slides.add_slide(layout)
    add_text(s, 0.75, 0.42, 11.0, 0.3, 'RESULT 1 OF 3', 10.5, True, ACCENT)
    add_text(s, 0.73, 0.72, 11.8, 0.8, 'What you get without training anything', 24, True, INK)
    add_text(s, 0.75, 1.8, 6.4, 4.0,
             'The decoder head is initialized so an untrained v3\n'
             'exactly reproduces bilinear upsampling of the coarse flow.\n\n'
             'Untrained v3, measured on VKITTI2 Scene18+20:\n\n'
             '     2.48 px EPE   (v2: 2.32 — only +0.15)\n'
             '     74.7% of pixels within 1px  (v2: 77.6%)\n\n'
             'And the query mechanism is exact: decoding N sparse\n'
             'points matches the dense output at those points to\n'
             '0.00 px, at O(N) cost.', 13.5, False, INK)
    add_text(s, 7.5, 1.8, 5.0, 4.0,
             'Why this matters\n\n'
             'Queryability is nearly free before any\n'
             'training: +0.15 px EPE buys sparse\n'
             'on-demand flow at ~35 ms total.\n\n'
             'Every training experiment is measured\n'
             'against this 2.48 baseline — training\n'
             'must beat "doing nothing."', 13, False, INK)
    add_text(s, 1.05, 6.3, 11.3, 0.6,
             'Takeaway   The architecture change alone delivers the capability; training only has to buy accuracy.',
             11.5, True, ACCENT)
    footer(s, 4)

    # ---- 5 · Results by dataset ----
    s = p.slides.add_slide(layout)
    add_text(s, 0.75, 0.42, 11.0, 0.3, 'RESULT 2 OF 3  ·  VKITTI2 SCENE18+20, 1174 PAIRS', 10.5, True, ACCENT)
    add_text(s, 0.73, 0.72, 11.8, 0.8, 'Training the decoder: what each dataset contributes', 24, True, INK)
    s.shapes.add_picture('results/epe_by_regime.png', Inches(0.75), Inches(1.7), width=Inches(7.4))
    add_text(s, 8.5, 1.85, 4.1, 4.3,
             'VKITTI2 only (12.7k pairs, 6 weather\n'
             'variants): 2.39 — first run to beat its\n'
             'own initialization.\n\n'
             'FlyingChairs only (22.2k pairs): 2.27 —\n'
             'beats v2 without seeing a single\n'
             'driving frame. Large synthetic motion\n'
             'crushes the error tail.\n\n'
             'Sequential finetune: 2.50 — forgets\n'
             'chairs robustness. Fix: mixed-dataset\n'
             'training (next).\n\n'
             'Open gap: 1px accuracy 69.7-74.7%\n'
             'vs v2 77.6% -> Fourier PE ablation\n'
             f'({PE_RESULT}).', 12, False, INK)
    footer(s, 5)


    # ---- 6 · Visual results ----
    s = p.slides.add_slide(layout)
    add_text(s, 0.75, 0.42, 11.0, 0.3, 'VISUAL RESULTS  ·  VKITTI2 SCENE18', 10.5, True, ACCENT)
    add_text(s, 0.73, 0.72, 11.8, 0.8, 'Seeing the output: dense fields and sparse queries', 24, True, INK)
    s.shapes.add_picture('results/visuals/sparse_queries.png', Inches(0.75), Inches(1.65), width=Inches(8.2))
    add_text(s, 9.2, 1.75, 3.5, 3.2,
             '300 corner-detected queries,\ndecoded in ONE 1.6 ms call\n(0.05% of a dense field).\n\n'
             'Arrow color = flow magnitude.\nLeft roadside streams left,\nright side streams right —\n'
             'forward ego-motion, correctly\nrecovered per point.', 12, False, INK)
    add_text(s, 0.75, 4.6, 11.8, 0.4,
             'Full-field comparisons (input / GT / v2 / v3 / error maps): results/visuals/compare_0.png, compare_1.png',
             11, False, MUTED)
    add_text(s, 1.05, 6.3, 11.3, 0.6,
             'Takeaway   v3 flow fields are structurally clean; both models fail in the same hard regions (dense foliage).',
             11.5, True, ACCENT)
    footer(s, 6)

    # ---- 7 · Compute benchmark ----
    s = p.slides.add_slide(layout)
    add_text(s, 0.75, 0.42, 11.0, 0.3, 'RESULT 3 OF 3  ·  RTX 4060 LAPTOP 8GB, FP16, 384x1248', 10.5, True, ACCENT)
    add_text(s, 0.73, 0.72, 11.8, 0.8, 'Compute: v2 vs v3', 24, True, INK)
    s.shapes.add_picture('results/latency_v2_v3.png', Inches(0.75), Inches(1.75), width=Inches(8.6))
    add_text(s, 9.6, 1.85, 3.0, 4.3,
             'Params:\n9.03M (v2)\n7.83M (v3)\n\n'
             'Sparse v3 costs the\nsame as v2 dense —\n'
             'but extra queries on\na processed pair are\n'
             '~20x cheaper than\nv2 recomputing.\n\n'
             'v3 dense (327 ms) is\nnot the use case.', 12, False, INK)
    add_text(s, 1.05, 6.35, 11.3, 0.6,
             'Takeaway   For sparse workloads (registration, tracking, mapping) v3 is strictly cheaper; parity latency, fewer params, O(N) scaling.',
             11.5, True, ACCENT)
    footer(s, 6)


    # ---- 8 · How to query ----
    s = p.slides.add_slide(layout)
    add_text(s, 0.75, 0.42, 11.0, 0.3, 'INTERFACE', 10.5, True, ACCENT)
    add_text(s, 0.73, 0.72, 11.8, 0.8, 'How querying works — sizes, API, interactivity', 24, True, INK)
    add_text(s, 0.75, 1.7, 6.3, 2.4,
             'Query = one continuous (x, y). N is free: 1 click to 479,232\n'
             '(= dense at 384x1248). Sub-pixel coords are first-class.\n\n'
             'state = model.infer_coarse_state(img0, img1)   # 33 ms, once\n'
             'flow  = model.decode_queries(state, query_coords=q)\n'
             '        # q: [B, N, 2] pixels -> [B, N, 2] flow, 1.6 ms\n'
             '        # or target_h/w (any resolution), or adaptive_n', 12, False, INK)
    add_text(s, 0.75, 4.15, 6.3, 2.0,
             'Training setup: batch 4 (8 GB VRAM) · 4,096 queries/image\n'
             '(3.1% of crop, half at motion boundaries) · datasets:\n'
             'VKITTI2 6-variant (12,726 pairs) + FlyingChairs (22,232)\n'
             '· AdamW OneCycle 2e-4 · backbone frozen', 12, False, INK)
    s.shapes.add_picture('results/visuals/query_gui_selftest.png', Inches(7.3), Inches(1.7), width=Inches(5.3))
    add_text(s, 7.3, 3.6, 5.3, 1.6,
             'Interactive query selector (PyQt5, working):\n'
             'click = flow at that pixel in ~1.6 ms; G = 32x32 grid.\n'
             'Viable precisely because of the two-pass API —\n'
             'v2 would recompute the full frame per click.', 11, False, INK)
    add_text(s, 1.05, 6.3, 11.3, 0.6,
             'Takeaway   scripts/query_gui.py demonstrates flow-on-demand; the same API serves robots and humans.',
             11.5, True, ACCENT)
    footer(s, 8)

    # ---- 9 · Objective scorecard ----
    s = p.slides.add_slide(layout)
    add_text(s, 0.75, 0.42, 11.0, 0.3, 'OBJECTIVES', 10.5, True, ACCENT)
    add_text(s, 0.73, 0.72, 11.8, 0.8, 'Better EPE at less compute, edge-capable — where we stand', 24, True, INK)
    add_text(s, 0.75, 1.9, 11.5, 3.9,
             'Beat v2 mean EPE                        DONE — 2.275 vs 2.324 (chairs-only training)\n\n'
             'Less compute (sparse use case)          DONE — ~35 ms total, O(N) queries, 13% fewer params, 2.2 GB VRAM\n\n'
             f'Sub-pixel precision parity (1px acc)    IN PROGRESS — Fourier PE ablation ({PE_RESULT})\n\n'
             'Combine chairs + vkitti2 strengths      NEXT — mixed-dataset training (forgetting measured and understood)\n\n'
             'Edge-device validation                  NEXT — Jetson Orin benchmark of the two-pass API', 14, False, INK)
    add_text(s, 1.05, 6.3, 11.3, 0.6,
             'Takeaway   Two of three headline objectives are met; the remaining gap is quantified and has a targeted fix in flight.',
             11.5, True, ACCENT)
    footer(s, 7)


    # ---- 10 · FAQ ----
    s = p.slides.add_slide(layout)
    add_text(s, 0.75, 0.42, 11.0, 0.3, 'ANTICIPATED QUESTIONS', 10.5, True, ACCENT)
    add_text(s, 0.73, 0.72, 11.8, 0.8, 'First-principles FAQ', 24, True, INK)
    add_text(s, 0.75, 1.7, 11.8, 4.4,
             'Why is flow queryable at continuous coords?  Bilinear feature interpolation + an MLP = a function defined at every real (x, y).\n\n'
             'Why freeze the backbone?  Joint training needs ~800K steps (InfiniDepth); at 30K the decoder chases moving features and diverges.\n\n'
             'Is sparse an approximation?  No — the identical function at fewer points; matches dense to 0.00 px at the same coordinates.\n\n'
             'Why does chairs training beat driving-data training on driving data?  Diverse large motions fix the error tail; 5 scenes teach memorization.\n\n'
             'If sparse is fast, why is dense slow?  Dense = 479k MLP calls (~293 ms); v2 upsampling is one fused conv. Dense is not the use case.\n\n'
             'What limits 1px accuracy?  The head could not see the sub-cell position — exactly what the Fourier PE ablation injects.',
             12.5, False, INK)
    add_text(s, 1.05, 6.3, 11.3, 0.6,
             'Takeaway   Full FAQ + derivations: docs/NeuFlow_v3_Report.md sections 5-6.',
             11.5, True, ACCENT)
    footer(s, 10)

    # ---- 11 · Next steps ----
    s = p.slides.add_slide(layout)
    add_text(s, 0.75, 0.42, 11.0, 0.3, 'NEXT', 10.5, True, ACCENT)
    add_text(s, 0.73, 0.72, 11.8, 0.8, 'Next steps', 24, True, INK)
    add_text(s, 0.75, 1.9, 11.5, 3.8,
             '1. Fourier PE ablation — same chairs recipe, +sub-cell encoding; targets 1px accuracy directly.\n\n'
             '2. Mixed chairs + vkitti2_all training — one dataloader change; expected to hold 2.27 EPE while recovering precision.\n\n'
             '3. Jetson Orin benchmark — the O(N) story is strongest where dense flow cannot run at all.\n\n'
             '4. Thesis experiments — registration/mapping demo on advisor-provided survey imagery; Spring dataset for\n'
             '    above-input-resolution querying (GT at 2x input resolution — a test only queryable decoders can take natively).',
             14, False, INK)
    add_text(s, 1.05, 6.3, 11.3, 0.6,
             'Takeaway   The architecture is validated; remaining work is training composition and edge deployment.',
             11.5, True, ACCENT)
    footer(s, 11)

    p.save(OUT)
    print(f'saved {OUT} with {len(p.slides._sldIdLst)} slides')


if __name__ == '__main__':
    main()
