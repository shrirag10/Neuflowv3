"""Build docs/NeuFlow_v3_status.pptx — complete rebuild, 2026-07-26.

Full revamp: linear narrative, every number sourced from docs/V3DEV_LOG.md,
five new comparison plots (results/plots/), no half-finished sections.
Monochrome academic style (Georgia, black/gray/white, thin rules).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from lxml import etree

INK    = RGBColor(0x1A, 0x1A, 0x1A)
ACCENT = RGBColor(0x33, 0x33, 0x33)
MUTED  = RGBColor(0x8A, 0x8A, 0x8A)
BOX_DK = RGBColor(0x33, 0x33, 0x33)
BOX_LT = RGBColor(0xEF, 0xEF, 0xEF)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
FONT   = 'Georgia'
OUT    = 'docs/NeuFlow_v3_status.pptx'


def add_text(s, x, y, w, h, text, size, bold=False, color=INK, line_spacing=1.0, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split('\n')):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.line_spacing = line_spacing
        para.alignment = align
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


def footer(s, num):
    add_text(s, 0.75, 7.02, 8.0, 0.30, 'NeuFlow v3, S. Srinivasan', 9, False, MUTED)
    add_text(s, 11.60, 7.02, 1.20, 0.30, str(num), 9, False, MUTED)


def box(s, x, y, w, h, text, dark=False, size=10, dashed=False):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.12
    shp.fill.solid()
    shp.fill.fore_color.rgb = BOX_DK if dark else BOX_LT
    shp.line.color.rgb = INK
    shp.line.width = Pt(1.0)
    if dashed:
        ln = shp.line._get_or_add_ln()
        d = etree.SubElement(ln, '{http://schemas.openxmlformats.org/drawingml/2006/main}prstDash')
        d.set('val', 'dash')
        shp.fill.background()
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Emu(27432)
    tf.margin_top = tf.margin_bottom = Emu(9144)
    for i, line in enumerate(text.split('\n')):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = PP_ALIGN.CENTER
        r = para.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = dark
        r.font.color.rgb = WHITE if dark else INK
    return shp


def arrow(s, x1, y1, x2, y2):
    conn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = INK
    conn.line.width = Pt(1.4)
    ln = conn.line._get_or_add_ln()
    tail = etree.SubElement(ln, '{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd')
    tail.set('type', 'triangle')
    tail.set('w', 'med')
    tail.set('len', 'med')
    return conn


def framed_pic(s, path, x, y, w, h):
    s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    fr = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    fr.fill.background()
    fr.line.color.rgb = INK
    fr.line.width = Pt(1.2)
    fr.shadow.inherit = False


def scene_grid(s, panels):
    W, H = 5.75, 1.74
    pos = [(0.75, 1.95), (6.85, 1.95), (0.75, 4.28), (6.85, 4.28)]
    for (path, label), (x, y) in zip(panels, pos):
        add_text(s, x, y - 0.32, W, 0.3, label, 11, True, INK)
        framed_pic(s, path, x, y, W, H)


def main():
    p = Presentation()
    p.slide_width = Inches(13.333)
    p.slide_height = Inches(7.5)
    blank = p.slide_layouts[6]
    counter = [0]

    def slide():
        counter[0] += 1
        return p.slides.add_slide(blank), counter[0]

    # ============================================ 1 · Title
    s, i = slide()
    add_text(s, 0.9, 2.5, 11.5, 1.0, 'NeuFlow v3', 40, True, INK)
    add_text(s, 0.9, 3.35, 11.5, 0.5, 'Queryable optical flow for edge devices', 18, False, ACCENT)
    add_text(s, 0.9, 4.0, 11.5, 0.9,
             'Status update. NeuFlow v2 backbone (frozen) + an implicit decoder that answers\n'
             'flow queries at arbitrary coordinates in O(N).', 13, False, INK, line_spacing=1.2)
    add_text(s, 0.9, 6.6, 11.5, 0.4, 'Shriman Raghav Srinivasan, MS Robotics, Northeastern University, Field Robotics Lab', 11, False, MUTED)

    # ============================================ 2 · Motivation
    s, i = slide()
    header(s, 'MOTIVATION', 'Why query optical flow?')
    add_text(s, 0.75, 1.65, 5.7, 4.8,
             'Optical flow tells you where each pixel in frame t moves\n'
             'to in frame t+1.\n\n'
             'Current networks always output the full dense map, at a\n'
             'fixed resolution. Many tasks never use most of it:\n\n'
             '-  registration wants a few hundred correspondences at\n'
             '    points it picks itself (corners, texture)\n'
             '-  sparse tracking only needs flow at feature points\n'
             '-  mosaicking needs matches in overlap regions, ideally\n'
             '    at sub-pixel positions\n\n'
             'On a small GPU, computing 479k values to use 800 of them\n'
             'is the difference between real time and not.', 13, False, INK, line_spacing=1.15)
    add_text(s, 6.85, 1.65, 5.7, 4.8,
             'Idea\n\n'
             'Keep everything in NeuFlow v2 that understands motion\n'
             '(backbone, matching, refinement), frozen.\n\n'
             'Swap only the last stage: instead of an upsampler that\n'
             'always produces the full map, use a decoder that returns\n'
             'flow at whatever coordinates you ask for.\n\n'
             'Cost then scales with the number of queries, O(N),\n'
             'instead of image area, O(HxW).', 13, False, INK, line_spacing=1.15)
    footer(s, i)

    # ============================================ 3 · Method: v2 pipeline
    s, i = slide()
    header(s, 'METHOD', 'NeuFlow v2: what stays, frozen and untouched')
    boxes = [
        (0.75, 1.9, 'Image pair'),
        (2.85, 1.9, 'CNN backbone\n(1/8, 1/16 features)'),
        (5.35, 1.9, 'Cross-attention\n+ global matching'),
        (7.95, 1.9, 'Refine x1 (1/16)\nRefine x8 (1/8)'),
        (10.55, 1.9, 'Coarse flow\n(1/8 resolution)'),
    ]
    for x, y, t in boxes:
        box(s, x, y, 2.0, 0.9, t, size=10.5)
    for k in range(len(boxes) - 1):
        arrow(s, boxes[k][0] + 2.0, boxes[k][1] + 0.45, boxes[k+1][0], boxes[k+1][1] + 0.45)
    add_text(s, 0.75, 3.2, 11.8, 0.35, 'Every block above is frozen in v3 -- identical weights, identical computation.', 12.5, True, INK)
    add_text(s, 0.75, 3.75, 11.8, 3.0,
             'NeuFlow v2 (Zhang, Gupta, Jiang, Singh, arXiv:2408.10161) is a real-time flow network\n'
             'built for edge devices:\n\n'
             '1.  Shallow CNN, features at 1/8 and 1/16\n'
             '2.  Cross-attention + global matching at 1/16 for the initial flow\n'
             '3.  Light recurrent refinement: 1 iteration at 1/16, 8 at 1/8\n'
             '4.  (v2 only) Learned convex upsampler: each output pixel is a blend of a 3x3\n'
             '     neighborhood of the 1/8 flow -- this last step is what v3 replaces.\n\n'
             'v2 alone: 9.03M parameters, 19.6 ms per frame pair (V100, 384x1248), 2.324 px\n'
             'mean EPE on VKITTI2. Its structural limit: output resolution and cost are fixed\n'
             'at design time -- no cheap way to ask a smaller question, no way to ask a finer one.', 12.5, False, INK, line_spacing=1.15)
    footer(s, i)

    # ============================================ 4 · Method: v3 decoder + pseudocode
    s, i = slide()
    header(s, 'METHOD', 'What v3 adds: a queryable decoder instead of the upsampler')
    add_text(s, 0.75, 1.6, 11.8, 0.3, 'Per query: sample features at (x, y), blend a bounded set of candidates', 12.5, True, INK)
    code = ('# ONCE per image pair (~16-17 ms, V100, 384x1248)\n'
            'coarse_flow, features = v2_pipeline(img0, img1)      # frozen, unchanged\n\n'
            '# PER QUERY BATCH (~2.6 ms for up to ~2,000 points)\n'
            'for each query (x, y):                                # continuous coords\n'
            '    feat = sample(features, x, y)                     # bilinear, 260-d\n'
            '    weights, [b] = decoder_head(feat)                 # softmax + optional log-scale\n'
            '    candidates = 3x3 coarse-flow window + bilinear value\n'
            '    flow = sum(weights * candidates)                  # bounded convex blend\n'
            'return flow, [b]                                       # b = predicted error (option G)')
    tb = s.shapes.add_textbox(Inches(0.9), Inches(2.0), Inches(11.5), Inches(2.2))
    tf = tb.text_frame
    tf.word_wrap = True
    for j, line in enumerate(code.split('\n')):
        para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        r = para.add_run(); r.text = line
        r.font.name = 'Consolas'; r.font.size = Pt(11)
        r.font.color.rgb = INK
        if line.lstrip().startswith('#'):
            r.font.bold = True
    add_text(s, 0.75, 4.35, 11.8, 2.5,
             'The head outputs weights over a 3x3 neighborhood of the coarse flow plus a bilinear\n'
             'candidate, softmax-blended -- bounded by construction, so it cannot hallucinate large\n'
             'flow. Zero-initialized so an untrained decoder reproduces plain bilinear upsampling\n'
             'exactly (verified: 2.476 px EPE at step 0, full validation set).\n\n'
             'Two-pass API: infer_coarse_state() once per frame, decode_queries() per query batch.\n'
             'Sparse output matches dense output at the same points exactly -- verified to 0.00 px.\n'
             'v3 has fewer parameters than v2 (7.83M vs 9.03M) despite adding this capability.', 12.5, False, INK, line_spacing=1.15)
    footer(s, i)

    # ============================================ 5 · Results: training curriculum
    s, i = slide()
    header(s, 'RESULTS', 'The full training curriculum, in order')
    s.shapes.add_picture('results/plots/curriculum_epe.png', Inches(0.6), Inches(1.55), width=Inches(12.1))
    add_text(s, 0.75, 6.7, 11.8, 0.7,
             'Every configuration tried, worst to best. Local runs (RTX 4060, batch 4) established the\n'
             'recipe; HPC runs (Explorer cluster, batch 16, 100K steps) scaled it. big18 is the best full run.', 11.5, False, INK)
    footer(s, i)

    # ============================================ 6 · Results: precision
    s, i = slide()
    header(s, 'RESULTS', 'Precision: closing the sub-pixel gap')
    s.shapes.add_picture('results/plots/precision_bars.png', Inches(2.58), Inches(1.55), width=Inches(8.18))
    add_text(s, 0.75, 6.6, 11.8, 0.7,
             'uncG (uncertainty head) is the first v3 checkpoint to beat v2 on 3px accuracy (90.02 vs\n'
             '89.8) and nearly closes the 1px gap (77.51 vs 77.6 -- 0.09 points, within measurement noise).', 11.5, False, INK)
    footer(s, i)

    # ============================================ 7 · Results: visual comparison
    s, i = slide()
    header(s, 'RESULTS', 'Same scene, four checkpoints, all beating v2 on EPE')
    scene_grid(s, [
        ('results/panels_hpc/scene_v2.png', 'NeuFlow v2, EPE 2.324 (full set) / 2.111 (this scene)'),
        ('results/panels_hpc/scene_v3_grandmix.png', 'v3 grandmix, EPE 2.166 (full set) / 2.077 (this scene)'),
        ('results/panels_hpc/scene_v3_big18.png', 'v3 big18, EPE 2.072 (full set, best) / 2.089 (this scene)'),
        ('results/panels_hpc/scene_v3_uncG.png', 'v3 uncG, EPE 2.082 (full set) / 2.053 (this scene, best here)'),
    ])
    add_text(s, 0.75, 6.35, 11.9, 0.75,
             'Batch 16, 100K steps, H200/V100 (Explorer cluster). All three HPC checkpoints beat v2\n'
             'on mean EPE, on the full 1,174-pair validation set, not just this one scene.', 11.5, False, INK)
    footer(s, i)

    # ============================================ 8 · Results: speed
    s, i = slide()
    header(s, 'RESULTS', 'Speed: the real deployment argument')
    s.shapes.add_picture('results/plots/speed_bars.png', Inches(3.0), Inches(1.5), width=Inches(7.3))
    add_text(s, 0.75, 6.15, 11.8, 0.9,
             'Measured on identical V100 hardware. The pitch is not "6% better EPE" -- it is that v2\n'
             'pays the full 19.6 ms on every single call, while v3 pays a similar cost once per frame\n'
             'and then answers follow-up queries on that same frame for ~2.6 ms each.', 11.5, False, INK)
    footer(s, i)

    # ============================================ 9 · Results: uncertainty calibration
    s, i = slide()
    header(s, 'RESULTS', 'Option G: a calibrated confidence signal v2 cannot express')
    s.shapes.add_picture('results/plots/calibration_bars.png', Inches(2.8), Inches(1.5), width=Inches(7.7))
    add_text(s, 0.75, 6.15, 11.8, 0.9,
             'One extra decoder output, trained with a self-calibrating loss (|error|/b + 2 log b).\n'
             'Bins by predicted b show mean REAL error rising monotonically from 0.22 to 7.38 px --\n'
             'the confidence output is not noise. Also the first checkpoint to beat v2 on 3px accuracy.', 11.5, False, INK)
    footer(s, i)

    # ============================================ 10 · Results: distillation
    s, i = slide()
    header(s, 'RESULTS', 'Option A: refinement self-distillation')
    s.shapes.add_picture('results/plots/distillation_bars.png', Inches(3.0), Inches(1.5), width=Inches(7.3))
    add_text(s, 0.75, 6.15, 11.8, 0.9,
             'The 8-iteration refinement loop is 59% of pipeline runtime; iterations 5-8 barely change\n'
             'the answer. Retrained ONLY the refinement module (teacher = the model itself, no ground\n'
             'truth) so 3 iterations approach 8-iteration quality at 3-iteration cost. Decoder untouched.', 11.5, False, INK)
    footer(s, i)

    # ============================================ 11 · Interface: how to query
    s, i = slide()
    header(s, 'INTERFACE', 'Querying in practice: sizes and API')
    add_text(s, 0.75, 1.7, 11.8, 1.6,
             'A query is one continuous (x, y) coordinate; sub-pixel positions are valid inputs. N\n'
             'ranges from 1 (a single click) to the full frame (479,232 at 384x1248). Decode cost is\n'
             'flat at ~2.6 ms per call for up to ~2,000 queries per call.\n\n'
             'Training: batch size 4-16 depending on hardware; 4,096 supervision queries per image,\n'
             'half placed at motion boundaries; AdamW with a one-cycle schedule peaking at 2e-4;\n'
             'backbone frozen throughout every result on this deck except the distillation experiment.',
             12.5, False, INK, line_spacing=1.15)
    code = ('state = model.infer_coarse_state(img0, img1)     # once, ~17 ms\n'
            'flow  = model.decode_queries(state, query_coords=q)   # q: [B, N, 2] -> [B, N, 2]')
    tb = s.shapes.add_textbox(Inches(0.9), Inches(4.7), Inches(11.5), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    for j, line in enumerate(code.split('\n')):
        para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        r = para.add_run(); r.text = line
        r.font.name = 'Consolas'; r.font.size = Pt(12)
        r.font.color.rgb = INK
    footer(s, i)

    # ============================================ 12 · Interface: the GUI
    s, i = slide()
    header(s, 'INTERFACE', 'The GUI')
    framed_pic(s, 'results/visuals/query_gui_selftest.png', 0.75, 1.7, 5.7, 4.0)
    framed_pic(s, 'results/visuals/query_gui_motion.png', 6.85, 1.7, 5.7, 4.0)
    add_text(s, 0.75, 5.85, 5.7, 0.6, 'Adaptive queries over a dense overlay, with the magnitude legend.', 10.5, False, MUTED)
    add_text(s, 6.85, 5.85, 5.7, 0.6, 'Playback with live motion detection, boxed from the coarse flow at zero extra decode cost.', 10.5, False, MUTED)
    add_text(s, 0.75, 6.5, 11.8, 0.6,
             'A model selector switches between v3 and the v2 baseline in place: with v2 every interaction\n'
             'recomputes the full dense map; with v3 the same interactions reuse the cached coarse state.', 11, False, INK)
    footer(s, i)

    # ============================================ 13 · Interface: live proof
    s, i = slide()
    header(s, 'INTERFACE', 'Live run on public video, with measurements')
    framed_pic(s, 'results/visuals/yt_run_viewer.png', 0.75, 1.7, 6.3, 3.55)
    framed_pic(s, 'results/visuals/yt_run_resources.png', 7.35, 1.7, 5.3, 3.55)
    add_text(s, 0.75, 5.45, 11.8, 1.3,
             'Forty consecutive frame pairs of a public YouTube highway video (640x360), processed live\n'
             'in the GUI: mean 29.8 FPS end to end with motion boxes on moving vehicles. The resource\n'
             'graphs were sampled during the same run; GPU memory stays flat because one cached backbone\n'
             'state serves every interaction.', 11.5, False, INK)
    footer(s, i)

    # ============================================ 14 · Objectives
    s, i = slide()
    header(s, 'OBJECTIVES', 'Where the three goals stand')
    add_text(s, 0.75, 1.75, 11.8, 4.9,
             'Goal 1, beat v2 accuracy.  Done, verified on the full validation set, batch-16/100K-step\n'
             'HPC training: best EPE 2.07 vs v2 2.32 (11% better, big18). 3px accuracy now BEATS v2\n'
             '(90.02 vs 89.8, uncG). 1px accuracy gap nearly closed (77.51 vs 77.6, uncG -- 0.09 points,\n'
             'within noise).\n\n'
             'Goal 2, less compute, edge-viable.  Done for the sparse workload, verified on identical\n'
             'V100 hardware: first query on a new frame already at parity with v2 (19.1 vs 19.6 ms);\n'
             'every additional query on the same frame costs ~7x less (2.6 ms vs 19.6 ms) since v2 has\n'
             'no cached state to reuse.\n\n'
             'Goal 3, do what v2 cannot.  Done: continuous-coordinate queries, output at any resolution,\n'
             'sparse matches dense exactly, a working interactive tool, and a calibrated per-query\n'
             'confidence signal (Pearson r=0.38 vs real error) that v2 cannot express at all.',
             13, False, INK, line_spacing=1.15)
    footer(s, i)

    # ============================================ 15 · Limitations
    s, i = slide()
    header(s, 'LIMITATIONS', 'What is not yet proven')
    add_text(s, 0.75, 1.75, 11.8, 4.9,
             '1px accuracy is close but not fully closed on every checkpoint (grandmix still trails v2\n'
             'by 1.35 points; only uncG gets within noise).\n\n'
             'The training-data confound is only partly resolved. grandmix (chairs+vkitti2+sintel) was\n'
             'the fair-comparison attempt but scored slightly below big18 -- more data did not clearly\n'
             'help at 100K steps, and the confound is not fully closed until v2 is retrained on the\n'
             'exact same mixture, which has not been done.\n\n'
             'The "auxiliary loss improves main accuracy" pattern seen in uncG is a single run, not a\n'
             'repeated result -- could be seed noise.\n\n'
             'All speed numbers are RTX 4060 / V100. No Jetson or embedded-device measurement exists yet.\n\n'
             'Spring (1080p, the dataset motivating above-input-resolution querying) was truncated at\n'
             '70,000 of 100,000 steps by the 8-hour job limit -- the score (2.080 EPE) is strong but the\n'
             'run never reached its final LR anneal, and the actual 4K-resolution query test has not run.',
             13, False, INK, line_spacing=1.15)
    footer(s, i)

    # ============================================ 16 · Next steps
    s, i = slide()
    header(s, 'NEXT STEPS', 'Planned work and two requests')
    add_text(s, 0.75, 1.7, 6.1, 4.6,
             'Immediate\n\n'
             '-  Resubmit spring from step 70,000 with a fresh 8h clock\n'
             '    to reach its full 100K steps.\n'
             '-  Run the Spring 4K-ground-truth query test: only a\n'
             '    queryable decoder can answer above input resolution;\n'
             '    v2 structurally cannot.\n'
             '-  Repeat the uncertainty-head run once to check whether\n'
             '    its accuracy gain is real or seed noise.\n'
             '-  Retrain v2 itself on the grand-mix data to fully close\n'
             '    the training-data confound.', 12.5, False, INK, line_spacing=1.15)
    add_text(s, 7.1, 1.7, 5.8, 4.8,
             'Requests\n\n'
             'Request 1 -- continued HPC access\n'
             'The remaining runs above (spring resumption, v2 retraining, a repeat run) are all 6-8 hour jobs; done on the Explorer cluster so far.\n\n'
             'Request 2 -- Field Robotics Lab survey data\n'
             'SeaBED AUV and UAV survey sequences (overlapping frames, slow motion, GPS/IMU tags) are exactly the data this method targets: sparse, on-demand correspondences at chosen points, not full dense flow over a whole seafloor mosaic.',
             12.5, False, INK, line_spacing=1.15)
    footer(s, i)

    # ============================================ 17 · Q&A prep
    s, i = slide()
    header(s, 'Q&A PREP', 'Questions I expect')
    add_text(s, 0.75, 1.7, 11.8, 5.0,
             'Why is flow defined at non-integer coordinates?\n'
             'The decoder bilinearly samples feature maps, so any real-valued (x, y) is a valid query --\n'
             'the same mechanism that lets it decode above input resolution.\n\n'
             'Is the 6-11% EPE improvement real, or a training-data confound?\n'
             'Partly resolved: grandmix (chairs+vkitti2+sintel, closer to v2\'s own training mix) still\n'
             'beats v2 by 7% on EPE. Not fully resolved: v2 has not been retrained on that exact mixture.\n\n'
             'Why does the uncertainty head also seem to improve the main flow output?\n'
             'Unconfirmed hypothesis: the auxiliary loss may regularize training. Single run; needs a repeat.\n\n'
             'What happens with fast real-world motion at 60 FPS?\n'
             'Track-maintain-replenish: propagate points via their own flow answers, drop points that fail\n'
             'a forward-backward consistency check, replenish only in image regions that lost coverage.\n\n'
             'Has this been tested on an actual edge device (Jetson)?\n'
             'No. Every number in this deck is RTX 4060 or V100. That validation has not been done.',
             13, False, INK, line_spacing=1.2)
    footer(s, i)

    p.save(OUT)
    print(f'saved {OUT} with {len(p.slides._sldIdLst)} slides')


if __name__ == '__main__':
    main()
