"""Build docs/NeuFlow_v3_Presentation.pptx, the defence deck.

Structured-argument deck. The argument lives in the titles, the evidence in one
exhibit per slide, and the detail in the speaker notes. Nothing on a slide
exists because a result exists; it exists because the argument needs it.

Argument spine, in five movements:

    1  the platform problem        slides 2 to 4
    2  the change and its control       5 to 6
    3  what it costs                    7 to 8
    4  what it buys                     9 to 15
    5  where that leaves it            16 to 18

Layout, palette, measurement and the build-time checks live in deck_style, so
that this deck and the status deck cannot drift apart. Deck figures come from
`DECK_FIGS=1 python3 scripts/make_final_plots.py`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pptx.enum.text import PP_ALIGN
from deck_style import (
    INK, MID, MUTED, RULE, PANEL, WHITE, FONT,
    F_TITLE, F_BODY, F_COL, F_TABLE, F_SMALL, F_CITE,
    W, H, MARGIN, CW, TITLE_TOP, TITLE_H, RULE_Y, CONTENT_Y, CITE_Y,
    FIG_X, FIG_W, COL_X, COL_W, PLOTS, MONTAGE, VISUALS,
    text, bullets, hrule, vrule, action_title, cite, note, panel,
    keyline, keyline_h, picture, exhibit, figure_bottom, column, table, chain,
    new_deck, finish,
)

OUT = 'docs/NeuFlow_v3_Presentation.pptx'


# ------------------------------------------------------------------ deck
def main():
    prs, slide = new_deck()

    # ============================================================= 1  title
    s, i = slide(footer=False)
    text(s, MARGIN, 1.85, CW, 2.30,
         'Optical flow you can query:\n'
         'calibrated confidence and 27x cheaper repeat\n'
         'access for 2.6% of mean accuracy',
         32, True, INK, 1.22, tight=True)
    hrule(s, 4.34, MARGIN, 2.6, pt=3.5, color=INK)
    text(s, MARGIN, 4.60, CW, 0.9,
         'Shriman Raghav Srinivasan\n'
         'MS Robotics, Northeastern University   |   Field Robotics Lab',
         F_COL, False, INK, 1.35, tight=True)
    text(s, MARGIN, 5.75, CW, 0.4,
         'Final project presentation   |   12 August 2026',
         F_CITE, False, MUTED, tight=True)
    note(s, "I replaced NeuFlow v2's fixed upsampler with a decoder you can query at any "
            "coordinate. The headline is that this buys three capabilities v2 cannot express, "
            "and costs 2.6 percent of mean accuracy. I will show you both sides. The result I "
            "would most like your reaction to is selective accuracy, which is slide eleven, so "
            "I will keep the earlier material tight.")

    # ============================================================= 2  the platform
    s, i = slide()
    action_title(s, 'A fast platform cannot afford the frame\nit is given')
    fig = f'{MONTAGE}/scenarios_illustrated_tight.png'
    picture(s, fig, 1.60, 1.78, w=10.1)
    cite(s, 'Left: the region a tracker nominates. Right: the flow returned. The grey area is '
            'never computed. Held-out Virtual KITTI 2.')
    note(s, "Start with the situation rather than the network. A drone or a fast vehicle has a "
            "millisecond budget per frame and a dense flow field does not fit in it. So it flows "
            "only the region that matters. Three situations it meets. First, something worth "
            "tracking enters the field of view. Second, the platform turns and that region now "
            "overlaps a second object. Third, a new object appears in a frame you are already "
            "part way through processing. On the right is what the model actually computes: flow "
            "inside the marked regions only. Everything in this deck is in service of that third "
            "case, which is the one a dense network cannot answer cheaply.")

    # ============================================================= 3  foreclosure
    s, i = slide()
    action_title(s, 'A dense output forecloses the two questions\nthat platform actually asks')
    bullets(s, MARGIN, CONTENT_Y, CW, 2.6, [
        ('Where exactly? ',
         'A corner tracked to (312.7, 188.2) cannot be requested. The pixel grid is a sampling '
         'artefact; the motion underneath it is continuous.'),
        ('And again, on a frame already processed? ',
         'A second question about the same pair costs the entire computation a second time, '
         'because nothing is kept between calls.'),
        ('Meanwhile every consumer discards most of the answer. ',
         'Registration, sparse tracking and survey mosaicking each select a few hundred points '
         'and throw away the other 478,000.'),
    ], F_BODY)
    panel(s, MARGIN, 5.35, CW, 0.86,
          'On constrained hardware, computing 479,232 correspondences in order to consume 800 '
          'is the difference between operating in real time and not.',
          F_BODY, PP_ALIGN.CENTER)
    cite(s, 'Frame size 384 x 1248. Baseline: Zhang, Gupta, Jiang and Singh (2024), NeuFlow v2.')
    note(s, "Every modern flow network computes a displacement for every pixel at one fixed "
            "resolution, and that contract is rarely questioned. It forecloses two things the "
            "platform on the last slide needs. It cannot be asked for a position between pixels, "
            "and it cannot be asked a second question about a frame it has already processed "
            "without paying for the whole thing again. On top of that, the consumers that matter "
            "here select a few hundred points themselves and discard the rest. On a small GPU "
            "that waste is the difference between real time and not.")

    # ============================================================= 4  question
    s, i = slide()
    action_title(s, 'Can the output stage be made queryable\nwithout surrendering accuracy or speed?')
    panel(s, MARGIN, CONTENT_Y, CW, 1.30,
          'Can NeuFlow v2’s fixed upsampler be replaced by a decoder evaluated at\n'
          'arbitrary continuous coordinates, and what does that cost?',
          F_BODY, PP_ALIGN.CENTER)
    bullets(s, MARGIN, 3.62, CW, 1.9, [
        ('Approach. ', 'Freeze everything in v2 that understands motion. Change only the last stage.'),
        ('Consequence. ', 'Cost scales with the number of queries, O(N), not with image area, O(HxW).'),
        ('Test. ', 'One variable per training run, and a front end verified unchanged.'),
    ], F_BODY)
    cite(s, 'Evaluated on 1,174 held-out Virtual KITTI 2 pairs, 460,573,660 valid pixels, and on '
            'TartanAir aerial sequences.')
    note(s, "So the question, stated precisely. Not whether I can build a better flow network, "
            "but whether the final stage of a real-time one can be made queryable, and what that "
            "costs. The approach is to keep everything that understands motion frozen and change "
            "only the output, so that any difference you see is attributable to one thing.")

    # ============================================================= 5  architecture
    s, i = slide()
    action_title(s, 'v3 keeps NeuFlow v2 frozen and replaces\nonly its fixed upsampler')
    text(s, MARGIN, CONTENT_Y, CW, 0.26,
         'Phase 1   once per frame pair, 33 ms, cached', F_SMALL, True, MID, tight=True)
    chain(s, 2.18, [('Image pair', False), ('CNN backbone\n1/8 and 1/16', False),
                    ('Cross-attention\nglobal match', False),
                    ('Refinement\n1x s16, 8x s8', False),
                    ('Coarse flow\nat 1/8', True)])
    text(s, MARGIN, 3.44, CW, 0.26,
         'Phase 2   once per query batch, 1.25 ms, O(N)', F_SMALL, True, MID, tight=True)
    chain(s, 3.82, [('Query (x, y)\ncontinuous', False), ('Sample 3x3\nfour sources', False),
                    ('Gated fusion\nMLP', False), ('Convex blend\nnine plus one', False),
                    ('Flow and\nconfidence', True)])
    panel(s, MARGIN, 5.20, CW, 0.90,
          'The output is a convex combination of neighbouring coarse-flow values, so the decoder '
          'cannot predict motion its inputs do not locally support.',
          F_BODY, PP_ALIGN.CENTER)
    cite(s, 'Convex-weight formulation follows Jung et al. (2023), AnyFlow. Gated fusion follows '
            'InfiniDepth (2025).')
    note(s, "The change is confined to the last stage. Phase one is v2 exactly as published, run "
            "once and cached. Phase two takes a coordinate, samples feature windows around it, "
            "and predicts weights that blend the nine neighbouring coarse flow values plus a "
            "bilinear sample. Because it is a convex blend it cannot hallucinate motion. And "
            "because phase one is cached, asking a second question about the same frame is "
            "cheap. That caching turns out to be the property that matters most, and it is the "
            "second of the two foreclosures from the previous slide.")

    # ============================================================= 6  controls
    s, i = slide()
    action_title(s, 'The front end is verified bit-identical to v2,\nso the comparison isolates the decoder')
    bullets(s, MARGIN, CONTENT_Y, 6.6, 3.1, [
        ('Frozen front end. ', 'All 137 shared tensors verified bit-identical, per checkpoint.'),
        ('Disjoint splits. ', 'Both evaluation scenes are excluded from every training set, '
                              'checked at pair and frame level before each run.'),
        ('One variable per run. ', 'Same seed, schedule, batch size and query budget throughout.'),
    ], F_COL)
    table(s, 7.55, CONTENT_Y + 0.02, [
        ['Evaluation', ''],
        ['Held-out pairs', '1,174'],
        ['Valid pixels', '460.6 M'],
        ['Cross-domain', 'TartanAir'],
        ['Hardware', 'RTX 4060'],
        ['Precision', 'fp16'],
    ], [3.10, 1.93])
    panel(s, MARGIN, 5.42, CW, 0.78,
          'Zero-initialised, the decoder is exactly bilinear upsampling: training can only move '
          'away from a known-good operating point by learning something.',
          F_BODY, PP_ALIGN.CENTER)
    cite(s, 'Cabon et al. (2020), Virtual KITTI 2. Wang et al. (2020), TartanAir.')
    note(s, "Two things make the rest of the deck interpretable. First, the front end is frozen "
            "and I verify per checkpoint that all 137 shared tensors are bit-identical to v2, so "
            "any difference you see is the decoder and nothing else. Second, both evaluation "
            "scenes are held out of every training set, checked at pair and frame level before "
            "each run. The zero-initialisation property is worth noting too: an untrained decoder "
            "is exactly bilinear upsampling, so it starts from a known-good operating point.")

    # ============================================================= 7  accuracy
    s, i = slide()
    action_title(s, 'The decoder costs 2.6 percent of mean error,\nfrom a model 13 percent smaller')
    fb = exhibit(s, f'{PLOTS}/accuracy_bars.png', 1.95)
    column(s, 'THE COST, STATED FIRST', [
        ('', 'Every v3 configuration sits above v2.'),
        ('', 'Like-for-like on FlyingChairs alone: 2.500, or 7.6% behind.'),
        ('', '7.83 M parameters against 9.03 M.'),
    ], key='2.384\nagainst 2.324', fig_bottom=fb)
    cite(s, 'Mean end-point error per pixel, 1,174 held-out VKITTI 2 pairs. Lower is better.')
    note(s, "Here is the central cost, and I am leading with it rather than burying it. Every v3 "
            "configuration is less accurate than the baseline on mean end-point error. The best "
            "is 2.384 against 2.324, so 2.6 percent worse. The like-for-like comparison, trained "
            "on FlyingChairs alone as v2 was trained on FlyingThings alone, is 7.6 percent "
            "behind. The model is 13 percent smaller, which is worth something but does not "
            "excuse it. Four slides on, the same model is more than twice as accurate as v2 over "
            "most of the frame, once it is allowed to abstain. That is the argument.")

    # ============================================================= 8  precision
    s, i = slide()
    action_title(s, 'The cost concentrates in sub-pixel precision,\nnot in large errors')
    fb = exhibit(s, f'{PLOTS}/precision_bars.png', 1.95)
    column(s, 'DIAGNOSED CAUSE', [
        ('', 'The decoder’s finest input is 1/8 scale, so evidence barely varies inside an '
             '8x8 cell. The upsampler it replaces reads the full-resolution frame.'),
        ('', 'A Fourier positional encoding changed nothing, which rules out a missing '
             'position signal.'),
    ], key='1.5 points below v2\non 1 px accuracy', fig_bottom=fb)
    cite(s, 'Percentage of pixels with end-point error below 1 px. Higher is better.')
    note(s, "This is where the cost actually sits. On one-pixel accuracy v3 is below v2 in every "
            "configuration. Mean error partly hides it, because v3 makes fewer catastrophic "
            "errors, which drags the average down while it is less exact on the majority of "
            "pixels. I know the mechanism: the decoder's finest input is at one-eighth "
            "resolution, so inside an eight by eight cell it is looking at almost the same "
            "evidence wherever you query, while the v2 upsampler reads the full-resolution frame "
            "directly. I tested the competing explanation, that it was simply missing positional "
            "information, by adding a Fourier encoding. It changed nothing, which rules that out.")

    # ============================================================= 9  speed
    s, i = slide()
    action_title(s, 'A repeat query on a cached frame costs\n1.25 ms, against 33.3 ms for a fresh pass')
    fb = exhibit(s, f'{PLOTS}/speed_bars.png', 2.02)
    column(s, 'WHERE THE WIN IS', [
        ('', 'A first sparse query is level with v2: both pay the coarse pass, and global '
             'matching needs the whole image, so it cannot be skipped.'),
        ('', 'v2 keeps no state, so a second question costs a full recomputation.'),
    ], key='27x on repeat access', fig_bottom=fb)
    cite(s, 'RTX 4060 laptop, fp16, 384 x 1248, median of 50 runs after warm-up.')
    note(s, "On speed I want to be precise about which claim I am making. Dense output is ten "
            "percent slower than v2, and that mode is not what the decoder is for. A first sparse "
            "query is level with v2, because 96 percent of the cost is the coarse pass, which "
            "both share and which cannot be skipped since global matching needs the whole image. "
            "The win is repeat access. Asking a second question about a frame you have already "
            "processed costs 1.25 milliseconds against 33.3, because v2 keeps no state and has "
            "to redo everything. Anything that revisits a frame gets that: iterative "
            "registration, RANSAC refinement, an operator inspecting a scene. That is "
            "structural, not a margin.")

    # ============================================================= 10 calibration
    s, i = slide()
    action_title(s, 'Predicted confidence tracks real error\nmonotonically across a 15-fold span')
    fb = exhibit(s, f'{PLOTS}/calibration_bars.png', 1.95)
    column(s, 'WHY IT IS USABLE', [
        ('', 'Weight correspondences in RANSAC by predicted scale.'),
        ('', 'Reject unreliable matches before they reach a solver.'),
        ('', 'v2 emits flow alone. There is no equivalent quantity to compare against.'),
    ], key='0.48 px to 7.10 px\nacross five bins', fig_bottom=fb)
    cite(s, 'Per-query error scale under a Laplace likelihood. 2,348,000 queries, Pearson r = 0.318.')
    note(s, "The head predicts an error scale for every query alongside the flow itself, trained "
            "under a Laplace likelihood. When you bin queries by predicted uncertainty, actual "
            "error rises monotonically across all five bins, from 0.48 pixels in the most "
            "confident to 7.10 in the least, a fifteen-fold span, over 2.35 million queries. It "
            "is not perfectly correlated, r is 0.318, but it is clearly informative and directly "
            "usable: weight correspondences, drop unreliable ones, or send more queries where the "
            "model is unsure. v2 has no equivalent output, so there is nothing to compare against.")

    # ============================================================= 11 selective
    s, i = slide()
    action_title(s, 'Abstaining on 20 percent of queries makes\nv3 2.2x more accurate than v2')
    fb = exhibit(s, f'{PLOTS}/selective_accuracy.png', 2.00)
    column(s, 'WHY THIS IS THE RIGHT COMPARISON', [
        ('', 'Mean error over all pixels is one operating point, and the only one a model '
             'without confidence has.'),
        ('', 'Registration picks a few hundred points anyway, so declining to answer costs '
             'nothing it would have used.'),
    ], key='1.06 px at 80% coverage\nagainst v2 at 2.32', fig_bottom=fb)
    cite(s, 'Same 2,348,000 queries as the calibration bins. Coverage is the share of queries answered.')
    note(s, "This is the slide I would most like your reaction to. The accuracy table earlier "
            "reports mean error over every pixel, which is the only operating point v2 has, "
            "because it cannot tell you which of its outputs to trust. v3 can, so instead of a "
            "point it has a curve. Discard the least confident fifth and error on what remains "
            "falls to 1.06 pixels against v2's 2.32 over the whole frame. More than twice as "
            "accurate over eighty percent of the image. Keep only the most confident fifth and it "
            "is 0.48. Why this is not a statistical trick: for registration and mapping you are "
            "choosing a few hundred points anyway, so declining to answer where the model is "
            "unsure costs you nothing you would otherwise have used.")

    # ============================================================= 12 region cost
    s, i = slide()
    action_title(s, 'Processing 14 percent of the frame costs\n0.034 px and runs 4.4x faster')
    table(s, MARGIN, CONTENT_Y, [
        ['Region processed', 'Area', 'Latency', 'Speedup', 'EPE in region'],
        ['Full frame', '100%', '33.3 ms', '1.0x', '0.657'],
        ['Region, no margin', '7.9%', '7.8 ms', '4.3x', '1.089'],
        ['Region + 32 px margin', '13.8%', '7.6 ms', '4.4x', '0.691'],
        ['Region + 64 px margin', '20.1%', '8.1 ms', '4.1x', '0.667'],
        ['Region + 128 px margin', '34.2%', '9.9 ms', '3.4x', '0.655'],
    ], [2.95, 0.83, 1.04, 1.12, 1.73], hl_row=3, avail=7.75)
    panel(s, 8.75, CONTENT_Y, 3.83, 1.98,
          'Design rule\n\nmargin ≈ expected motion\n≈ speed / frame rate',
          F_COL, PP_ALIGN.CENTER)
    panel(s, MARGIN, 4.90, CW, 1.00,
          'The margin is not optional. Without one, error rises 65 percent and a quarter of '
          'large-motion pixels fail outright, because global matching loses the context it uses '
          'to locate distant correspondences.',
          F_BODY, PP_ALIGN.CENTER)
    cite(s, 'Applies to any network in this family, v2 included. A platform technique, not a '
            'property of the decoder.')
    note(s, "Here is what flowing only a region actually buys. Keep full resolution, process "
            "fourteen percent of the frame, and it costs thirty-four thousandths of a pixel. That "
            "is the enabling result for the whole scenario. The margin is the part worth dwelling "
            "on. Without one, error jumps sixty-five percent and a quarter of the large-motion "
            "pixels fail completely, because this network finds large displacement using "
            "attention over the whole image and a tight crop takes that away. I should be clear "
            "that this applies to any flow network including v2 unchanged. It is a platform "
            "technique, not something my decoder provides. What the decoder adds comes two slides "
            "from here.")

    # ============================================================= 13 margin rule
    s, i = slide()
    action_title(s, 'The margin a region needs is set by motion,\nnot by image size')
    picture(s, f'{PLOTS}/margin_rule.png', 1.57, 1.88, w=10.2)
    panel(s, MARGIN, 5.74, CW, 1.00,
          'Driving averages 26.6 px of motion and needs about 32 px of margin. Aerial averages '
          '9.26 px and needs about 8. Both shed most of the penalty by one frame of motion.',
          F_BODY, PP_ALIGN.CENTER)
    cite(s, 'Measured with the baseline network. Agreement holds to a ratio of one; residuals '
            'beyond that differ by scene.')
    note(s, "I wanted to know whether the margin rule was a curve fitted to one dataset or "
            "something more general, so I tested it on aerial sequences, which move very "
            "differently: nine point three pixels per frame against twenty-six point six for "
            "driving. The prediction was that the required margin should scale with motion, so "
            "the knee should move from thirty-two pixels down to about nine. It did. On the left "
            "the two need different absolute margins. On the right, divided by mean motion, both "
            "start at the same penalty and both have lost most of it by the point where the "
            "margin equals one frame of motion. I will be precise about the limits: the agreement "
            "is good up to that point and residuals differ beyond it, so this predicts the scale "
            "rather than the exact curve. That is still enough to size a margin from speed and "
            "frame rate before running anything.")

    # ============================================================= 14 scenario 3
    s, i = slide()
    action_title(s, 'A region discovered mid-frame costs 1.7 ms,\nagainst 7.3 ms for a fresh crop')
    picture(s, f'{PLOTS}/scenario3_marginal.png', 1.57, 1.88, w=10.2)
    panel(s, MARGIN, 5.74, CW, 1.00,
          'The cached coarse state answers a question the platform did not have when the frame '
          'arrived. A cropped pipeline must run a whole new pass, and is less accurate doing it.',
          F_BODY, PP_ALIGN.CENTER)
    cite(s, 'Two disjoint crops cost more than one full frame, which gives a simple policy: '
            'crop once, or not at all.')
    note(s, "This is the third situation from the opening slide, and the one result specific to "
            "this architecture rather than to cropping. A frame is already in flight when a new "
            "object enters the field of view. Because the coarse pass is cached, answering costs "
            "one point seven milliseconds. A cropped pipeline cannot reuse anything, since the "
            "new object is outside its box, so it runs a whole new pass at seven point three "
            "milliseconds, four times more, and the answer is worse, three point six pixels "
            "against two point one, because a fresh crop has lost the surrounding context. v2 "
            "keeps no state at all between calls. The general point is that this buys the ability "
            "to answer a question you did not know you would have when the frame arrived, which "
            "on a moving platform is most of them.")

    # ============================================================= 15 aerial
    s, i = slide()
    action_title(s, 'On a domain neither model trained for,\nv3 leads at all seven margin settings')
    table(s, MARGIN, 2.16, [
        ['Crop margin', 'NeuFlow v2', 'NeuFlow v3'],
        ['full frame', '0.781', '0.777'],
        ['0 px', '1.205', '1.176'],
        ['8 px', '0.901', '0.881'],
        ['16 px', '0.872', '0.865'],
        ['24 px', '0.879', '0.854'],
        ['32 px', '0.834', '0.814'],
        ['64 px', '0.793', '0.778'],
    ], [2.55, 2.15, 2.15], avail=7.20)
    column(s, 'HOW MUCH IT COUNTS', [
        ('', 'The only unconfounded comparison in the deck. Every other table is on a domain v3 '
             'trained on and v2 did not.'),
        ('', 'Consistent rather than large: 0.004 to 0.029 px, over 40 pairs and one seed.'),
        ('', 'v3 is still worse on large-motion failures, 7.2% against 6.5%.'),
    ])
    cite(s, 'Mean end-point error, px. v2 trained on FlyingThings; v3 on FlyingChairs, VKITTI 2 '
            'and Sintel. Neither trained on aerial imagery.')
    note(s, "One result I did not expect, and I am putting it last in the evidence because it is "
            "the cleanest. Everything else in this deck is measured on driving sequences, and v3 "
            "trained on driving data while v2 did not, which is a confound I have flagged "
            "throughout. Aerial data is neutral ground: neither model has seen anything like it. "
            "On that neutral ground v3 comes out ahead at all seven margin settings. Appendix E "
            "has the same three situations from the opening slide in that domain, with the region "
            "geometry and decoder configuration held identical, if anyone wants to see them. The "
            "likely reason v3 leads is that the decoder saw three datasets where v2 "
            "saw one, and the convex head is bounded so it cannot invent motion its inputs do not "
            "support, which should degrade more gracefully out of domain. I want to be careful "
            "about how much weight this carries: the margins are small, it is forty pairs and one "
            "seed, and v3 is still slightly worse on the large-motion pixels. What makes it worth "
            "showing is that it is consistent across seven independent settings.")

    # ============================================================= 16 discussion
    s, i = slide()
    action_title(s, 'The decoder earns its place when the\nconsumer chooses its own points')
    text(s, MARGIN, CONTENT_Y, 5.6, 0.28, 'USE THE DECODER WHEN', F_SMALL, True, MID, tight=True)
    bullets(s, MARGIN, CONTENT_Y + 0.38, 5.6, 2.6, [
        ('', 'The consumer selects its own query points.'),
        ('', 'A frame is revisited after it has been processed.'),
        ('', 'Each correspondence needs a confidence value.'),
        ('', 'A position between pixels is required.'),
    ], F_COL)
    vrule(s, 6.85, CONTENT_Y, 3.0, pt=0.75, color=RULE)
    text(s, 7.15, CONTENT_Y, 5.43, 0.28, 'USE THE BASELINE WHEN', F_SMALL, True, MID, tight=True)
    bullets(s, 7.15, CONTENT_Y + 0.38, 5.43, 2.6, [
        ('', 'One complete dense field per frame is consumed, and nothing else is needed.'),
        ('', 'v2 is more accurate and 10% faster in exactly that mode, and remains the correct '
             'choice for it.'),
    ], F_COL)
    panel(s, MARGIN, 5.32, CW, 0.90,
          'Selective prediction reframes the comparison: mean error over all pixels is the metric '
          'the decoder loses on, and the only one a model without confidence can report.',
          F_BODY, PP_ALIGN.CENTER)
    note(s, "So what is the contribution. It is an operating point, not a leaderboard position. "
            "For an application that consumes one complete dense field per frame and nothing "
            "else, v2 remains the correct choice and I will say so plainly: it is more accurate "
            "and ten percent faster in that mode. The decoder earns its place when the consumer "
            "picks its own points, revisits a frame, needs a position between pixels, or needs a "
            "confidence value. And the selective prediction result reframes the whole accuracy "
            "comparison, because mean error over all pixels is exactly the metric a model "
            "without confidence is stuck with.")

    # ============================================================= 17 limitations
    s, i = slide()
    action_title(s, 'The accuracy gap has a single cause, and\nthree experiments converge on it')
    bullets(s, MARGIN, CONTENT_Y, CW, 3.0, [
        ('Accuracy. ', 'The decoder’s finest input is 1/8 resolution; the upsampler it replaces '
                       'reads the full-resolution frame.'),
        ('Sub-pixel querying. ', 'For the same reason, evaluation between pixels is presently '
                                 'closer to interpolation than inference. The interface is exact; '
                                 'the extra information recoverable is small.'),
        ('Embedded validation. ', 'All latency is from a laptop GPU. The edge premise is a design '
                                  'target, not a result.'),
        ('Statistical resolution. ', 'One seed per configuration. Differences below about 0.05 px '
                                     'are not resolved by this study.'),
    ], F_COL, space_after=11)
    panel(s, MARGIN, 5.40, CW, 0.82,
          'Three independent nulls point at the same mechanism: Fourier encoding, sub-pixel '
          'querying, and reduced-resolution querying. The remedy is a full-resolution stem.',
          F_BODY, PP_ALIGN.CENTER)
    note(s, "Limitations, stated plainly rather than buried. The accuracy gap is structural: the "
            "decoder never sees full-resolution features. Three separate experiments returned "
            "null for the same reason, which is what makes me confident in the diagnosis rather "
            "than merely suspicious. Sub-pixel querying is thinner than it sounds for the same "
            "reason: the interface is exact but the extra information recoverable between pixels "
            "is small until the decoder gets full-resolution input. There is no embedded "
            "measurement at all, so the word edge is a design target. And one seed per "
            "configuration, so please do not read much into differences below about five "
            "hundredths of a pixel.")

    # ============================================================= 18 conclusions
    s, i = slide()
    text(s, MARGIN, TITLE_TOP, CW, 0.6, 'Conclusions', F_TITLE, True, INK, tight=True)
    hrule(s, RULE_Y, MARGIN, CW, pt=3.5, color=INK)
    bullets(s, MARGIN, 1.95, CW, 3.4, [
        ('1.  Queryable output is affordable. ',
         'Arbitrary continuous coordinates in O(N), for 2.6% of mean accuracy, from a model 13% '
         'smaller.'),
        ('2.  Confidence is worth more than the mean. ',
         'Abstaining on 20% of queries gives 2.2x lower error over the remaining 80% of the frame.'),
        ('3.  Region margins are predictable in advance. ',
         'Margin ≈ speed / frame rate, verified across a threefold difference in motion scale.'),
    ], F_COL, space_after=20)
    hrule(s, 5.65)
    text(s, MARGIN, 5.85, CW, 0.7,
         'Next: a full-resolution feature stem, then embedded measurement.\n'
         'srinivasan.shrim@northeastern.edu   |   Code and report: github.com/shrirag10/Neuflowv3',
         F_CITE, False, MUTED, 1.4, tight=True)
    note(s, "Three conclusions. Queryable output is affordable: arbitrary continuous coordinates "
            "for 2.6 percent of mean accuracy, from a smaller model. Confidence is worth more "
            "than the mean, because abstaining on a fifth of queries more than halves the error "
            "on the rest. And the region margin a fast platform needs is predictable from speed "
            "and frame rate before any flow is computed. The next experiment is the "
            "full-resolution stem, which is the direct attack on the one gap I have not closed. "
            "I will stop there. Are there questions or feedback?")

    # ============================================================= 19 references
    s, i = slide()
    text(s, MARGIN, TITLE_TOP, CW, 0.6, 'References', F_TITLE, True, INK, tight=True)
    hrule(s, RULE_Y)
    refs = [
        'Zhang, Z., Gupta, A., Jiang, H., & Singh, H. (2024). NeuFlow v2: High-Efficiency Optical '
        'Flow Estimation on Edge Devices. arXiv:2408.10161.',
        'Jung, H., Hui, Z., Luo, L., Yang, H., Liu, F., Yoo, S., Ranjan, R., & Demandolx, D. (2023). '
        'AnyFlow: Arbitrary Scale Optical Flow with Implicit Neural Representation. CVPR.',
        'Teed, Z., & Deng, J. (2020). RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. ECCV.',
        'Yu et al. (2025). InfiniDepth: Implicit Neural Upsampling for Monocular Depth Estimation.',
        'Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for '
        'Computer Vision? NeurIPS.',
        'Cabon, Y., Murray, N., & Humenberger, M. (2020). Virtual KITTI 2. arXiv:2001.10773.',
        'Wang, W., Zhu, D., Wang, X., Hu, Y., Qiu, Y., Wang, C., Hu, Y., Kapoor, A., & Scherer, S. '
        '(2020). TartanAir: A Dataset to Push the Limits of Visual SLAM. IROS.',
    ]
    y = 1.90
    for r in refs:
        text(s, MARGIN, y, CW, 0.5, r, F_CITE, False, INK, 1.20)
        y += 0.60
    note(s, "The work this builds on. NeuFlow v2 is the baseline and the frozen front end, and "
            "it comes from this lab. AnyFlow is where the convex-weight formulation comes from, "
            "the part that stops the decoder predicting motion its inputs do not support. "
            "InfiniDepth is where the gated multi-scale fusion comes from. Kendall and Gal is the "
            "uncertainty loss. Virtual KITTI 2 and TartanAir are the two evaluation datasets.")

    # ============================================================= A1 interface
    s, i = slide(label='Appendix A:  query cost and interface')
    action_title(s, 'Decode cost is flat to 2,048 queries:\n2,048 points cost what 800 do')
    exhibit(s, f'{PLOTS}/decode_flat.png', 2.05)
    text(s, COL_X, CONTENT_Y, COL_W, 0.28, 'THE ENTIRE API', F_SMALL, True, MID, tight=True)
    panel(s, COL_X, 2.22, COL_W, 1.85,
          'state = model.infer_coarse_state(\n            img0, img1)\n\n'
          'flow  = model.decode_queries(\n            state, query_coords=q)\n\n'
          'flow, b = ... return_uncertainty=True',
          11, PP_ALIGN.LEFT, spacing=1.14)
    bullets(s, COL_X, 4.30, COL_W, 1.4, [
        ('', 'Launch-overhead bound below roughly 10,000 points, compute-bound above, where '
             'dense decoding is the better choice.'),
    ], F_SMALL)
    cite(s, '1.254 ms at N = 800; 1.285 ms at N = 2,048. RTX 4060 laptop, fp16.')
    note(s, "The interface is two calls: one coarse pass per frame pair, then as many decode "
            "calls as you like against it. The plot shows decode cost is flat between 800 and "
            "2,048 queries, which tells you it is dominated by kernel launch overhead rather "
            "than arithmetic, so you get two thousand points for the price of eight hundred. "
            "Above about ten thousand points it becomes compute bound and dense mode is the "
            "better choice.")

    # ============================================================= A2 tool
    s, i = slide(label='Appendix B:  working implementation')
    action_title(s, 'The decoder runs as an interactive tool:\ndraw a region, get flow there')
    exhibit(s, f'{VISUALS}/region_gui_window.png', 2.05)
    text(s, COL_X, CONTENT_Y, COL_W, 0.28, 'TWO TIMED MODES', F_SMALL, True, MID, tight=True)
    bullets(s, COL_X, CONTENT_Y + 0.40, COL_W, 3.0, [
        ('QUERY ', 'decodes only inside the box against a full-frame coarse pass. Exact, and '
                   'identical to the full-frame result inside the box.'),
        ('CROP ', 'feeds only the box through the pipeline, so cost scales with area, at the '
                  'price of the lost context.'),
    ], F_COL)
    cite(s, 'The panel reports the measured breakdown live, including where the decoder is not '
            'the faster option.')
    note(s, "This is a working tool, not a mock-up. Load a video, step to a frame, drag a box, "
            "and it computes flow in that box. Query mode runs the full coarse pass and decodes "
            "only inside the box, which is exact. Crop mode feeds just the box through the whole "
            "pipeline, so cost scales with the area requested but it loses the surrounding "
            "context that global matching uses. The panel shows the actual breakdown live, "
            "including the parts where v3 is not the faster option.")

    # ============================================================= A3 scorecard
    s, i = slide(label='Appendix C:  objectives set at the start')
    action_title(s, 'Two of three original objectives were not met;\nthe project delivers three others')
    table(s, MARGIN, CONTENT_Y, [
        ['Objective', 'Verdict', 'Evidence'],
        ['Better accuracy than v2', 'not met', 'best 2.384 against 2.324; like-for-like 7.6% behind'],
        ['Less compute than v2', 'partly', 'dense 10% slower; repeat query 27x cheaper'],
        ['Runs on edge devices', 'unproven', 'every figure is from a laptop RTX 4060'],
    ], [3.40, 1.55, 6.88], F_SMALL)
    text(s, MARGIN, 3.75, CW, 0.28, 'WHAT IT DOES DELIVER', F_SMALL, True, MID, tight=True)
    for k, (t, d) in enumerate([
        ('Queryable output', 'any continuous coordinate;\nsparse equals dense exactly'),
        ('Cheap repeat access', '1.25 ms per further query\nbatch on a cached frame'),
        ('Calibrated confidence', 'a per-query error estimate,\nmonotonic against real error'),
    ]):
        panel(s, MARGIN + k * 4.03, 4.18, 3.78, 1.45, f'{t}\n\n{d}', F_SMALL,
              PP_ALIGN.CENTER)
    cite(s, 'Sparse queries reproduce the dense field to 0.00057 px maximum deviation.')
    note(s, "The scorecard against what I set out to do, and two of three are not met. Better "
            "accuracy: not met. Less compute: partly, dense is slower, a first query is level, "
            "but repeat queries are twenty-seven times cheaper. Edge capable: unproven, "
            "everything is on a laptop 4060. What the project does deliver is the three things "
            "at the bottom, and those are measured.")

    # ============================================================= A4 questions
    s, i = slide(label='Appendix D:  anticipated questions')
    text(s, MARGIN, TITLE_TOP, CW, 0.6, 'Four questions I expect, answered',
         F_TITLE, True, INK, tight=True)
    hrule(s, RULE_Y)
    qa = [
        ('Why is flow defined between pixels at all?',
         'Motion is continuous and the pixel grid is a sampling artefact. A corner tracked to '
         '(312.7, 188.2) is a real answer, and registration consumes it directly.'),
        ('Why freeze the backbone?',
         'To make the decoder the only variable. Any difference between v2 and v3 is then '
         'attributable to the output stage and nothing else.'),
        ('Why is dense v3 slower than v2?',
         'v2 upsamples with one convolution that shares work across neighbouring pixels. v3 '
         'answers each query independently, which gives up that sharing.'),
        ('Is 2.6 percent worth the added complexity?',
         'Not on its own. It is worth it if you need queryability, cheap repeat access, or a '
         'confidence value. For plain dense flow, v2 remains the better choice.'),
    ]
    y = 1.95
    for q, a in qa:
        text(s, MARGIN, y, CW, 0.28, q, F_COL, True, INK, tight=True)
        text(s, MARGIN + 0.28, y + 0.32, CW - 0.28, 0.55, a, F_SMALL, False, INK,
             1.20, tight=True)
        y += 1.18
    note(s, "Four questions I expect, with the answers on a slide rather than recalled under "
            "pressure. The first is the one I most want to get right: the pixel grid is a "
            "sampling decision, not a property of the motion, so asking for a position between "
            "pixels is a well-posed request even though the current decoder answers it less fully "
            "than I would like. The last one is the honest summary of the whole project: 2.6 "
            "percent is not worth paying for nothing, and it is worth paying for the three "
            "capabilities on the scorecard.")

    # ============================================================= A5 aerial
    s, i = slide(label='Appendix E:  the three situations in the aerial domain')
    action_title(s, 'Same region geometry, same decoder, only\nthe domain changes')
    picture(s, f'{MONTAGE}/scenarios_tartanair_tight.png', MARGIN, 1.86, h=4.62)
    bullets(s, 6.10, 2.10, 6.48, 3.4, [
        ('What is held fixed. ', 'Region geometry, margin, decode stride and checkpoint are '
                                 'identical to the driving figure on slide 2.'),
        ('What changes. ', 'Mean inter-frame motion falls from 26.6 px to 9.26 px, about a third.'),
        ('What to notice. ', 'Flow inside each region is much closer to uniform than in the '
                             'driving frames, because a drone rotating through a static scene '
                             'produces near-constant displacement across a small window.'),
        ('Why it matters. ', 'No model in this study trained on this imagery, which is what makes '
                             'the head-to-head on slide 15 worth having.'),
    ], F_SMALL, space_after=13)
    cite(s, 'TartanAir aerial sequences. Flow colour encodes direction, so these two panels stay '
            'in colour where the rest of the deck does not.')
    note(s, "The same three situations on aerial data, held here rather than in the main line "
            "because it is context rather than a claim. Region geometry and decoder configuration "
            "are identical to the driving figure; only the domain changes. Two things to notice. "
            "The flow inside each region is much closer to uniform than in the driving frames, "
            "because a drone rotating through a static scene produces near-constant displacement "
            "over a small window. And no model in this study trained on this imagery, which is "
            "exactly what makes the seven-margin comparison on slide fifteen the cleanest number "
            "in the deck.")

    finish(prs, OUT)


if __name__ == '__main__':
    main()
