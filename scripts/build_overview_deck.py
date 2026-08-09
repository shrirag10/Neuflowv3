"""Build docs/NeuFlow_v3_Overview.pptx.

A third deck, kept separate from the defence and status decks rather than
replacing either. This one is built to a different brief: every point is a
bullet, the text budget per slide is small and enforced, and anything that can
be drawn is drawn instead of written. Where the other decks explain a mechanism
in a paragraph, this one puts a flowchart on the slide and leaves the paragraph
in the speaker notes, which is where a paragraph belongs.

Two rules are enforced at build time rather than trusted:

  Word budget. No slide face may exceed WORD_BUDGET words outside its tables.
  Prose blocks are what made the earlier decks feel like documents, and a
  budget is the only thing that reliably stops them coming back.

  Bullets. Points are set with `bullet_list`, which draws a marker and indents
  the wrapped lines to clear it, so a list reads as a list rather than as a
  paragraph with bold bits in it.

The domain comparison on slide 10 needs results/plots/deck/domain_comparison.png
from `python3 scripts/make_domain_comparison.py`.

Layout, palette and every number come from deck_style.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Inches
from deck_style import (
    INK, MID, MUTED, WHITE, FACTS,
    F_TITLE, F_BODY, F_COL, F_SMALL, F_CITE,
    W, MARGIN, CW, TITLE_TOP, RULE_Y, CONTENT_Y, CITE_Y,
    FIG_W, COL_X, COL_W, PLOTS,
    text, hrule, vrule, action_title, cite, note, panel,
    keyline, picture, exhibit, bullet_list, chain, reuse_diagram,
    img_aspect, new_deck, finish,
)

OUT = 'docs/NeuFlow_v3_Overview.pptx'
F = FACTS
WORD_BUDGET = 62


def body_words(sl):
    """Words of running prose on a slide face.

    Counts text boxes only, and only above the citation line. A label inside a
    flowchart node is not prose and should not be penalised, since drawing the
    thing instead of writing it is the point; and the citation and page number
    are fine print nobody reads from the back of a room.
    """
    return sum(len(sh.text_frame.text.split()) for sh in sl.shapes
               if sh.shape_type == MSO_SHAPE_TYPE.TEXT_BOX
               and sh.top < Inches(CITE_Y - 0.1))


def budget(prs):
    """Refuse to ship a slide that has quietly turned back into a document."""
    counts = [body_words(sl) for sl in prs.slides]
    over = [(k + 1, n) for k, n in enumerate(counts) if n > WORD_BUDGET]
    assert not over, f'over the {WORD_BUDGET}-word budget: {over}'
    print('   body words per slide: ' + ' '.join(str(n) for n in counts))


def main():
    prs, slide = new_deck()

    # ============================================================= 1  title
    s, i = slide(footer=False)
    text(s, MARGIN, 2.05, CW, 1.3, 'NeuFlow v3', 44, True, INK, 1.10, tight=True)
    text(s, MARGIN, 3.10, CW, 1.1,
         'Optical flow you can query,\nwith a confidence value attached',
         26, False, INK, 1.26, tight=True)
    hrule(s, 4.52, MARGIN, 2.6, pt=3.5, color=INK)
    text(s, MARGIN, 4.85, CW, 0.9,
         'Shriman Raghav Srinivasan\nMS Robotics, Northeastern University   |   '
         'Field Robotics Lab',
         F_COL, False, INK, 1.36, tight=True)
    note(s, "I replaced NeuFlow v2's fixed upsampler with a decoder you can query at any "
            "coordinate. This version of the deck is deliberately visual: the argument is in "
            "the diagrams and the figures, and I will carry the detail verbally. Three things "
            "to take away. It costs 2.6 percent of mean accuracy. It buys three capabilities "
            "the baseline cannot express. And there is one limitation I have diagnosed but not "
            "closed.")

    # ============================================================= 2  objectives
    s, i = slide()
    action_title(s, 'The three objectives set at the start')
    bullet_list(s, MARGIN, 2.35, CW, [
        ('Better accuracy ', 'than NeuFlow v2'),
        ('Less compute ', 'than NeuFlow v2'),
        ('Runs on edge devices', ''),
    ], 26, gap=0.52, indent=0.42, marker=0.105)
    note(s, "These are the three objectives I wrote down at the start, and I am putting them up "
            "on their own so they are fixed in your mind before any results. Better accuracy "
            "than v2. Less compute than v2. And runs on an edge device. I will come back to "
            "every one of them at the end and tell you plainly which were met, because two of "
            "them were not, and the honest answer to that is more interesting than the "
            "objectives themselves.")

    # ============================================================= 3  the problem
    s, i = slide()
    action_title(s, 'A fast platform cannot afford the frame\nit is given')
    chain(s, 2.15, [('Camera\nframe pair', False), ('Dense flow\nevery pixel', False),
                    ('479,232\nvectors', True), ('Consumer\nuses ~800', False),
                    ('478,432\ndiscarded', False)], h=1.05, size=13.5)
    bullet_list(s, MARGIN, 3.80, CW, [
        ('', 'Registration, tracking and mosaicking each pick their own few hundred points.'),
        ('', 'Flow between pixels cannot be requested at all.'),
        ('', 'A second question about the same frame costs the whole computation again.'),
    ], F_BODY)
    cite(s, f'Frame size {F["frame"]}. Baseline: Zhang, Gupta, Jiang and Singh (2024), '
            f'NeuFlow v2.')
    note(s, "The situation, before the network. A drone or a fast vehicle has a millisecond "
            "budget per frame, and a dense flow field does not fit inside it. The chain at the "
            "top is what actually happens: the network computes four hundred and seventy-nine "
            "thousand vectors and the consumer uses about eight hundred of them. Two things "
            "follow from the fixed dense contract, and they are the two bullets underneath. You "
            "cannot ask for a position between pixels, and you cannot ask a second question "
            "about a frame the network has already processed without paying for all of it "
            "again.")

    # ============================================================= 4  what changed
    s, i = slide()
    action_title(s, 'Only the last stage changes')
    text(s, MARGIN, 2.05, CW, 0.26,
         f'PHASE 1   once per frame pair, cached', F_SMALL, True, MID, tight=True)
    chain(s, 2.38, [('Image pair', False), ('CNN backbone\n1/8 and 1/16', False),
                    ('Cross-attention\nglobal match', False),
                    ('Refinement\n1x s16, 8x s8', False),
                    ('Coarse flow\nat 1/8', True)], h=0.98, size=12.5)
    text(s, MARGIN, 3.72, CW, 0.26,
         f'PHASE 2   once per query batch, O(N)', F_SMALL, True, MID, tight=True)
    chain(s, 4.05, [('Query (x, y)\ncontinuous', False), ('Sample 3x3\nfour sources', False),
                    ('Gated fusion\nMLP', False), ('Convex blend\nnine plus one', False),
                    ('Flow and\nconfidence', True)], h=0.98, size=12.5)
    bullet_list(s, MARGIN, 5.42, CW, [
        ('', 'Everything in Phase 1 is frozen and verified bit-identical to v2.'),
        ('', 'A convex blend cannot predict motion its inputs do not support.'),
    ], F_COL)
    cite(s, 'Convex weights follow AnyFlow (2023). Gated fusion follows InfiniDepth (2026).')
    note(s, "What actually changed, in one picture. The top chain is v2 exactly as published, up "
            "to its one-eighth resolution coarse flow. All 137 tensors shared with v2 are "
            "verified bit-identical per checkpoint, so any difference you see later is the "
            "decoder and nothing else. The bottom chain is the new part: a coordinate goes in, "
            "and out comes a convex blend of the nine neighbouring coarse-flow values plus a "
            "bilinear sample. Because the blend is convex it cannot hallucinate motion, and "
            "zero-initialised it is exactly bilinear upsampling, so it starts from a known-good "
            "operating point.")

    # ============================================================= 5  state reuse
    s, i = slide()
    action_title(s, 'Three questions about one frame')
    reuse_diagram(s, 2.35)
    cite(s, f'{F["device"]}, fp16, {F["frame"]}. Per-call latencies measured; totals are their '
            f'sum.')
    note(s, "This is the single most important slide in the deck, and it is why the decoder "
            "exists. Suppose the platform asks three questions about one frame: a tracker, then "
            "a second region, then a refinement. The baseline keeps nothing between calls, so "
            "each answer costs a full pass, three times thirty-three milliseconds. v3 pays the "
            "coarse pass once, caches it, and each further answer is a decode at one and a "
            "quarter milliseconds. Ninety-nine point nine against thirty-seven. That gap is "
            "structural rather than a margin, and it widens with every extra question.")

    # ============================================================= 6  accuracy
    s, i = slide()
    action_title(s, f'The price: {F["gap_pct"]} of mean error')
    fb = exhibit(s, f'{PLOTS}/accuracy_bars.png', 2.05)
    text(s, COL_X, CONTENT_Y + 0.18, COL_W, 0.28, 'WHAT THIS SHOWS', F_SMALL, True, MID,
         tight=True)
    bullet_list(s, COL_X, 2.56, COL_W, [
        ('', 'Every configuration sits above v2.'),
        ('', f'Like-for-like: {F["v3_epe_chairs"]}, or {F["gap_pct_like"]} behind.'),
        ('', f'{F["v3_params"]} against {F["v2_params"]} parameters.'),
    ], F_COL)
    keyline(s, COL_X, fb - 0.72, COL_W, f'{F["v3_epe"]}\nagainst {F["v2_epe"]}')
    cite(s, f'Mean end-point error per pixel, {F["pairs"]} held-out VKITTI 2 pairs. Lower is '
            f'better.')
    note(s, "The cost, and I would rather lead with it than have you find it. Every v3 "
            "configuration is less accurate than the baseline on mean end-point error. The best "
            "is 2.384 against 2.324, so 2.6 percent worse. The fair comparison is the "
            "like-for-like one, trained on FlyingChairs alone as v2 was trained on FlyingThings "
            "alone, and that is 7.6 percent behind. The model is thirteen percent smaller, "
            "which is worth something but does not excuse it. I know the mechanism and it is on "
            "the open-problem slide.")

    # ============================================================= 7  speed
    s, i = slide()
    action_title(s, f'A repeat query costs {F["v3_repeat_ms"]} ms')
    fb = exhibit(s, f'{PLOTS}/speed_bars.png', 2.05)
    text(s, COL_X, CONTENT_Y + 0.18, COL_W, 0.28, 'WHICH CLAIM I MAKE', F_SMALL, True, MID,
         tight=True)
    bullet_list(s, COL_X, 2.56, COL_W, [
        ('', f'Dense output is {F["dense_penalty"]} slower.'),
        ('', f'A first query is level: {F["coarse_share"]} of it is the shared coarse pass.'),
        ('', 'The win is the second question.'),
    ], F_COL)
    keyline(s, COL_X, fb - 0.42, COL_W, f'{F["repeat_speedup"]} on repeat access')
    cite(s, f'{F["device"]}, fp16, {F["frame"]}, median of 50 runs after warm-up.')
    note(s, "On speed there are three numbers here and only one of them is a win, so I want to "
            "be precise about which claim I am making. Dense output is ten percent slower than "
            "v2, and that mode is not what the decoder is for. A first sparse query is level "
            "with v2, because ninety-six percent of the cost is the coarse pass and both models "
            "pay it. The win is repeat access, which is the diagram from two slides ago.")

    # ============================================================= 8  confidence
    s, i = slide()
    action_title(s, 'Confidence buys the accuracy back')
    fb = exhibit(s, f'{PLOTS}/selective_accuracy.png', 2.05)
    text(s, COL_X, CONTENT_Y + 0.18, COL_W, 0.28, 'WHAT THIS SHOWS', F_SMALL, True, MID,
         tight=True)
    bullet_list(s, COL_X, 2.56, COL_W, [
        ('', f'Error rises monotonically across five bins, {F["bins"]}.'),
        ('', 'v2 has one operating point; v3 has a curve.'),
        ('', 'Abstaining is free when the consumer picks its own points.'),
    ], F_COL)
    keyline(s, COL_X, fb - 0.72, COL_W, f'{F["sel_80"]} px over\n80% of the frame')
    cite(s, f'Per-query error scale under a Laplace likelihood, {F["queries"]} queries. '
            f'Coverage is the share answered.')
    note(s, "The head predicts an error scale for every query alongside the flow, trained under "
            "a Laplace likelihood. Binned by predicted uncertainty, actual error rises "
            "monotonically across all five bins, a fifteen-fold span over 2.35 million queries. "
            "What that buys is the curve. Mean error over all pixels is one operating point and "
            "it is the only one v2 has, because it cannot tell you which of its outputs to "
            "trust. Discard the least confident fifth and error on the remainder is 1.06 pixels. "
            "For registration you are choosing a few hundred points anyway, so declining to "
            "answer costs you nothing you would have used.")

    # ============================================================= 9  region
    s, i = slide()
    action_title(s, f'Flow {F["roi_area"]} of the frame, not all of it')
    chain(s, 2.20, [('Full frame\n100%, 33.3 ms', False), ('Region only\n7.9%, 4.3x', False),
                    ('Region + margin\n13.8%, 4.4x', True),
                    ('Wider margin\n34.2%, 3.4x', False)], h=1.05, size=13.5)
    bullet_list(s, MARGIN, 3.85, CW, [
        ('', f'Processing {F["roi_area"]} of the frame costs {F["roi_cost"]} and runs '
             f'{F["roi_speedup"]} faster.'),
        ('', 'Without a margin, error rises 65% and a quarter of large-motion pixels fail.'),
        ('', 'Design rule: margin ≈ expected motion ≈ speed / frame rate.'),
    ], F_BODY)
    cite(s, 'Applies to any network in this family, v2 included. A platform technique, not a '
            'property of the decoder.')
    note(s, "This is the enabling result for the whole fast-platform framing. Keep full "
            "resolution, process fourteen percent of the frame, and it costs thirty-four "
            "thousandths of a pixel while running four and a half times faster. The margin is "
            "the part worth dwelling on. Without one, error jumps sixty-five percent and a "
            "quarter of the large-motion pixels fail completely, because this network locates "
            "large displacement using attention over the whole image and a tight crop takes "
            "that away. I should be clear that this applies to any flow network including v2 "
            "unchanged.")

    # ============================================================= 10 both domains
    s, i = slide()
    action_title(s, 'The same three situations, two domains')
    fig = f'{PLOTS}/domain_comparison.png'
    fw = 10.85
    fy = 2.02
    picture(s, fig, (W - fw) / 2, fy, w=fw)
    bullet_list(s, MARGIN, fy + fw / img_aspect(fig) + 0.24, CW, [
        ('', 'Region geometry, margin and decoder configuration are identical across both.'),
        ('', 'Only the domain changes, and no model in this study trained on aerial imagery.'),
    ], F_COL)
    cite(s, 'Left: Virtual KITTI 2. Right: TartanAir. Flow colour encodes direction.')
    note(s, "The three situations, run on both datasets with the region geometry and the decoder "
            "configuration held identical. Only the domain changes. Reading across: the first "
            "row is a region entering the field of view, the second is the platform turning so "
            "that region now overlaps a second object, the third is a new object appearing in a "
            "frame already being processed. Two things to notice. The grey area is never "
            "computed at all, in either domain. And the flow inside each aerial region is much "
            "closer to uniform than in the driving frames, because a drone rotating through a "
            "static scene produces near-constant displacement across a small window.")

    # ============================================================= 11 margin rule
    s, i = slide()
    action_title(s, 'Margin is set by motion, not image size')
    fb = exhibit(s, f'{PLOTS}/margin_rule.png', 2.02, w=9.9, x=(W - 9.9) / 2)
    bullet_list(s, MARGIN, fb + 0.24, CW, [
        ('Driving. ', f'{F["motion_drive"]} mean motion, needs about {F["margin_drive"]}.'),
        ('Aerial. ', f'{F["motion_air"]} mean motion, needs about {F["margin_air"]}.'),
    ], F_COL)
    cite(s, 'Measured with the baseline network. Predicts the scale, not the exact curve.')
    note(s, "I wanted to know whether the margin rule was fitted to one dataset or was something "
            "more general, so I tested it on aerial sequences, which move about a third as fast. "
            "The prediction was that the required margin scales with motion, so the knee should "
            "move from thirty-two pixels down to about nine. It did. On the right, divided by "
            "mean motion, both domains start at the same penalty and both have shed most of it "
            "by the point where the margin equals one frame of motion. To be precise about the "
            "limit, this predicts the scale rather than the exact curve, because residuals "
            "beyond a ratio of one differ by scene. It is still enough to size a margin in "
            "advance from speed and frame rate.")

    # ============================================================= 12 scenario 3
    s, i = slide()
    action_title(s, f'A region found mid-frame costs {F["s3_sparse_ms"]} ms')
    fb = exhibit(s, f'{PLOTS}/scenario3_marginal.png', 2.02, w=9.9, x=(W - 9.9) / 2)
    bullet_list(s, MARGIN, fb + 0.24, CW, [
        ('', f'A fresh crop runs a whole new pass at {F["s3_crop_ms"]} ms, and is less '
             f'accurate doing it.'),
        ('', 'Two disjoint crops cost more than one full frame: crop once, or not at all.'),
    ], F_COL)
    cite(s, 'Held-out VKITTI 2. The second region is disjoint, so a cropped pipeline can reuse '
            'nothing.')
    note(s, "This is the one result specific to the architecture rather than to cropping, and it "
            "is scenario three from the domain slide. A frame is already in flight when a new "
            "object enters the field of view. Because the coarse pass is cached, answering costs "
            "one point seven milliseconds. A cropped pipeline cannot reuse anything, since the "
            "new object is outside its box, so it runs an entire new pass at seven point three "
            "milliseconds and the answer is worse, three point six pixels against two point one, "
            "because a fresh crop has lost the surrounding context. This buys the ability to "
            "answer a question you did not know you would have when the frame arrived.")

    # ============================================================= 13 open problem
    s, i = slide()
    action_title(s, 'One cause, three experiments converging')
    chain(s, 2.25, [('Full-resolution\nframe', False), ('Backbone\n1/8 and 1/16', False),
                    ('Decoder sees\n1/8 only', True), ('Evidence flat\nin an 8x8 cell', False),
                    ('Sub-pixel gain\nis small', False)], h=1.08, size=13)
    bullet_list(s, MARGIN, 3.92, CW, [
        ('', 'Fourier positional encoding, sub-pixel querying and reduced-resolution querying '
             'all returned null, and one mechanism explains all three.'),
        ('Remedy. ', 'A full-resolution stem, roughly 1 to 2 ms. Not yet built.'),
    ], F_BODY)
    cite(s, f'Also open: no embedded measurement, one evaluation domain, and {F["seed_note"]}, '
            f'so differences below {F["resolution"]} are not resolved.')
    note(s, "This is the part I have not solved, and it is one mechanism rather than a list of "
            "unrelated weaknesses. The decoder never sees full-resolution features. Its finest "
            "input is at one-eighth resolution, so inside an eight by eight cell it is looking "
            "at almost the same evidence wherever you query, while the upsampler it replaces "
            "reads the full-resolution frame directly. That single fact explains the accuracy "
            "gap and it explains why three separate experiments returned null. Three nulls with "
            "one explanation is what makes me confident in the diagnosis rather than merely "
            "suspicious. The remedy is a cheap full-resolution stem, and it is the only "
            "candidate on my list that is specific to this architecture.")

    # ============================================================= 14 scorecard
    s, i = slide()
    action_title(s, 'Back to the three objectives')
    bullet_list(s, MARGIN, 2.15, CW, [
        ('Better accuracy than v2.  ', f'Not met. {F["v3_epe"]} against {F["v2_epe"]}.'),
        ('Less compute than v2.  ', f'Partly. Dense slower, repeat query {F["repeat_speedup"]} '
                                    f'cheaper.'),
        ('Runs on edge devices.  ', 'Unproven. Every figure is from a laptop GPU.'),
    ], F_COL, gap=0.26)
    text(s, MARGIN, 4.25, CW, 0.28, 'DELIVERED INSTEAD, AND MEASURED', F_SMALL, True, MID,
         tight=True)
    for k, (t, d) in enumerate([
        ('Queryable output', 'any continuous\ncoordinate'),
        ('Cheap repeat access', f'{F["v3_repeat_ms"]} ms per\nfurther query'),
        ('Calibrated confidence', 'a per-query\nerror estimate'),
    ]):
        panel(s, MARGIN + k * 4.03, 4.68, 3.78, 1.32, f'{t}\n\n{d}', F_COL,
              PP_ALIGN.CENTER)
    cite(s, 'Next: a full-resolution stem, then embedded measurement.')
    note(s, "Back to the three objectives from slide two, and the honest answer. Better accuracy: "
            "not met. Less compute: partly, and it depends entirely on which mode, because dense "
            "is slower while repeat queries are twenty-seven times cheaper. Edge capable: "
            "unproven, because every number I have is from a laptop GPU. What the project "
            "delivered instead is the three things at the bottom, none of which were on the "
            "original list, and all three are measured rather than asserted. The next experiment "
            "is the full-resolution stem, which is the direct attack on the one gap I have not "
            "closed.")

    budget(prs)
    finish(prs, OUT)


if __name__ == '__main__':
    main()
