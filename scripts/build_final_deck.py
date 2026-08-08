"""Build docs/NeuFlow_v3_status.pptx, the advisor status deck.

A different deck from the defence one, not a shorter cut of it. The defence
deck argues a case to people meeting the work for the first time, so it spends
its first four slides establishing why the question is worth asking. An advisor
already knows that. What an advisor wants is the standing of the project, the
evidence behind it, what is still open, and where their judgement would change
what happens next.

So this deck inverts the order: the scorecard is slide 2 rather than an
appendix, the evidence is compressed to the seven results that carry weight,
and it ends on the two slides the defence deck does not have, next steps with
costs attached and the specific questions where a decision is wanted.

Results forward, no development chronology: the debugging history lives in
docs/V3DEV_LOG.md and is not what a status meeting is for.

Layout, palette and every number come from deck_style, so this deck and the
defence deck cannot disagree. Deck figures come from
`DECK_FIGS=1 python3 scripts/make_final_plots.py`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pptx.enum.text import PP_ALIGN
from deck_style import (
    INK, MID, MUTED, PANEL, WHITE, FACTS,
    F_TITLE, F_BODY, F_COL, F_TABLE, F_SMALL, F_CITE,
    W, MARGIN, CW, TITLE_TOP, RULE_Y, CONTENT_Y, CITE_Y,
    FIG_W, COL_X, COL_W, PLOTS, MONTAGE,
    text, bullets, hrule, vrule, action_title, cite, note, panel,
    keyline, picture, exhibit, column, table, width_in,
    new_deck, finish,
)

OUT = 'docs/NeuFlow_v3_status.pptx'
F = FACTS


def main():
    prs, slide = new_deck()

    # ============================================================= 1  title
    s, i = slide(footer=False)
    text(s, MARGIN, 1.70, CW, 1.5,
         'NeuFlow v3', 40, True, INK, 1.10, tight=True)
    text(s, MARGIN, 2.62, CW, 1.2,
         'A queryable, uncertainty-aware flow decoder\nfor edge robotics',
         26, False, INK, 1.24, tight=True)
    hrule(s, 4.10, MARGIN, 2.6, pt=3.5, color=INK)
    panel(s, MARGIN, 4.40, CW, 1.02,
          'The decoder works and is measured. It costs 2.6 percent of mean accuracy, '
          'buys three capabilities the baseline cannot express, and has one diagnosed '
          'limitation that is not yet closed.',
          F_BODY, PP_ALIGN.CENTER)
    text(s, MARGIN, 5.80, CW, 0.9,
         'Shriman Raghav Srinivasan   |   MS Robotics, Northeastern University\n'
         'Field Robotics Lab   |   Status review, August 2026',
         F_CITE, False, MUTED, 1.40, tight=True)
    note(s, "This is where the project stands rather than a full presentation of it. I will do "
            "the scorecard first so you know immediately what is and is not done, then seven "
            "results, then the one thing I have not solved, then what I would do next and the "
            "handful of places where I would like your judgement. Stop me anywhere.")

    # ============================================================= 2  scorecard
    s, i = slide()
    action_title(s, 'Two of the three objectives I set are not met.\nThree capabilities were not on the list')
    table(s, MARGIN, CONTENT_Y, [
        ['Objective set at the start', 'Verdict', 'Evidence'],
        ['Better accuracy than v2', 'not met',
         f'{F["v3_epe"]} against {F["v2_epe"]}; like-for-like {F["gap_pct_like"]} behind'],
        ['Less compute than v2', 'partly',
         f'dense {F["dense_penalty"]} slower; repeat query {F["repeat_speedup"]} cheaper'],
        ['Runs on edge devices', 'unproven',
         f'every figure is from a {F["device"]}'],
    ], [3.55, 1.55, 6.73], F_SMALL, avail=CW, align='lcr')
    text(s, MARGIN, 3.95, CW, 0.28, 'DELIVERED, AND MEASURED', F_SMALL, True, MID, tight=True)
    for k, (t, d) in enumerate([
        ('Queryable output',
         'flow at any continuous coordinate;\nsparse reproduces dense exactly'),
        ('Cheap repeat access',
         f'{F["v3_repeat_ms"]} ms per further query batch\nagainst a cached frame'),
        ('Calibrated confidence',
         'a per-query error estimate,\nmonotonic against real error'),
    ]):
        panel(s, MARGIN + k * 4.03, 4.38, 3.78, 1.42, f'{t}\n\n{d}', F_SMALL,
              PP_ALIGN.CENTER)
    cite(s, f'Accuracy on {F["pairs"]} held-out Virtual KITTI 2 pairs, {F["pixels"]} valid '
            f'pixels. Timings on a {F["device"]}, fp16, {F["frame"]}.')
    note(s, "Starting with the scorecard because it is the honest summary. Against the three "
            "objectives I wrote down at the start, two are not met and one is partly met. Better "
            "accuracy than v2: not met, and I will show you exactly how much and why. Less "
            "compute: it depends entirely on which mode, dense is slower and repeat queries are "
            "twenty-seven times cheaper. Edge capable: unproven, because every number I have is "
            "from a laptop GPU. What the project did deliver is the three things at the bottom, "
            "none of which were on the original list, and all three are measured rather than "
            "asserted.")

    # ============================================================= 3  the change
    s, i = slide()
    action_title(s, 'The change is confined to the last stage,\nand the front end is verified unchanged')
    bullets(s, MARGIN, CONTENT_Y, 6.75, 3.2, [
        ('Phase 1, once per frame pair. ',
         'v2 exactly as published, up to its 1/8 coarse flow. 33 ms, and the result is cached.'),
        ('Phase 2, once per query batch. ',
         f'A coordinate goes in, a convex blend of the nine neighbouring coarse-flow values '
         f'plus a bilinear sample comes out. {F["v3_repeat_ms"]} ms, and cost scales with the '
         f'number of queries rather than with image area.'),
        ('The control. ',
         'All 137 tensors shared with v2 are verified bit-identical per checkpoint, and both '
         'evaluation scenes are held out of every training set.'),
    ], F_COL)
    panel(s, 7.85, CONTENT_Y, 4.73, 1.55,
          'Because the blend is convex,\nthe decoder cannot predict motion\n'
          'its inputs do not locally support.',
          F_COL, PP_ALIGN.CENTER)
    panel(s, 7.85, 3.60, 4.73, 1.42,
          'Zero-initialised, the decoder is\nexactly bilinear upsampling.',
          F_COL, PP_ALIGN.CENTER)
    panel(s, MARGIN, 5.55, CW, 0.85,
          f'Because Phase 1 is cached, a second question about a frame already processed costs '
          f'{F["v3_repeat_ms"]} ms instead of {F["v2_ms"]}. That is the property the rest of '
          f'this deck turns on.',
          F_BODY, PP_ALIGN.CENTER)
    cite(s, 'Convex-weight formulation follows AnyFlow (2023). Gated multi-scale fusion follows '
            'InfiniDepth (2025).')
    note(s, "A one-slide reminder of what actually changed. Everything in v2 that understands "
            "motion is frozen and verified: all 137 shared tensors bit-identical to v2, checked "
            "per checkpoint, so any difference you see in the results is the decoder and nothing "
            "else. The new part is phase two, which takes a coordinate and returns a convex blend "
            "of the neighbouring coarse flow values. Two properties matter. Convexity means it "
            "cannot invent motion, and zero-initialised it is exactly bilinear upsampling, so it "
            "starts from a known-good operating point. And phase one being cached is what makes "
            "repeat queries cheap, which is the thread through the whole deck.")

    # ============================================================= 4  accuracy
    s, i = slide()
    action_title(s, f'The price is {F["gap_pct"]} of mean error, from a\nmodel {F["smaller_pct"]} smaller')
    fb = exhibit(s, f'{PLOTS}/accuracy_bars.png', 1.95)
    column(s, 'THE COST, STATED FIRST', [
        ('', 'Every configuration sits above v2. The best is '
             f'{F["v3_epe"]} against {F["v2_epe"]}.'),
        ('', f'Like-for-like, trained on FlyingChairs alone as v2 was trained on FlyingThings '
             f'alone: {F["v3_epe_chairs"]}, or {F["gap_pct_like"]} behind.'),
        ('', f'{F["v3_params"]} parameters against {F["v2_params"]}.'),
    ], key=f'{F["v3_epe"]}\nagainst {F["v2_epe"]}', fig_bottom=fb)
    cite(s, f'Mean end-point error per pixel, {F["pairs"]} held-out VKITTI 2 pairs. Lower is better.')
    note(s, "The cost, and I would rather lead with it than have you find it. Every v3 "
            "configuration is less accurate than the baseline on mean end-point error. The best "
            "is 2.384 against 2.324, so 2.6 percent worse. The honest comparison is the "
            "like-for-like one, chairs only against v2's things only, and that is 7.6 percent "
            "behind. The model is thirteen percent smaller, which is worth something but does "
            "not excuse it. I know the mechanism and it is on the open-problem slide.")

    # ============================================================= 5  compute
    s, i = slide()
    action_title(s, 'State reuse is the win: a repeat query costs\n'
                    f'{F["v3_repeat_ms"]} ms against {F["v2_ms"]} ms')
    fb = exhibit(s, f'{PLOTS}/speed_bars.png', 2.02)
    column(s, 'WHICH CLAIM I AM MAKING', [
        ('', f'Dense output is {F["dense_penalty"]} slower than v2. That mode is not what the '
             f'decoder is for.'),
        ('', f'A first sparse query is level with v2: {F["coarse_share"]} of the cost is the '
             f'coarse pass, which both pay and which cannot be skipped.'),
        ('', 'The win is the second question about the same frame. v2 keeps no state, so it '
             'recomputes everything.'),
    ], key=f'{F["repeat_speedup"]} on repeat access', fig_bottom=fb)
    cite(s, f'{F["device"]}, fp16, {F["frame"]}, median of 50 runs after warm-up.')
    note(s, "On speed I want to be precise about which claim I am making, because there are three "
            "numbers here and only one of them is a win. Dense output is ten percent slower than "
            "v2. A first sparse query is level with v2, because ninety-six percent of the cost is "
            "the coarse pass and both models pay it. The win is repeat access: asking a second "
            "question about a frame you have already processed costs one and a quarter "
            "milliseconds against thirty-three, because v2 keeps nothing between calls. Anything "
            "that revisits a frame gets that, and that is structural rather than a margin.")

    # ============================================================= 6  confidence
    s, i = slide()
    action_title(s, 'Confidence is the capability v2 cannot express,\nand it buys accuracy back')
    fb = exhibit(s, f'{PLOTS}/selective_accuracy.png', 2.00)
    column(s, 'CALIBRATION BEHIND THE CURVE', [
        ('', f'Real error rises monotonically across five bins, {F["bins"]}, over '
             f'{F["queries"]} queries.'),
        ('', f'Pearson r = {F["pearson"]}: informative rather than tight, and usable for '
             f'weighting or rejecting matches.'),
        ('', 'v2 emits flow alone, so it has one operating point where v3 has a curve.'),
    ], key=f'{F["sel_80"]} px over 80%\nof the frame', fig_bottom=fb)
    cite(s, f'Per-query error scale under a Laplace likelihood. Coverage is the share of queries '
            f'answered. Same {F["queries"]} queries as the calibration bins.')
    note(s, "This is the result I would most like your reaction to, so I have put the calibration "
            "and what it buys on one slide. The head predicts an error scale for every query, "
            "trained under a Laplace likelihood. Binned by predicted uncertainty, actual error "
            "rises monotonically across all five bins, a fifteen-fold span over 2.35 million "
            "queries. What that buys is the curve: mean error over all pixels is one operating "
            "point and it is the only one v2 has, because it cannot tell you which outputs to "
            "trust. Discard the least confident fifth and error on the remainder is 1.06 pixels. "
            "The reason I do not think this is a selection trick is that for registration and "
            "mapping you are choosing a few hundred points anyway.")

    # ============================================================= 7  region
    s, i = slide()
    action_title(s, f'Flowing {F["roi_area"]} of the frame costs '
                    f'{F["roi_cost"]}\nand runs {F["roi_speedup"]} faster')
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
    note(s, "This is the enabling result for the whole fast-platform framing. Keep full "
            "resolution, process fourteen percent of the frame, and it costs thirty-four "
            "thousandths of a pixel and runs four and a half times faster. The margin is the part "
            "worth dwelling on: without one, error jumps sixty-five percent and a quarter of the "
            "large-motion pixels fail completely, because this network locates large displacement "
            "with attention over the whole image and a tight crop takes that away. I should be "
            "clear that this applies to any flow network including v2 unchanged, so it is a "
            "platform technique rather than something the decoder provides.")

    # ============================================================= 8  margin rule
    s, i = slide()
    action_title(s, 'The margin needed is set by motion,\nnot by image size')
    picture(s, f'{PLOTS}/margin_rule.png', 1.57, 1.88, w=10.2)
    panel(s, MARGIN, 5.74, CW, 1.00,
          f'Driving averages {F["motion_drive"]} of motion and needs about {F["margin_drive"]} '
          f'of margin. Aerial averages {F["motion_air"]} and needs about {F["margin_air"]}. Both '
          f'shed most of the penalty by one frame of motion, so a margin can be sized from speed '
          f'and frame rate before anything is run.',
          F_BODY, PP_ALIGN.CENTER)
    cite(s, 'Measured with the baseline network. Agreement holds to a ratio of one; residuals '
            'beyond that differ by scene.')
    note(s, "I wanted to know whether the margin rule was fitted to one dataset or was something "
            "more general, so I tested it on aerial sequences, which move about a third as fast. "
            "The prediction was that the required margin scales with motion, so the knee should "
            "move from thirty-two pixels down to about nine. It did. On the right, divided by "
            "mean motion, both domains start at the same penalty and both have shed most of it by "
            "the point where the margin equals one frame of motion. To be precise about the "
            "limit: this predicts the scale rather than the exact curve, because the residuals "
            "beyond a ratio of one differ by scene. It is still enough to size a margin in "
            "advance.")

    # ============================================================= 9  scenario 3
    s, i = slide()
    action_title(s, f'A region found mid-frame costs {F["s3_sparse_ms"]} ms,\n'
                    f'against {F["s3_crop_ms"]} ms for a fresh crop')
    picture(s, f'{PLOTS}/scenario3_marginal.png', 1.57, 1.88, w=10.2)
    panel(s, MARGIN, 5.74, CW, 1.00,
          'The cached coarse state answers a question the platform did not have when the frame '
          'arrived. A cropped pipeline must run an entire new pass, and is less accurate doing '
          'it. Two disjoint crops cost more than one full frame: crop once, or not at all.',
          F_BODY, PP_ALIGN.CENTER)
    cite(s, 'Held-out VKITTI 2. The second region is disjoint from the first, so nothing computed '
            'for the first can be reused by a cropped pipeline.')
    note(s, "This is the one result that is specific to the architecture rather than to cropping, "
            "and it is the case I think is most relevant to a moving platform. A frame is already "
            "in flight when a new object enters the field of view. Because the coarse pass is "
            "cached, answering about it costs one point seven milliseconds. A cropped pipeline "
            "cannot reuse anything, because the new object is outside its box, so it runs a whole "
            "new pass at seven point three milliseconds and the answer is worse, three point six "
            "pixels against two point one, because a fresh crop has lost the surrounding context. "
            "The general point is that this buys the ability to answer a question you did not "
            "know you would have when the frame arrived.")

    # ============================================================= 10 aerial
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
        ('', 'The only unconfounded comparison I have. Every other table is on a domain v3 '
             'trained on and v2 did not.'),
        ('', 'Consistent rather than large: 0.004 to 0.029 px, over 40 pairs and one seed.'),
        ('', 'v3 is still worse on large-motion failures, 7.2% against 6.5%.'),
    ])
    cite(s, 'Mean end-point error, px. v2 trained on FlyingThings; v3 on FlyingChairs, VKITTI 2 '
            'and Sintel. Neither trained on aerial imagery.')
    note(s, "One result I did not expect. Everything else is measured on driving sequences, where "
            "v3 trained on driving data and v2 did not, which is a confound I have flagged "
            "throughout. Aerial is neutral ground: neither model has seen anything like it. There "
            "v3 comes out ahead at all seven margin settings. My guess at the reason is that the "
            "decoder saw three datasets where v2 saw one, and the convex head is bounded so it "
            "cannot invent motion, which should degrade more gracefully out of domain. I want to "
            "be careful how much weight this carries: the margins are small, it is forty pairs "
            "and one seed, and v3 is still worse on the large-motion pixels. What makes it worth "
            "showing is that it is consistent across seven independent settings.")

    # ============================================================= 11 open problem
    s, i = slide()
    action_title(s, 'One diagnosed cause, three experiments\nconverging on it, not yet fixed')
    bullets(s, MARGIN, CONTENT_Y, 6.9, 3.4, [
        ('The cause. ',
         'The decoder’s finest input is at 1/8 resolution, so the evidence available to it '
         'barely varies inside an 8x8 cell. The upsampler it replaces reads the '
         'full-resolution frame directly.'),
        ('Three independent nulls. ',
         'A Fourier positional encoding, sub-pixel querying, and reduced-resolution querying '
         'all returned null, and all three are explained by that one mechanism.'),
        ('What it also explains. ',
         'Querying between pixels is presently closer to interpolation than inference. The '
         'interface is exact; the extra information recoverable is small.'),
    ], F_COL)
    panel(s, 7.95, CONTENT_Y, 4.63, 2.30,
          'Proposed remedy\n\nA full-resolution stem: a cheap\nstride-8 convolution over the\n'
          'full-resolution image, fed to\nthe decoder. Roughly 1 to 2 ms.',
          F_COL, PP_ALIGN.CENTER)
    panel(s, MARGIN, 5.62, CW, 0.82,
          'This is the only candidate that is specific to v3. The baseline’s upsampler already '
          'reads full resolution, so nothing here transfers to it.',
          F_BODY, PP_ALIGN.CENTER)
    cite(s, f'Other limitations: no embedded measurement, one evaluation domain, and '
            f'{F["seed_note"]}, so differences below about {F["resolution"]} are not resolved.')
    note(s, "This is the part I have not solved, and it is one mechanism rather than a list of "
            "unrelated weaknesses. The decoder never sees full-resolution features. Its finest "
            "input is at one-eighth resolution, so inside an eight by eight cell it is looking at "
            "almost the same evidence wherever you query. That single fact explains the accuracy "
            "gap, and it explains why three separate experiments returned null: the Fourier "
            "encoding, sub-pixel querying, and reduced-resolution querying. Three nulls with one "
            "explanation is what makes me confident in the diagnosis rather than merely "
            "suspicious. The remedy I would try is a cheap full-resolution stem, and it is the "
            "only candidate on my list that is specific to this architecture.")

    # ============================================================= 12 next
    s, i = slide()
    action_title(s, 'What I would do next, in priority order,\nwith what each costs')
    rows = [
        ['Next experiment', 'Why it is next', 'Cost'],
        ['Full-resolution stem', 'the direct attack on the one diagnosed cause',
         'half a day, 5 h train'],
        ['Repeat uncertainty run, new seed', 'the 2.392 to 2.384 gain is inside checkpoint noise',
         '5 h train'],
        ['Jetson measurement', 'the word edge is currently a design target, not a result',
         'hardware dependent'],
        ['Spring 4K evaluation', 'the one test the baseline structurally cannot take',
         '1 h, script exists'],
        ['Tiled refinement', 'roughly 2x, but it would benefit the baseline equally',
         'not built'],
    ]
    table(s, MARGIN, CONTENT_Y, rows, [3.85, 5.55, 2.43], F_SMALL, avail=CW,
          align='llr')
    panel(s, MARGIN, 4.75, CW, 1.05,
          'Only the first is a claim about this architecture. The second decides whether the '
          'uncertainty head’s accuracy gain is real. The third decides whether the edge premise '
          'can be stated at all.',
          F_BODY, PP_ALIGN.CENTER)
    cite(s, 'Training costs are on the Explorer cluster; evaluation on the laptop GPU.')
    note(s, "Next steps in the order I would take them, with costs, so you can tell me if you "
            "disagree with the ordering. The full-resolution stem is first because it is the only "
            "one that attacks the diagnosed cause and the only one specific to this architecture. "
            "Second is repeating the uncertainty run with a different seed, because the apparent "
            "gain from the uncertainty head is inside the checkpoint-to-checkpoint variation I "
            "measured, so at the moment I cannot claim it. Third is a Jetson measurement, because "
            "edge is in the title and I have no embedded number. Fourth is the Spring four-K "
            "evaluation, which is the one test v2 structurally cannot take, and the script is "
            "already written. Tiled refinement is last because it would help the baseline just as "
            "much.")

    # ============================================================= 13 asks
    s, i = slide()
    action_title(s, 'Four places where your judgement would\nchange what I do')
    qa = [
        ('Is selective prediction a fair reframing of the accuracy result, or does it need '
         'a different baseline?',
         'It is the strongest thing I have and the thing I am least certain is being presented '
         'correctly.'),
        ('Full-resolution stem first, or embedded measurement first?',
         'The stem closes the scientific gap. The measurement closes the claim in the title. '
         'There is not time for both to be leisurely.'),
        ('Is one seed per configuration acceptable for the write-up?',
         f'It is what I have. It means nothing below about {F["resolution"]} is resolved, and I '
         f'have said so wherever a difference is smaller than that.'),
        ('Is the aerial result worth foregrounding, given it is 40 pairs?',
         'It is the only comparison in the study without a training-domain confound, but it is '
         'also the smallest sample.'),
    ]
    # Spacing follows the measured text: one of these questions wraps to two
    # lines and the rest do not, so a fixed offset puts the answers at four
    # different distances from their questions.
    avail = CW - 0.30
    y = CONTENT_Y
    for q, a in qa:
        qh = (int(width_in(q, F_COL, True) / avail) + 1) * (F_COL / 72) * 1.18
        ah = (int(width_in(a, F_SMALL) / avail) + 1) * (F_SMALL / 72) * 1.20
        vrule(s, MARGIN, y, qh + ah + 0.10)
        text(s, MARGIN + 0.24, y, avail, qh, q, F_COL, True, INK, 1.18, tight=True)
        text(s, MARGIN + 0.24, y + qh + 0.08, avail, ah, a, F_SMALL, False, MID, 1.20,
             tight=True)
        y += qh + ah + 0.48
    note(s, "Rather than a generic questions slide, these are the four decisions where what you "
            "say actually changes what I do next. The first is the one I care most about: "
            "selective prediction is the strongest result I have and it is also the one I am "
            "least sure I am presenting correctly, so I would rather you push on it here than in "
            "the defence. The second is a genuine fork given the time left. The third and fourth "
            "are about how much weight to put on results I have flagged as thin.")

    # ============================================================= A1 numbers
    s, i = slide(label='Appendix:  every number in this deck, in one place')
    action_title(s, 'Full result table')
    table(s, MARGIN, CONTENT_Y, [
        ['Model', 'Training data', 'EPE', '1 px', 'Params'],
        ['NeuFlow v2', 'FlyingThings', '2.324', '77.63', '9.03 M'],
        ['NeuFlow v3', 'FlyingChairs', '2.500', '72.81', '7.83 M'],
        ['NeuFlow v3', '+ VKITTI 2', '2.398', '75.74', '7.83 M'],
        ['NeuFlow v3', '+ MPI-Sintel', '2.392', '75.83', '7.83 M'],
        ['NeuFlow v3', '+ uncertainty head', '2.384', '76.13', '7.83 M'],
    ], [2.35, 3.05, 1.35, 1.35, 1.45], F_SMALL, avail=9.55, align='llrrr')
    text(s, MARGIN, 4.20, CW, 0.28, 'COMPUTE, SAME DEVICE THROUGHOUT', F_SMALL, True, MID,
         tight=True)
    table(s, MARGIN, 4.58, [
        ['Mode', 'Latency', 'Note'],
        ['v2, full frame', '33.3 ms', 'its only mode'],
        ['v3 sparse, first query on a new pair', '34.1 ms', 'equivalent to v2'],
        ['v3 sparse, repeat query on a cached pair', '1.25 ms', '27x cheaper'],
        ['v3 dense output, stride 2', '37.1 ms', 'not the intended mode'],
    ], [5.05, 1.85, 4.93], F_SMALL, avail=CW, align='lrl')
    cite(s, f'{F["pairs"]} held-out VKITTI 2 pairs, {F["pixels"]} valid pixels. '
            f'{F["device"]}, fp16, {F["frame"]}.')
    note(s, "The full table, kept here so that the results slides could each carry one claim "
            "rather than a grid of numbers. Two things worth pointing at if we end up here. The "
            "ordering is monotonic as training data is added, which is what you would expect and "
            "is a weak check that nothing is broken. And the gap between the uncertainty row and "
            "the row above it is 0.008, which is smaller than the checkpoint-to-checkpoint "
            "variation, so I do not claim the uncertainty head improves accuracy.")

    finish(prs, OUT)


if __name__ == '__main__':
    main()
