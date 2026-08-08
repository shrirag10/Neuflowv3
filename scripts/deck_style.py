"""Shared design system for the NeuFlow v3 decks.

Both generators import this so that a layout fix or a corrected number lands in
both decks at once. The two decks previously drifted: they disagreed on the
coarse-pass share (96 vs 87 percent), on the dense penalty (10 vs 11 percent)
and even on the input frame size, because each carried its own copy of every
figure.

The rules encoded here, in one place:

  Canvas 13.333 x 7.5 in, so a plot authored 9 in wide is placed near its
  intended size instead of being shrunk until its labels are half the size of
  the body text beside them.

  Widths are measured against the real font file before placement, and the
  build fails if a title line or a table cell does not fit. A cell that
  silently wraps is how a table starts looking homemade.

  Monochrome, serif, white background, no rounded callouts, no highlight
  fills. Categories that a report separates by hue are separated here by
  value, which survives a projector and a greyscale handout.

FACTS holds every number either deck quotes. Quote it from here or not at all.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ------------------------------------------------------------------ palette
INK   = RGBColor(0x1A, 0x1A, 0x1A)   # titles and body
MID   = RGBColor(0x5A, 0x5A, 0x5A)   # secondary text, column headers
MUTED = RGBColor(0x80, 0x80, 0x80)   # citations, captions, page numbers
RULE  = RGBColor(0xC8, 0xC8, 0xC8)
PANEL = RGBColor(0xF2, 0xF2, 0xF2)   # flat block fill, no border, no radius
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = 'DejaVu Serif'
FONT_FILE = {
    False: '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    True:  '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
}

F_TITLE, F_BODY, F_COL, F_TABLE, F_SMALL, F_CITE = 30, 18, 16.5, 15, 13, 11.5

W, H = 13.333, 7.5
MARGIN = 0.75
CW = W - 2 * MARGIN                  # content width, 11.833

TITLE_TOP, TITLE_H = 0.40, 1.06      # two lines at 30 pt
RULE_Y = 1.54
CONTENT_Y = 1.82
CITE_Y = 6.92

FIG_X, FIG_W = MARGIN, 7.40          # 0.82 of a 9 in authored figure
COL_X, COL_W = 8.45, 4.13            # right edge lands exactly on the margin

PLOTS = 'results/plots/deck'         # title-free, monochrome variants
MONTAGE = 'results/plots'            # scenario montages, already tightened
VISUALS = 'results/visuals'

# --------------------------------------------------------------------- facts
# Every number either deck is allowed to quote. All from the 2026-08-03 runs:
# both evaluation scenes held out, and a front end verified unchanged against
# v2. Anything not in here does not go on a slide.
FACTS = dict(
    # accuracy, VKITTI 2 Scene18+20, 1,174 pairs, 460,573,660 valid px
    v2_epe=2.324, v2_a1=77.63, v2_params='9.03 M',
    v3_epe=2.384, v3_a1=76.13, v3_params='7.83 M',
    v3_epe_chairs=2.500,               # like-for-like, chairs only
    gap_pct='2.6%', gap_pct_like='7.6%', smaller_pct='13%',
    # compute, RTX 4060 laptop, fp16, 384 x 1248
    v2_ms=33.3, v3_dense_ms=37.1, v3_first_ms=34.1, v3_repeat_ms=1.25,
    repeat_speedup='27x', dense_penalty='10%', coarse_share='96%',
    # confidence
    bins='0.48 px to 7.10 px', bin_span='15-fold', pearson=0.318,
    queries='2,348,000', sel_80=1.058, sel_20=0.480,
    # region-directed inference
    roi_area='14%', roi_cost='0.034 px', roi_speedup='4.4x',
    motion_drive='26.6 px', motion_air='9.26 px',
    margin_drive='32 px', margin_air='8 px',
    s3_sparse_ms=1.7, s3_crop_ms=7.3,
    # scope
    pairs='1,174', pixels='460,573,660', frame='384 x 1248',
    device='RTX 4060 laptop', seed_note='one seed per configuration',
    resolution='0.05 px',
)


# ------------------------------------------------------------------ measuring
def width_in(s, size, bold=False):
    """Rendered width of a string in inches, from the actual font file.

    Oversampled 4x because ImageFont works in integer pixels and a 15 pt cell
    measured to the nearest pixel is only accurate to about 1.5%, which is the
    same order as the headroom being checked.
    """
    f = ImageFont.truetype(FONT_FILE[bold], int(round(size * 4)))
    return f.getlength(s) / 4 / 72


def fits(s, size, avail, bold=False, what='text'):
    w = width_in(s, size, bold)
    if w > avail:
        raise SystemExit(
            f'{what} overflows: {w:.3f} in needed, {avail:.3f} available '
            f'at {size} pt\n  {s!r}')
    return w


def img_aspect(path):
    with Image.open(path) as im:
        return im.size[0] / im.size[1]


# ------------------------------------------------------------------ primitives
def text(s, x, y, w, h, body, size, bold=False, color=INK, spacing=1.0,
         align=PP_ALIGN.LEFT, wrap=True, tight=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    if tight:
        tf.margin_left = tf.margin_right = 0
        tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(body.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = spacing
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def bullets(s, x, y, w, h, items, size=F_BODY, space_after=13, color=INK):
    """items: (lead, rest). The lead is bold. Either may be empty."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    for i, (lead, rest) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.14
        p.space_after = Pt(space_after)
        for chunk, bold in ((lead, True), (rest, False)):
            if not chunk:
                continue
            r = p.add_run()
            r.text = chunk
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = color
    return tb


def hrule(s, y, x=MARGIN, w=CW, pt=0.75, color=RULE):
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                             Inches(w), Pt(pt))
    bar.fill.solid(); bar.fill.fore_color.rgb = color
    bar.line.fill.background(); bar.shadow.inherit = False
    return bar


def vrule(s, x, y, h, pt=2.5, color=INK):
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                             Pt(pt), Inches(h))
    bar.fill.solid(); bar.fill.fore_color.rgb = color
    bar.line.fill.background(); bar.shadow.inherit = False
    return bar


def action_title(s, title):
    """Two-line claim, measured against the real font before placement."""
    for ln in title.split('\n'):
        fits(ln, F_TITLE, CW, bold=True, what='title line')
    text(s, MARGIN, TITLE_TOP, CW, TITLE_H, title, F_TITLE, True, INK, 1.10,
         tight=True)
    hrule(s, RULE_Y)


def cite(s, body, y=CITE_Y):
    text(s, MARGIN, y, CW - 0.55, 0.26, body, F_CITE, False, MUTED, tight=True)


def note(s, body):
    s.notes_slide.notes_text_frame.text = body


def panel(s, x, y, w, h, body, size=F_SMALL, align=PP_ALIGN.LEFT, bold=False,
          color=INK, spacing=1.20):
    """Flat block. No border, no corner radius, no fill beyond a light grey."""
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                           Inches(w), Inches(h))
    b.fill.solid(); b.fill.fore_color.rgb = PANEL
    b.line.fill.background(); b.shadow.inherit = False
    tf = b.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.20)
    tf.margin_top = tf.margin_bottom = Inches(0.08)
    for i, line in enumerate(body.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run(); r.text = line
        r.font.name = FONT; r.font.size = Pt(size)
        r.font.bold = bold; r.font.color.rgb = color
    return b


def keyline_h(body, size=F_COL):
    return len(body.split('\n')) * (size / 72) * 1.34 + 0.10


def keyline(s, x, y, w, body, size=F_COL):
    """The single number a results slide turns on. A rule, not a coloured box."""
    h = keyline_h(body, size)
    vrule(s, x, y, h)
    text(s, x + 0.20, y - 0.03, w - 0.20, h, body, size, True, INK, 1.30,
         tight=True)
    return y + h


def picture(s, path, x, y, w=None, h=None):
    if not os.path.exists(path):
        # results/ is gitignored, so a fresh clone has no deck figures. Fail
        # here rather than shipping a deck with a placeholder box on it.
        raise SystemExit(
            f'missing figure: {path}\n'
            'Deck figures are generated, not committed. Run:\n'
            '    DECK_FIGS=1 python3 scripts/make_final_plots.py')
    kw = {}
    if w:
        kw['width'] = Inches(w)
    if h:
        kw['height'] = Inches(h)
    return s.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def figure_bottom(y, path, w=None, h=None):
    """Where a placed figure actually ends, so nothing can be laid under it."""
    return y + (w / img_aspect(path) if w else h)


def exhibit(s, path, y, w=FIG_W, x=FIG_X):
    """Place the slide's one exhibit and report where it ends."""
    picture(s, path, x, y, w=w)
    return figure_bottom(y, path, w=w)


# Rendered line height as a multiple of the point size, for DejaVu Serif at
# line_spacing 1.14. The paragraph multiple applies to the font's own line
# height, not to the point size, and this font's is generous, so the two
# compound. Measured against LibreOffice output rather than assumed.
LEADING = 1.42


def wrap_lines(chunks, size, width):
    """Greedy word wrap, counting lines the way a renderer would.

    chunks is a list of (text, bold) so that a bold lead-in is measured with
    the bold metrics. Estimating instead from total width over column width
    undercounts every time a long word forces an early break, which is exactly
    the case that pushes a block lower than expected.
    """
    words = [(w, b) for txt, b in chunks for w in txt.split()]
    lines, cur = 1, 0.0
    space = width_in(' ', size)
    for w, b in words:
        ww = width_in(w, size, b)
        if cur and cur + space + ww > width:
            lines += 1
            cur = ww
        else:
            cur += (space if cur else 0) + ww
    return lines


def block_height(items, size, width, space_after=12):
    """Rendered height of a bullet block.

    Text boxes are given generous heights so they never clip, which means their
    declared geometry says nothing about where the text actually ends, and the
    overlap check cannot see inside them. So anything placed under a bullet
    block has to measure it.
    """
    total = sum(wrap_lines([(lead, True), (rest, False)], size, width)
                for lead, rest in items) * (size / 72) * LEADING
    return total + (len(items) - 1) * space_after / 72


def column(s, header, items, key=None, size=F_COL, y=CONTENT_Y, fig_bottom=None):
    """Interpretive column to the right of an exhibit.

    The key number sits on the figure's bottom edge, so the two halves of the
    slide finish on the same line whatever the figure's aspect ratio turns out
    to be. Unless the text runs lower than the figure, in which case it sits
    under the text instead: a key number laid over the last bullet is the one
    fault the overlap check cannot see, because both are unfilled text boxes.
    """
    top = y + 0.36
    text(s, COL_X, y, COL_W, 0.28, header, F_SMALL, True, MID, tight=True)
    bullets(s, COL_X, top, COL_W, 3.4, items, size, space_after=12)
    if key:
        base = fig_bottom if fig_bottom else 6.10
        ky = max(base - keyline_h(key, size),
                 top + block_height(items, size, COL_W) + 0.22)
        end = ky + keyline_h(key, size)
        if end > CITE_Y - 0.12:
            raise SystemExit(
                f'column overflows: key number ends at {end:.2f} in, '
                f'citation line is at {CITE_Y}. Shorten the column.')
        keyline(s, COL_X, ky, COL_W, key)


def table(s, x, y, rows, col_w, size=F_TABLE, hl_row=None, head_color=MID,
          avail=None, align=None):
    """Booktabs-style rule structure, with every cell width verified.

    Row height follows from the point size rather than being a constant that
    happens to work at one size, and cells do not wrap: a cell too wide for its
    column stops the build instead of silently spilling into the row below.
    `avail` is the horizontal room the caller has actually reserved, so a table
    cannot grow into whatever is sitting to its right. `align` is one letter per
    column from 'lcr'; the default right-aligns everything after the first
    column, which is correct for numbers and wrong for prose.
    """
    ALIGN = {'l': PP_ALIGN.LEFT, 'c': PP_ALIGN.CENTER, 'r': PP_ALIGN.RIGHT}
    if align is None:
        align = 'l' + 'r' * (len(col_w) - 1)
    assert len(align) == len(col_w), 'one alignment letter per column'
    if avail is not None and sum(col_w) > avail:
        raise SystemExit(f'table is {sum(col_w):.2f} in wide, {avail:.2f} reserved')
    row_h = (size / 72) * 1.42
    pad = 0.10
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            fits(cell, size, col_w[ci] - pad,
                 bold=(ri == 0 or ri == hl_row), what=f'table cell r{ri}c{ci}')

    top = y
    hrule(s, top, x, sum(col_w), pt=1.1, color=INK)
    yy = top + 0.05
    for ri, row in enumerate(rows):
        head = ri == 0
        bold = head or ri == hl_row
        color = head_color if head else INK
        xx = x
        for ci, cell in enumerate(row):
            al = ALIGN[align[ci]]
            text(s, xx, yy, col_w[ci], row_h, cell, size, bold, color,
                 align=al, tight=True)
            xx += col_w[ci]
        yy += row_h
        if head:
            hrule(s, yy + 0.03, x, sum(col_w), pt=0.75, color=RULE)
            yy += 0.11
    hrule(s, yy + 0.04, x, sum(col_w), pt=1.1, color=INK)
    return yy + 0.04


def chain(s, y, labels, h=0.92, size=12.5):
    """Pipeline strip. Terminal stage solid, the rest outlined in grey."""
    gap = 0.34
    n = len(labels)
    w = (CW - (n - 1) * gap) / n
    x = MARGIN
    for i, (lab, solid) in enumerate(labels):
        b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                               Inches(w), Inches(h))
        b.fill.solid()
        b.fill.fore_color.rgb = INK if solid else WHITE
        b.line.color.rgb = INK if solid else RULE
        b.line.width = Pt(0.9)
        b.shadow.inherit = False
        tf = b.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.06)
        for k, line in enumerate(lab.split('\n')):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            p.line_spacing = 1.12
            r = p.add_run(); r.text = line
            r.font.name = FONT; r.font.size = Pt(size)
            r.font.bold = solid
            r.font.color.rgb = WHITE if solid else INK
        if i < n - 1:
            text(s, x + w, y + h / 2 - 0.16, gap, 0.3, '>', 14, True, MUTED,
                 align=PP_ALIGN.CENTER, tight=True)
        x += w + gap



# ------------------------------------------------------------------ scaffold
def new_deck():
    """A 13.333 x 7.5 presentation and a slide factory that numbers itself."""
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]
    n = [0]

    def slide(footer=True, label=None):
        n[0] += 1
        s = prs.slides.add_slide(blank)
        bg = s.background.fill
        bg.solid(); bg.fore_color.rgb = WHITE
        if label:
            text(s, MARGIN, 0.10, CW, 0.22, label, F_CITE, False, MUTED,
                 tight=True)
        if footer:
            text(s, W - MARGIN - 0.45, CITE_Y, 0.45, 0.24, str(n[0]), F_CITE,
                 False, MUTED, align=PP_ALIGN.RIGHT, tight=True)
        return s, n[0]

    return prs, slide


def finish(prs, out, min_note_words=25):
    """Save, then refuse to ship anything the audience would notice.

    Every one of these fired at least once during development, which is the
    only reason to keep them: a placeholder note, an en dash, a table grown
    into its neighbour, a figure laid over its own caption.
    """
    os.makedirs(os.path.dirname(out), exist_ok=True)
    prs.save(out)

    total = len(prs.slides._sldIdLst)
    notes = sum(1 for sl in prs.slides
                if sl.has_notes_slide and sl.notes_slide.notes_text_frame.text.strip())
    thin = [k + 1 for k, sl in enumerate(prs.slides)
            if len(sl.notes_slide.notes_text_frame.text.split()) < min_note_words]
    assert notes == total, 'every slide must carry a speaker note'
    assert not thin, f'placeholder speaker notes on slides {thin}'

    bad = [k + 1 for k, sl in enumerate(prs.slides) for sh in sl.shapes
           if sh.has_text_frame and ('—' in sh.text_frame.text
                                     or '–' in sh.text_frame.text)]
    assert not bad, f'en or em dash on slides {sorted(set(bad))}'

    def box_of(sh):
        return (sh.left / 914400, sh.top / 914400,
                (sh.left + sh.width) / 914400, (sh.top + sh.height) / 914400)

    off, hits = [], []
    for k, sl in enumerate(prs.slides):
        solid = []
        for sh in sl.shapes:
            x0, y0, x1, y1 = box_of(sh)
            if x0 < -0.01 or y0 < -0.01 or x1 > W + 0.01 or y1 > H + 0.01:
                off.append((k + 1, round(x1, 2), round(y1, 2)))
            if sh.shape_type == 13 or (sh.has_text_frame and sh.fill.type == 1
                                       and sh.height > Inches(0.3)):
                solid.append((x0, y0, x1, y1))
        for a in range(len(solid)):
            for b in range(a + 1, len(solid)):
                ax0, ay0, ax1, ay1 = solid[a]
                bx0, by0, bx1, by1 = solid[b]
                if (min(ax1, bx1) - max(ax0, bx0) > 0.02
                        and min(ay1, by1) - max(ay0, by0) > 0.02):
                    hits.append(k + 1)
    assert not off, f'shapes outside the slide: {off}'
    assert not hits, f'overlapping exhibits on slides {sorted(set(hits))}'

    print(f'saved {out}: {total} slides, {notes} with speaker notes, '
          f'{W}x{H} in, {FONT}')


# ------------------------------------------------------------------ bullets
def bullet_list(s, x, y, w, items, size=F_BODY, gap=0.20, indent=0.30,
                marker=0.070, color=INK):
    """Bullets with a visible marker and a real hanging indent.

    The plain `bullets` helper distinguishes items by a bold lead-in alone,
    which reads as a paragraph when the lead-in is absent. This one puts a
    square against each item and indents the wrapped lines to clear it, so a
    three-item list looks like three items from the back of a room.

    Returns the y the list ends at, so whatever follows can be placed against
    measured text rather than a guess.
    """
    tw = w - indent
    line_h = (size / 72) * LEADING
    for lead, rest in items:
        n = wrap_lines([(lead, True), (rest, False)], size, tw)
        sq = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x),
                                Inches(y + line_h / 2 - marker / 2),
                                Inches(marker), Inches(marker))
        sq.fill.solid(); sq.fill.fore_color.rgb = color
        sq.line.fill.background(); sq.shadow.inherit = False
        tb = text(s, x + indent, y, tw, n * line_h, '', size, tight=True)
        p = tb.text_frame.paragraphs[0]
        p.line_spacing = 1.14
        for chunk, bold in ((lead, True), (rest, False)):
            if not chunk:
                continue
            r = p.add_run(); r.text = chunk
            r.font.name = FONT; r.font.size = Pt(size)
            r.font.bold = bold; r.font.color.rgb = color
        y += n * line_h + gap
    end = y - gap
    if end > CITE_Y - 0.06:
        raise SystemExit(
            f'bullet list ends at {end:.2f} in, citation line is at {CITE_Y}. '
            f'Shorten it or shrink what is above it.')
    return end


# ------------------------------------------------------------------ diagrams
def _dbox(s, x, y, w, h, label, size=12, solid=False, dashed=False):
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                           Inches(w), Inches(h))
    b.fill.solid()
    b.fill.fore_color.rgb = INK if solid else WHITE
    b.line.color.rgb = RULE if dashed else INK
    b.line.width = Pt(0.9)
    b.shadow.inherit = False
    tf = b.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = 0
    for i, ln in enumerate(label.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.line_spacing = 1.10
        r = p.add_run(); r.text = ln
        r.font.name = FONT; r.font.size = Pt(size)
        r.font.bold = solid
        r.font.color.rgb = WHITE if solid else INK
    return b


def reuse_diagram(s, y, h=2.45):
    """Why state reuse is the whole argument, drawn rather than described.

    Three questions about one frame. The baseline has no state, so each answer
    costs a full pass. v3 pays the pass once and each further answer is a
    decode. Both per-call numbers are measured; the totals are their sum.
    """
    lab_w, box_w, ans_w = 1.62, 2.35, 1.30
    row_h, step = 0.54, 0.68
    rows = [y + i * step for i in range(3)]

    # ---- baseline: one full pass per question
    text(s, MARGIN, y - 0.38, 4.0, 0.26, 'NEUFLOW V2', F_SMALL, True, MID, tight=True)
    x = MARGIN + lab_w
    for i, ry in enumerate(rows):
        _dbox(s, x, ry, box_w, row_h, f'full pass\n{FACTS["v2_ms"]} ms')
        text(s, x + box_w, ry + row_h / 2 - 0.14, 0.34, 0.28, '>', 13, True, MUTED,
             align=PP_ALIGN.CENTER, tight=True)
        _dbox(s, x + box_w + 0.34, ry, ans_w, row_h, f'answer {i + 1}', 11.5)
    text(s, MARGIN, rows[1] + 0.06, lab_w - 0.12, 0.5,
         'nothing is\nkept between\ncalls', F_SMALL, False, MID, 1.12, tight=True)
    keyline(s, x + box_w + 0.34 + ans_w + 0.42, rows[1] + 0.06,
            2.6, f'{FACTS["v2_ms"] * 3:.1f} ms', F_COL)

    # ---- v3: pay the pass once, then decode
    y2 = y + h
    text(s, MARGIN, y2 - 0.38, 4.0, 0.26, 'NEUFLOW V3', F_SMALL, True, MID, tight=True)
    mid = y2 + step + row_h / 2
    _dbox(s, MARGIN, mid - row_h / 2, lab_w + 0.5, row_h,
          f'coarse pass\n{FACTS["v2_ms"]} ms', 12, solid=True)
    text(s, MARGIN + lab_w + 0.5, mid - 0.14, 0.34, 0.28, '>', 13, True, MUTED,
         align=PP_ALIGN.CENTER, tight=True)
    _dbox(s, MARGIN + lab_w + 0.84, mid - row_h / 2, 1.55, row_h, 'cached\nstate', 12)
    stem = MARGIN + lab_w + 0.84 + 1.55
    rows2 = [y2 + i * step for i in range(3)]
    vrule(s, stem + 0.22, rows2[0] + row_h / 2, rows2[2] - rows2[0], pt=0.9, color=INK)
    for i, ry in enumerate(rows2):
        hrule(s, ry + row_h / 2, stem, 0.44, pt=0.9, color=INK)
        _dbox(s, stem + 0.44, ry, 1.72, row_h, f'decode\n{FACTS["v3_repeat_ms"]} ms')
        text(s, stem + 2.16, ry + row_h / 2 - 0.14, 0.34, 0.28, '>', 13, True, MUTED,
             align=PP_ALIGN.CENTER, tight=True)
        _dbox(s, stem + 2.50, ry, ans_w, row_h, f'answer {i + 1}', 11.5)
    keyline(s, stem + 2.50 + ans_w + 0.42, rows2[1] + 0.06, 2.6,
            f'{FACTS["v2_ms"] + 3 * FACTS["v3_repeat_ms"]:.2f} ms', F_COL)
