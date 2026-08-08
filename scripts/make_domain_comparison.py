"""Compose the two scenario montages into one side-by-side domain comparison.

The same three situations, S1 first encounter, S2 turn with overlap, S3 a new
object mid-frame, run on driving and on aerial sequences with identical region
geometry and decoder configuration. Only the domain changes.

This composes the two montages that make_scenario_figure.py already produced
rather than regenerating them, because the aerial montage needs TartanAir and
that dataset is not on every machine. Both inputs are tightened PNGs, so the
only work here is scaling them to a common height and labelling the columns.

    python3 scripts/make_domain_comparison.py
"""

import os

from PIL import Image, ImageDraw, ImageFont

SRC = 'results/plots'
OUT = 'results/plots/deck/domain_comparison.png'

LEFT = f'{SRC}/scenarios_illustrated_tight.png'
RIGHT = f'{SRC}/scenarios_tartanair_tight.png'

SERIF = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
SERIF_B = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'

HEAD_H = 104          # band for the column headings
GAP = 56              # gutter between the two domains
PAD = 18
INK = (26, 26, 26)
MUTED = (128, 128, 128)
RULE = (200, 200, 200)


def main():
    for p in (LEFT, RIGHT):
        if not os.path.exists(p):
            raise SystemExit(f'missing input: {p}')

    a, b = Image.open(LEFT).convert('RGB'), Image.open(RIGHT).convert('RGB')

    # Scale to a common height so the three scenario rows line up across the
    # two domains. Without this the rows drift and the comparison stops being
    # readable as a comparison.
    h = max(a.size[1], b.size[1])
    a = a.resize((round(a.size[0] * h / a.size[1]), h), Image.LANCZOS)
    b = b.resize((round(b.size[0] * h / b.size[1]), h), Image.LANCZOS)

    w = PAD * 2 + a.size[0] + GAP + b.size[0]
    canvas = Image.new('RGB', (w, HEAD_H + h + PAD), 'white')
    canvas.paste(a, (PAD, HEAD_H))
    canvas.paste(b, (PAD + a.size[0] + GAP, HEAD_H))

    d = ImageDraw.Draw(canvas)
    head = ImageFont.truetype(SERIF_B, 34)
    sub = ImageFont.truetype(SERIF, 25)

    for x0, width, title, note in (
        (PAD, a.size[0], 'Driving', 'Virtual KITTI 2, 26.6 px mean motion'),
        (PAD + a.size[0] + GAP, b.size[0], 'Aerial',
         'TartanAir, 9.26 px mean motion'),
    ):
        d.text((x0, 4), title, font=head, fill=INK)
        d.text((x0, 50), note, font=sub, fill=MUTED)
        d.line([(x0, HEAD_H - 12), (x0 + width, HEAD_H - 12)], fill=RULE, width=2)

    # divider between the domains, so the eye reads two panels rather than six
    xd = PAD + a.size[0] + GAP // 2
    d.line([(xd, 4), (xd, HEAD_H + h)], fill=RULE, width=2)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    canvas.save(OUT)
    print(f'{OUT}  {canvas.size[0]}x{canvas.size[1]}  '
          f'AR {canvas.size[0] / canvas.size[1]:.2f}')


if __name__ == '__main__':
    main()
