"""Tighten a figure for print: drop the white border and any wide internal gutter.

Matplotlib sizes each axes from the gridspec, but imshow centres the image
inside its axes, so a figure whose panels are narrower than the columns they
sit in ends up with a wide band of white between the columns and another down
one side. That reads as sloppy at print size and wastes column inches.

This collapses any all-white vertical band wider than --gutter down to a fixed
separation, and trims the outer margin. It is geometry-independent: no pixel
offsets are hard-coded, so it survives the figure being regenerated at a
different size.

    python3 scripts/tighten_figure.py results/plots/scenarios_tartanair.png \
        --drop-title --out results/plots/scenarios_tartanair_tight.png
"""

import argparse

import numpy as np
from PIL import Image


def white_runs(mask, min_len):
    """Start/stop indices of runs of False (all-white) at least min_len long."""
    runs, start = [], None
    for i, filled in enumerate(list(mask) + [True]):
        if not filled and start is None:
            start = i
        elif filled and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    return runs


def tighten(path, out, gutter_frac=0.05, keep=24, pad=10, drop_title=False,
            thresh=245):
    im = Image.open(path).convert('RGB')
    a = np.array(im)
    filled = (np.array(im.convert('L')) < thresh)

    if drop_title:
        # The suptitle sits above the panels, separated from them by a blank
        # band. Cut everything above the first such band. Once it is gone the
        # left margin it alone occupied becomes white and the border trim below
        # reclaims it.
        rows = filled.any(1)
        first = rows.argmax()
        gaps = white_runs(rows[first:], max(int(0.015 * a.shape[0]), 4))
        if gaps:
            filled[:first + gaps[0][1], :] = False

    cols, rows = filled.any(0), filled.any(1)
    if not cols.any():
        raise SystemExit(f'{path}: no content found')

    # collapse wide internal white bands between the first and last content column
    x0, x1 = cols.argmax(), len(cols) - cols[::-1].argmax()
    min_gap = max(int(gutter_frac * a.shape[1]), 1)
    drop = np.zeros(a.shape[1], bool)
    for s, e in white_runs(cols[x0:x1], min_gap):
        drop[x0 + s + keep // 2: x0 + e - keep // 2] = True

    y0, y1 = rows.argmax(), len(rows) - rows[::-1].argmax()
    y0, y1 = max(0, y0 - pad), min(a.shape[0], y1 + pad)
    x0, x1 = max(0, x0 - pad), min(a.shape[1], x1 + pad)

    keep_cols = ~drop[x0:x1]
    result = a[y0:y1, x0:x1][:, keep_cols]
    Image.fromarray(result).save(out)
    print(f'{path}  {a.shape[1]}x{a.shape[0]}  ->  {out}  '
          f'{result.shape[1]}x{result.shape[0]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gutter', type=float, default=0.05,
                    help='collapse white bands wider than this fraction of the width')
    ap.add_argument('--keep', type=int, default=24, help='px of separation to leave')
    ap.add_argument('--drop-title', action='store_true',
                    help='also cut the figure suptitle, when the caption already carries it')
    a = ap.parse_args()
    tighten(a.path, a.out, a.gutter, a.keep, drop_title=a.drop_title)
