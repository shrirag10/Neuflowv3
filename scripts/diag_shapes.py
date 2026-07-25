"""Print sample shapes from a training stage to find collate mismatches.

Usage: python3 scripts/diag_shapes.py vkitti2_mix
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from data_utils.datasets import build_train_dataset

stage = sys.argv[1] if len(sys.argv) > 1 else 'vkitti2_mix'
ds = build_train_dataset(stage)
print(f'stage={stage} len={len(ds)}')

shapes = Counter()
bad = 0
random.seed(0)
idxs = random.sample(range(len(ds)), min(60, len(ds)))
for i in idxs:
    try:
        s = ds[i]
        key = tuple(tuple(x.shape) for x in s[:3])
        shapes[key] += 1
        if len(shapes) > 1 and shapes[key] == 1:
            print(f'NEW SHAPE at idx {i}: {key}')
    except Exception as e:
        bad += 1
        print(f'idx {i}: LOAD ERROR {type(e).__name__}: {e}')

print('\nshape histogram:')
for k, v in shapes.items():
    print(f'  {v:3d} x {k}')
print(f'load errors: {bad}/{len(idxs)}')
