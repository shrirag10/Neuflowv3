"""Confirm no training pair appears in the evaluation split.

Run before any training job. Must print OVERLAP 0.

    python3 scripts/check_leak.py --stage FlyingChairs+VKITTI2 \
        --dataset_root /scratch/$USER/neuflow_datasets/vkitti2
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils.datasets import build_train_dataset
from eval_vkitti2 import build_vkitti2_val_pairs


def collect(ds):
    """Flatten a possibly-concatenated dataset into its image pair list."""
    pairs = set()
    stack = [ds]
    while stack:
        d = stack.pop()
        if hasattr(d, 'datasets'):
            stack.extend(d.datasets)
            continue
        if hasattr(d, 'dataset'):
            stack.append(d.dataset)
            continue
        for x in getattr(d, 'image_list', []):
            pairs.add(tuple(x))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='FlyingChairs+VKITTI2')
    ap.add_argument('--dataset_root', default='datasets/vkitti2')
    ap.add_argument('--val_scenes', nargs='+', default=['Scene18', 'Scene20'])
    args = ap.parse_args()

    val_pairs = build_vkitti2_val_pairs(args.dataset_root, args.val_scenes)
    val = set((a, b) for a, b, _ in val_pairs)
    val_files = set()
    for a, b, f in val_pairs:
        val_files.update([a, b, f])

    train = collect(build_train_dataset(args.stage))
    overlap = train & val

    # a training pair could also reuse a val FRAME without being the same pair
    train_files = set()
    for p in train:
        train_files.update(p)
    file_overlap = train_files & val_files

    print(f'stage           : {args.stage}')
    print(f'train pairs     : {len(train):,}')
    print(f'val pairs       : {len(val):,}')
    print(f'OVERLAP (pairs) : {len(overlap)}')
    print(f'OVERLAP (frames): {len(file_overlap)}')
    for p in list(overlap)[:5]:
        print('  leaked pair:', p)
    for f in list(file_overlap)[:5]:
        print('  leaked frame:', f)

    ok = not overlap and not file_overlap
    print('RESULT: ' + ('CLEAN' if ok else 'LEAK DETECTED -- do not train'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
