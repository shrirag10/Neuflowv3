"""Splice the distilled refine_s8 weights into a checkpoint with a trained
decoder, producing one checkpoint that can be evaluated end-to-end.

Why this is a valid splice, not a hack: both source checkpoints share the
exact same frozen backbone/matching/refine_s16/merge weights (neither run
ever touched them), so there is no conflict -- only refine_s8 and the decoder
differ between the two, and each source contributes the piece it was
actually optimized for.

Usage:
    python3 scripts/merge_distill_decoder.py \
        --distilled /scratch/$USER/neuflow_ckpts/distill3/step_060000.pth \
        --decoder /scratch/$USER/neuflow_ckpts/big18_v3dev/step_100000.pth \
        --out /scratch/$USER/neuflow_ckpts/merged_distill_big18.pth
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch

from utils.load_model import my_load_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distilled', required=True, help='checkpoint with the trained refine_s8')
    ap.add_argument('--decoder', required=True, help='checkpoint with the trained decoder')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    distilled = my_load_weights(args.distilled)
    decoder_ck = my_load_weights(args.decoder)

    refine_keys = [k for k in distilled if k.startswith('refine_s8.')]
    if not refine_keys:
        raise RuntimeError('No refine_s8.* keys found in --distilled checkpoint')

    # sanity: everything OUTSIDE refine_s8/decoder must match between the two
    # sources exactly (both frozen from the same v2 base) -- if it doesn't,
    # they are not mergeable and this would silently produce a broken model.
    skip_prefixes = ('refine_s8.', 'implicit_decoder_module.')
    mismatches = []
    for k, v in decoder_ck.items():
        if k.startswith(skip_prefixes):
            continue
        if k not in distilled:
            mismatches.append(f'{k}: missing from --distilled')
        elif not torch.equal(v, distilled[k]):
            mismatches.append(f'{k}: value differs between checkpoints')
    if mismatches:
        print('ABORT: checkpoints are not mergeable, found differences outside refine_s8/decoder:')
        for m in mismatches[:20]:
            print(' ', m)
        sys.exit(1)

    merged = dict(decoder_ck)
    for k in refine_keys:
        merged[k] = distilled[k]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'model': merged}, args.out)
    print(f'Merged {len(refine_keys)} refine_s8 tensors from {args.distilled}')
    print(f'into decoder+backbone from {args.decoder}')
    print(f'Saved: {args.out}')
    print('\nEvaluate with:')
    print(f'  python3 scripts/eval_vkitti2.py --head convex --iters_s16 1 --iters_s8 3 \\')
    print(f'    --checkpoint {args.out} --dataset_root <vkitti2 root>')


if __name__ == '__main__':
    main()
