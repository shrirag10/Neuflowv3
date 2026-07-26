"""Sparse-query speed benchmark: the actual deployment-mode number.

Measures, for a real checkpoint:
  - coarse pass latency (infer_coarse_state, paid once per new image pair)
  - sparse decode latency at N queries (paid per query batch, reusable
    against the SAME cached state for free extra queries)
  - total = coarse + decode, the realistic "new frame, N queries" cost,
    directly comparable to v2's dense forward time on the same hardware.

Usage:
    python3 scripts/benchmark_sparse.py --checkpoint <path> --head convex \
        [--uncertainty] --n 800 --height 384 --width 1248
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch

from NeuFlow.neuflow import NeuFlow
from utils.load_model import my_load_weights, load_with_new_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--head', default='convex', choices=['regress', 'convex'])
    ap.add_argument('--uncertainty', action='store_true')
    ap.add_argument('--pe', action='store_true')
    ap.add_argument('--n', type=int, nargs='+', default=[800, 2048])
    ap.add_argument('--height', type=int, default=384)
    ap.add_argument('--width', type=int, default=1248)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--runs', type=int, default=30)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda'

    model = NeuFlow(use_implicit=True, head_mode=args.head, use_pe=args.pe,
                    predict_uncertainty=args.uncertainty).to(device)
    load_with_new_keys(model, my_load_weights(args.checkpoint),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
    model.eval()
    model.init_bhwd(1, args.height, args.width, device, amp=amp)

    img0 = torch.randint(0, 255, (1, 3, args.height, args.width), dtype=torch.float32, device=device)
    img1 = torch.randint(0, 255, (1, 3, args.height, args.width), dtype=torch.float32, device=device)

    def timeit(fn, warmup, runs):
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=amp):
            for _ in range(warmup):
                fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) / runs * 1000  # ms

    coarse_ms = timeit(lambda: model.infer_coarse_state(img0, img1), args.warmup, args.runs)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Input: {args.height}x{args.width}, device: {device}')
    print(f'\nCoarse pass (once per new frame pair): {coarse_ms:.2f} ms  ({1000/coarse_ms:.1f} FPS ceiling)')

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=amp):
        state = model.infer_coarse_state(img0, img1)

    print(f'\n{"N queries":>10}  {"decode (ms)":>12}  {"total ms":>10}  {"total FPS":>10}')
    for n in args.n:
        q = torch.rand(1, n, 2, device=device) * torch.tensor([args.width - 1., args.height - 1.], device=device)
        decode_ms = timeit(lambda: model.decode_queries(state, query_coords=q), args.warmup, args.runs)
        total_ms = coarse_ms + decode_ms
        print(f'{n:>10}  {decode_ms:>12.3f}  {total_ms:>10.2f}  {1000/total_ms:>10.1f}')

    print('\nNote: "total" = paying the coarse pass fresh for THIS frame + one decode batch.')
    print('Additional decode calls against the SAME state (e.g. a second query batch on the')
    print('same frame) cost only the decode ms above, not the coarse pass again.')


if __name__ == '__main__':
    main()
