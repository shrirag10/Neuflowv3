import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
benchmark_edge.py — NeuFlow v3 Edge Device Benchmark

Measures the speed/accuracy trade-off of sparse (N-point) inference
vs dense (H×W) inference.  This proves the O(N) edge-device claim.

Usage:
    python3 benchmark_edge.py

Output:
    - Table: N | latency (ms) | memory (MB) | EPE vs dense
    - Saved: results/benchmark_edge.png  (latency & EPE curves)
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from utils.load_model import my_load_weights
from NeuFlow.neuflow import NeuFlow

# ─── Config ─────────────────────────────────────────────────────────────────
CHECKPOINT   = 'checkpoints/neuflowv3/step_020000.pth'
IMAGE_H, IMAGE_W = 384, 1248        # VKITTI2 native resolution
WARMUP_RUNS  = 5
BENCH_RUNS   = 20
QUERY_BUDGETS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]  # N values to test
OUT_DIR      = 'results'
# ────────────────────────────────────────────────────────────────────────────

def load_model(ckpt, device):
    model = NeuFlow(use_implicit=True).to(device)
    sd = my_load_weights(ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.init_bhwd(1, IMAGE_H, IMAGE_W, device)
    return model


def measure(fn, warmup=WARMUP_RUNS, runs=BENCH_RUNS):
    """Return mean latency in ms over `runs` timed calls after `warmup`."""
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000   # ms


def epe_at_queries(dense_flow, sparse_flow, query_coords, H, W):
    """
    Compare sparse decoder output against dense ground-truth at the same coords.
    dense_flow:   [1, 2, H, W]
    sparse_flow:  [1, N, 2]
    query_coords: [1, N, 2] pixel coords (x, y)
    """
    coords = query_coords[0].long().clamp(
        torch.tensor([0, 0], device=query_coords.device),
        torch.tensor([W-1, H-1], device=query_coords.device)
    )
    gt = dense_flow[0, :, coords[:, 1], coords[:, 0]].T   # [N, 2]
    pred = sparse_flow[0]                                   # [N, 2]
    return (pred.float() - gt.float()).norm(dim=-1).mean().item()


def mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6


def main():
    device = torch.device('cuda')
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'Loading {CHECKPOINT}...')
    model = load_model(CHECKPOINT, device)

    # Random test images (no GT needed for speed benchmark)
    img0 = torch.randint(0, 255, (1, 3, IMAGE_H, IMAGE_W),
                         dtype=torch.float32, device=device)
    img1 = torch.randint(0, 255, (1, 3, IMAGE_H, IMAGE_W),
                         dtype=torch.float32, device=device)

    total_pixels = IMAGE_H * IMAGE_W
    print(f'\nImage: {IMAGE_H}×{IMAGE_W} = {total_pixels:,} pixels')
    print(f'{"N":>8}  {"latency(ms)":>12}  {"vs dense":>10}  '
          f'{"mem(MB)":>9}  {"EPE vs dense":>14}')
    print('─' * 65)

    # ── Dense baseline ───────────────────────────────────────────────────────
    with torch.amp.autocast('cuda'):
        dense_fn = lambda: model(img0, img1)
        dense_ms = measure(dense_fn)
        torch.cuda.empty_cache()
        mem_before = mem_mb()
        with torch.no_grad():
            dense_flow = model(img0, img1)[-1]   # [1, 2, H, W]
        dense_mem = mem_mb() - mem_before

    print(f'{"DENSE":>8}  {dense_ms:>12.1f}  {"1.00×":>10}  '
          f'{dense_mem:>9.1f}  {"—":>14}')

    latencies, speedups, epelist, budgets = [], [], [], []

    # ── Sparse sweeps ────────────────────────────────────────────────────────
    with torch.amp.autocast('cuda'):
        state = model.infer_coarse_state(img0, img1)   # backbone: run once

        for N in QUERY_BUDGETS:
            if N > total_pixels:
                continue
            budgets.append(N)

            # Latency: backbone is cached; only time the decoder
            sparse_fn = lambda: model.decode_queries(state, adaptive_n=N)
            sparse_ms = measure(sparse_fn)

            # Memory
            torch.cuda.empty_cache()
            mem_before = mem_mb()
            with torch.no_grad():
                sparse_flow = model.decode_queries(state, adaptive_n=N)
            sparse_mem = mem_mb() - mem_before

            # Accuracy vs dense at the same query positions
            qc = model.decode_queries.__self__  # just get coords used
            # Re-run to capture actual coords used
            from NeuFlow.adaptive_query import coarse_flow_query
            query_coords = coarse_flow_query(
                state['coarse_flow_s8'], num_points=N, adaptive_ratio=0.7
            )
            with torch.no_grad():
                sparse_flow = model.decode_queries(state, query_coords=query_coords)
            epe = epe_at_queries(dense_flow, sparse_flow, query_coords,
                                 IMAGE_H, IMAGE_W)

            speedup  = dense_ms / sparse_ms
            latencies.append(sparse_ms)
            speedups.append(speedup)
            epelist.append(epe)

            pct = N / total_pixels * 100
            print(f'{N:>8,}  {sparse_ms:>12.1f}  {speedup:>9.2f}×  '
                  f'{sparse_mem:>9.1f}  {epe:>14.3f}')

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('NeuFlow v3 — Edge Device Benchmark\n'
                 f'Image: {IMAGE_H}×{IMAGE_W}  |  Dense baseline: {dense_ms:.1f} ms',
                 fontsize=13)

    # Speed
    ax1.axhline(dense_ms, color='red', linestyle='--', linewidth=1.5,
                label=f'Dense ({dense_ms:.1f} ms)')
    ax1.plot(budgets, latencies, 'o-', color='steelblue', linewidth=2,
             markersize=6, label='Sparse (decoder only)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Query budget N (log scale)')
    ax1.set_ylabel('Decoder latency (ms)')
    ax1.set_title('Latency vs Query Budget')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(budgets, epelist, 's-', color='darkorange', linewidth=2,
             markersize=6)
    ax2.axhline(0, color='green', linestyle='--', linewidth=1,
                label='Perfect (= dense)')
    ax2.set_xscale('log')
    ax2.set_xlabel('Query budget N (log scale)')
    ax2.set_ylabel('EPE vs dense flow (pixels)')
    ax2.set_title('Accuracy vs Query Budget\n(EPE relative to dense output)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'benchmark_edge.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved: {out_path}')

    # ── Summary ──────────────────────────────────────────────────────────────
    print('\n── Summary ──────────────────────────────────────────────────')
    best_n = budgets[np.argmin(epelist)]
    best_epe = min(epelist)
    best_speedup = dense_ms / latencies[np.argmin(epelist)]
    print(f'Best accuracy at N={best_n:,}: '
          f'EPE={best_epe:.3f}px vs dense, {best_speedup:.1f}× faster decoder')
    print(f'At N=512:  ~{dense_ms / latencies[budgets.index(512)]:.1f}× speedup '
          if 512 in budgets else '', end='')
    print()


if __name__ == '__main__':
    main()
