"""Equivalence test for local-window feature sampling.

Verifies four properties before any training run:
  1. center-init makes _sample_local_window output match _sample_features
     within float32 tolerance — warm-start causes zero regression at step 0.
  2. Output shape is [B, N, C].
  3. Non-center cells are sampled at distinct locations (window is real).
  4. Gradients flow through both grid_sample and the projection linear.

Run on CPU, no dataset required:
    python3 scripts/test_window_equiv.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from NeuFlow.implicit_decoder import ImplicitFlowDecoder


def make_decoder(window_size=3):
    dec = ImplicitFlowDecoder(
        feat_dim_s8=128,
        feat_dim_ctx=64,
        hidden_dim=128,
        window_size=window_size,
    )
    dec.eval()
    return dec


def random_feat(B, C, H, W):
    return torch.randn(B, C, H, W)


def random_coords_norm(B, N):
    # Uniform in (-0.9, 0.9) to stay away from border
    return (torch.rand(B, N, 2) * 1.8 - 0.9)


# ─── Test 1: center-init equivalence ────────────────────────────────────────

def test_center_init_equivalence():
    torch.manual_seed(0)
    B, N, C, H, W = 2, 64, 128, 32, 32

    dec = make_decoder(window_size=3)
    dec.reset_window_projections_to_center()

    feat = random_feat(B, C, H, W)
    coords = random_coords_norm(B, N)

    with torch.no_grad():
        out_window = dec._sample_local_window(feat, coords, dec.window_size, dec.win_proj_s8)
        out_point  = dec._sample_features(feat, coords)

    max_diff = (out_window - out_point).abs().max().item()
    assert max_diff < 1e-5, (
        f'center-init equivalence FAILED: max abs diff = {max_diff:.2e} (expected < 1e-5)'
    )
    print(f'[PASS] center-init equivalence  max_diff={max_diff:.2e}')


# ─── Test 2: output shape ────────────────────────────────────────────────────

def test_output_shape():
    torch.manual_seed(1)
    B, N, C_s8, C_ctx, H8, W8 = 2, 128, 128, 64, 30, 40

    dec = make_decoder(window_size=3)
    dec.reset_window_projections_to_center()

    feat_s8  = random_feat(B, C_s8, H8, W8)
    feat_s16 = random_feat(B, C_s8, H8 // 2, W8 // 2)
    ctx_s8   = random_feat(B, C_ctx, H8, W8)
    coords   = random_coords_norm(B, N)

    with torch.no_grad():
        f8   = dec._sample_local_window(feat_s8,  coords, 3, dec.win_proj_s8)
        f16  = dec._sample_local_window(feat_s16, coords, 3, dec.win_proj_s16)
        fctx = dec._sample_local_window(ctx_s8,   coords, 3, dec.win_proj_ctx)

    assert f8.shape   == (B, N, C_s8),  f'win_proj_s8 shape wrong: {f8.shape}'
    assert f16.shape  == (B, N, C_s8),  f'win_proj_s16 shape wrong: {f16.shape}'
    assert fctx.shape == (B, N, C_ctx), f'win_proj_ctx shape wrong: {fctx.shape}'
    print(f'[PASS] output shapes  f8={tuple(f8.shape)}  f16={tuple(f16.shape)}  fctx={tuple(fctx.shape)}')


# ─── Test 3: non-center cells sample distinct locations ──────────────────────

def test_window_samples_distinct_locations():
    """Verify the 3x3 window samples 9 distinct spatial locations.

    Uses a feature map where each spatial cell has a unique value. After
    center-init, win_proj passes through only the center cell. We temporarily
    set all 9 cells' weights equal to verify the 9 samples are distinct.
    """
    torch.manual_seed(2)
    B, N, C, H, W = 1, 4, 128, 16, 16

    # Nonlinear spatial fingerprint: a symmetric 3x3 average differs from center.
    feat = torch.zeros(B, C, H, W)
    for y in range(H):
        for x in range(W):
            feat[:, :, y, x] = float(y * y * W + x * x)

    dec = make_decoder(window_size=3)

    # Set all k*k blocks equal so all 9 cells contribute equally
    k2 = 9
    with torch.no_grad():
        for proj, c in [(dec.win_proj_s8, C)]:
            proj.weight.zero_()
            for i in range(k2):
                proj.weight[:, i * c : (i + 1) * c] = torch.eye(c) / k2
            if proj.bias is not None:
                proj.bias.zero_()

    # Pick a center coord that has 8 neighbors clearly within bounds
    cx, cy = 0.0, 0.0  # normalized center
    coords = torch.tensor([[[cx, cy]]]).expand(B, N, -1).clone()

    with torch.no_grad():
        out = dec._sample_local_window(feat, coords, 3, dec.win_proj_s8)

    # If all 9 samples were the same location, out would equal center cell value exactly.
    # With distinct neighbors, out should be the mean of 9 different values.
    center_val = dec._sample_features(feat, coords).mean().item()
    window_val = out.mean().item()

    assert abs(center_val - window_val) > 1e-3, (
        f'Window appears to sample only the center: center={center_val:.4f} window={window_val:.4f}'
    )
    print(f'[PASS] distinct window locations  center_val={center_val:.3f}  window_mean={window_val:.3f}')


# ─── Test 4: gradients flow ──────────────────────────────────────────────────

def test_gradients_flow():
    torch.manual_seed(3)
    B, N, C, H, W = 1, 32, 128, 20, 20

    dec = make_decoder(window_size=3)
    dec.reset_window_projections_to_center()

    feat = random_feat(B, C, H, W).requires_grad_(True)
    coords = random_coords_norm(B, N)

    out = dec._sample_local_window(feat, coords, 3, dec.win_proj_s8)
    loss = out.sum()
    loss.backward()

    assert feat.grad is not None, 'No gradient on feat'
    assert feat.grad.abs().sum().item() > 0, 'Zero gradient on feat'
    assert dec.win_proj_s8.weight.grad is not None, 'No gradient on win_proj_s8.weight'
    assert dec.win_proj_s8.weight.grad.abs().sum().item() > 0, 'Zero gradient on win_proj_s8.weight'
    print('[PASS] gradients flow through grid_sample and win_proj')


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Running window equivalence tests...\n')
    test_center_init_equivalence()
    test_output_shape()
    test_window_samples_distinct_locations()
    test_gradients_flow()
    print('\nAll tests passed.')
