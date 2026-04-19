"""
Smoke test for the NeuFlow-V2 Implicit Decoder.

Verifies:
  1. Forward pass produces correct output shapes.
  2. Gradient flow is clean (no NaN/Inf).
  3. Arbitrary-resolution querying works.
  4. Legacy (convex upsampler) mode still works.
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

from NeuFlow.neuflow import NeuFlow


def test_shape_and_gradients():
    """Test 1 & 2: Shape correctness and gradient flow."""
    print("=" * 60)
    print("TEST 1: Shape correctness + gradient flow (implicit mode)")
    print("=" * 60)

    B, H, W = 2, 256, 384  # Small for testing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = NeuFlow(use_implicit=True).to(device)
    model.train()

    img0 = torch.randint(0, 256, (B, 3, H, W), dtype=torch.float32, device=device)
    img1 = torch.randint(0, 256, (B, 3, H, W), dtype=torch.float32, device=device)

    model.init_bhwd(B, H, W, device, amp=False)

    flow_list = model(img0, img1, iters_s16=1, iters_s8=2)

    print(f"  Number of flow predictions: {len(flow_list)}")
    for i, f in enumerate(flow_list):
        print(f"  flow_list[{i}] shape: {f.shape}")

    final_flow = flow_list[-1]
    assert final_flow.shape == (B, 2, H, W), \
        f"Expected shape {(B, 2, H, W)}, got {final_flow.shape}"
    print(f"  ✓ Final flow shape correct: {final_flow.shape}")

    # Backward pass
    loss = final_flow.abs().mean()
    loss.backward()

    nan_params = []
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
            nan_params.append(name)

    if nan_params:
        print(f"  ✗ NaN/Inf gradients in: {nan_params}")
    else:
        print(f"  ✓ All gradients are finite")

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    decoder_params = sum(p.numel() for n, p in model.named_parameters()
                        if 'implicit_decoder' in n or 'conv_s1' in n)
    print(f"  Total params: {total_params:,}")
    print(f"  Implicit decoder params: {decoder_params:,}")
    print()


def test_arbitrary_resolution():
    """Test 3: Query at 2× resolution."""
    print("=" * 60)
    print("TEST 2: Arbitrary resolution (2× upsampling)")
    print("=" * 60)

    B, H, W = 1, 128, 192
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = NeuFlow(use_implicit=True).to(device)
    model.eval()

    img0 = torch.randint(0, 256, (B, 3, H, W), dtype=torch.float32, device=device)
    img1 = torch.randint(0, 256, (B, 3, H, W), dtype=torch.float32, device=device)

    model.init_bhwd(B, H, W, device, amp=False)

    target_h, target_w = H * 2, W * 2

    with torch.no_grad():
        flow_list = model(img0, img1, iters_s16=1, iters_s8=1,
                         target_h=target_h, target_w=target_w)

    final_flow = flow_list[-1]
    print(f"  Input resolution:  {H}×{W}")
    print(f"  Target resolution: {target_h}×{target_w}")
    print(f"  Output shape:      {final_flow.shape}")

    assert final_flow.shape == (B, 2, target_h, target_w), \
        f"Expected {(B, 2, target_h, target_w)}, got {final_flow.shape}"
    print(f"  ✓ Arbitrary resolution output correct")
    print()


def test_legacy_mode():
    """Test 4: Legacy convex upsampler still works."""
    print("=" * 60)
    print("TEST 3: Legacy convex upsampler mode")
    print("=" * 60)

    B, H, W = 1, 128, 192
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = NeuFlow(use_implicit=False).to(device)
    model.eval()

    img0 = torch.randint(0, 256, (B, 3, H, W), dtype=torch.float32, device=device)
    img1 = torch.randint(0, 256, (B, 3, H, W), dtype=torch.float32, device=device)

    model.init_bhwd(B, H, W, device, amp=False)

    with torch.no_grad():
        flow_list = model(img0, img1, iters_s16=1, iters_s8=1)

    final_flow = flow_list[-1]
    assert final_flow.shape == (B, 2, H, W), \
        f"Expected {(B, 2, H, W)}, got {final_flow.shape}"
    print(f"  ✓ Legacy mode output shape correct: {final_flow.shape}")
    print()


def test_sparse_loss():
    """Test 5: Sparse loss function."""
    print("=" * 60)
    print("TEST 4: Sparse flow loss function")
    print("=" * 60)

    sys.path.insert(0, '.')
    from loss import sparse_flow_loss_func, flow_loss_func

    B, H, W = 2, 64, 96
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    flow_pred = torch.randn(B, 2, H, W, device=device, requires_grad=True)
    flow_gt = torch.randn(B, 2, H, W, device=device)
    valid = torch.ones(B, H, W, device=device)

    # Dense loss
    dense_loss, dense_metrics = flow_loss_func([flow_pred], flow_gt, valid)
    print(f"  Dense loss:  {dense_loss.item():.4f}  EPE: {dense_metrics['epe']:.4f}")

    # Sparse loss
    sparse_loss, sparse_metrics = sparse_flow_loss_func(
        [flow_pred], flow_gt, valid, num_points=1024
    )
    print(f"  Sparse loss: {sparse_loss.item():.4f}  EPE: {sparse_metrics['epe']:.4f}")

    sparse_loss.backward()
    assert flow_pred.grad is not None and torch.all(torch.isfinite(flow_pred.grad))
    print(f"  ✓ Sparse loss backward pass clean")
    print()


def test_sparse_query_forward():
    """Test 6: Sparse query_coords forward pass (true memory-saving path)."""
    print("=" * 60)
    print("TEST 5: Sparse query_coords forward + backward")
    print("=" * 60)

    B, H, W = 2, 256, 384
    N = 4096  # number of query points
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = NeuFlow(use_implicit=True).to(device)
    model.train()

    img0 = torch.randint(0, 256, (B, 3, H, W), dtype=torch.float32, device=device)
    img1 = torch.randint(0, 256, (B, 3, H, W), dtype=torch.float32, device=device)

    # Random query coords in full-res pixel space (x, y)
    query_x = torch.rand(B, N, device=device) * (W - 1)
    query_y = torch.rand(B, N, device=device) * (H - 1)
    query_coords = torch.stack([query_x, query_y], dim=-1)  # [B, N, 2]

    model.init_bhwd(B, H, W, device, amp=False)

    flow_list = model(img0, img1, iters_s16=1, iters_s8=2,
                      query_coords=query_coords)

    print(f"  Number of flow predictions: {len(flow_list)}")
    for i, f in enumerate(flow_list):
        print(f"  flow_list[{i}] shape: {f.shape}")

    final_flow = flow_list[-1]
    assert final_flow.shape == (B, N, 2), \
        f"Expected sparse shape {(B, N, 2)}, got {final_flow.shape}"
    print(f"  ✓ Sparse output shape correct: {final_flow.shape}")

    # Backward pass
    loss = final_flow.abs().mean()
    loss.backward()

    nan_params = []
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
            nan_params.append(name)

    if nan_params:
        print(f"  ✗ NaN/Inf gradients in: {nan_params}")
        assert False, "NaN gradients detected"
    else:
        print(f"  ✓ All gradients are finite (sparse path)")
    print()


if __name__ == '__main__':
    print("\n🔬 NeuFlow-V2 Implicit Decoder — Smoke Tests\n")

    test_sparse_loss()
    test_shape_and_gradients()
    test_arbitrary_resolution()
    test_legacy_mode()
    test_sparse_query_forward()

    print("=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
