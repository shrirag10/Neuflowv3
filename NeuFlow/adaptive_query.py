"""
Adaptive Flow-Aware Query Strategy (InfiniDepth §3.3 Adaptation).

Instead of sampling query points uniformly, allocate more queries where
the optical flow has high spatial gradients (object boundaries, occluding
edges, fast-moving objects).  This concentrates the implicit decoder's
limited query budget where it matters most — critical for edge devices
running with N ≪ H×W.

Usage in training loop:
    from NeuFlow.adaptive_query import adaptive_flow_query

    coords = adaptive_flow_query(
        flow_gt, valid_mask,
        num_points=8192,
        adaptive_ratio=0.5,   # 50% adaptive, 50% uniform
    )
"""

import torch
import torch.nn.functional as F


def flow_gradient_magnitude(flow: torch.Tensor) -> torch.Tensor:
    """Compute spatial gradient magnitude of a flow field.

    Args:
        flow: [B, 2, H, W] optical flow.
    Returns:
        [B, H, W] gradient magnitude (sum of |∇u| + |∇v|).
    """
    # Sobel-like finite differences (central difference approximation)
    # Pad with replicate to avoid border artifacts
    flow_pad = F.pad(flow, (1, 1, 1, 1), mode='replicate')

    # Horizontal gradient: f(x+1) - f(x-1)
    grad_x = flow_pad[:, :, 1:-1, 2:] - flow_pad[:, :, 1:-1, :-2]
    # Vertical gradient: f(y+1) - f(y-1)
    grad_y = flow_pad[:, :, 2:, 1:-1] - flow_pad[:, :, :-2, 1:-1]

    # Magnitude per channel, then sum u and v
    grad_mag = (grad_x.abs() + grad_y.abs()).sum(dim=1)  # [B, H, W]
    return grad_mag


def adaptive_flow_query(
    flow_or_grad: torch.Tensor,
    valid_mask: torch.Tensor,
    num_points: int = 8192,
    adaptive_ratio: float = 0.5,
    jitter: bool = True,
    is_gradient: bool = False,
) -> torch.Tensor:
    """Sample query coordinates with flow-gradient-weighted importance sampling.

    Mixes adaptive (gradient-concentrated) and uniform random sampling.

    Args:
        flow_or_grad: [B, 2, H, W] GT flow or [B, H, W] precomputed gradient magnitude.
        valid_mask:   [B, H, W] boolean mask of valid pixels.
        num_points:   Total number of query points per batch element.
        adaptive_ratio: Fraction of points sampled adaptively (0=all uniform, 1=all adaptive).
        jitter: Whether to add sub-pixel jitter U(-0.5, 0.5).
        is_gradient: If True, flow_or_grad is already a gradient magnitude map.
    Returns:
        query_coords: [B, num_points, 2] pixel coordinates (x, y).
    """
    if is_gradient:
        grad_mag = flow_or_grad
    else:
        grad_mag = flow_gradient_magnitude(flow_or_grad)  # [B, H, W]

    B, H, W = grad_mag.shape
    device = grad_mag.device

    n_adaptive = int(num_points * adaptive_ratio)
    n_uniform = num_points - n_adaptive

    query_coords_list = []

    for b in range(B):
        valid_b = valid_mask[b]  # [H, W]
        grad_b = grad_mag[b]    # [H, W]

        # Get valid pixel indices
        valid_idx = valid_b.nonzero(as_tuple=False)  # [K, 2] (y, x)
        K = valid_idx.shape[0]

        coords_parts = []

        # --- Adaptive sampling (gradient-weighted) ---
        if n_adaptive > 0 and K > 0:
            # Build importance weights from gradient magnitude at valid pixels
            weights = grad_b[valid_idx[:, 0], valid_idx[:, 1]]  # [K]
            # Add small epsilon to ensure non-zero probability everywhere
            weights = weights + 1e-3
            weights = weights / weights.sum()

            # Multinomial sampling (with replacement for efficiency)
            idx = torch.multinomial(weights, min(n_adaptive, K), replacement=True)
            sel = valid_idx[idx]  # [n_adaptive, 2]
            yy = sel[:, 0].float()
            xx = sel[:, 1].float()
            coords_parts.append(torch.stack([xx, yy], dim=-1))

        # --- Uniform sampling ---
        if n_uniform > 0 and K > 0:
            if K >= n_uniform:
                perm = torch.randperm(K, device=device)[:n_uniform]
            else:
                perm = torch.randint(0, K, (n_uniform,), device=device)
            sel = valid_idx[perm]
            yy = sel[:, 0].float()
            xx = sel[:, 1].float()
            coords_parts.append(torch.stack([xx, yy], dim=-1))

        # --- Fallback if no valid pixels ---
        if K == 0:
            yy = torch.randint(0, H, (num_points,), device=device).float()
            xx = torch.randint(0, W, (num_points,), device=device).float()
            coords_parts.append(torch.stack([xx, yy], dim=-1))

        coords_b = torch.cat(coords_parts, dim=0)  # [num_points, 2]

        # Ensure exact count (handle rounding)
        if coords_b.shape[0] > num_points:
            coords_b = coords_b[:num_points]
        elif coords_b.shape[0] < num_points:
            # Pad by repeating
            deficit = num_points - coords_b.shape[0]
            pad_idx = torch.randint(0, coords_b.shape[0], (deficit,), device=device)
            coords_b = torch.cat([coords_b, coords_b[pad_idx]], dim=0)

        # --- Sub-pixel jitter ---
        if jitter:
            coords_b[:, 0] = (coords_b[:, 0] + torch.rand_like(coords_b[:, 0]) - 0.5).clamp(0, W - 1)
            coords_b[:, 1] = (coords_b[:, 1] + torch.rand_like(coords_b[:, 1]) - 0.5).clamp(0, H - 1)

        query_coords_list.append(coords_b)

    return torch.stack(query_coords_list, dim=0)  # [B, num_points, 2]


def coarse_flow_query(
    coarse_flow: torch.Tensor,
    num_points: int = 1000,
    adaptive_ratio: float = 0.7,
) -> torch.Tensor:
    """Inference-time adaptive query using the coarse flow's gradient.

    For edge-device inference where N is very small (100-1000),
    concentrate queries at motion boundaries detected from the 1/8 coarse flow.

    Args:
        coarse_flow: [B, 2, H8, W8] flow at 1/8 scale.
        num_points: Number of query points in full-res pixel space.
        adaptive_ratio: Fraction of points at high-gradient regions.
    Returns:
        query_coords: [B, num_points, 2] pixel coordinates (x, y) in full-res.
    """
    B, _, H8, W8 = coarse_flow.shape
    H_full = H8 * 8
    W_full = W8 * 8
    device = coarse_flow.device

    grad_mag = flow_gradient_magnitude(coarse_flow)  # [B, H8, W8]
    valid_mask = torch.ones(B, H8, W8, dtype=torch.bool, device=device)

    # Sample in coarse space
    coarse_coords = adaptive_flow_query(
        grad_mag, valid_mask,
        num_points=num_points,
        adaptive_ratio=adaptive_ratio,
        jitter=True,
        is_gradient=True,
    )  # [B, N, 2] in coarse (1/8) pixel space

    # Scale to full resolution
    coarse_coords[..., 0] = (coarse_coords[..., 0] * (W_full / W8)).clamp(0, W_full - 1)
    coarse_coords[..., 1] = (coarse_coords[..., 1] * (H_full / H8)).clamp(0, H_full - 1)

    return coarse_coords
