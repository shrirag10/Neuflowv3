"""
ImplicitFlowDecoder v2 — Edge-optimized multi-scale local implicit decoder.

Key change from v1: Eliminated the full-resolution dense feature map dependency.
Instead of running conv_s1 on ALL H×W pixels and then sampling, we now extract
small local patches (k×k) from the raw image directly at each query point and
project them through a lightweight linear layer.  This makes the decoder's
compute cost O(N) in the number of query points, independent of image resolution.

References:
  - InfiniDepth (Yu et al., 2026): multi-scale local implicit depth decoder.
  - AnyFlow (Hui et al., 2023): implicit neural flow upsampler with PE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Positional Encoding (sinusoidal Fourier features, as in AnyFlow / NeRF)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding of 2-D relative coordinates.

    Given (dx, dy) ∈ ℝ², produces a vector of size 2 + 4*L:
        [dx, dy, sin(2^0·π·dx), cos(2^0·π·dx), ...,
                  sin(2^{L-1}·π·dy), cos(2^{L-1}·π·dy)]
    """

    def __init__(self, num_bands: int = 6):
        super().__init__()
        self.num_bands = num_bands
        # Pre-compute frequency multipliers: 2^0 … 2^{L-1}
        freqs = torch.tensor([2.0 ** i for i in range(num_bands)]) * math.pi
        self.register_buffer("freqs", freqs)  # [L]

    @property
    def out_dim(self) -> int:
        """Output dimensionality: raw 2 + sin/cos for each band & each coord."""
        return 2 + 2 * 2 * self.num_bands  # 2 + 4L

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [..., 2]  (dx, dy) relative coordinates.
        Returns:
            [..., out_dim] positional-encoded vector.
        """
        # coords shape: [..., 2]
        # Expand freqs for broadcasting: [1, ..., 1, L] * [..., 2, 1]
        x = coords.unsqueeze(-1) * self.freqs  # [..., 2, L]
        # sin / cos
        enc = torch.cat([x.sin(), x.cos()], dim=-1)  # [..., 2, 2L]
        enc = enc.flatten(-2)  # [..., 4L]
        return torch.cat([coords, enc], dim=-1)  # [..., 2 + 4L]


# ---------------------------------------------------------------------------
#  Gated Feed-Forward Fusion Block (InfiniDepth §3.2, Eq. 3)
# ---------------------------------------------------------------------------
class GatedFusionBlock(nn.Module):
    """
    h_{k+1} = FFN( f_{k+1}  +  g_k ⊙ Linear(h_k) )

    Where:
        f_{k+1}: feature from the *next* (deeper / lower-res) scale.
        h_k:     running fused representation from the previous (shallower) scale.
        g_k:     learnable channel-wise sigmoid gate.
        FFN:     2-layer feed-forward network with GELU.
    """

    def __init__(self, in_dim_h: int, in_dim_f: int, out_dim: int, expansion: int = 2):
        super().__init__()
        self.proj_h = nn.Linear(in_dim_h, out_dim)
        # Learnable gate initialised at 0 → sigmoid(0)=0.5 (balanced start)
        self.gate = nn.Parameter(torch.zeros(out_dim))
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * expansion),
            nn.GELU(),
            nn.Linear(out_dim * expansion, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, f_next: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:      [N, in_dim_h]  fused repr from shallower scale.
            f_next: [N, in_dim_f]  queried feature from deeper scale.
                    (in_dim_f == out_dim assumed after external projection)
        Returns:
            [N, out_dim]
        """
        g = torch.sigmoid(self.gate)  # [out_dim]
        fused = f_next + g * self.proj_h(h)  # [N, out_dim]
        fused = self.ffn(fused)
        fused = self.norm(fused)
        return fused


# ---------------------------------------------------------------------------
#  Implicit Flow Decoder v2 (Edge-Optimized)
# ---------------------------------------------------------------------------
class ImplicitFlowDecoder(nn.Module):
    """Multi-scale local implicit decoder for optical flow.

    v2 changes (edge-optimized):
      - Removed dependency on dense 1/1-scale feature map.
      - Instead, extracts a local k×k RGB patch at each query point and
        projects it through a linear layer.  Compute is O(N) not O(H×W).
      - Compatible with the same training loop and inference API as v1.
    """

    def __init__(
        self,
        feat_dim_s8: int = 128,
        feat_dim_s16: int = 128,
        pe_bands: int = 6,
        ffn_expansion: int = 2,
        mlp_hidden: int = 128,
        patch_size: int = 5,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.pe = PositionalEncoding(num_bands=pe_bands)
        pe_dim = self.pe.out_dim  # 2 + 4*6 = 26

        # --- Local patch → feature projection (replaces conv_s1) -----------
        # A k×k RGB patch has 3*k*k dimensions
        patch_feat_dim = 3 * patch_size * patch_size  # e.g., 75 for 5×5
        hidden = feat_dim_s8  # 128 — consistent hidden size

        self.proj_patch = nn.Sequential(
            nn.Linear(patch_feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # --- Projection layers for each backbone scale to common dim -------
        self.proj_s8 = nn.Linear(feat_dim_s8, hidden)
        self.proj_s16 = nn.Linear(feat_dim_s16, hidden)

        # --- Hierarchical fusion blocks -----------------------------------
        # shallow (local patch) → mid (1/8)
        self.fuse_patch_to_8 = GatedFusionBlock(
            in_dim_h=hidden, in_dim_f=hidden, out_dim=hidden,
            expansion=ffn_expansion,
        )
        # mid (1/8) → deep (1/16)
        self.fuse_8_to_16 = GatedFusionBlock(
            in_dim_h=hidden, in_dim_f=hidden, out_dim=hidden,
            expansion=ffn_expansion,
        )

        # --- Flow MLP head ------------------------------------------------
        # Input: fused_feat (hidden) + PE (pe_dim) + coarse_flow (2)
        mlp_in = hidden + pe_dim + 2
        self.flow_head = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Linear(mlp_hidden // 2, 2),  # (du, dv)
        )

    # ----- helper: sample features with a normalized grid ------
    @staticmethod
    def _sample_with_norm_grid(
        feat_map: torch.Tensor, norm_grid: torch.Tensor
    ) -> torch.Tensor:
        """Bilinear-sample a feature map using an already-normalised grid.

        Args:
            feat_map:  [B, C, H_f, W_f]
            norm_grid: [B, N, 1, 2]  grid in [-1, 1] (x, y order).
        Returns:
            [B, N, C]
        """
        sampled = F.grid_sample(
            feat_map, norm_grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )  # [B, C, N, 1]
        return sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, C]

    def _extract_local_patches(
        self, img: torch.Tensor, query_coords: torch.Tensor
    ) -> torch.Tensor:
        """Extract k×k RGB patches around each query point via grid_sample.

        Args:
            img:           [B, 3, H, W] normalized image (0-1 range).
            query_coords:  [B, N, 2] pixel coordinates (x, y) in image space.
        Returns:
            [B, N, 3*k*k] flattened patch features.
        """
        B, _, H, W = img.shape
        N = query_coords.shape[1]
        k = self.patch_size
        half_k = k // 2

        # Build k×k offset grid: [-half_k, ..., +half_k]
        offsets = torch.arange(-half_k, half_k + 1, dtype=img.dtype, device=img.device)
        oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')  # [k, k]
        # Flatten to [k*k, 2] (dx, dy) → but grid_sample wants (x, y) order
        offset_grid = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)  # [k*k, 2]

        # Expand query_coords: [B, N, 1, 2] + [1, 1, k*k, 2] → [B, N, k*k, 2]
        coords_expanded = query_coords.unsqueeze(2) + offset_grid.unsqueeze(0).unsqueeze(0)

        # Normalize to [-1, 1] for grid_sample
        coords_norm = coords_expanded.clone()
        coords_norm[..., 0] = 2.0 * coords_norm[..., 0] / max(W - 1, 1) - 1.0
        coords_norm[..., 1] = 2.0 * coords_norm[..., 1] / max(H - 1, 1) - 1.0

        # Reshape for grid_sample: [B, N*k*k, 1, 2]
        grid = coords_norm.reshape(B, N * k * k, 1, 2)

        # Sample: [B, 3, N*k*k, 1] → [B, 3, N*k*k] → [B, N*k*k, 3]
        sampled = F.grid_sample(
            img, grid, mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze(-1).permute(0, 2, 1)  # [B, N*k*k, 3]

        # Reshape to [B, N, k*k*3]
        patches = sampled.reshape(B, N, k * k * 3)
        return patches

    def forward(
        self,
        img: torch.Tensor,             # [B, 3, H, W]        — raw normalized image
        feat_s8: torch.Tensor,          # [B, C8, H/8, W/8]   — 1/8 features
        feat_s16: torch.Tensor,         # [B, C16, H/16, W/16] — 1/16 features
        coarse_flow: torch.Tensor,      # [B, 2, H/8, W/8]    — refined flow at 1/8
        query_coords: torch.Tensor | None = None,  # [B, N, 2] full-res pixel coords
        target_h: int | None = None,
        target_w: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img:       Raw normalized image (0-1).
            feat_s8:   1/8 feature map (from backbone merge).
            feat_s16:  1/16 feature map (from cross-attention).
            coarse_flow: Refined flow at 1/8 scale.
            query_coords: Optional explicit query points [B, N, 2] in full-res
                          pixel space (x, y).  If None, a dense grid at
                          (target_h × target_w) is generated.
            target_h, target_w: Target output resolution.
        Returns:
            flow: [B, 2, target_h, target_w]  (dense) or [B, N, 2] (sparse).
        """
        B, _, H_full, W_full = img.shape

        if target_h is None:
            target_h = H_full
        if target_w is None:
            target_w = W_full

        dense_query = query_coords is None
        if dense_query:
            # Build a dense grid of query pixel coordinates
            ys = torch.arange(target_h, dtype=img.dtype, device=img.device)
            xs = torch.arange(target_w, dtype=img.dtype, device=img.device)

            if target_h != H_full or target_w != W_full:
                ys = ys * ((H_full - 1) / max(target_h - 1, 1))
                xs = xs * ((W_full - 1) / max(target_w - 1, 1))

            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            query_coords = torch.stack([grid_x, grid_y], dim=-1)
            query_coords = query_coords.reshape(1, -1, 2).expand(B, -1, -1)

        N = query_coords.shape[1]

        # === Compute normalised grid in [-1, 1] for feature sampling ======
        norm_grid = query_coords.clone()
        norm_grid[..., 0] = 2.0 * norm_grid[..., 0] / max(W_full - 1, 1) - 1.0
        norm_grid[..., 1] = 2.0 * norm_grid[..., 1] / max(H_full - 1, 1) - 1.0
        norm_grid = norm_grid.unsqueeze(2)  # [B, N, 1, 2]

        # --- 1. Bilinearly upsample coarse flow to query locations ---------
        coarse_at_query = self._sample_with_norm_grid(coarse_flow, norm_grid)  # [B, N, 2]
        _, _, H8, W8 = coarse_flow.shape
        scale_x = W_full / W8
        scale_y = H_full / H8
        coarse_at_query[..., 0] = coarse_at_query[..., 0] * scale_x
        coarse_at_query[..., 1] = coarse_at_query[..., 1] * scale_y

        # --- 2. Extract local patches at query points (replaces conv_s1) ---
        f_local = self._extract_local_patches(img, query_coords)  # [B, N, 3*k*k]
        f_local = self.proj_patch(f_local)  # [B, N, hidden]

        # --- 3. Query multi-scale backbone features ------------------------
        f8  = self._sample_with_norm_grid(feat_s8,  norm_grid)  # [B, N, C8]
        f16 = self._sample_with_norm_grid(feat_s16, norm_grid)  # [B, N, C16]

        # --- 4. Project to common hidden dim -------------------------------
        f8  = self.proj_s8(f8)     # [B, N, hidden]
        f16 = self.proj_s16(f16)   # [B, N, hidden]

        # Flatten batch & query dims for MLP processing
        f_local = f_local.reshape(B * N, -1)
        f8  = f8.reshape(B * N, -1)
        f16 = f16.reshape(B * N, -1)

        # --- 5. Hierarchical fusion: shallow → deep -----------------------
        h = f_local                                # start: local patch
        h = self.fuse_patch_to_8(h, f8)            # fuse with 1/8
        h = self.fuse_8_to_16(h, f16)              # fuse with 1/16

        # --- 6. Positional encoding of LOCAL relative coords ---------------
        cell_w = W_full / W8   # ≈ 8.0
        cell_h = H_full / H8   # ≈ 8.0
        query_in_coarse_x = query_coords[..., 0] / cell_w
        query_in_coarse_y = query_coords[..., 1] / cell_h
        rel_x = query_in_coarse_x - torch.round(query_in_coarse_x)  # ∈ [-0.5, 0.5]
        rel_y = query_in_coarse_y - torch.round(query_in_coarse_y)
        local_coords = torch.stack([rel_x * 2.0, rel_y * 2.0], dim=-1)  # [B, N, 2]
        pe = self.pe(local_coords)  # [B, N, pe_dim]
        pe = pe.reshape(B * N, -1)

        coarse_flat = coarse_at_query.reshape(B * N, 2)

        # --- 7. MLP head → residual flow ---------------------------------
        mlp_input = torch.cat([h, pe, coarse_flat], dim=-1)
        delta_flow = self.flow_head(mlp_input)  # [B*N, 2]

        # --- 8. Add residual to coarse flow & reshape ---------------------
        flow = coarse_flat + delta_flow
        flow = flow.reshape(B, N, 2)

        if dense_query:
            flow = flow.reshape(B, target_h, target_w, 2).permute(0, 3, 1, 2)

        return flow
