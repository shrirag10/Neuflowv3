# Implicit flow decoder adapted from InfiniDepth (Yu et al., 2025).
# InfiniDepth uses ViT features (256/512/1024d) for depth — here we use
# NeuFlow's CNN backbone features (64/128/128d) for optical flow.
#
# Three feature scales:
#   ctx_s8   (64d, 1/8)  — appearance context
#   feat_s8  (128d, 1/8) — cross-frame matching
#   feat_s16 (128d, 1/16) — global/semantic
#
# Hierarchical fusion (InfiniDepth Eq. 3, applied twice):
#   h2 = FFN1(feat_s8  + σ(g1) * Linear(ctx_s8))
#   h3 = FFN2(feat_s16 + σ(g2) * Linear(h2))
#
# Local-window sampling (3x3 by default):
#   Each query point samples a k×k neighborhood in feature-map pixel space.
#   Offsets are per-feature-pixel (2/Hf per step), NOT per full-res pixel.
#   At s8 stride: 3×3 window = ±8 full-res pixels of spatial context.
#   win_proj_* collapses [k*k*C → C], so downstream dims are unchanged.
#   center-init on win_proj_* means behavior is identical to point-sampling
#   at step 0 — gradient descent expands the effective receptive field.
#
# MLP input: [h3 | feat1_warped | (x,y)_norm | u_coarse_norm] = 260d
# Output: delta_flow, added to bilinear-upsampled coarse flow.
# Zeroing the output layer starts from the bilinear coarse-flow base, not from
# the legacy v2 convex upsampler.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_list:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ImplicitFlowDecoder(nn.Module):

    def __init__(
        self,
        feat_dim_s8: int = 128,   # feat_s8 / feat_s16 channel dim (same in NeuFlow)
        feat_dim_ctx: int = 64,   # context_s8 channel dim
        hidden_dim: int = 128,    # hidden dim throughout fusion
        hidden_list: list = None,
        window_size: int = 3,     # local-window size (must be odd)
    ):
        super().__init__()

        if window_size % 2 != 1:
            raise ValueError(f'window_size must be odd, got {window_size}')

        if hidden_list is None:
            hidden_list = [256, 128, 64]

        self.feat_dim_s8  = feat_dim_s8
        self.feat_dim_ctx = feat_dim_ctx
        self.window_size  = window_size

        # --- Local-window projectors: k*k*C → C ---
        # Do NOT call center-init here; NeuFlow.__init__ runs Xavier over all
        # parameters after constructing this module, which would overwrite it.
        # reset_window_projections_to_center() is called there after the Xavier pass.
        k2 = window_size ** 2
        self.win_proj_ctx   = nn.Linear(k2 * feat_dim_ctx, feat_dim_ctx)   # 576→64
        self.win_proj_s8    = nn.Linear(k2 * feat_dim_s8,  feat_dim_s8)    # 1152→128
        self.win_proj_s16   = nn.Linear(k2 * feat_dim_s8,  feat_dim_s8)    # 1152→128
        self.win_proj_feat1 = nn.Linear(k2 * feat_dim_s8,  feat_dim_s8)    # 1152→128

        # --- Fusion: ctx_s8 -> feat_s8 -> feat_s16 (shallow to deep) ---
        self.proj_ctx = nn.Linear(feat_dim_ctx, hidden_dim)
        self.gate1    = nn.Parameter(torch.ones(hidden_dim))
        self.ffn1     = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.proj_s8 = nn.Linear(feat_dim_s8, hidden_dim)
        self.gate2   = nn.Parameter(torch.ones(hidden_dim))
        self.ffn2    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # MLP: cat([h3 | feat1_warped | coords_norm | coarse_norm]) → delta
        self.flow_head = MLP(
            in_dim=hidden_dim + feat_dim_s8 + 2 + 2,   # 128+128+2+2 = 260
            out_dim=2,
            hidden_list=hidden_list,
        )

    def reset_window_projections_to_center(self):
        """Set win_proj_* to identity on the center cell, zero elsewhere.

        After this call, _sample_local_window produces the same output as
        _sample_features (single-point bilinear), so warm-starting from a
        pre-window checkpoint causes zero regression at step 0.

        Must be called AFTER NeuFlow's global Xavier init, not inside __init__.
        """
        k2 = self.window_size ** 2
        center = k2 // 2  # index 4 for 3×3

        for proj, C in [
            (self.win_proj_ctx,   self.feat_dim_ctx),
            (self.win_proj_s8,    self.feat_dim_s8),
            (self.win_proj_s16,   self.feat_dim_s8),
            (self.win_proj_feat1, self.feat_dim_s8),
        ]:
            with torch.no_grad():
                proj.weight.zero_()
                # Only the center cell's block contributes — equivalent to single-point sampling
                proj.weight[:, center * C : (center + 1) * C] = torch.eye(C)
                if proj.bias is not None:
                    proj.bias.zero_()

    @staticmethod
    def _sample_features(feat_map: torch.Tensor, coords_norm: torch.Tensor) -> torch.Tensor:
        """Bilinear sample feat_map at normalized coords (x,y) in [-1,1]."""
        grid = coords_norm.to(feat_map.dtype).unsqueeze(1)  # [B, 1, N, 2]
        sampled = F.grid_sample(
            feat_map, grid,
            mode='bilinear', padding_mode='border', align_corners=False
        )  # [B, C, 1, N]
        return sampled.squeeze(2).permute(0, 2, 1)  # [B, N, C]

    @staticmethod
    def _sample_local_window(
        feat_map: torch.Tensor,
        coords_norm: torch.Tensor,
        window_size: int,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        """Sample a k×k local window around each query point, project to C.

        Offsets are in feature-map pixel space: step = 2/Hf (y) and 2/Wf (x).
        At s8 stride a 1-feature-pixel step = 8 full-res pixels.
        At s16 stride a 1-feature-pixel step = 16 full-res pixels.

        Args:
            feat_map:   [B, C, Hf, Wf]
            coords_norm:[B, N, 2]  normalized (x,y) in [-1,1]
            window_size: k (must be odd)
            out_proj:   nn.Linear(k*k*C → C)
        Returns:
            [B, N, C]
        """
        B, C, Hf, Wf = feat_map.shape
        N = coords_norm.shape[1]
        r = window_size // 2
        device = feat_map.device

        # Per-feature-pixel offsets in normalized coord space
        oy = torch.arange(-r, r + 1, device=device).float() * (2.0 / Hf)
        ox = torch.arange(-r, r + 1, device=device).float() * (2.0 / Wf)
        grid_oy, grid_ox = torch.meshgrid(oy, ox, indexing='ij')
        offsets = torch.stack([grid_ox.flatten(), grid_oy.flatten()], dim=-1)  # [k*k, 2]
        k2 = offsets.shape[0]

        # [B, N, 1, 2] + [1, 1, k*k, 2] → [B, N, k*k, 2]
        win_coords = coords_norm.float().unsqueeze(2) + offsets[None, None]
        win_coords = win_coords.clamp(-1 + 1e-6, 1 - 1e-6)

        # Single grid_sample call over all N*k*k positions
        flat = win_coords.reshape(B, 1, N * k2, 2).to(feat_map.dtype)
        sampled = F.grid_sample(
            feat_map, flat,
            mode='bilinear', padding_mode='border', align_corners=False
        )  # [B, C, 1, N*k*k]

        # [B, C, 1, N*k*k] → [B, N, k*k*C]
        sampled = sampled.squeeze(2).permute(0, 2, 1).reshape(B, N, k2 * C)

        return out_proj(sampled.to(out_proj.weight.dtype))  # [B, N, C]

    def _fuse_features(self, feat_s16, feat_s8, ctx_s8):
        # InfiniDepth Eq. 3 — two gated residual fusion steps
        h2 = self.ffn1(feat_s8 + torch.sigmoid(self.gate1) * self.proj_ctx(ctx_s8))
        h3 = feat_s16 + torch.sigmoid(self.gate2) * self.proj_s8(h2)
        return self.ffn2(h3)

    def forward(self, img, feat_s8, feat1_s8, feat_s16, ctx_s8, coarse_flow,
                query_coords=None, target_h=None, target_w=None,
                zero_residual: bool = False):
        # Infer full-res spatial dims from feat_s8 stride (×8)
        B = feat_s8.shape[0]
        _, _, H8, W8 = feat_s8.shape
        H_full = H8 * 8
        W_full = W8 * 8

        if target_h is None:
            target_h = H_full
        if target_w is None:
            target_w = W_full

        dense_query = query_coords is None
        if dense_query:
            ys = torch.arange(target_h, dtype=torch.float32, device=img.device)
            xs = torch.arange(target_w, dtype=torch.float32, device=img.device)
            if target_h != H_full or target_w != W_full:
                # Center-based rescaling for align_corners=False semantics
                ys = (ys + 0.5) * (H_full / target_h) - 0.5
                xs = (xs + 0.5) * (W_full / target_w) - 0.5
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            query_coords = torch.stack([grid_x, grid_y], dim=-1)
            query_coords = query_coords.reshape(1, -1, 2).expand(B, -1, -1)  # [B, H*W, 2]

        # Normalize to [-1, 1] (align_corners=False convention)
        coords_norm = query_coords.clone().float()
        coords_norm[..., 0] = 2.0 * (coords_norm[..., 0] + 0.5) / W_full - 1.0
        coords_norm[..., 1] = 2.0 * (coords_norm[..., 1] + 0.5) / H_full - 1.0
        coords_norm.clamp_(-1 + 1e-6, 1 - 1e-6)

        # Local-window sample all four feature sources
        ws = self.window_size
        f16      = self._sample_local_window(feat_s16, coords_norm, ws, self.win_proj_s16)
        f8       = self._sample_local_window(feat_s8,  coords_norm, ws, self.win_proj_s8)
        fctx     = self._sample_local_window(ctx_s8,   coords_norm, ws, self.win_proj_ctx)

        fused = self._fuse_features(f16, f8, fctx)

        # Coarse flow at query points — single-point sample is correct here
        coarse_at_q = self._sample_features(coarse_flow, coords_norm)
        coarse_at_q[..., 0] = coarse_at_q[..., 0] * (W_full / W8)
        coarse_at_q[..., 1] = coarse_at_q[..., 1] * (H_full / H8)

        # Warped img1 features: local window at the warped correspondence
        warp_x = coords_norm[..., 0] + 2.0 * coarse_at_q[..., 0] / W_full
        warp_y = coords_norm[..., 1] + 2.0 * coarse_at_q[..., 1] / H_full
        warped_coords = torch.stack([warp_x, warp_y], dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        f1_warped = self._sample_local_window(feat1_s8, warped_coords, ws, self.win_proj_feat1)

        coarse_norm = coarse_at_q.clone().float()
        coarse_norm[..., 0] /= W_full
        coarse_norm[..., 1] /= H_full

        mlp_in = torch.cat([fused, f1_warped, coords_norm, coarse_norm], dim=-1)
        delta_norm = self.flow_head(mlp_in)
        if zero_residual:
            delta_norm = torch.zeros_like(delta_norm)

        delta_flow = delta_norm.clone().float()
        delta_flow[..., 0] = delta_norm[..., 0] * W_full
        delta_flow[..., 1] = delta_norm[..., 1] * H_full

        flow = coarse_at_q + delta_flow

        if dense_query:
            flow = flow.reshape(B, target_h, target_w, 2).permute(0, 3, 1, 2)

        return flow
