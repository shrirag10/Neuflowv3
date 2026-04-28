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
# MLP input: [h3 | feat1_warped | (x,y)_norm | u_coarse_norm] = 260d
# Output: delta_flow, added to bilinear-upsampled coarse flow.
# Output layer is zero-init so the model starts identical to v2.

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
        feat_dim_ctx: int = 64,   # context_s8 channel dim (3rd / finest scale)
        hidden_dim: int = 128,    # hidden dim throughout fusion
        hidden_list: list = None,
    ):
        super().__init__()

        if hidden_list is None:
            hidden_list = [256, 128, 64]

        self.feat_dim_s8  = feat_dim_s8
        self.feat_dim_ctx = feat_dim_ctx

        # fusion: ctx_s8 -> feat_s8 -> feat_s16 (shallow to deep)
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

    @staticmethod
    def _sample_features(feat_map: torch.Tensor, coords_norm: torch.Tensor) -> torch.Tensor:
        """
        Bilinear sample feat_map at normalized coords (x,y) in [-1,1].
        grid_sample expects last-dim as (x,y) — no flip needed.
        """
        grid = coords_norm.to(feat_map.dtype).unsqueeze(1)  # [B, 1, N, 2]  (x,y) order
        sampled = F.grid_sample(
            feat_map, grid,
            mode='bilinear', padding_mode='border', align_corners=False
        )  # [B, C, 1, N]
        return sampled.squeeze(2).permute(0, 2, 1)  # [B, N, C]

    def _fuse_features(self, feat_s16, feat_s8, ctx_s8):
        # InfiniDepth Eq. 3 — two gated residual fusion steps
        h2 = self.ffn1(feat_s8 + torch.sigmoid(self.gate1) * self.proj_ctx(ctx_s8))
        h3 = feat_s16 + torch.sigmoid(self.gate2) * self.proj_s8(h2)
        return self.ffn2(h3)

    def forward(self, img, feat_s8, feat1_s8, feat_s16, ctx_s8, coarse_flow,
                query_coords=None, target_h=None, target_w=None):
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
        # align_corners=False pixel-center normalization
        coords_norm[..., 0] = 2.0 * (coords_norm[..., 0] + 0.5) / W_full - 1.0
        coords_norm[..., 1] = 2.0 * (coords_norm[..., 1] + 0.5) / H_full - 1.0
        coords_norm.clamp_(-1 + 1e-6, 1 - 1e-6)

        # sample all three scales
        f16  = self._sample_features(feat_s16, coords_norm)
        f8   = self._sample_features(feat_s8,  coords_norm)
        fctx = self._sample_features(ctx_s8,   coords_norm)

        fused = self._fuse_features(f16, f8, fctx)

        # coarse flow at query points, scaled to pixel space
        coarse_at_q = self._sample_features(coarse_flow, coords_norm)
        coarse_at_q[..., 0] = coarse_at_q[..., 0] * (W_full / W8)
        coarse_at_q[..., 1] = coarse_at_q[..., 1] * (H_full / H8)

        # sample img1 features at the warped location
        warp_x = coords_norm[..., 0] + 2.0 * coarse_at_q[..., 0] / W_full
        warp_y = coords_norm[..., 1] + 2.0 * coarse_at_q[..., 1] / H_full
        warped_coords = torch.stack([warp_x, warp_y], dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        f1_warped = self._sample_features(feat1_s8, warped_coords)

        coarse_norm = coarse_at_q.clone().float()
        coarse_norm[..., 0] /= W_full
        coarse_norm[..., 1] /= H_full

        mlp_in = torch.cat([fused, f1_warped, coords_norm, coarse_norm], dim=-1)
        delta_norm = self.flow_head(mlp_in)

        delta_flow = delta_norm.clone().float()
        delta_flow[..., 0] = delta_norm[..., 0] * W_full
        delta_flow[..., 1] = delta_norm[..., 1] * H_full

        flow = coarse_at_q + delta_flow

        if dense_query:
            flow = flow.reshape(B, target_h, target_w, 2).permute(0, 3, 1, 2)

        return flow
