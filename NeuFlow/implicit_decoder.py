"""
ImplicitFlowDecoder — mirrors InfiniDepth's local implicit decoder architecture,
adapted for optical flow (2D vector output instead of scalar depth).

InfiniDepth source: github.com/zju3dv/InfiniDepth
  - Feature sources: ViT layers 4 (256d), 11 (512d), 23 (1024d)
  - Fusion (Eq. 3): h^{k+1} = FFN^k( f^{k+1} + g^k ⊙ Linear(h^k) )
    runs L-1 = 2 times, shallow→deep
  - MLP input: fused feature h^L

NeuFlow mapping (3 scales, all available from the frozen backbone):
  Scale 1 (finest )  context_s8   64d at 1/8  — appearance / refinement context
  Scale 2 (mid    )  feat_s8     128d at 1/8  — cross-frame matching features
  Scale 3 (deepest)  feat_s16    128d at 1/16 — semantic / global features

  Fusion chain (exactly Eq. 3, twice):
    h1 = context_s8
    h2 = FFN1( feat_s8  + g1 ⊙ Linear_{64→128}(h1) )
    h3 = FFN2( feat_s16 + g2 ⊙ Linear_{128→128}(h2) )

  Extra flow-specific signal (not in InfiniDepth — single-image task):
    feat1_s8 sampled at (x + u_coarse, y + v_coarse): img1 correspondence

  MLP input: cat([ h3(128d) | feat1_warped(128d) | coords_norm(2) | coarse_norm(2) ])
             = 260d

Why NOT widen to InfiniDepth's 256/512/1024d:
  NeuFlow's CNN backbone produces 128d features. Projecting 128→256 adds
  parameters but zero new information. The capacity ceiling is the backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_list: list):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_list:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ImplicitFlowDecoder(nn.Module):
    """
    Local implicit decoder for optical flow, mirroring InfiniDepth's ImplicitHead.

    Key ingredient vs. naive bilinear upsampler:
        Continuous coordinate conditioning — the normalized query (x,y) is
        concatenated to the MLP input, letting the network learn sub-pixel
        refinement that is spatially aware and boundary-preserving.

    MLP input: cat([fused(hidden_dim), coords_norm(2), coarse_flow_norm(2)])
    MLP output: delta flow (pixel space), added to bilinearly upsampled coarse flow.
    """

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

        # ------------------------------------------------------------------ #
        # InfiniDepth Eq. 3 — 3-scale hierarchical fusion (shallow→deep):
        #
        #  h1 = context_s8                              (64d, finest scale)
        #  h2 = FFN1( feat_s8  + g1 ⊙ Linear(h1) )    (128d, mid scale)
        #  h3 = FFN2( feat_s16 + g2 ⊙ Linear(h2) )    (128d, deep scale)
        #
        # Two gated residual fusion steps mirror InfiniDepth's L-1=2 iterations.
        # Gates are learnable static channel-wise scalars (not input-dependent).
        # ------------------------------------------------------------------ #

        # Scale 1 → 2: context_s8 (64d) → feat_s8 (128d)
        self.proj_ctx  = nn.Linear(feat_dim_ctx, hidden_dim)      # Linear(h1)
        self.gate1     = nn.Parameter(torch.ones(hidden_dim))     # g1
        self.ffn1      = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Scale 2 → 3: fused_s8 (128d) → feat_s16 (128d)
        self.proj_s8   = nn.Linear(feat_dim_s8, hidden_dim)       # Linear(h2)
        self.gate2     = nn.Parameter(torch.ones(hidden_dim))     # g2
        self.ffn2      = nn.Sequential(
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

    def _fuse_features(
        self,
        feat_s16: torch.Tensor,    # [B, N, 128] — deep, semantic (scale 3)
        feat_s8:  torch.Tensor,    # [B, N, 128] — mid, matching  (scale 2)
        ctx_s8:   torch.Tensor,    # [B, N, 64]  — fine, context  (scale 1)
    ) -> torch.Tensor:
        """InfiniDepth Eq. 3, applied twice (L=3 scales, L-1=2 fusion steps).

        Step 1 (finest → mid):
            h2 = FFN1( feat_s8  + g1 ⊙ Linear_{64→128}(ctx_s8) )
        Step 2 (mid → deepest):
            h3 = FFN2( feat_s16 + g2 ⊙ Linear_{128→128}(h2) )
        """
        # Step 1: context_s8 → feat_s8
        g1 = torch.sigmoid(self.gate1)                   # [128]
        h2 = feat_s8 + g1 * self.proj_ctx(ctx_s8)        # [B, N, 128]
        h2 = self.ffn1(h2)

        # Step 2: fused_s8 → feat_s16
        g2 = torch.sigmoid(self.gate2)                   # [128]
        h3 = feat_s16 + g2 * self.proj_s8(h2)            # [B, N, 128]
        return self.ffn2(h3)                              # [B, N, 128]

    def forward(
        self,
        img: torch.Tensor,                          # [B, 3, H, W]  (used for device/dtype)
        feat_s8: torch.Tensor,                      # [B, 128, H/8,  W/8]  img0 matching
        feat1_s8: torch.Tensor,                     # [B, 128, H/8,  W/8]  img1 matching
        feat_s16: torch.Tensor,                     # [B, 128, H/16, W/16] img0 semantic
        ctx_s8: torch.Tensor,                       # [B,  64, H/8,  W/8]  img0 context
        coarse_flow: torch.Tensor,                  # [B,   2, H/8,  W/8]
        query_coords: torch.Tensor | None = None,   # [B, N, 2] full-res pixel coords (x,y)
        target_h: int | None = None,
        target_w: int | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            [B, 2, H, W] dense flow  or  [B, N, 2] sparse flow.
        """
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

        # 1. Sample all 3 feature scales at query coords
        f16  = self._sample_features(feat_s16, coords_norm)  # [B, N, 128] deep
        f8   = self._sample_features(feat_s8,  coords_norm)  # [B, N, 128] mid
        fctx = self._sample_features(ctx_s8,   coords_norm)  # [B, N,  64] finest

        # 2. 3-scale gated fusion (InfiniDepth Eq. 3, twice)
        fused = self._fuse_features(f16, f8, fctx)           # [B, N, 128]

        # 3. Sample coarse flow at query coords and scale to full-res pixel space
        coarse_at_q = self._sample_features(coarse_flow, coords_norm)  # [B, N, 2]
        coarse_at_q[..., 0] = coarse_at_q[..., 0] * (W_full / W8)
        coarse_at_q[..., 1] = coarse_at_q[..., 1] * (H_full / H8)

        # 4. Sample img1 features at WARPED position: (x,y) + coarse_flow
        #    This is the core correspondence signal — what img1 looks like where
        #    img0's pixel has moved to.  Warp in normalized [-1,1] coords:
        #      warp_norm = coords_norm + 2 * coarse_pixel / image_size
        warp_x = coords_norm[..., 0] + 2.0 * coarse_at_q[..., 0] / W_full
        warp_y = coords_norm[..., 1] + 2.0 * coarse_at_q[..., 1] / H_full
        warped_coords = torch.stack([warp_x, warp_y], dim=-1)
        warped_coords = warped_coords.clamp(-1 + 1e-6, 1 - 1e-6)
        f1_warped = self._sample_features(feat1_s8, warped_coords)      # [B, N, C8]

        # Normalize coarse flow for MLP input
        coarse_norm = coarse_at_q.clone().float()
        coarse_norm[..., 0] = coarse_norm[..., 0] / W_full
        coarse_norm[..., 1] = coarse_norm[..., 1] / H_full

        # 5. MLP: [img0_fused | img1_warped | coord | coarse_flow] → delta
        mlp_in = torch.cat([fused, f1_warped, coords_norm, coarse_norm], dim=-1)
        delta_norm = self.flow_head(mlp_in)                             # [B, N, 2]

        # Delta is predicted in the same normalized space as coarse_norm (÷W, ÷H).
        # Scale back to pixel space — same convention as coarse_norm.
        delta_flow = delta_norm.clone().float()
        delta_flow[..., 0] = delta_norm[..., 0] * W_full
        delta_flow[..., 1] = delta_norm[..., 1] * H_full

        flow = coarse_at_q + delta_flow

        if dense_query:
            flow = flow.reshape(B, target_h, target_w, 2).permute(0, 3, 1, 2)

        return flow
