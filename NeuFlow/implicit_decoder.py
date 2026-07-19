# Queryable flow decoder — v3 rebuild (2026-07-19).
#
# Design contract: docs/v3_rebuild_audit.md Part 2. Every choice below is backed
# by a measured result; rejected ideas (Fourier PE, unbounded regression head)
# are documented there with their evidence.
#
# One compute path for everything:
#   precompute(): window projections applied as 3x3 convs over the feature maps
#     (mathematically identical to window-sample-then-Linear — both are linear),
#     then gated fusion evaluated ONCE on the 1/8 grid.
#     Known approximation: sampling fused-features instead of fusing sampled
#     features costs +0.02 px EPE (measured, VKITTI2).
#   decode(): per query — one point sample of the fused map, one of the warped
#     frame-1 map, 3x3 coarse-flow candidates, softmax convex combination.
#
# Head: convex weights over the 3x3 coarse-flow neighborhood + a bilinear
# candidate, biased so a zero-initialized head reproduces bilinear upsampling
# exactly (the 2.476 px operating point; full-set verified).
#
# Parameter names match the pre-rebuild decoder, so existing convex-head
# checkpoints (e.g. neuflowv3_mix) load without key surgery. Their weights were
# trained on the exact per-query fusion path; retraining on this unified path
# is PENDING — until then, loaded checkpoints run with the +0.02 px approximation.

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
        feat_dim_s8: int = 128,
        feat_dim_ctx: int = 64,
        hidden_dim: int = 128,
        hidden_list: list = None,
        window_size: int = 3,
    ):
        super().__init__()

        if window_size % 2 != 1:
            raise ValueError(f'window_size must be odd, got {window_size}')
        if hidden_list is None:
            hidden_list = [256, 128, 64]

        self.feat_dim_s8 = feat_dim_s8
        self.feat_dim_ctx = feat_dim_ctx
        self.window_size = window_size

        k2 = window_size ** 2
        # window projections (stored as Linear for checkpoint compatibility;
        # applied as convs via _win_conv)
        self.win_proj_ctx = nn.Linear(k2 * feat_dim_ctx, feat_dim_ctx)
        self.win_proj_s8 = nn.Linear(k2 * feat_dim_s8, feat_dim_s8)
        self.win_proj_s16 = nn.Linear(k2 * feat_dim_s8, feat_dim_s8)
        self.win_proj_feat1 = nn.Linear(k2 * feat_dim_s8, feat_dim_s8)

        # gated hierarchical fusion (InfiniDepth Eq. 3 applied twice)
        self.proj_ctx = nn.Linear(feat_dim_ctx, hidden_dim)
        self.gate1 = nn.Parameter(torch.ones(hidden_dim))
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.proj_s8 = nn.Linear(feat_dim_s8, hidden_dim)
        self.gate2 = nn.Parameter(torch.ones(hidden_dim))
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # convex head: k2 window candidates + 1 bilinear candidate
        self.convex_head = MLP(
            in_dim=hidden_dim + feat_dim_s8 + 2 + 2,   # fused | f1_warped | xy | coarse
            out_dim=k2 + 1,
            hidden_list=hidden_list,
        )
        prior = torch.zeros(k2 + 1)
        prior[k2] = 6.0   # zero-init head => ~0.9975 weight on the bilinear candidate
        self.register_buffer('convex_prior', prior)

    def reset_window_projections_to_center(self):
        """Identity on the center cell, zero elsewhere: window projection output
        equals a single point sample at initialization. Call AFTER any global init."""
        k2 = self.window_size ** 2
        center = k2 // 2
        for proj, C in [
            (self.win_proj_ctx, self.feat_dim_ctx),
            (self.win_proj_s8, self.feat_dim_s8),
            (self.win_proj_s16, self.feat_dim_s8),
            (self.win_proj_feat1, self.feat_dim_s8),
        ]:
            with torch.no_grad():
                proj.weight.zero_()
                proj.weight[:, center * C:(center + 1) * C] = torch.eye(C)
                if proj.bias is not None:
                    proj.bias.zero_()

    # ---- primitives -------------------------------------------------------

    def _win_conv(self, feat_map, proj):
        """Window projection as a conv (exact: both forms are linear in the map)."""
        B, C, Hf, Wf = feat_map.shape
        k = self.window_size
        w = proj.weight.view(-1, k, k, C).permute(0, 3, 1, 2).to(feat_map.dtype)
        x = F.pad(feat_map, (k // 2,) * 4, mode='replicate')
        return F.conv2d(x, w, bias=proj.bias.to(feat_map.dtype))

    @staticmethod
    def _sample(feat_map, coords_norm):
        """Bilinear point sample at normalized (x, y) in [-1, 1] -> [B, N, C]."""
        grid = coords_norm.to(feat_map.dtype).unsqueeze(1)
        out = F.grid_sample(feat_map, grid, mode='bilinear',
                            padding_mode='border', align_corners=False)
        return out.squeeze(2).permute(0, 2, 1)

    def _sample_window_raw(self, feat_map, coords_norm):
        """Raw k x k neighborhood values around each query -> [B, N, k*k, C]."""
        B, C, Hf, Wf = feat_map.shape
        N = coords_norm.shape[1]
        r = self.window_size // 2
        device = feat_map.device
        oy = torch.arange(-r, r + 1, device=device).float() * (2.0 / Hf)
        ox = torch.arange(-r, r + 1, device=device).float() * (2.0 / Wf)
        gy, gx = torch.meshgrid(oy, ox, indexing='ij')
        offsets = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
        k2 = offsets.shape[0]
        win = (coords_norm.float().unsqueeze(2) + offsets[None, None]).clamp(-1 + 1e-6, 1 - 1e-6)
        flat = win.reshape(B, 1, N * k2, 2).to(feat_map.dtype)
        out = F.grid_sample(feat_map, flat, mode='bilinear',
                            padding_mode='border', align_corners=False)
        return out.squeeze(2).permute(0, 2, 1).reshape(B, N, k2, C)

    def _fuse(self, f16, f8, fctx):
        h2 = self.ffn1(f8 + torch.sigmoid(self.gate1) * self.proj_ctx(fctx))
        h3 = f16 + torch.sigmoid(self.gate2) * self.proj_s8(h2)
        return self.ffn2(h3)

    # ---- two-phase API -----------------------------------------------------

    def precompute(self, feat_s8, feat1_s8, feat_s16, ctx_s8):
        """Once per image pair: conv-form projections + fusion on the 1/8 grid."""
        m8 = self._win_conv(feat_s8, self.win_proj_s8)
        mctx = self._win_conv(ctx_s8, self.win_proj_ctx)
        m16 = self._win_conv(feat_s16, self.win_proj_s16)
        m16 = F.interpolate(m16, size=m8.shape[-2:], mode='bilinear', align_corners=False)
        fused = self._fuse(m16.permute(0, 2, 3, 1), m8.permute(0, 2, 3, 1),
                           mctx.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        mf1 = self._win_conv(feat1_s8, self.win_proj_feat1)
        return {'fused': fused, 'mf1': mf1}

    def decode(self, maps, coarse_flow, query_coords):
        """Flow at continuous pixel coords [B, N, 2] -> [B, N, 2]."""
        B, _, H8, W8 = coarse_flow.shape
        H_full, W_full = H8 * 8, W8 * 8

        cn = query_coords.clone().float()
        cn[..., 0] = 2.0 * (cn[..., 0] + 0.5) / W_full - 1.0
        cn[..., 1] = 2.0 * (cn[..., 1] + 0.5) / H_full - 1.0
        cn.clamp_(-1 + 1e-6, 1 - 1e-6)

        coarse_at_q = self._sample(coarse_flow, cn)
        coarse_at_q[..., 0] = coarse_at_q[..., 0] * (W_full / W8)
        coarse_at_q[..., 1] = coarse_at_q[..., 1] * (H_full / H8)

        warp = torch.stack([
            cn[..., 0] + 2.0 * coarse_at_q[..., 0] / W_full,
            cn[..., 1] + 2.0 * coarse_at_q[..., 1] / H_full,
        ], dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)

        fused = self._sample(maps['fused'], cn)
        f1w = self._sample(maps['mf1'], warp)

        coarse_norm = coarse_at_q.clone().float()
        coarse_norm[..., 0] /= W_full
        coarse_norm[..., 1] /= H_full

        mlp_in = torch.cat([fused, f1w, cn, coarse_norm], dim=-1)

        win_flow = self._sample_window_raw(coarse_flow, cn).float()
        win_flow[..., 0] = win_flow[..., 0] * (W_full / W8)
        win_flow[..., 1] = win_flow[..., 1] * (H_full / H8)
        candidates = torch.cat([win_flow, coarse_at_q.float().unsqueeze(2)], dim=2)

        logits = self.convex_head(mlp_in).float() + self.convex_prior
        weights = torch.softmax(logits, dim=-1)
        return (weights.unsqueeze(-1) * candidates).sum(dim=2)

    def decode_dense(self, maps, coarse_flow, target_h=None, target_w=None, stride=2):
        """Dense grid decode. stride=2 + bilinear upsample measured at no EPE cost."""
        B, _, H8, W8 = coarse_flow.shape
        H_full, W_full = H8 * 8, W8 * 8
        th, tw = target_h or H_full, target_w or W_full
        dh, dw = th // stride, tw // stride
        dev = coarse_flow.device

        ys = torch.arange(dh, dtype=torch.float32, device=dev)
        xs = torch.arange(dw, dtype=torch.float32, device=dev)
        if dh != H_full or dw != W_full:
            ys = (ys + 0.5) * (H_full / dh) - 0.5
            xs = (xs + 0.5) * (W_full / dw) - 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([gx, gy], -1).reshape(1, -1, 2).expand(B, -1, -1)

        flow = self.decode(maps, coarse_flow, coords)
        flow = flow.reshape(B, dh, dw, 2).permute(0, 3, 1, 2)
        if (dh, dw) != (th, tw):
            flow = F.interpolate(flow, size=(th, tw), mode='bilinear', align_corners=False)
        return flow

    # ---- training/compat entry point --------------------------------------

    def forward(self, img, feat_s8, feat1_s8, feat_s16, ctx_s8, coarse_flow,
                query_coords=None, target_h=None, target_w=None):
        maps = self.precompute(feat_s8, feat1_s8, feat_s16, ctx_s8)
        if query_coords is not None:
            return self.decode(maps, coarse_flow, query_coords)
        return self.decode_dense(maps, coarse_flow, target_h, target_w, stride=1)
