import torch
import torch.nn.functional as F

from NeuFlow import backbone_v7
from NeuFlow import transformer
from NeuFlow import matching
from NeuFlow import corr
from NeuFlow import refine
from NeuFlow import upsample
from NeuFlow import implicit_decoder
from NeuFlow import config
from NeuFlow.adaptive_query import coarse_flow_query

from huggingface_hub import PyTorchModelHubMixin


class NeuFlow(torch.nn.Module,
              PyTorchModelHubMixin,
              repo_url="https://github.com/neufieldrobotics/NeuFlow_v2", license="apache-2.0", pipeline_tag="image-to-image"):
    def __init__(self, use_implicit: bool = True):
        super(NeuFlow, self).__init__()

        self.use_implicit = use_implicit

        self.backbone = backbone_v7.CNNEncoder(config.feature_dim_s16, config.context_dim_s16, config.feature_dim_s8, config.context_dim_s8)
        
        self.cross_attn_s16 = transformer.FeatureAttention(config.feature_dim_s16+config.context_dim_s16, num_layers=2, ffn=True, ffn_dim_expansion=1, post_norm=True)
        
        self.matching_s16 = matching.Matching()

        # self.flow_attn_s16 = transformer.FlowAttention(config.feature_dim_s16)

        self.corr_block_s16 = corr.CorrBlock(radius=4, levels=1)
        self.corr_block_s8 = corr.CorrBlock(radius=4, levels=1)
        
        self.merge_s8 = torch.nn.Sequential(torch.nn.Conv2d(config.feature_dim_s16 + config.feature_dim_s8, config.feature_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.GELU(),
                                              torch.nn.Conv2d(config.feature_dim_s8, config.feature_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.BatchNorm2d(config.feature_dim_s8))

        self.context_merge_s8 = torch.nn.Sequential(torch.nn.Conv2d(config.context_dim_s16 + config.context_dim_s8, config.context_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.GELU(),
                                           torch.nn.Conv2d(config.context_dim_s8, config.context_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.BatchNorm2d(config.context_dim_s8))

        self.refine_s16 = refine.Refine(config.context_dim_s16, config.iter_context_dim_s16, num_layers=5, levels=1, radius=4, inter_dim=128)
        self.refine_s8 = refine.Refine(config.context_dim_s8, config.iter_context_dim_s8, num_layers=5, levels=1, radius=4, inter_dim=96)

        if self.use_implicit:
            # ---- Implicit decoder v2 path (edge-optimized) ----
            # No dense full-res conv — the decoder extracts local patches
            # at query points internally, making compute O(N) not O(H×W).
            self.implicit_decoder_module = implicit_decoder.ImplicitFlowDecoder(
                feat_dim_s8=config.feature_dim_s8,
                feat_dim_ctx=config.context_dim_s8,
                hidden_dim=config.feature_dim_s16,
                hidden_list=config.implicit_mlp_hidden_list,
            )
        else:
            # ---- Legacy convex-upsampler path ----
            self.conv_s8 = backbone_v7.ConvBlock(3, config.feature_dim_s1_legacy, kernel_size=8, stride=8, padding=0)
            self.upsample_s8 = upsample.UpSample(config.feature_dim_s1_legacy, upsample_factor=8)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def init_bhwd(self, batch_size, height, width, device, amp=True):

        self.backbone.init_bhwd(batch_size*2, height//16, width//16, device, amp)

        self.matching_s16.init_bhwd(batch_size, height//16, width//16, device, amp)

        self.corr_block_s16.init_bhwd(batch_size, height//16, width//16, device, amp)
        self.corr_block_s8.init_bhwd(batch_size, height//8, width//8, device, amp)

        self.refine_s16.init_bhwd(batch_size, height//16, width//16, device, amp)
        self.refine_s8.init_bhwd(batch_size, height//8, width//8, device, amp)

        self.init_iter_context_s16 = torch.zeros(batch_size, config.iter_context_dim_s16, height//16, width//16, device=device, dtype=torch.half if amp else torch.float)
        self.init_iter_context_s8 = torch.zeros(batch_size, config.iter_context_dim_s8, height//8, width//8, device=device, dtype=torch.half if amp else torch.float)

    def split_features(self, features, context_dim, feature_dim):

        context, features = torch.split(features, [context_dim, feature_dim], dim=1)

        context, _ = context.chunk(chunks=2, dim=0)
        feature0, feature1 = features.chunk(chunks=2, dim=0)

        return features, torch.relu(context)

    def infer_coarse_state(self, img0, img1, iters_s16=1, iters_s8=8):
        """Compute and cache implicit-decoder inputs once for decode-only query passes.

        Returns a dict containing normalized image, coarse flow (1/8 scale), and
        sampled feature maps needed by the implicit decoder.
        """
        if not self.use_implicit:
            raise RuntimeError('infer_coarse_state is only available in implicit mode.')

        img0 = img0 / 255.
        img1 = img1 / 255.

        features_s16, features_s8 = self.backbone(torch.cat([img0, img1], dim=0))

        features_s16 = self.cross_attn_s16(features_s16)

        features_s16, context_s16 = self.split_features(features_s16, config.context_dim_s16, config.feature_dim_s16)
        features_s8, context_s8 = self.split_features(features_s8, config.context_dim_s8, config.feature_dim_s8)

        feature0_s16, feature1_s16 = features_s16.chunk(chunks=2, dim=0)

        flow0 = self.matching_s16.global_correlation_softmax(feature0_s16, feature1_s16)

        corr_pyr_s16 = self.corr_block_s16.init_corr_pyr(feature0_s16, feature1_s16)

        iter_context_s16 = self.init_iter_context_s16

        for i in range(iters_s16):
            corrs = self.corr_block_s16(corr_pyr_s16, flow0)
            iter_context_s16, delta_flow = self.refine_s16(corrs, context_s16, iter_context_s16, flow0)
            flow0 = flow0 + delta_flow

        flow0 = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2

        features_s16 = F.interpolate(features_s16, scale_factor=2, mode='nearest')
        features_s8 = self.merge_s8(torch.cat([features_s8, features_s16], dim=1))

        feature0_s8, feature1_s8 = features_s8.chunk(chunks=2, dim=0)

        corr_pyr_s8 = self.corr_block_s8.init_corr_pyr(feature0_s8, feature1_s8)

        context_s16 = F.interpolate(context_s16, scale_factor=2, mode='nearest')
        context_s8 = self.context_merge_s8(torch.cat([context_s8, context_s16], dim=1))

        iter_context_s8 = self.init_iter_context_s8

        for i in range(iters_s8):
            corrs = self.corr_block_s8(corr_pyr_s8, flow0)
            iter_context_s8, delta_flow = self.refine_s8(corrs, context_s8, iter_context_s8, flow0)
            flow0 = flow0 + delta_flow

        return {
            'img0': img0,
            'feature0_s8': feature0_s8,
            'feature1_s8': feature1_s8,
            'feature0_s16': feature0_s16,
            'context0_s8': context_s8,
            'coarse_flow_s8': flow0,
        }

    def decode_queries(self, state, query_coords=None, target_h=None, target_w=None,
                       adaptive_n=None, adaptive_ratio=0.7):
        """Decode flow from cached coarse state at arbitrary query coords/resolution.

        Args:
            state:          Dict from infer_coarse_state().
            query_coords:   [B, N, 2] explicit pixel coords, or None.
            target_h/w:     Dense output resolution (used when query_coords=None
                            and adaptive_n=None).
            adaptive_n:     If set and query_coords=None, use coarse-flow-gradient-
                            weighted query allocation (edge-device mode).
                            Returns [B, adaptive_n, 2] sparse flow.
            adaptive_ratio: Fraction of adaptive_n at high-gradient regions.
        """
        if not self.use_implicit:
            raise RuntimeError('decode_queries is only available in implicit mode.')

        # Inference-time adaptive query: allocate budget at motion boundaries
        # using the coarse flow’s own gradient — mirrors InfiniDepth §3.3.
        if query_coords is None and adaptive_n is not None:
            query_coords = coarse_flow_query(
                state['coarse_flow_s8'],
                num_points=adaptive_n,
                adaptive_ratio=adaptive_ratio,
            )  # [B, adaptive_n, 2]

        return self.implicit_decoder_module(
            img=state['img0'],
            feat_s8=state['feature0_s8'],
            feat1_s8=state['feature1_s8'],
            feat_s16=state['feature0_s16'],
            ctx_s8=state['context0_s8'],
            coarse_flow=state['coarse_flow_s8'],
            query_coords=query_coords,
            target_h=target_h,
            target_w=target_w,
        )

    def forward(self, img0, img1, iters_s16=1, iters_s8=8,
                query_coords=None, target_h=None, target_w=None,
                adaptive_n=None, adaptive_ratio=0.7):
        """
        Args:
            img0, img1: [B, 3, H, W] input images (uint8-range, i.e. 0-255).
            iters_s16:  Number of refinement iterations at 1/16 scale.
            iters_s8:   Number of refinement iterations at 1/8 scale.
            query_coords: (implicit mode only) Optional [B, N, 2] pixel coords
                          for arbitrary-point flow querying.
            target_h, target_w: (implicit mode only) Target output resolution.
                                Defaults to input resolution.
            adaptive_n: If set and query_coords=None, use coarse-flow-gradient-
                        weighted query allocation at inference (edge-device mode).
            adaptive_ratio: Fraction of adaptive_n at high-gradient regions.
        Returns:
            flow_list: list of flow predictions (coarse → fine).  The last
                       entry is the final full-resolution flow.
        """
        flow_list = []

        img0 = img0 / 255.
        img1 = img1 / 255.

        features_s16, features_s8 = self.backbone(torch.cat([img0, img1], dim=0))

        features_s16 = self.cross_attn_s16(features_s16)

        features_s16, context_s16 = self.split_features(features_s16, config.context_dim_s16, config.feature_dim_s16)
        features_s8, context_s8 = self.split_features(features_s8, config.context_dim_s8, config.feature_dim_s8)

        feature0_s16, feature1_s16 = features_s16.chunk(chunks=2, dim=0)

        flow0 = self.matching_s16.global_correlation_softmax(feature0_s16, feature1_s16)

        # flow0 = self.flow_attn_s16(feature0_s16, flow0)

        corr_pyr_s16 = self.corr_block_s16.init_corr_pyr(feature0_s16, feature1_s16)

        iter_context_s16 = self.init_iter_context_s16

        for i in range(iters_s16):

            if self.training and i > 0:
                flow0 = flow0.detach()
                # iter_context_s16 = iter_context_s16.detach()

            corrs = self.corr_block_s16(corr_pyr_s16, flow0)

            iter_context_s16, delta_flow = self.refine_s16(corrs, context_s16, iter_context_s16, flow0)

            flow0 = flow0 + delta_flow

            if self.training:
                up_flow0 = F.interpolate(flow0, scale_factor=16, mode='bilinear') * 16
                flow_list.append(up_flow0)

        flow0 = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2

        features_s16 = F.interpolate(features_s16, scale_factor=2, mode='nearest')

        features_s8 = self.merge_s8(torch.cat([features_s8, features_s16], dim=1))

        feature0_s8, feature1_s8 = features_s8.chunk(chunks=2, dim=0)

        corr_pyr_s8 = self.corr_block_s8.init_corr_pyr(feature0_s8, feature1_s8)

        context_s16 = F.interpolate(context_s16, scale_factor=2, mode='nearest')

        context_s8 = self.context_merge_s8(torch.cat([context_s8, context_s16], dim=1))
        context0_s8 = context_s8  # already img0-only (split_features discards img1 context)

        iter_context_s8 = self.init_iter_context_s8

        for i in range(iters_s8):

            if self.training and i > 0:
                flow0 = flow0.detach()
                # iter_context_s8 = iter_context_s8.detach()

            corrs = self.corr_block_s8(corr_pyr_s8, flow0)

            iter_context_s8, delta_flow = self.refine_s8(corrs, context_s8, iter_context_s8, flow0)

            flow0 = flow0 + delta_flow

            if self.training or i == iters_s8 - 1:

                if self.use_implicit:
                    # At inference, resolve adaptive_n → coords from coarse flow gradient
                    _qc = query_coords
                    if _qc is None and adaptive_n is not None and not self.training:
                        _qc = coarse_flow_query(
                            flow0, num_points=adaptive_n, adaptive_ratio=adaptive_ratio,
                        )
                    up_flow0 = self.implicit_decoder_module(
                        img=img0,
                        feat_s8=feature0_s8,
                        feat1_s8=feature1_s8,
                        feat_s16=feature0_s16,
                        ctx_s8=context0_s8,
                        coarse_flow=flow0,
                        query_coords=_qc,
                        target_h=target_h,
                        target_w=target_w,
                    )
                    flow_list.append(up_flow0)
                else:
                    # ---- Legacy convex-upsampler path ----
                    feature0_s1 = self.conv_s8(img0)
                    up_flow0 = self.upsample_s8(feature0_s1, flow0) * 8
                    flow_list.append(up_flow0)

        return flow_list
