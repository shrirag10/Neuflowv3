feature_dim_s16 = 128
context_dim_s16 = 64
iter_context_dim_s16 = 64
feature_dim_s8 = 128
context_dim_s8 = 64
iter_context_dim_s8 = 64

# --- Original convex-upsampler feature dim (kept for backward compat) ---
feature_dim_s1_legacy = 128

# --- Implicit decoder configuration (InfiniDepth-style, 3-scale) ---
# 3 feature scales (context_s8 64d, feat_s8 128d, feat_s16 128d).
# MLP hidden widened proportionally: [256, 128, 64]
implicit_mlp_hidden_list = [256, 128, 64]
