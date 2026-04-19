feature_dim_s16 = 128
context_dim_s16 = 64
iter_context_dim_s16 = 64
feature_dim_s8 = 128
context_dim_s8 = 64
iter_context_dim_s8 = 64

# --- Original convex-upsampler feature dim (kept for backward compat) ---
feature_dim_s1_legacy = 128

# --- Implicit decoder v2 configuration ---
implicit_patch_size = 5        # Local patch size (k×k) around each query point
implicit_pe_bands = 6          # Fourier positional-encoding frequency bands
implicit_ffn_expansion = 2     # FFN hidden-dim expansion factor
implicit_mlp_hidden = 128      # MLP head hidden dimension
