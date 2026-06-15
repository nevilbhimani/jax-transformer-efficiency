"""
model.py — Transformer in Flax for H100 GPU.

KEY DIFFERENCES FROM CPU VERSION:
  1. bf16 dtype throughout
     H100 peak: 989 TFLOP/s bf16 (sparse), 312 TFLOP/s dense bf16
     float32 peak: 78 TFLOP/s — 4x slower
     Using bf16 gives honest MFU against real H100 peak.

  2. Larger default config
     d_model=2048, num_heads=16 → head_dim=128
     128 is the H100 tensor core tile size
     head_dim=128 means full tensor core utilization
     head_dim=32 (CPU version) = 25% tensor core utilization

  3. Everything else identical
     Same sharding constraints, same architecture, same patterns
     The hardware changes, the logic doesn't
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class TransformerConfig:
    # H100-optimized defaults
    # head_dim = d_model / num_heads = 2048 / 16 = 128
    # 128 = H100 tensor core tile size → maximum MXU utilization
    vocab_size:   int   = 32_000
    seq_len:      int   = 2048
    d_model:      int   = 2048
    num_heads:    int   = 16
    num_layers:   int   = 8
    d_ff:         int   = 8192    # 4 × d_model
    dropout_rate: float = 0.0     # disabled for benchmarking
    
    # bf16 for H100 peak throughput
    # H100 dense bf16: 312 TFLOP/s
    # H100 dense fp32:  78 TFLOP/s
    # Always use bf16 for training benchmarks on H100
    dtype: any = jnp.bfloat16

    @property
    def head_dim(self):
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        return self.d_model // self.num_heads


# Smaller config for quick smoke tests
SMALL_CONFIG = TransformerConfig(
    vocab_size=4096,
    seq_len=512,
    d_model=512,
    num_heads=8,     # head_dim = 64
    num_layers=4,
    d_ff=2048,
)

# Full config for real benchmarks
FULL_CONFIG = TransformerConfig(
    vocab_size=32_000,
    seq_len=2048,
    d_model=2048,
    num_heads=16,    # head_dim = 128 → perfect H100 tile alignment
    num_layers=8,
    d_ff=8192,
)


# ─────────────────────────────────────────────────────────────
# Multi-Head Attention
# ─────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    config: TransformerConfig
    mesh:   any

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        B, T, D = x.shape
        H  = self.config.num_heads
        Dh = self.config.head_dim
        dtype = self.config.dtype

        # Cast input to bf16
        # All computation runs in bf16 for H100 tensor core throughput
        x = x.astype(dtype)

        # QKV projection
        # Weight dtype matches computation dtype (bf16)
        qkv = nn.DenseGeneral(
            features=(3, H, Dh),
            axis=-1,
            dtype=dtype,
            kernel_init=nn.initializers.normal(0.02),
            name="qkv_proj"
        )(x)
        # shape: (B, T, 3, H, Dh)

        # Enforce head sharding — prevents all-gather before attention
        qkv = jax.lax.with_sharding_constraint(
            qkv,
            NamedSharding(self.mesh, P("data", None, None, "model", None))
        )

        q, k, v = [qkv[:, :, i, :, :].transpose(0, 2, 1, 3) for i in range(3)]
        # shape: (B, H, T, Dh)

        # Scaled dot-product attention
        # Running in bf16 — fast on H100 tensor cores
        scale       = jnp.sqrt(Dh).astype(dtype)
        attn_weights = jnp.einsum("bhtd,bhsd->bhts", q, k) / scale

        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, jnp.finfo(dtype).min)

        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(dtype)

        # Enforce sharding after attention computation
        attn_weights = jax.lax.with_sharding_constraint(
            attn_weights,
            NamedSharding(self.mesh, P("data", "model", None, None))
        )

        attn_output = jnp.einsum("bhts,bhsd->bhtd", attn_weights, v)
        # shape: (B, H, T, Dh)

        attn_output = jax.lax.with_sharding_constraint(
            attn_output,
            NamedSharding(self.mesh, P("data", "model", None, None))
        )

        # Reshape: (B, H, T, Dh) → (B, T, D)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)

        # Output projection (row parallel)
        output = nn.DenseGeneral(
            features=D,
            axis=-1,
            dtype=dtype,
            kernel_init=nn.initializers.normal(0.02),
            name="out_proj"
        )(attn_output)

        output = jax.lax.with_sharding_constraint(
            output,
            NamedSharding(self.mesh, P("data", None, None))
        )

        return output


# ─────────────────────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    config: TransformerConfig
    mesh:   any

    @nn.compact
    def __call__(self, x, deterministic=True):
        D    = self.config.d_model
        D_ff = self.config.d_ff
        dtype = self.config.dtype

        x = x.astype(dtype)

        # Column parallel up-projection
        hidden = nn.Dense(
            features=D_ff,
            dtype=dtype,
            kernel_init=nn.initializers.normal(0.02),
            name="up_proj"
        )(x)

        # Enforce feature sharding before GELU
        # GELU is elementwise — runs on local shard, no gather needed
        hidden = jax.lax.with_sharding_constraint(
            hidden,
            NamedSharding(self.mesh, P("data", None, "model"))
        )

        hidden = jax.nn.gelu(hidden)

        # Row parallel down-projection
        output = nn.Dense(
            features=D,
            dtype=dtype,
            kernel_init=nn.initializers.normal(0.02),
            name="down_proj"
        )(hidden)

        output = jax.lax.with_sharding_constraint(
            output,
            NamedSharding(self.mesh, P("data", None, None))
        )

        return output


# ─────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    config: TransformerConfig
    mesh:   any

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        dtype = self.config.dtype

        # Pre-norm attention
        residual = x
        x = nn.LayerNorm(dtype=dtype, name="attn_norm")(x)
        x = MultiHeadAttention(
            config=self.config, mesh=self.mesh, name="attention"
        )(x, mask=mask, deterministic=deterministic)
        x = x + residual

        # Pre-norm FFN
        residual = x
        x = nn.LayerNorm(dtype=dtype, name="ffn_norm")(x)
        x = FeedForward(
            config=self.config, mesh=self.mesh, name="ffn"
        )(x, deterministic=deterministic)
        x = x + residual

        return x


# ─────────────────────────────────────────────────────────────
# Full Transformer
# ─────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    config: TransformerConfig
    mesh:   any

    @nn.compact
    def __call__(self, token_ids, deterministic=True):
        B, T  = token_ids.shape
        D     = self.config.d_model
        dtype = self.config.dtype

        # Embeddings (float32 → cast to bf16)
        token_embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=D,
            dtype=dtype,
            embedding_init=nn.initializers.normal(0.02),
            name="token_embedding"
        )(token_ids)

        pos_ids   = jnp.arange(T)[None, :]
        pos_embed = nn.Embed(
            num_embeddings=self.config.seq_len,
            features=D,
            dtype=dtype,
            embedding_init=nn.initializers.normal(0.02),
            name="pos_embedding"
        )(pos_ids)

        x = token_embed + pos_embed

        x = jax.lax.with_sharding_constraint(
            x,
            NamedSharding(self.mesh, P("data", None, None))
        )

        # Causal mask
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]

        # Transformer blocks
        for i in range(self.config.num_layers):
            x = TransformerBlock(
                config=self.config,
                mesh=self.mesh,
                name=f"layer_{i}"
            )(x, mask=mask, deterministic=deterministic)

        x = nn.LayerNorm(dtype=dtype, name="final_norm")(x)

        # Output projection
        # Cast back to float32 for stable softmax/loss computation
        logits = nn.Dense(
            features=self.config.vocab_size,
            dtype=jnp.float32,
            kernel_init=nn.initializers.normal(0.02),
            name="lm_head"
        )(x.astype(jnp.float32))

        logits = jax.lax.with_sharding_constraint(
            logits,
            NamedSharding(self.mesh, P("data", None, None))
        )

        return logits


# ─────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────

def init_model(config, mesh, rng_key):
    """Initialize model and return (model, params)."""
    model = Transformer(config=config, mesh=mesh)

    # Small dummy input for tracing
    dummy_input = jnp.zeros((2, min(64, config.seq_len)), dtype=jnp.int32)

    variables = model.init(rng_key, dummy_input)
    params    = variables['params']

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model initialized: {n_params:,} parameters ({n_params/1e6:.1f}M)")
    print(f"  d_model={config.d_model}, num_heads={config.num_heads}, "
          f"head_dim={config.head_dim}, num_layers={config.num_layers}")
    print(f"  dtype={config.dtype}")
    print(f"  head_dim alignment: {config.head_dim} "
          f"({'128-aligned ✓' if config.head_dim % 128 == 0 else str(config.head_dim) + ' (not 128-aligned)'})")

    return model, params


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sharding import setup_devices, create_mesh

    setup_devices()
    mesh, _, _ = create_mesh()

    config = SMALL_CONFIG  # use small config for quick test

    with jax.set_mesh(mesh):
        rng    = jax.random.PRNGKey(0)
        model, params = init_model(config, mesh, rng)

        dummy  = jnp.zeros((2, 64), dtype=jnp.int32)
        logits = model.apply({'params': params}, dummy)
        print(f"\nOutput shape: {logits.shape}")
        print(f"Output dtype: {logits.dtype}")

    print("\n✓ model.py OK")
