"""
train.py — Training loop for H100 GPU.

KEY DIFFERENCES FROM CPU VERSION:
  1. No XLA_FLAGS — real GPUs
  2. bf16 params and optimizer state
  3. Larger model config by default
  4. donate_argnums re-enabled — safe now because we thread
     params correctly through the loop (not using lambda)
"""

import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import time

from model import Transformer, TransformerConfig, init_model, SMALL_CONFIG, FULL_CONFIG
from sharding import setup_devices, create_mesh


# ─────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────

def cross_entropy_loss(logits, targets):
    """Cross-entropy loss for language modeling."""
    B, T, V    = logits.shape
    logits_2d  = logits.reshape(B * T, V)
    targets_1d = targets.reshape(B * T)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits_2d, targets_1d
    ).mean()


# ─────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────

def make_optimizer(learning_rate=3e-4, warmup_steps=100):
    """Adam with warmup and gradient clipping."""
    schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule, b1=0.9, b2=0.95, eps=1e-8),
    )


# ─────────────────────────────────────────────────────────────
# Train step
# ─────────────────────────────────────────────────────────────

def make_train_step(model, optimizer):
    """Create jit-compiled train step."""

    def loss_fn(params, batch):
        logits = model.apply({'params': params}, batch['input_ids'])
        return cross_entropy_loss(logits, batch['target_ids'])

    grad_fn = jax.value_and_grad(loss_fn, argnums=0)

    def train_step(params, opt_state, batch):
        loss, grads         = grad_fn(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params          = optax.apply_updates(params, updates)
        grad_norm           = optax.global_norm(grads)
        return new_params, new_opt_state, {'loss': loss, 'grad_norm': grad_norm}

    # donate_argnums safe here — we thread params correctly
    return jax.jit(train_step, donate_argnums=(0, 1))


# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────

def make_fake_batch(rng, batch_size, seq_len, vocab_size, mesh):
    """Random token batch, sharded on data axis."""
    tokens    = jax.random.randint(rng, (batch_size, seq_len + 1), 0, vocab_size)
    sharding  = NamedSharding(mesh, P("data", None))
    return {
        'input_ids':  jax.device_put(tokens[:, :-1], sharding),
        'target_ids': jax.device_put(tokens[:, 1:],  sharding),
    }


# ─────────────────────────────────────────────────────────────
# Param sharding
# ─────────────────────────────────────────────────────────────

def shard_params(params, mesh):
    """Replicate params across all devices initially."""
    replicated = NamedSharding(mesh, P())
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, replicated), params
    )


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train(config, num_steps=20, batch_size=8, learning_rate=3e-4):
    """Full training loop on H100."""
    print("=" * 60)
    print("JAX Transformer Training — H100 GPU")
    print("=" * 60)

    setup_devices()
    mesh, n_data, n_model = create_mesh()

    # batch_size must be divisible by n_data for even sharding
    assert batch_size % n_data == 0, \
        f"batch_size ({batch_size}) must be divisible by n_data ({n_data})"

    rng = jax.random.PRNGKey(42)

    with jax.set_mesh(mesh):
        # Initialize
        print("\n[1/4] Initializing model...")
        rng, init_rng = jax.random.split(rng)
        model, params = init_model(config, mesh, init_rng)
        params        = shard_params(params, mesh)

        print("[2/4] Initializing optimizer...")
        optimizer = make_optimizer(learning_rate=learning_rate)
        opt_state = optimizer.init(params)

        print("[3/4] Compiling train step...")
        train_step = make_train_step(model, optimizer)

        rng, data_rng = jax.random.split(rng)
        batch = make_fake_batch(
            data_rng, batch_size, config.seq_len, config.vocab_size, mesh
        )

        t0 = time.perf_counter()
        params, opt_state, _ = train_step(params, opt_state, batch)
        jax.block_until_ready(params)
        print(f"    Compile time: {time.perf_counter()-t0:.2f}s")

        # Training loop
        print(f"\n[4/4] Training for {num_steps} steps...")
        print(f"{'Step':>6}  {'Loss':>10}  {'Grad Norm':>12}  {'ms/step':>10}")
        print("-" * 45)

        total_start = time.perf_counter()

        for step in range(num_steps):
            rng, data_rng = jax.random.split(rng)
            batch = make_fake_batch(
                data_rng, batch_size, config.seq_len, config.vocab_size, mesh
            )

            t0 = time.perf_counter()
            params, opt_state, metrics = train_step(params, opt_state, batch)
            jax.block_until_ready(params)
            step_ms = (time.perf_counter() - t0) * 1000

            loss      = float(metrics['loss'])
            grad_norm = float(metrics['grad_norm'])

            if step % 5 == 0 or step == num_steps - 1:
                print(f"{step:>6}  {loss:>10.4f}  {grad_norm:>12.4f}  {step_ms:>10.2f}")

        total_time    = time.perf_counter() - total_start
        steps_per_sec = num_steps / total_time

        print(f"\nTotal time:    {total_time:.2f}s")
        print(f"Steps/sec:     {steps_per_sec:.1f}")
        print(f"Final loss:    {loss:.4f}")

    return params, opt_state


if __name__ == "__main__":
    # Use small config for quick test, full config for real benchmark
    config = SMALL_CONFIG

    params, opt_state = train(config, num_steps=20, batch_size=8)
    print("\n✓ train.py OK")
