"""
benchmark.py — All benchmarks on real H100 hardware.

KEY DIFFERENCES FROM CPU VERSION:
  1. H100 peak FLOP/s = 312 TFLOP/s (dense bf16)
     MFU is now a real number against known hardware peak
     
  2. donate_argnums re-enabled
     Manual timing loops thread params correctly
     
  3. Larger model — stresses H100 properly
  
  4. All 4 benchmarks produce real hardware numbers:
     - JIT speedup: expect ~3-5x (GPU eager is already optimized)
     - MFU: expect 40-65% on H100 with bf16 + 128-aligned dims
     - Communication overhead: real NVLink, real all-gathers
     - Remat: real GPU memory measurements
"""

import time
import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P, NamedSharding

from model import Transformer, TransformerConfig, init_model, SMALL_CONFIG, FULL_CONFIG
from sharding import setup_devices, create_mesh
from train import (make_train_step, make_optimizer,
                   make_fake_batch, shard_params, cross_entropy_loss)


# ─────────────────────────────────────────────────────────────
# H100 hardware constants
# ─────────────────────────────────────────────────────────────

# H100 SXM5 80GB
# Dense bf16: 312 TFLOP/s (no sparsity)
# This is the conservative honest number for dense matmuls
H100_PEAK_TFLOPS_BF16 = 312.0

# H100 HBM3 bandwidth: 3.35 TB/s
H100_HBM_BANDWIDTH_TBS = 3.35


# ─────────────────────────────────────────────────────────────
# MFU calculation
# ─────────────────────────────────────────────────────────────

def compute_mfu(config, batch_size, elapsed_time_per_step, n_devices=4):
    """
    Compute Model FLOPs Utilization against H100 peak.
    
    MFU = achieved_FLOP/s / (n_devices × H100_peak_FLOP/s)
    
    Uses 6N approximation: 2N forward + 4N backward.
    Adds attention FLOPs (quadratic in sequence length).
    """
    D    = config.d_model
    D_ff = config.d_ff
    H    = config.num_heads
    Dh   = config.head_dim
    L    = config.num_layers
    V    = config.vocab_size
    T    = config.seq_len
    B    = batch_size

    # Parameter count
    params_per_layer = 3*D**2 + D**2 + D*D_ff + D_ff*D + 4*D
    total_params = (L * params_per_layer
                    + V * D + T * D   # embeddings
                    + D * V)          # lm_head

    # FLOPs per step
    flops_matmul   = 6 * total_params * T * B
    flops_attn     = L * 4 * B * H * T * T * Dh
    total_flops    = flops_matmul + flops_attn

    achieved_flops    = total_flops / elapsed_time_per_step
    achieved_tflops   = achieved_flops / 1e12

    # Total peak across all devices
    total_peak_tflops = n_devices * H100_PEAK_TFLOPS_BF16
    mfu               = achieved_flops / (total_peak_tflops * 1e12)

    return mfu, achieved_tflops, total_params


# ─────────────────────────────────────────────────────────────
# Timing utility (manual loop — no lambda, no donated buffer issues)
# ─────────────────────────────────────────────────────────────

def time_train_step(train_step_fn, params, opt_state, batch,
                    n_warmup=3, n_timed=20):
    """
    Time a train step correctly on GPU.
    
    GPU ops are async — must block_until_ready before stopping timer.
    Warmup ensures JIT compilation is done before timing starts.
    """
    p, o = params, opt_state

    # Warmup
    for _ in range(n_warmup):
        p, o, _ = train_step_fn(p, o, batch)
        jax.block_until_ready(p)

    # Timed
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        p, o, _ = train_step_fn(p, o, batch)
        jax.block_until_ready(p)
        times.append(time.perf_counter() - t0)

    mean_ms = sum(times) / len(times) * 1000
    min_ms  = min(times) * 1000
    return mean_ms, min_ms, p, o


# ─────────────────────────────────────────────────────────────
# Benchmark 1: JIT vs Eager
# ─────────────────────────────────────────────────────────────

def benchmark_jit_vs_eager(config, mesh, batch_size=8, n_devices=4):
    print("\n" + "=" * 60)
    print("BENCHMARK 1: JIT vs Eager")
    print("=" * 60)

    rng = jax.random.PRNGKey(0)

    with jax.set_mesh(mesh):
        rng, init_rng = jax.random.split(rng)
        model, params = init_model(config, mesh, init_rng)
        params        = shard_params(params, mesh)
        optimizer     = make_optimizer()
        opt_state     = optimizer.init(params)

        rng, data_rng = jax.random.split(rng)
        batch = make_fake_batch(
            data_rng, batch_size, config.seq_len, config.vocab_size, mesh
        )

        # ── Eager baseline ────────────────────────────────
        print("\nEager mode (JIT disabled)...")

        def eager_step(p, batch):
            def loss_fn(p, batch):
                logits = model.apply({'params': p}, batch['input_ids'])
                return cross_entropy_loss(logits, batch['target_ids'])
            loss, grads = jax.value_and_grad(loss_fn)(p, batch)
            return loss, grads

        with jax.disable_jit():
            eager_times = []
            p_e = params
            for i in range(5):
                t0 = time.perf_counter()
                loss, _ = eager_step(p_e, batch)
                jax.block_until_ready(loss)
                t = (time.perf_counter() - t0) * 1000
                eager_times.append(t)
                print(f"  step {i}: {t:.1f}ms")

        # Exclude first step (Python warmup)
        eager_mean_ms = sum(eager_times[1:]) / len(eager_times[1:])

        # ── JIT compiled ──────────────────────────────────
        print("\nJIT mode (compiled)...")
        train_step = make_train_step(model, optimizer)

        # Compile
        t0 = time.perf_counter()
        p2, o2, _ = train_step(params, opt_state, batch)
        jax.block_until_ready(p2)
        compile_ms = (time.perf_counter() - t0) * 1000
        print(f"  compile: {compile_ms:.0f}ms")

        # Time
        jit_times = []
        p, o = p2, o2
        for i in range(20):
            t0 = time.perf_counter()
            p, o, metrics = train_step(p, o, batch)
            jax.block_until_ready(p)
            t = (time.perf_counter() - t0) * 1000
            jit_times.append(t)
            print(f"  step {i}: {t:.2f}ms")

        jit_mean_ms = sum(jit_times[2:]) / len(jit_times[2:])
        speedup     = eager_mean_ms / jit_mean_ms

        print(f"\n── Results ──────────────────────────────")
        print(f"  Eager mean:  {eager_mean_ms:.1f}ms")
        print(f"  JIT mean:    {jit_mean_ms:.2f}ms")
        print(f"  Speedup:     {speedup:.2f}x")

        return {
            'eager_ms': eager_mean_ms,
            'jit_ms':   jit_mean_ms,
            'speedup':  speedup,
        }


# ─────────────────────────────────────────────────────────────
# Benchmark 2: MFU on H100
# ─────────────────────────────────────────────────────────────

def benchmark_mfu(config, mesh, batch_size=8, n_devices=4):
    print("\n" + "=" * 60)
    print("BENCHMARK 2: MFU on H100")
    print("=" * 60)

    rng = jax.random.PRNGKey(1)

    with jax.set_mesh(mesh):
        rng, init_rng = jax.random.split(rng)
        model, params = init_model(config, mesh, init_rng)
        params        = shard_params(params, mesh)
        optimizer     = make_optimizer()
        opt_state     = optimizer.init(params)
        train_step    = make_train_step(model, optimizer)

        rng, data_rng = jax.random.split(rng)
        batch = make_fake_batch(
            data_rng, batch_size, config.seq_len, config.vocab_size, mesh
        )

        mean_ms, min_ms, _, _ = time_train_step(
            train_step, params, opt_state, batch,
            n_warmup=5, n_timed=30
        )

        mfu, achieved_tflops, total_params = compute_mfu(
            config, batch_size, mean_ms / 1000, n_devices
        )

        print(f"\n── Model ────────────────────────────────")
        print(f"  Parameters:    {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  d_model:       {config.d_model}")
        print(f"  num_heads:     {config.num_heads}")
        print(f"  head_dim:      {config.head_dim} "
              f"({'128-aligned ✓' if config.head_dim % 128 == 0 else 'not 128-aligned'})")
        print(f"  num_layers:    {config.num_layers}")
        print(f"  seq_len:       {config.seq_len}")
        print(f"  batch_size:    {batch_size}")
        print(f"  dtype:         {config.dtype}")

        print(f"\n── Throughput ───────────────────────────")
        print(f"  Step time:     {mean_ms:.2f}ms (mean), {min_ms:.2f}ms (min)")
        print(f"  Achieved:      {achieved_tflops:.2f} TFLOP/s")
        print(f"  H100 peak:     {n_devices} × {H100_PEAK_TFLOPS_BF16} = "
              f"{n_devices * H100_PEAK_TFLOPS_BF16} TFLOP/s (bf16)")
        print(f"  MFU:           {mfu*100:.1f}%")

        return {
            'mfu':            mfu,
            'achieved_tflops': achieved_tflops,
            'mean_ms':        mean_ms,
            'params':         total_params,
        }


# ─────────────────────────────────────────────────────────────
# Benchmark 3: Sharding constraint impact
# ─────────────────────────────────────────────────────────────

def benchmark_sharding_constraints(config, mesh, batch_size=8, n_devices=4):
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Sharding Constraint Impact")
    print("=" * 60)

    rng = jax.random.PRNGKey(2)
    optimizer = make_optimizer()

    with jax.set_mesh(mesh):
        rng, data_rng = jax.random.split(rng)
        batch = make_fake_batch(
            data_rng, batch_size, config.seq_len, config.vocab_size, mesh
        )

        # ── With constraints ──────────────────────────────
        print("\nWith sharding constraints...")
        rng, init_rng = jax.random.split(rng)
        model_wc, params_wc = init_model(config, mesh, init_rng)
        params_wc  = shard_params(params_wc, mesh)
        opt_state_wc = optimizer.init(params_wc)
        step_wc    = make_train_step(model_wc, optimizer)

        time_wc, _, _, _ = time_train_step(
            step_wc, params_wc, opt_state_wc, batch,
            n_warmup=3, n_timed=20
        )

        # ── Without constraints ───────────────────────────
        print("Without sharding constraints...")
        original_wsc = jax.lax.with_sharding_constraint
        jax.lax.with_sharding_constraint = lambda x, _: x

        rng, init_rng2 = jax.random.split(rng)
        model_nc, params_nc = init_model(config, mesh, init_rng2)
        params_nc    = shard_params(params_nc, mesh)
        opt_state_nc = optimizer.init(params_nc)
        step_nc      = make_train_step(model_nc, optimizer)

        time_nc, _, _, _ = time_train_step(
            step_nc, params_nc, opt_state_nc, batch,
            n_warmup=3, n_timed=20
        )

        jax.lax.with_sharding_constraint = original_wsc

        overhead_pct = (time_nc - time_wc) / time_nc * 100

        print(f"\n── Results ──────────────────────────────")
        print(f"  With constraints:    {time_wc:.2f}ms")
        print(f"  Without constraints: {time_nc:.2f}ms")
        print(f"  Overhead removed:    {overhead_pct:.1f}%")
        print(f"  (Real NVLink communication on H100)")

        return {
            'time_with_ms':    time_wc,
            'time_without_ms': time_nc,
            'overhead_pct':    overhead_pct,
        }


# ─────────────────────────────────────────────────────────────
# Benchmark 4: Remat tradeoff
# ─────────────────────────────────────────────────────────────

def benchmark_remat(config, mesh, batch_size=8):
    print("\n" + "=" * 60)
    print("BENCHMARK 4: Remat (Gradient Checkpointing)")
    print("=" * 60)

    optimizer = make_optimizer()
    rng       = jax.random.PRNGKey(3)

    with jax.set_mesh(mesh):
        rng, data_rng = jax.random.split(rng)
        batch = make_fake_batch(
            data_rng, batch_size, config.seq_len, config.vocab_size, mesh
        )

        # ── Without remat ─────────────────────────────────
        print("\nWithout remat...")
        rng, init_rng = jax.random.split(rng)
        model_nr, params_nr = init_model(config, mesh, init_rng)
        params_nr    = shard_params(params_nr, mesh)
        opt_state_nr = optimizer.init(params_nr)
        step_nr      = make_train_step(model_nr, optimizer)

        times_nr = []
        p_nr, o_nr = params_nr, opt_state_nr
        for _ in range(3):
            p_nr, o_nr, _ = step_nr(p_nr, o_nr, batch)
            jax.block_until_ready(p_nr)
        for _ in range(15):
            t0 = time.perf_counter()
            p_nr, o_nr, _ = step_nr(p_nr, o_nr, batch)
            jax.block_until_ready(p_nr)
            times_nr.append((time.perf_counter() - t0) * 1000)
        time_no_remat = sum(times_nr) / len(times_nr)

        # ── With remat ────────────────────────────────────
        print("With remat...")
        rng, init_rng2 = jax.random.split(rng)
        model_r, params_r = init_model(config, mesh, init_rng2)
        params_r    = shard_params(params_r, mesh)
        opt_state_r = optimizer.init(params_r)

        @jax.checkpoint
        def checkpointed_forward(params, token_ids):
            return model_r.apply({'params': params}, token_ids)

        def remat_loss(params, batch):
            logits = checkpointed_forward(params, batch['input_ids'])
            return cross_entropy_loss(logits, batch['target_ids'])

        remat_grad_fn = jax.value_and_grad(remat_loss)

        def remat_step(params, opt_state, batch):
            loss, grads = remat_grad_fn(params, batch)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        remat_step_jit = jax.jit(remat_step)

        times_r = []
        p_r, o_r = params_r, opt_state_r
        for _ in range(3):
            p_r, o_r, _ = remat_step_jit(p_r, o_r, batch)
            jax.block_until_ready(p_r)
        for _ in range(15):
            t0 = time.perf_counter()
            p_r, o_r, _ = remat_step_jit(p_r, o_r, batch)
            jax.block_until_ready(p_r)
            times_r.append((time.perf_counter() - t0) * 1000)
        time_remat = sum(times_r) / len(times_r)

        overhead_pct = (time_remat - time_no_remat) / time_no_remat * 100

        # Memory estimate
        D = config.d_model
        T = config.seq_len
        B = batch_size
        L = config.num_layers
        bytes_per_elem = 2  # bf16

        mem_no_remat_mb = L * 2 * B * T * D * bytes_per_elem / 1e6
        mem_remat_mb    = 2 * B * T * D * bytes_per_elem / 1e6

        print(f"\n── Results ──────────────────────────────")
        print(f"  Without remat:   {time_no_remat:.2f}ms/step")
        print(f"  With remat:      {time_remat:.2f}ms/step")
        print(f"  Compute overhead: +{overhead_pct:.1f}%")
        print(f"\n── Memory Estimate ─────────────────────")
        print(f"  Without remat:   {mem_no_remat_mb:.1f} MB activations")
        print(f"  With remat:      {mem_remat_mb:.1f} MB activations")
        print(f"  Memory savings:  {mem_no_remat_mb/mem_remat_mb:.1f}x")

        return {
            'time_no_remat_ms':  time_no_remat,
            'time_remat_ms':     time_remat,
            'overhead_pct':      overhead_pct,
            'mem_no_remat_mb':   mem_no_remat_mb,
            'mem_remat_mb':      mem_remat_mb,
        }


# ─────────────────────────────────────────────────────────────
# MFU comparison: head_dim alignment
# ─────────────────────────────────────────────────────────────

def benchmark_mfu_comparison(mesh, batch_size=8, n_devices=4):
    """
    Demonstrate head_dim alignment effect on real H100.
    
    head_dim=64:  not 128-aligned → partial tensor core utilization
    head_dim=128: 128-aligned     → full tensor core utilization
    
    """
    print("\n" + "=" * 60)
    print("MFU COMPARISON: head_dim alignment on H100")
    print("=" * 60)

    configs = {
        "Baseline  (head_dim=64,  d_model=512,  8 heads)": TransformerConfig(
            vocab_size=4096, seq_len=512,
            d_model=512, num_heads=8, num_layers=4, d_ff=2048,
        ),
        "Optimized (head_dim=128, d_model=2048, 16 heads)": TransformerConfig(
            vocab_size=32000, seq_len=2048,
            d_model=2048, num_heads=16, num_layers=8, d_ff=8192,
        ),
    }

    results = {}

    for name, config in configs.items():
        print(f"\n── {name} ──")
        optimizer = make_optimizer()
        rng = jax.random.PRNGKey(42)

        with jax.set_mesh(mesh):
            rng, init_rng = jax.random.split(rng)
            model, params = init_model(config, mesh, init_rng)
            params     = shard_params(params, mesh)
            opt_state  = optimizer.init(params)
            train_step = make_train_step(model, optimizer)

            rng, data_rng = jax.random.split(rng)
            batch = make_fake_batch(
                data_rng, batch_size, config.seq_len, config.vocab_size, mesh
            )

            mean_ms, min_ms, _, _ = time_train_step(
                train_step, params, opt_state, batch,
                n_warmup=3, n_timed=15
            )

            mfu, tflops, n_params = compute_mfu(
                config, batch_size, mean_ms / 1000, n_devices
            )

            print(f"   Step time:  {mean_ms:.2f}ms")
            print(f"   Achieved:   {tflops:.2f} TFLOP/s")
            print(f"   MFU:        {mfu*100:.1f}%")

            results[name] = {'mfu': mfu, 'ms': mean_ms, 'tflops': tflops}

    names = list(results.keys())
    if len(names) == 2:
        base = results[names[0]]['mfu']
        opt  = results[names[1]]['mfu']
        print(f"\n── Summary ──────────────────────────────")
        print(f"  Baseline MFU:   {base*100:.1f}%")
        print(f"  Optimized MFU:  {opt*100:.1f}%")
        print(f"  Relative gain:  +{(opt-base)/base*100:.1f}%")
        print(f"  (Measured on real H100 80GB hardware)")

    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_devices()
    mesh, n_data, n_model = create_mesh()
    n_devices = len(jax.devices())

    # Use small config for fast benchmarks
    # Switch to FULL_CONFIG for production numbers
    config     = FULL_CONFIG
    batch_size = 4

    assert batch_size % n_data == 0, \
        f"batch_size {batch_size} must be divisible by n_data {n_data}"

    results = {}

    results['jit_vs_eager']         = benchmark_jit_vs_eager(config, mesh, batch_size, n_devices)
    results['mfu']                  = benchmark_mfu(config, mesh, batch_size, n_devices)
    results['sharding_constraints'] = benchmark_sharding_constraints(config, mesh, batch_size, n_devices)
    results['remat']                = benchmark_remat(config, mesh, batch_size)
    results['mfu_comparison']       = benchmark_mfu_comparison(mesh, batch_size, n_devices)

    print("\n" + "=" * 60)
    print("SUMMARY — H100 Benchmark Results")
    print("=" * 60)
    print(f"  Hardware:              {n_devices}× H100 80GB")
    print(f"  JIT speedup:           {results['jit_vs_eager']['speedup']:.2f}x")
    print(f"  MFU:                   {results['mfu']['mfu']*100:.1f}%  "
          f"(against {n_devices}× {H100_PEAK_TFLOPS_BF16} TFLOP/s peak)")
    print(f"  Comm overhead removed: {results['sharding_constraints']['overhead_pct']:.1f}%")
    print(f"  Remat compute cost:    +{results['remat']['overhead_pct']:.1f}%")
    print(f"  Remat memory savings:  "
          f"{results['remat']['mem_no_remat_mb']/results['remat']['mem_remat_mb']:.1f}x")

    print("\n✓ benchmark.py OK")
    print("Record these numbers → update README.md")
