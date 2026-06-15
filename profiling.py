"""
profiling.py — XProf trace capture and MFU comparison on H100.

Same structure as CPU version but:
  - Real GPU trace shows actual CUDA kernel launches
  - Tensor core utilization visible in trace
  - Real NVLink communication ops visible
  - No fake devices — real device timelines
"""

import time
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

from model import Transformer, TransformerConfig, init_model, SMALL_CONFIG
from sharding import setup_devices, create_mesh
from train import make_train_step, make_optimizer, make_fake_batch, shard_params
from benchmark import compute_mfu, H100_PEAK_TFLOPS_BF16


def capture_trace(config, mesh, batch_size=8, trace_dir="/tmp/jax-trace-h100"):
    """
    Capture execution trace on H100.
    
    View with:
      tensorboard --logdir=/tmp/jax-trace-h100
    Or drag .json.gz to https://ui.perfetto.dev
    
    On H100, the trace shows:
      - Real CUDA kernel launches (not CPU ops)
      - Tensor core active vs idle time
      - NVLink all-reduce ops between GPUs
      - XLA fusion clusters as single custom calls
    """
    print(f"\nCapturing H100 trace → {trace_dir}")

    rng = jax.random.PRNGKey(99)

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

        # Warmup outside trace
        params, opt_state, _ = train_step(params, opt_state, batch)
        jax.block_until_ready(params)

        # Capture
        with jax.profiler.trace(trace_dir, create_perfetto_link=False):
            for _ in range(3):
                params, opt_state, _ = train_step(params, opt_state, batch)
            jax.block_until_ready(params)

    print(f"✓ Trace written to {trace_dir}")
    print(f"  View: drag {trace_dir}/plugins/profile/*/**.json.gz")
    print(f"        to https://ui.perfetto.dev")
    return trace_dir


def report_device_placement(config, mesh, batch_size=4):
    """Print tensor shardings across H100 devices."""
    print("\n" + "=" * 60)
    print("DEVICE PLACEMENT REPORT — H100")
    print("=" * 60)

    rng = jax.random.PRNGKey(7)

    with jax.set_mesh(mesh):
        rng, init_rng = jax.random.split(rng)
        model, params = init_model(config, mesh, init_rng)
        params        = shard_params(params, mesh)

        print("\n── Parameter Shardings ──")
        for path, leaf in jax.tree_util.tree_leaves_with_path(params):
            path_str    = ".".join(str(p.key) for p in path)
            sharding_str = str(leaf.sharding)[:50] if hasattr(leaf, 'sharding') else "replicated"
            print(f"  {path_str:45s}  {str(leaf.shape):20s}  {sharding_str}")

        rng, data_rng = jax.random.split(rng)
        batch = make_fake_batch(
            data_rng, batch_size, config.seq_len, config.vocab_size, mesh
        )

        print("\n── Batch Shardings ──")
        for key, arr in batch.items():
            print(f"  {key:20s}  {str(arr.shape):20s}  {arr.sharding}")


if __name__ == "__main__":
    setup_devices()
    mesh, n_data, n_model = create_mesh()
    n_devices = len(jax.devices())

    config     = SMALL_CONFIG
    batch_size = 8

    report_device_placement(config, mesh, batch_size=4)

    capture_trace(config, mesh, batch_size=batch_size,
                  trace_dir="/tmp/jax-trace-h100")

    print("\n✓ profiling.py OK")
