"""
sharding.py — Device mesh setup for real GPU hardware.

KEY DIFFERENCE FROM CPU VERSION:
  No XLA_FLAGS simulation. JAX sees real H100 GPUs directly.
  We detect however many GPUs are available and build the
  largest valid 2D mesh from them.

  CPU version: forced 4 fake devices via XLA_FLAGS
  GPU version: uses actual hardware devices JAX finds
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils


def setup_devices():
    """
    Detect and return available GPU devices.
    
    No XLA_FLAGS needed — JAX sees real GPUs automatically
    when jax[cuda12] is installed and CUDA is available.
    """
    devices = jax.devices()
    backend  = jax.default_backend()
    
    print(f"JAX backend:       {backend}")
    print(f"Devices available: {len(devices)}")
    for d in devices:
        print(f"  {d}")
    
    if backend != "gpu":
        raise RuntimeError(
            f"Expected GPU backend, got '{backend}'. "
            "Make sure jax[cuda12] is installed and you're on a GPU node."
        )
    
    return devices


def get_mesh_shape(n_devices):
    """
    Return the best 2D mesh shape for n_devices.
    
    We want the mesh to be as square as possible to balance
    data parallelism and model parallelism.
    
    Examples:
      4 devices → (2, 2)   ← data=2, model=2
      8 devices → (4, 2)   ← data=4, model=2
      2 devices → (2, 1)   ← data=2, model=1 (data parallel only)
      1 device  → (1, 1)   ← single device (no sharding)
    """
    if n_devices == 8:
        return (4, 2)   # 4 data shards × 2 model shards
    elif n_devices == 4:
        return (2, 2)   # 2 data shards × 2 model shards
    elif n_devices == 2:
        return (2, 1)   # 2 data shards, no model sharding
    elif n_devices == 1:
        return (1, 1)   # single device
    else:
        # Generic: find largest factor pair closest to square
        for i in range(int(n_devices**0.5), 0, -1):
            if n_devices % i == 0:
                return (n_devices // i, i)
        return (n_devices, 1)


def create_mesh():
    """
    Create device mesh from available GPUs.
    
    Returns:
        mesh: Mesh with axes ('data', 'model')
        n_data: number of data-parallel shards
        n_model: number of model-parallel shards
    """
    devices  = jax.devices()
    n        = len(devices)
    shape    = get_mesh_shape(n)
    n_data, n_model = shape
    
    device_array = mesh_utils.create_device_mesh(shape)
    mesh = Mesh(device_array, axis_names=("data", "model"))
    
    print(f"\nMesh shape:  {mesh.shape}")
    print(f"  data axis:  {n_data} devices  (batch sharding)")
    print(f"  model axis: {n_model} devices  (tensor parallelism)")
    print(f"Devices:\n{device_array}")
    
    return mesh, n_data, n_model


def get_partition_specs():
    """
    PartitionSpecs for every tensor in the Transformer.
    Same as CPU version — sharding strategy is hardware-independent.
    """
    return {
        "input":                P("data", None, None),
        "output":               P("data", None, None),
        "embedding":            P(None, None),
        "attention_qkv_weight": P(None, "model", None),
        "attention_qkv_bias":   P("model", None),
        "attention_out_weight": P("model", None, None),
        "attention_out_bias":   P(None,),
        "ffn_up_weight":        P(None, "model"),
        "ffn_up_bias":          P("model",),
        "ffn_down_weight":      P("model", None),
        "ffn_down_bias":        P(None,),
        "layer_norm_scale":     P(None,),
        "layer_norm_bias":      P(None,),
    }


def make_sharding(mesh, pspec):
    return NamedSharding(mesh, pspec)


def describe_sharding(array, name="tensor"):
    print(f"\n── {name} ──")
    print(f"  Shape:    {array.shape}")
    print(f"  Dtype:    {array.dtype}")
    if hasattr(array, 'sharding'):
        print(f"  Sharding: {array.sharding}")


if __name__ == "__main__":
    devices       = setup_devices()
    mesh, nd, nm  = create_mesh()
    specs         = get_partition_specs()
    
    print("\n── Partition specs ──")
    for name, spec in specs.items():
        print(f"  {name:30s} → {spec}")
    
    with jax.set_mesh(mesh):
        sharding = NamedSharding(mesh, P("data", None))
        x = jax.device_put(jnp.ones((4, 8)), sharding)
        describe_sharding(x, "test array")
    
    print("\n✓ sharding.py OK")
