# H100 Setup Instructions

## Step 1 — Get an interactive GPU session

```bash
# Request 4 H100s interactively
srun --gres=gpu:4 --ntasks=1 --cpus-per-task=16 --mem=128G --time=02:00:00 --pty bash

# Or with specific GPU type if cluster requires it
srun --gres=gpu:h100:4 --ntasks=1 --cpus-per-task=16 --mem=128G --time=02:00:00 --pty bash
```

## Step 2 — Set up environment

```bash
# Check Python version
python3 --version   # need 3.9+

# Create virtualenv
python3 -m venv venv
source venv/bin/activate

# Install JAX with CUDA 12 support
pip install "jax[cuda12]" flax optax

# Verify JAX sees GPUs
python -c "import jax; print(jax.devices())"
# Should show: [CudaDevice(id=0), CudaDevice(id=1), ...]
```

## Step 3 — Copy your project files

```bash
# From your MacBook, scp the src/ folder to the cluster
scp -r src/ username@cluster.edu:~/jax-transformer/src/
```

## Step 4 — Run in order

```bash
cd src/

# 1. Verify devices and mesh
python sharding.py

# 2. Verify model forward pass
python model.py

# 3. Run training
python train.py

# 4. Run all benchmarks (this is the important one)
python benchmark.py

# 5. Capture profiling trace
python profiling.py
```

## Troubleshooting

**"No GPU backend found"**
```bash
# Check CUDA is visible
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Reinstall JAX with correct CUDA version
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**"Only 1 device visible"**
```bash
# Check all GPUs are allocated
nvidia-smi
# If CUDA_VISIBLE_DEVICES is set, unset it
unset CUDA_VISIBLE_DEVICES
```

**"batch_size not divisible by n_data"**
- With 4 GPUs: batch_size must be divisible by 2 (n_data=2 in 2×2 mesh)
- Use batch_size = 4, 8, 16, 32...

**Out of memory on SMALL_CONFIG**
- Reduce batch_size in benchmark.py from 8 to 4
- Or reduce seq_len in SMALL_CONFIG from 512 to 256

## Expected output on H100

```
JAX backend: gpu
Devices available: 4
  CudaDevice(id=0) ... H100

Mesh shape: {'data': 2, 'model': 2}

BENCHMARK 1: JIT vs Eager
  Speedup: ~3-5x

BENCHMARK 2: MFU on H100
  MFU: ~40-65%  ← real number against 312 TFLOP/s peak

BENCHMARK 3: Sharding Constraint Impact
  Overhead removed: ~25-35%

BENCHMARK 4: Remat
  Compute overhead: +20-35%
```
