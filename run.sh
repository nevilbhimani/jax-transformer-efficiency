#!/bin/bash
#SBATCH --job-name=jax-transformer
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_output.txt
#SBATCH --error=logs/%j_error.txt

# ─────────────────────────────────────────────────────────────
# SLURM job script for H100 cluster
#
# USAGE:
#   sbatch run.sh
#
# OR for interactive session (recommended first time):
#   srun --gres=gpu:4 --ntasks=1 --cpus-per-task=16 \
#        --mem=128G --time=02:00:00 --pty bash
#   Then run commands manually
#
# ADJUST:
#   --gres=gpu:4    → request 4 GPUs (change to gpu:2 or gpu:8 if needed)
#   --time=02:00:00 → 2 hour limit (adjust based on cluster policy)
#   --mem=128G      → system RAM (not GPU memory)
# ─────────────────────────────────────────────────────────────

echo "============================================"
echo "JAX/XLA Transformer Efficiency Optimizer"
echo "H100 GPU Benchmark Run"
echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $SLURM_GPUS"
echo "Time:      $(date)"
echo ""

# Create logs directory
mkdir -p logs

# ── Environment setup ────────────────────────────────────────
# Option A: module system (common on university clusters)
# module load cuda/12.0
# module load python/3.11

# Option B: conda environment
# conda activate jax-env

# Option C: virtualenv (what we use)
source venv/bin/activate

# Verify GPU visibility
echo "── nvidia-smi ──"
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader
echo ""

# Verify JAX sees GPUs
echo "── JAX device check ──"
python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'Backend:     {jax.default_backend()}')
print(f'Devices:     {jax.devices()}')
"
echo ""

# ── Run benchmarks ───────────────────────────────────────────
cd src/

echo "── Running sharding.py ──"
python sharding.py
echo ""

echo "── Running model.py ──"
python model.py
echo ""

echo "── Running train.py ──"
python train.py
echo ""

echo "── Running benchmark.py ──"
python benchmark.py
echo ""

echo "── Running profiling.py ──"
python profiling.py
echo ""

echo "============================================"
echo "All benchmarks complete"
echo "Time: $(date)"
echo "============================================"
