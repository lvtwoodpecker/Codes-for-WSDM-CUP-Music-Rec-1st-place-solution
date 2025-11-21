#!/bin/bash
#SBATCH --job-name=train_nn
#SBATCH --output=logs/nn_%j.out
#SBATCH --error=logs/nn_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8        # Data generator needs CPU cores
#SBATCH --mem=64GB               # Large RAM for pandas read_csv
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --time=48:00:00          # 2 days

# Create logs directory so the script doesn't fail
mkdir -p logs

module purge
module load singularity/3.7.4

# Define paths
OVERLAY_IMG="my_env.ext3"
SIF_IMG="tensorflow_latest-gpu.sif"

# Execute
# :ro means read-only for the overlay (safer for batch jobs)
singularity exec --nv --overlay $OVERLAY_IMG:ro $SIF_IMG python nn_training_multi_seed.py