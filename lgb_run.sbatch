#!/bin/bash
#SBATCH --job-name=train_lgb
#SBATCH --output=logs/lgb_%j.out
#SBATCH --error=logs/lgb_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16       # LightGBM scales well with more CPUs
#SBATCH --mem=128GB              # 128GB RAM for GBDT
#SBATCH --time=48:00:00          # 2 days

mkdir -p logs

module purge
module load singularity/3.7.4

# Define paths
OVERLAY_IMG="my_env.ext3"
SIF_IMG="tensorflow_latest-gpu.sif"

echo "Starting LightGBM Training..."

# Execute
singularity exec --overlay $OVERLAY_IMG:ro $SIF_IMG python -u lgb_training_multi_seed.py