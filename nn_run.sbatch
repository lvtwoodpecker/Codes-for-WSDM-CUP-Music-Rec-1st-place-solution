#!/bin/bash
#SBATCH --job-name=TrainNN
#SBATCH --output=logs_nn_%j.out    # Output log file (%j = Job ID)
#SBATCH --error=logs_nn_%j.err     # Error log file
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8          # 8 CPUs for data loading
#SBATCH --mem=64GB                 # 64GB RAM for big CSVs
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --time=48:00:00            # Run for 48 hours

# Create a logs folder so files don't clutter your main dir
mkdir -p logs

# Clean environment
module purge
module load singularity/3.7.4

# Define file names
SIF_IMAGE="tensorflow_latest-gpu.sif"
OVERLAY="my_env.ext3"

echo "Starting NN Training on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run the Python script
# --nv: Enable GPU
# :ro:  Read-Only overlay (Critical for running multiple jobs)
singularity exec --nv --overlay $OVERLAY:ro $SIF_IMAGE python nn_training_multi_seed.py