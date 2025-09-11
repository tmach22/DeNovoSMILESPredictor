#!/bin/bash -l

# SBATCH --job-name="train decoder model"  # Name of your job
#SBATCH -p gpu                        # Partition to submit to (GPU partition)
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=2             # Request 1 CPU core for the task
#SBATCH --mem=16G                      # Request 4GB of memory (adjust if needed for larger data)
#SBATCH --time=0-04:00:00             # Max job runtime: 30 minutes (adjust if needed)
#SBATCH --gres=gpu:1                  # Request 1 GPU of any type (e.g., gpu:a100:1 for a specific A100 GPU)
#SBATCH --output=/bigdata/jianglab/shared/ExploreData/slurm_logs/testing_run_%j.out  # Standard output file
#SBATCH --error=/bigdata/jianglab/shared/ExploreData/slurm_logs/testing_run_%j.err   # Standard error file

# Load any necessary modules (e.g., CUDA, if not automatically loaded by conda)
# module load cuda/11.8 # Example, uncomment and adjust if needed

# Activate your conda environment
echo "Activating conda environment..."
source venv/bin/activate  # Use the name of your DreaMS environment
echo "Conda environment 'dreams' activated."

# Print current date and hostname for logging
date
hostname

# Run your Python script
echo "Running Python script..."
python /bigdata/jianglab/shared/ExploreData/scripts/predict.py

echo "Script finished."
date