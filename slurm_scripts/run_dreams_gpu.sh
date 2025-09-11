#!/bin/bash -l

# SBATCH --job-name="dreams_embed_viz"  # Name of your job
#SBATCH -p gpu                        # Partition to submit to (GPU partition)
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=1             # Request 1 CPU core for the task
#SBATCH --mem=4G                      # Request 4GB of memory (adjust if needed for larger data)
#SBATCH --time=0-00:30:00             # Max job runtime: 30 minutes (adjust if needed)
#SBATCH --gres=gpu:1                  # Request 1 GPU of any type (e.g., gpu:a100:1 for a specific A100 GPU)
#SBATCH --output=dreams_embed_viz_%j.out  # Standard output file
#SBATCH --error=dreams_embed_viz_%j.err   # Standard error file
#SBATCH --mail-user=tmach007@ucr.edu # Your email address for notifications
#SBATCH --mail-type=ALL               # Email notifications for all job states (BEGIN, END, FAIL, REQUEUE)

# Load any necessary modules (e.g., CUDA, if not automatically loaded by conda)
# module load cuda/11.8 # Example, uncomment and adjust if needed

# Activate your conda environment
echo "Activating conda environment..."
source ~/.bashrc # Or your shell's equivalent to ensure conda is initialized
conda activate dreams
echo "Conda environment 'dreams' activated."

# Print current date and hostname for logging
date
hostname

# Run your Python script
echo "Running Python script..."
python visualizeEmbeddings.py

echo "Script finished."
date