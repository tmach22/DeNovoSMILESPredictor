#!/bin/bash -l

# SBATCH --job-name="train decoder model"  # Name of your job
#SBATCH -p gpu                        # Partition to submit to (GPU partition)
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # Request 1 CPU core for the task
#SBATCH --mem=16G                      # Request 4GB of memory (adjust if needed for larger data)
#SBATCH --time=0-20:00:00             # Max job runtime: 30 minutes (adjust if needed)
#SBATCH --gres=gpu:1                  # Request 1 GPU of any type (e.g., gpu:a100:1 for a specific A100 GPU)
#SBATCH --output=/bigdata/jianglab/shared/ExploreData/slurm_logs/split_data_%j.out  # Standard output file
#SBATCH --error=/bigdata/jianglab/shared/ExploreData/slurm_logs/split_data_%j.err   # Standard error file

# Load any necessary modules (e.g., CUDA, if not automatically loaded by conda)
# module load cuda/11.8 # Example, uncomment and adjust if needed

# Activate your conda environment
echo "Activating conda environment..."
source venv/bin/activate  # Use the name of your DreaMS environment
echo "Conda environment 'dreams' activated."

# Print current date and hostname for logging
date
hostname

input_file="/bigdata/jianglab/shared/ExploreData/hdf5_files/dreams_embeddings.hdf5"
output_dir="/bigdata/jianglab/shared/ExploreData/large_data_split/"

# Run your Python script
echo "Running Python script..."
python /bigdata/jianglab/shared/ExploreData/scripts/split_spectra.py --input_file "${input_file}" --output_dir "${output_dir}"

echo "Script finished."
date