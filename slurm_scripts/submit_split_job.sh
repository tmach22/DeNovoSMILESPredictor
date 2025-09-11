#!/bin/bash

#SBATCH --job-name="split_dataset_into_folds" # Name of your job
#SBATCH -p gpu                        # Partition to submit to (GPU partition)
#SBATCH --nodes=1                     # Request 1 node per task
#SBATCH --ntasks=1                    # Run a single task per array element
#SBATCH --cpus-per-task=4             # Request 4 CPU cores per task (DreaMS benefits from some CPU for data loading/processing)
#SBATCH --mem=16G                     # Request 16GB of memory per task (adjust based on chunk size, DreaMS needs RAM)
#SBATCH --time=0-00:10:00             # Max runtime per task: 30 minutes (adjust based on chunk processing time)
#SBATCH --gres=gpu:1                  # Request 1 GPU per task (DreaMS leverages GPU heavily for embedding computation) [1]
#SBATCH --output=/rhome/tmach007/bigdata/jianglab/tmach007/DreaMS_Project_TejasMachkar/slurm_logs/embed_gen_%A_%a.out # Standard output file
#SBATCH --error=/rhome/tmach007/bigdata/jianglab/tmach007/DreaMS_Project_TejasMachkar/slurm_logs/embed_gen_%A_%a.err  # Standard error file
#SBATCH --mail-type=END,FAIL          # Email notifications on job end or failure

# Load necessary modules (adjust for your cluster's environment)
# This typically includes modules for Conda, Python, or specific compilers if needed.
module purge
# Example: module load anaconda3 # Uncomment and adjust if your cluster requires loading anaconda module

# Activate your conda environment
echo "Activating conda environment..."
source ~/.bashrc # Or your shell's equivalent to ensure conda is initialized
conda activate dreams
echo "Conda environment 'dreams' activated."

# Define input and output paths for the script
INPUT_HDF5_FILE="/rhome/tmach007/bigdata/jianglab/tmach007/DreaMS_Project_TejasMachkar/final_dataset/all_dreams_embeddings_with_smiles.hdf5"
OUTPUT_HDF5_FILE="/rhome/tmach007/bigdata/jianglab/tmach007/DreaMS_Project_TejasMachkar/final_dataset/all_dreams_embeddings_with_smiles_murcko_split.hdf5"

# Run your Python splitting script
# --num_workers should ideally match --cpus-per-task for optimal CPU utilization.
python split_dataset.py \
    --input_hdf5_path "${INPUT_HDF5_FILE}" \
    --output_hdf5_path "${OUTPUT_HDF5_FILE}" \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --smiles_col_name "smiles" \
    --spec_id_col_name "spectrum_id" \
    --num_workers 4 # Set this to match --cpus-per-task for optimal parallelization