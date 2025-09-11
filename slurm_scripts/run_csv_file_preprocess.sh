#!/bin/bash -l

#SBATCH --job-name="generate_dreamings" # Name of your job
#SBATCH -p gpu                        # Partition to submit to (GPU partition)
#SBATCH --nodes=1                     # Request 1 node per task
#SBATCH --ntasks=1                    # Run a single task per array element
#SBATCH --cpus-per-task=2             # Request 4 CPU cores per task (DreaMS benefits from some CPU for data loading/processing)
#SBATCH --mem=16G                     # Request 16GB of memory per task (adjust based on chunk size, DreaMS needs RAM)
#SBATCH --time=0-02:00:00             # Max runtime per task: 30 minutes (adjust based on chunk processing time)
#SBATCH --gres=gpu:1                  # Request 1 GPU per task (DreaMS leverages GPU heavily for embedding computation) [1]
#SBATCH --output=/bigdata/jianglab/shared/ExploreData/slurm_logs/dreams_embed_%A_%a.out # Standard output file
#SBATCH --error=/bigdata/jianglab/shared/ExploreData/slurm_logs/dreams_embed_%A_%a.err  # Standard error file

echo "=========================================================="
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Job executing on: $(hostname)"
echo "Starting at: $(date)"
echo "=========================================================="
echo

# --- 2. Set up Conda Environment ---
echo "Activating Conda environment..."
# Sourcing your shell's rc file ensures 'conda' command is available
# This might be ~/.bashrc, ~/.bash_profile, or ~/.profile depending on your setup
source ~/.bashrc
conda activate dreams
echo "Conda environment 'dreams' activated."
echo

# --- 3. Define File Paths ---
# IMPORTANT: Set the full paths to your Python script, input JSON, and output HDF5 file
PYTHON_SCRIPT="/bigdata/jianglab/shared/DreaMS/generate_embeddings.py"
INPUT_MGF_DIRECTORY="/bigdata/jianglab/shared/ExploreData/raw_data_for_dreams/"
OUTPUT_HDF5="/bigdata/jianglab/shared/ExploreData/hdf5_files/dreams_dataset.hdf5"

# --- 4. Run the Python Script ---
echo "Running Python script: ${PYTHON_SCRIPT}"
echo "Input directory: ${INPUT_MGF_DIRECTORY}"
echo "Output file: ${OUTPUT_HDF5}"
echo

python "${PYTHON_SCRIPT}" -i "${INPUT_MGF_DIRECTORY}" -o "${OUTPUT_HDF5}"

echo
echo "Python script finished."

# --- 5. Deactivate Environment ---
conda deactivate
echo "Conda environment deactivated."

echo
echo "=========================================================="
echo "Job finished at: $(date)"
echo "=========================================================="