#!/bin/bash -l

#SBATCH --job-name="dreams_embed_gen_array" # Name of your job
#SBATCH -p gpu                        # Partition to submit to (GPU partition)
#SBATCH --nodes=1                     # Request 1 node per task
#SBATCH --ntasks=1                    # Run a single task per array element
#SBATCH --cpus-per-task=2             # Request 4 CPU cores per task (DreaMS benefits from some CPU for data loading/processing)
#SBATCH --mem=64G                     # Request 16GB of memory per task (adjust based on chunk size, DreaMS needs RAM)
#SBATCH --time=0-25:00:00             # Max runtime per task: 30 minutes (adjust based on chunk processing time)
#SBATCH --gres=gpu:1                  # Request 1 GPU per task (DreaMS leverages GPU heavily for embedding computation) [1]
#SBATCH --output=/bigdata/jianglab/shared/ExploreData/slurm_logs/embed_gen_%A_%a.out # Standard output file
#SBATCH --error=/bigdata/jianglab/shared/ExploreData/slurm_logs/embed_gen_%A_%a.err  # Standard error file

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
INPUT_MGF_FILE="/bigdata/jianglab/shared/ExploreData/data_for_dreams_in_silica/"
OUTPUT_HDF5="/bigdata/jianglab/shared/ExploreData/hdf5_files/dreams_embeddings.hdf5"

# --- 4. Run the Python Script ---
echo "Running Python script: ${PYTHON_SCRIPT}"
echo "Input file: ${INPUT_MGF_FILE}"
echo "Output file: ${OUTPUT_HDF5}"
echo

python "${PYTHON_SCRIPT}" --input_dir "${INPUT_MGF_FILE}" --output_file "${OUTPUT_HDF5}"

echo
echo "Python script finished."

# --- 5. Deactivate Environment ---
conda deactivate
echo "Conda environment deactivated."

echo
echo "=========================================================="
echo "Job finished at: $(date)"
echo "=========================================================="