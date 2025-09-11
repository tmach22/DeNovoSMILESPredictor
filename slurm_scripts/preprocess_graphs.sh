#!/bin/bash -l

#SBATCH --job-name="preprocess_graphs" # Name of your job
#SBATCH -p gpu                        # Partition to submit to (GPU partition)
#SBATCH --nodes=1                     # Request 1 node per task
#SBATCH --ntasks=1                    # Run a single task per array element
#SBATCH --cpus-per-task=4             # Request 4 CPU cores per task (DreaMS benefits from some CPU for data loading/processing)
#SBATCH --mem=96G                     # Request 16GB of memory per task (adjust based on chunk size, DreaMS needs RAM)
#SBATCH --time=0-05:00:00             # Max runtime per task: 30 minutes (adjust based on chunk processing time)
#SBATCH --gres=gpu:1                  # Request 1 GPU per task (DreaMS leverages GPU heavily for embedding computation) [1]
#SBATCH --output=/bigdata/jianglab/shared/ExploreData/slurm_logs/preprocess_graphs_train_%A_%a.out # Standard output file
#SBATCH --error=/bigdata/jianglab/shared/ExploreData/slurm_logs/preprocess_graphs_train44_%A_%a.err  # Standard error file

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
source venv/bin/activate
echo "Conda environment 'dreams' activated."
echo

# --- 3. Define File Paths ---
# IMPORTANT: Set the full paths to your Python script, input JSON, and output HDF5 file
PYTHON_SCRIPT="/bigdata/jianglab/shared/ExploreData/scripts/preprocess_and_collate_graphs.py"
HDF5_PATH="/bigdata/jianglab/shared/ExploreData/data_splits/train_set.hdf5"
OUTPUT_TEXT_PATH="/bigdata/jianglab/shared/ExploreData/scaffold_library/master_scaffold_library.txt"
similarity_matrix_path="/bigdata/jianglab/shared/ExploreData/hdf5_files/similarity_matrix.hdf5"
output_dir="/bigdata/jianglab/shared/ExploreData/data_splits/training_pt_files/"
shard_size=50000
num_workers=4

python "${PYTHON_SCRIPT}" --hdf5_path "${HDF5_PATH}" --scaffold_library_path "${OUTPUT_TEXT_PATH}" --similarity_matrix_path "${similarity_matrix_path}" --output_dir "${output_dir}" --shard_size "${shard_size}" --num_workers "${num_workers}"

echo
echo "Python script finished."

# --- 5. Deactivate Environment ---
deactivate
echo "Conda environment deactivated."

echo
echo "=========================================================="
echo "Job finished at: $(date)"
echo "=========================================================="