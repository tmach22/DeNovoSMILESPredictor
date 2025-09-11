#!/bin/bash -l

#SBATCH --job-name="train retriever" # Name of your job
#SBATCH -p gpu                        # Partition to submit to (GPU partition)
#SBATCH --nodes=1                     # Request 1 node per task
#SBATCH --ntasks=1                    # Run a single task per array element
#SBATCH --cpus-per-task=4             # Request 4 CPU cores per task (DreaMS benefits from some CPU for data loading/processing)
#SBATCH --mem=96G                     # Request 16GB of memory per task (adjust based on chunk size, DreaMS needs RAM)
#SBATCH --time=0-25:00:00             # Max runtime per task: 30 minutes (adjust based on chunk processing time)
#SBATCH --gres=gpu:1                  # Request 1 GPU per task (DreaMS leverages GPU heavily for embedding computation) [1]
#SBATCH --output=/bigdata/jianglab/shared/ExploreData/slurm_logs/train_retriever_%A_%a.out # Standard output file
#SBATCH --error=/bigdata/jianglab/shared/ExploreData/slurm_logs/train_retriever_%A_%a.err  # Standard error file

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

python_script="/bigdata/jianglab/shared/ExploreData/scripts/train_retriever.py"
train_processed_dir="/bigdata/jianglab/shared/ExploreData/data_splits/training_pt_files/"
val_processed_dir="/bigdata/jianglab/shared/ExploreData/data_splits/validation_pt_files/"
scaffold_library_path="/bigdata/jianglab/shared/ExploreData/scaffold_library/master_scaffold_library.txt"
save_dir="/bigdata/jianglab/shared/ExploreData/models/"  

python "${python_script}" \
    --train_processed_dir "${train_processed_dir}" \
    --val_processed_dir "${val_processed_dir}" \
    --scaffold_library_path "${scaffold_library_path}" \
    --save_dir "${save_dir}"

echo
echo "Python script finished."

echo
echo "=========================================================="
echo "Job finished at: $(date)"
echo "=========================================================="