#!/bin/bash -l

#SBATCH --job-name="dreams_embed_gen_array" # Name of your job
#SBATCH -p gpu                        # Partition to submit to (GPU partition)
#SBATCH --nodes=1                     # Request 1 node per task
#SBATCH --ntasks=1                    # Run a single task per array element
#SBATCH --cpus-per-task=2             # Request 4 CPU cores per task (DreaMS benefits from some CPU for data loading/processing)
#SBATCH --mem=64G                     # Request 16GB of memory per task (adjust based on chunk size, DreaMS needs RAM)
#SBATCH --time=0-25:00:00             # Max runtime per task: 30 minutes (adjust based on chunk processing time)
#SBATCH --gres=gpu:1                  # Request 1 GPU per task (DreaMS leverages GPU heavily for embedding computation) [1]
#SBATCH --output=/bigdata/jianglab/shared/ExploreData/slurm_logs/train_model_tri_gnn_%A_%a.out # Standard output file
#SBATCH --error=/bigdata/jianglab/shared/ExploreData/slurm_logs/train_model_tri_gnn_%A_%a.err  # Standard error file

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

python_script="/bigdata/jianglab/shared/ExploreData/scripts/insilica_train.py"
train_data_path="/bigdata/jianglab/shared/ExploreData/data_splits/train_set.hdf5"
val_data_path="/bigdata/jianglab/shared/ExploreData/data_splits/val_set.hdf5"
smiled_vocab_path="/bigdata/jianglab/shared/ExploreData/vocab/smiles_vocab.json"
formula_vocab_path="/bigdata/jianglab/shared/ExploreData/vocab/formula_vocab.json"
model_output_path="/bigdata/jianglab/shared/ExploreData/models/"
patience=10
learning_rate=2e-4

python "${python_script}" --train_path "${train_data_path}" --val_path "${val_data_path}" --smiles_vocab_path "${smiled_vocab_path}" --formula_vocab_path "${formula_vocab_path}" --save_dir "${model_output_path}" --patience "${patience}" --learning_rate "${learning_rate}"

echo
echo "Python script finished."

echo
echo "=========================================================="
echo "Job finished at: $(date)"
echo "=========================================================="