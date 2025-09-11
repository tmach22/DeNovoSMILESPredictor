#!/bin/bash -l

#SBATCH --job-name=dreams_pipeline_array
#SBATCH --output=/bigdata/jianglab/shared/ExploreData/slurm_logs/job_%A_%a.out
#SBATCH --error=/bigdata/jianglab/shared/ExploreData/slurm_logs/job_%A_%a.err
#SBATCH -p gpu                             # Partition to submit to (GPU partition)
#SBATCH --nodes=1                          # Request 1 node
#SBATCH --ntasks=1                         # Run a single task
#SBATCH --cpus-per-task=1                 # Request 4 CPU cores per task
#SBATCH --mem=16G                          # Request 16GB of memory
#SBATCH --time=0-01:00:00                  # Max runtime: 1 hour
#SBATCH --gres=gpu:1                       # Request 1 GPU per task

# --- Job Execution ---

echo "=========================================================="
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job executing on: $(hostname)"
echo "=========================================================="

# 1. Set up the Conda environment
echo "Activating Conda environment..."
conda activate dreams # Use the name of your DreaMS environment

# 2. Identify the specific JSON file for this job
# The array task ID is used to pick a line from our file list.
JSON_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" file_list.txt)
BASENAME=$(basename "$JSON_FILE".json)

# 3. Define the full paths for input and output files
MGF_FILE="/bigdata/jianglab/shared/ExploreData/mgf_files/${BASENAME}.mgf"
HDF5_FILE="/bigdata/jianglab/shared/ExploreData/hdf5_files/${BASENAME}.hdf5"

echo "Processing file: $JSON_FILE"
echo "Intermediate MGF will be: $MGF_FILE"
echo "Final HDF5 will be: $HDF5_FILE"

# 4. Run the two-step pipeline
echo "--- Step 1: Running preprocess_spectra.py ---"
python preprocess_json_file.py --input_json "$JSON_FILE" --output_mgf "$MGF_FILE"

# Check if the MGF file was created successfully before proceeding
if [ -f "$MGF_FILE" ]; then
    echo "--- Step 2: Running generate_embeddings.py ---"
    python generate_embeddings.py --input_mgf "$MGF_FILE" --output_hdf5 "$HDF5_FILE"
else
    echo "Error: MGF file was not created. Skipping embedding generation."
fi

rm -r /bigdata/jianglab/shared/ExploreData/slurm_logs/*

echo "--- Task complete for $JSON_FILE ---"