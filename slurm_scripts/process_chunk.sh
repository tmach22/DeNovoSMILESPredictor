#!/bin/bash

#SBATCH --job-name=MassFormer_Chunk      # A consistent name for all jobs in this pipeline
#SBATCH --partition=gpu                  # The GPU partition (confirm this name)
#SBATCH --gres=gpu:1                     # Request one GPU
#SBATCH --cpus-per-task=4                # Request 4 CPU cores
#SBATCH --mem=16G                        # Increased memory request
#SBATCH --time=04:00:00                  # 4-hour runtime per chunk (adjust if needed)
#SBATCH --output=/bigdata/jianglab/shared/massformer/logs/mf_chunk_%j.out    # Log file for standard output (%j is the job ID)
#SBATCH --error=/bigdata/jianglab/shared/massformer/logs/mf_chunk_%j.err     # Log file for standard error

set -e

# --- Safety Check: Ensure an input file is provided ---
if [ -z "$1" ]; then
    echo "Error: No input chunk file provided to the script."
    exit 1
fi

# --- Environment Setup ---
echo "Job $SLURM_JOB_ID started on $(hostname)"
echo "Processing chunk: $1"

# Load necessary modules
module purge
module load cuda/11.7 # Or the version you found earlier

# Initialize and activate Conda environment
eval "$(conda shell.bash hook)"
conda activate MF-GPU

# --- Define File Paths ---
INPUT_SMILES_CHUNK=$1
PROCESSED_DIR="/bigdata/jianglab/shared/massformer/smiles_chunks_processed"
CHUNK_BASENAME=$(basename "$INPUT_SMILES_CHUNK")
echo "Chunk basename: $CHUNK_BASENAME"

# Define directories for intermediate and final outputs
PREDICTIONS_DIR="predicted_spectra"
FINAL_DIR="final_data_chunks"

# Define output paths for this specific chunk
MASSFORMER_OUTPUT="${PREDICTIONS_DIR}/${CHUNK_BASENAME}_predicted.csv"
echo "MassFormer output will be saved to: $MASSFORMER_OUTPUT"
FINAL_OUTPUT="${FINAL_DIR}/${CHUNK_BASENAME}_final.csv"
echo "Final output will be saved to: $FINAL_OUTPUT"

# --- Main Commands ---

# 1. Run MassFormer Inference on the chunk
echo "Starting MassFormer prediction for ${CHUNK_BASENAME}..."
python scripts/run_inference.py \
  -c config/demo/demo_eval.yml \
  -s "$INPUT_SMILES_CHUNK" \
  -o "$MASSFORMER_OUTPUT" \
  --prec_types "[M+H]+" \
  -d 0

# 2. Check if inference was successful before proceeding
if [ -z "$MASSFORMER_OUTPUT" ]; then
    echo "MassFormer output file was not created. Exiting with an error."
    exit 1
fi

# 3. Run the Post-Processing and Merge Script
echo "Starting post-processing for ${CHUNK_BASENAME}..."
python scripts/merge_and_finalize.py \
  --smiles_input "$INPUT_SMILES_CHUNK" \
  --predictions_input "$MASSFORMER_OUTPUT" \
  --output "$FINAL_OUTPUT"

echo "Processing successful. Moving input chunk to processed directory."
mv "$INPUT_SMILES_CHUNK" "$PROCESSED_DIR/"

# 4. Clean up the intermediate predictions file to save space
rm "$MASSFORMER_OUTPUT"

echo "Job for chunk ${CHUNK_BASENAME} finished successfully."