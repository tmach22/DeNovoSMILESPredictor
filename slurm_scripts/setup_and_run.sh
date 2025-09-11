#!/bin/bash

# --- Configuration ---
# The name of your input tar file containing all the JSON files.
# This will be passed as the first argument to the script.
TAR_FILE=$1

# Names for the directories we will create.
JSON_DIR="/bigdata/jianglab/shared/ExploreData/json_files"
MGF_DIR="/bigdata/jianglab/shared/ExploreData/mgf_files"
HDF5_DIR="/bigdata/jianglab/shared/ExploreData/hdf5_files"
LOG_DIR="/bigdata/jianglab/shared/ExploreData/slurm_logs"

# --- Script Logic ---

echo "--- Starting Workflow Setup ---"

# 1. Create directories for our workflow
echo "Creating directories..."
mkdir -p "$JSON_DIR" "$MGF_DIR" "$HDF5_DIR" "$LOG_DIR"

# 2. Unpack the tar file into the JSON directory
echo "Unpacking $TAR_FILE into $JSON_DIR..."
tar -xf "$TAR_FILE" -C "$JSON_DIR"

# 3. Create a list of all JSON files to be processed
# This list will be used by the Slurm job array.
echo "Generating a list of JSON files to process..."
find "$JSON_DIR" -name "*.json" > file_list.txt

# 4. Count the number of files to determine the array size
NUM_FILES=$(wc -l < file_list.txt)
echo "Found $NUM_FILES JSON files to process."

# The Slurm array is 0-indexed, so we subtract 1
ARRAY_LIMIT=$((NUM_FILES - 1))

# 5. Submit the job array to Slurm
echo "Submitting job array to Slurm with $NUM_FILES tasks..."
sbatch --array=0-$ARRAY_LIMIT process_files.slurm

echo "--- Workflow Submitted ---"
echo "Monitor your jobs with the command: squeue -u $USER"
echo "Output and error logs will be saved in the '$LOG_DIR' directory."