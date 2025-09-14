#!/bin/bash -l

#SBATCH --job-name="dreams_embed_gen_array" # Name of your job
#SBATCH -p gpu                        # Partition to submit to (GPU partition)
#SBATCH --nodes=1                     # Request 1 node per task
#SBATCH --ntasks=1                    # Run a single task per array element
#SBATCH --cpus-per-task=4             # Request 4 CPU cores per task (DreaMS benefits from some CPU for data loading/processing)
#SBATCH --mem=16G                     # Request 16GB of memory per task (adjust based on chunk size, DreaMS needs RAM)
#SBATCH --time=0-01:00:00             # Max runtime per task: 30 minutes (adjust based on chunk processing time)
#SBATCH --gres=gpu:1                  # Request 1 GPU per task (DreaMS leverages GPU heavily for embedding computation) [1]
#SBATCH --output=/rhome/tmach007/bigdata/jianglab/tmach007/DreaMS_Project_TejasMachkar/slurm_logs/embed_gen_%A_%a.out # Standard output file
#SBATCH --error=/rhome/tmach007/bigdata/jianglab/tmach007/DreaMS_Project_TejasMachkar/slurm_logs/embed_gen_%A_%a.err  # Standard error file
#SBATCH --mail-user=tmach007@ucr.edu # Your e   mail address for notifications
#SBATCH --mail-type=END,FAIL          # Email notifications on job end or failure

# --- Array Job Configuration ---
# Define the directory where MGF chunks are stored (output of split_mgf_file.py)
MGF_CHUNKS_DIR="/rhome/tmach007/bigdata/jianglab/tmach007/DreaMS_Project_TejasMachkar/processed_ms_data/mgf_chunks"
# Define the directory where each chunk's HDF5 results will be saved
EMBEDDINGS_OUTPUT_DIR="/rhome/tmach007/bigdata/jianglab/tmach007/DreaMS_Project_TejasMachkar/dreams_embeddings_chunks"

# Create output directories if they don't exist (these will be created by each array task)
mkdir -p "${MGF_CHUNKS_DIR}"
mkdir -p "${EMBEDDINGS_OUTPUT_DIR}"
mkdir -p "/rhome/tmach007/bigdata/jianglab/tmach007/DreaMS_Project_TejasMachkar/slurm_logs"

# This SBATCH directive MUST be at the top with other SBATCH directives.
# It will be evaluated using the NUM_CHUNKS value determined above.
#SBATCH --array=0-$(($NUM_CHUNKS - 1))

echo "Starting DreaMS embedding generation for chunk ${SLURM_ARRAY_TASK_ID}..."
date

# Activate your conda environment
echo "Activating conda environment..."
source ~/.bashrc # Or your shell's equivalent to ensure conda is initialized
conda activate dreams
echo "Conda environment 'dreams' activated."

ITERATIONS=36

# The for loop iterates from 1 to the value of ITERATIONS
for i in $(seq 0 $ITERATIONS); do

    CHUNK_ID="$(printf "%05d" $i)"

    # Define the input MGF chunk file for this specific array task
    CHUNK_FILE="${MGF_CHUNKS_DIR}/chunk_${CHUNK_ID}.mgf"

    # Define the output HDF5 file for this specific chunk's results
    CHUNK_OUTPUT_HDF5="${EMBEDDINGS_OUTPUT_DIR}/chunk_${CHUNK_ID}.hdf5"

    # Run the Python script for this chunk
    echo "Running generate_dreams_embeddings_chunk.py for ${CHUNK_FILE}..."
    python /rhome/tmach007/bigdata/jianglab/tmach007/DreaMS/generate_dreams_embeddings_chunk.py "${CHUNK_FILE}" "${CHUNK_OUTPUT_HDF5}" "${CHUNK_ID}"

    echo "Embedding generation for chunk ${CHUNK_ID} finished."
    date
done

echo "Generating embedding script executed."