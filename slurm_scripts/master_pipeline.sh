#!/bin/bash

# --- Configuration ---
MAX_JOBS=5
CHUNKS_DIR="smiles_chunks"
PROCESSED_DIR="smiles_chunks_processed"
SUBMITTED_DIR="smiles_chunks_submitted"
JOB_NAME="MassFormer_Chunk" # Must match the --job-name in process_chunk.sh

# --- Setup ---
mkdir -p "$PROCESSED_DIR"
mkdir -p "$SUBMITTED_DIR"

# --- Main Loop ---
while true; do
  # Count how many of YOUR jobs with the specified name are currently running or pending
  NUM_RUNNING=$(squeue -u "$USER" --name="$JOB_NAME" -h -t RUNNING,PENDING | wc -l)
  
  # Find an unprocessed chunk file
  CHUNK_TO_PROCESS=$(find "$CHUNKS_DIR" -name "chunk_*.csv" | head -n 1)

  # If there are fewer than MAX_JOBS running AND there is a chunk to process...
  if [[ -n "$NUM_RUNNING" && "$NUM_RUNNING" -lt 5 ]]; then
    echo "Slots available ($NUM_RUNNING/$MAX_JOBS). Submitting job for: $CHUNK_TO_PROCESS"

    # Atomically move the chunk to the 'submitted' directory to "lock" it
    SUBMITTED_CHUNK_PATH="${SUBMITTED_DIR}/$(basename "$CHUNK_TO_PROCESS")"
    mv "$CHUNK_TO_PROCESS" "$SUBMITTED_CHUNK_PATH"
    
    echo "Submitting job for ${SUBMITTED_CHUNK_PATH}..."
    # Submit the job, passing the NEW path in the 'submitted' directory
    sbatch slurm_scripts/process_chunk.sh "$SUBMITTED_CHUNK_PATH"
    
    # Wait a moment to avoid overwhelming the scheduler
    sleep 5

  # If there are no more chunks to process...
  elif [[ -n "$CHUNK_TO_PROCESS" ]] ; then
    echo "All chunks have been submitted."
    # Check if any jobs are still running before we exit
    if [[ -n "$NUM_RUNNING" && "$NUM_RUNNING" -eq 0 ]]; then
      echo "All jobs have finished. Exiting pipeline."
      break
    else
      echo "$NUM_RUNNING jobs are still running. Waiting..."
    fi
  else
    # If job queue is full, just wait
    echo "Job queue is full ($NUM_RUNNING/$MAX_JOBS). Waiting..."
  fi
  
  # Wait for 1 minute before checking again
  sleep 60
done

echo "Master pipeline finished."