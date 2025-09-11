import os
import glob
from tqdm import tqdm
import traceback
import dreams.utils.data as du # Import dreams.utils.data for MSData operations
from pathlib import Path # Import Path for type hinting and path manipulation
import logging

# --- Configuration ---
EMBEDDINGS_CHUNKS_DIR = '/rhome/tmach007/bigdata/jianglab/tmach007/DreaMS_Project_TejasMachkar/dreams_embeddings_chunks'
FINAL_HDF5_PATH = '/rhome/tmach007/bigdata/jianglab/tmach007/DreaMS_Project_TejasMachkar/final_dataset/all_dreams_embeddings_with_smiles.hdf5'

# Define the columns that should be present in the merged HDF5 file.
# This should include all original columns from your MGFs plus 'dreams_embedding'.
# Ensure these match the column names used by MSData.from_pandas in generate_dreams_embeddings_chunk.py
# and the default column names expected by MSData.load().
# Common DreaMS columns are defined in dreams.definitions, but we list them explicitly here for clarity
# and to ensure 'dreams_embedding' is included.
EXPECTED_COLUMNS = ['spectrum', 'precursor_mz', 'smiles', 'ionmode', 'ms_level', 'spectrum_id', 'dreams_embedding']

# --- Setup Logging ---
# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(os.path.dirname(FINAL_HDF5_PATH), exist_ok=True)

    print(f"--- Combining DreaMS Embeddings from {EMBEDDINGS_CHUNKS_DIR} ---")

    # Find all HDF5 chunk files (now expected to be in DreaMS MSData format)
    # Convert paths to pathlib.Path objects as expected by MSData.merge
    chunk_files = sorted(glob.glob(os.path.join(EMBEDDINGS_CHUNKS_DIR, 'chunk_*.hdf5')))
    chunk_files = [Path(f) for f in chunk_files]
    
    if not chunk_files:
        print(f"No HDF5 chunk files found in {EMBEDDINGS_CHUNKS_DIR}. Aborting combination.")
        exit()

    print(f"Found {len(chunk_files)} HDF5 chunk files to merge.")

    # --- Use du.MSData.merge() for memory-efficient combination ---
    print(f"Starting memory-efficient merge to {os.path.basename(FINAL_HDF5_PATH)}...")
    try:
        # Remove any existing final HDF5 file before merging
        if os.path.exists(FINAL_HDF5_PATH):
            os.remove(FINAL_HDF5_PATH)
            print(f"Removed existing final HDF5 file: {os.path.basename(FINAL_HDF5_PATH)}")

        # MSData.merge handles reading from multiple input HDF5s and writing to a single output HDF5.
        # It appends data to datasets on disk, avoiding large in-memory DataFrames.
        # 'in_mem=False' is crucial for memory efficiency.
        # 'cols' ensures all necessary columns are merged.
        # 'add_dataset_col=False' prevents adding a 'dataset' column if you don't need it.
        # 'show_tqdm=True' provides a progress bar.
        
        final_msdata_obj = du.MSData.merge(
            pths=chunk_files,
            out_pth=Path(FINAL_HDF5_PATH),
            cols=EXPECTED_COLUMNS,
            show_tqdm=True,
            logger=logger, # You can pass a logger object here if you have one
            add_dataset_col=False, # Set to True if you want a column indicating original chunk file
            in_mem=False, # Keep data on disk
            spectra_already_trimmed=True # Assuming spectra are already trimmed/padded by generate_dreams_embeddings_chunk.py
        )
        
        # Close the final MSData object's file handle to ensure all changes are flushed
        final_msdata_obj.close()
        
        print(f"Successfully combined all chunks into {os.path.basename(FINAL_HDF5_PATH)}.")
        print(f"Total spectra in combined file: {len(final_msdata_obj)}")

    except Exception as e:
        print(f"Error during HDF5 merge operation: {e}")
        traceback.print_exc()
        # Attempt to clean up partially created final file if an error occurs
        if os.path.exists(FINAL_HDF5_PATH):
            print(f"Attempting to remove partially created file: {os.path.basename(FINAL_HDF5_PATH)}")
            os.remove(FINAL_HDF5_PATH)
        exit()

    print("--- Combination Complete ---")