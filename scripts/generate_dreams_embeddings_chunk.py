import numpy as np
import os
from dreams.api import dreams_embeddings
import dreams.utils.data as du
import dreams.utils.io as io
import argparse
from tqdm import tqdm
import traceback
import h5py
import pandas as pd

# --- Configuration ---
EMBEDDING_DIM = 1024 # [1]

# Define the full set of columns we expect in the final HDF5 chunks
# These should match EXPECTED_COLUMNS in combine_embeddings.py
# and the columns used by the training script.
# Note: 'spectrum' is handled specially by MSData.from_pandas
# 'dreams_embedding' is added later.
CORE_MSDATA_COLUMNS = [
    'spectrum', 'precursor_mz', 'smiles', 'ionmode', 'ms_level', 'spectrum_id'
]

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DreaMS embeddings for an MGF chunk and save to HDF5 in DreaMS MSData format.")
    parser.add_argument("input_mgf_path", type=str, help="Path to the input MGF chunk file.")
    parser.add_argument("output_hdf5_path", type=str, help="Path to the output HDF5 file for this chunk (in DreaMS MSData format).")
    parser.add_argument("chunk_id", type=str, help="ID of the chunk being processed")
    args = parser.parse_args()

    INPUT_MGF_PATH = args.input_mgf_path
    OUTPUT_HDF5_PATH = args.output_hdf5_path
    CHUNK_ID = args.chunk_id

    os.makedirs(os.path.dirname(OUTPUT_HDF5_PATH), exist_ok=True)

    print(f"--- Processing MGF Chunk: {os.path.basename(INPUT_MGF_PATH)} (Chunk ID: {CHUNK_ID}) ---")

    # Define a temporary path for the MSData object's internal HDF5 file.
    # This is where MSData will store its data in its own format.
    temp_dreams_hdf5_path = OUTPUT_HDF5_PATH.replace(".hdf5", "_temp_dreams.hdf5")
    if not temp_dreams_hdf5_path.endswith(".hdf5"):
        temp_dreams_hdf5_path = f"{OUTPUT_HDF5_PATH}_temp_dreams.hdf5"

    msdata_obj = None # Initialize to None for cleanup in case of early exit

    try:
        # --- Step 1: Load MGF into Pandas DataFrame and prepare for MSData.from_pandas ---
        print(f"Reading MGF file: {os.path.basename(INPUT_MGF_PATH)} into Pandas DataFrame...")
        df_from_mgf = io.read_mgf(INPUT_MGF_PATH)

        # Standardize column names to lowercase as expected by MSData.from_pandas
        df_from_mgf.columns = [col.lower() for col in df_from_mgf.columns]

        # --- Prepare DataFrame for MSData.from_pandas ---
        # Ensure all CORE_MSDATA_COLUMNS are present with appropriate dtypes
        print("  Verifying and adding missing columns to DataFrame...")
        for col_name in CORE_MSDATA_COLUMNS:
            if col_name not in df_from_mgf.columns:
                print(f"    Column '{col_name}' not found in MGF. Adding with default values.")
                # Add missing columns with appropriate default values
                if col_name == 'ionmode':
                    df_from_mgf[col_name] = 'unknown' # Default string
                elif col_name == 'ms_level':
                    df_from_mgf[col_name] = 2 # Default integer for MS2
                elif col_name == 'smiles':
                    df_from_mgf[col_name] = '' # Default empty string
                elif col_name == 'spectrum_id':
                    # Generate a simple ID if missing, using chunk_id for better uniqueness
                    df_from_mgf[col_name] = CHUNK_ID
                elif col_name == 'spectrum': # This should almost always be present from MGF
                    print(f"    CRITICAL WARNING: 'spectrum' column missing. This MGF might be malformed.")
                    df_from_mgf[col_name] = [np.array(0, dtype=np.float32).T] * len(df_from_mgf) # Empty spectrum
                elif col_name == 'precursor_mz': # This should almost always be present from MGF
                    print(f"    CRITICAL WARNING: 'precursor_mz' column missing. This MGF might be malformed.")
                    df_from_mgf[col_name] = 0.0 # Default float
                else:
                    df_from_mgf[col_name] = np.nan # For other unexpected missing columns

        # Ensure string columns are of Python string type (object dtype in Pandas)
        for col in ['smiles', 'ionmode', 'spectrum_id']:
            if col in df_from_mgf.columns:
                df_from_mgf[col] = df_from_mgf[col].astype(str)

        print(f"DataFrame prepared with {len(df_from_mgf)} spectra.")

        # Remove existing temporary file to ensure a clean start
        if os.path.exists(temp_dreams_hdf5_path):
            os.remove(temp_dreams_hdf5_path)
            print(f"Removed existing temporary DreaMS HDF5 file: {os.path.basename(temp_dreams_hdf5_path)}")

        # Create MSData object from the prepared DataFrame (this writes to temp_dreams_hdf5_path)
        # Crucially, set mode='a' to allow adding new columns later.
        msdata_obj = du.MSData.from_pandas(
            df_from_mgf,
            hdf5_pth=temp_dreams_hdf5_path,
            in_mem=False,
            mode='a', # Open in append mode to allow adding 'dreams_embedding'
            # Explicitly map column names to ensure MSData knows their roles
            spec_col='spectrum',
            prec_mz_col='precursor_mz',
            mol_col='smiles', # Use mol_col for SMILES
        )
        print(f"MSData object created from DataFrame with {len(msdata_obj)} spectra.")
        
        if len(msdata_obj) == 0:
            print("No spectra found in this MGF chunk after loading into DreaMS MSData. Aborting.")
            raise ValueError("Empty MSData object after MGF processing.") # Raise error to trigger cleanup

        # --- Step 2: Generate DreaMS embeddings ---
        embeddings = dreams_embeddings(msdata_obj)
        print(f"Successfully computed {embeddings.shape} embeddings.")

        if embeddings.shape[0]!= len(msdata_obj): # Check first dimension of embeddings
            print(f"Error: Mismatch between number of computed embeddings ({embeddings.shape}) and spectra in MSData ({len(msdata_obj)}). Aborting.")
            raise ValueError("Embedding count mismatch.") # Raise error to trigger cleanup

        # --- Step 3: Add embeddings as a new column/dataset to the MSData object ---
        print("Adding DreaMS embeddings to MSData object...")
        msdata_obj.add_column('dreams_embedding', embeddings)
        print("DreaMS embeddings successfully added to MSData object.")

        # --- Step 4: Finalize and rename the HDF5 file ---
        print(f"Finalizing and renaming HDF5 file to {os.path.basename(OUTPUT_HDF5_PATH)}...")
        
        # Close the MSData object's internal HDF5 file handle to flush all changes
        # This is crucial for ensuring all data is written before renaming.
        msdata_obj.close() # Call the close() method (assuming it's added to dreams/utils/data.py)

        # Rename the temporary HDF5 file to the final output path
        if os.path.exists(OUTPUT_HDF5_PATH):
            os.remove(OUTPUT_HDF5_PATH) # Remove any existing final file
        os.rename(temp_dreams_hdf5_path, OUTPUT_HDF5_PATH)
        print(f"Successfully saved chunk to {os.path.basename(OUTPUT_HDF5_PATH)} in DreaMS MSData format.")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        traceback.print_exc()
        # --- AGGRESSIVE CLEANUP ON ANY ERROR ---
        print(f"Attempting aggressive cleanup of temporary/output files due to error.")
        if msdata_obj:
            try:
                msdata_obj.close() # Try to close if it's still open
            except Exception as close_e:
                print(f"Warning: Error during msdata_obj.close() in cleanup: {close_e}")
        
        if os.path.exists(temp_dreams_hdf5_path):
            try:
                os.remove(temp_dreams_hdf5_path)
                print(f"Removed problematic temporary file: {os.path.basename(temp_dreams_hdf5_path)}")
            except Exception as rm_e:
                print(f"Warning: Could not remove temporary file {os.path.basename(temp_dreams_hdf5_path)}: {rm_e}")
        
        # Only remove OUTPUT_HDF5_PATH if it's not the same as temp_dreams_hdf5_path (i.e., if rename succeeded partially)
        if os.path.exists(OUTPUT_HDF5_PATH) and OUTPUT_HDF5_PATH!= temp_dreams_hdf5_path:
            try:
                os.remove(OUTPUT_HDF5_PATH)
                print(f"Removed problematic output file: {os.path.basename(OUTPUT_HDF5_PATH)}")
            except Exception as rm_e:
                print(f"Warning: Could not remove output file {os.path.basename(OUTPUT_HDF5_PATH)}: {rm_e}")
        
        exit(1) # Exit with a non-zero status to indicate failure
    
    print("--- Processing Complete ---")