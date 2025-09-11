import os
import glob
import argparse
import h5py
import numpy as np
from tqdm import tqdm
from dreams.api import dreams_embeddings
from matchms.importing import load_from_mgf

def create_or_open_hdf5(output_path):
    """
    Creates an HDF5 file with resizable datasets if it doesn't exist.
    Opens it in append mode if it does.
    """
    # Open in append mode, which creates the file if it doesn't exist
    with h5py.File(output_path, 'a') as hf:
        # Check if datasets need to be created
        if 'embeddings' not in hf:
            # DreaMS embeddings are 1024-dimensional
            hf.create_dataset('embeddings', (0, 1024), maxshape=(None, 1024), dtype=np.float32, chunks=True)
            
            # Use special string dtype for variable-length strings
            string_dt = h5py.string_dtype(encoding='utf-8')
            hf.create_dataset('smiles', (0,), maxshape=(None,), dtype=string_dt, chunks=True)
            hf.create_dataset('adduct', (0,), maxshape=(None,), dtype=string_dt, chunks=True)
            
            hf.create_dataset('precursor_mz', (0,), maxshape=(None,), dtype=np.float32, chunks=True)
            
            # Use variable-length dtypes for ragged peak arrays
            vlen_float_dt = h5py.vlen_dtype(np.float32)
            hf.create_dataset('peaks_mz', (0,), maxshape=(None,), dtype=vlen_float_dt, chunks=True)
            hf.create_dataset('peaks_intensities', (0,), maxshape=(None,), dtype=vlen_float_dt, chunks=True)
            
            print(f"Created new HDF5 file at {output_path}")

def process_chunks_to_hdf5(mgf_dir, output_path):
    """
    Processes all.mgf files, computes embeddings, and appends to a single HDF5 file.

    Args:
        mgf_dir (str): Path to the directory containing preprocessed.mgf files.
        output_path (str): Path for the output consolidated.hdf5 file.
    """
    # Ensure the HDF5 file and datasets are initialized
    create_or_open_hdf5(output_path)
    
    mgf_files = sorted(glob.glob(os.path.join(mgf_dir, '*.mgf')))

    if not mgf_files:
        print(f"Error: No.mgf files found in '{mgf_dir}'.")
        return

    print(f"Found {len(mgf_files)} .mgf files to process.")

    with h5py.File(output_path, 'a') as hf:
        for mgf_path in tqdm(mgf_files, desc="Processing Chunks into HDF5"):
            try:
                # 1. Load spectra and metadata from the.mgf file
                spectra = list(load_from_mgf(mgf_path))
                if not spectra:
                    print(f"Warning: No spectra found in {os.path.basename(mgf_path)}. Skipping.")
                    continue

                # 2. Compute DreaMS embeddings for the entire file
                embeddings = dreams_embeddings(mgf_path)

                # 3. Sanity check: Ensure order and count are preserved
                if len(spectra)!= embeddings.shape[0]:
                    print(f"CRITICAL ERROR: Mismatch in spectrum count for {os.path.basename(mgf_path)}.")
                    print(f"Spectra found: {len(spectra)}, Embeddings generated: {embeddings.shape}")
                    continue

                # 4. Prepare data for writing
                num_new_spectra = len(spectra)
                smiles_list = [s.get('smiles', '') for s in spectra]
                adduct_list = [s.get('adduct', '') for s in spectra]
                mz_list = [s.get('precursor_mz', 0.0) for s in spectra]
                peaks_mz_list = [s.peaks.mz.astype(np.float32) for s in spectra]
                peaks_intensities_list = [s.peaks.intensities.astype(np.float32) for s in spectra]

                # 5. Append data to the HDF5 file
                # Get current size and resize datasets
                current_size = hf['embeddings'].shape
                new_size = current_size[0] + num_new_spectra
                
                hf['embeddings'].resize(new_size, axis=0)
                hf['smiles'].resize(new_size, axis=0)
                hf['adduct'].resize(new_size, axis=0)
                hf['precursor_mz'].resize(new_size, axis=0)
                hf['peaks_mz'].resize(new_size, axis=0)
                hf['peaks_intensities'].resize(new_size, axis=0)

                # Write the new data into the newly allocated space
                hf['embeddings'][current_size:] = embeddings
                hf['smiles'][current_size:] = smiles_list
                hf['adduct'][current_size:] = adduct_list
                hf['precursor_mz'][current_size:] = mz_list
                hf['peaks_mz'][current_size:] = peaks_mz_list
                hf['peaks_intensities'][current_size:] = peaks_intensities_list

            except Exception as e:
                print(f"\nAn error occurred while processing {os.path.basename(mgf_path)}: {e}")

    with h5py.File(output_path, 'r') as hf:
        total_spectra = hf['embeddings'].shape
        print(f"\nProcessing complete. Total spectra in HDF5 file: {total_spectra}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate DreaMS embeddings and consolidate into a single HDF5 file.")
    parser.add_argument(
        "-i", "--input_dir",
        default="preprocessed_mgf_chunks",
        help="Directory containing the preprocessed.mgf chunk files."
    )
    parser.add_argument(
        "-o", "--output_file",
        default="dreams_dataset.hdf5",
        help="Path for the final, consolidated HDF5 dataset."
    )
    args = parser.parse_args()

    # Ensure you have activated the 'dreams' conda environment before running
    process_chunks_to_hdf5(args.input_dir, args.output_file)