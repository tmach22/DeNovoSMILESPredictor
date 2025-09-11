import pandas as pd
import numpy as np
import ast
import os
import glob
import argparse
from tqdm import tqdm
from matchms import Spectrum
from matchms.exporting import save_as_mgf

def process_spectrum(spectrum_str, min_peaks=5, max_peaks=100):
    """
    Parses, filters, and normalizes a single mass spectrum.
    
    Args:
        spectrum_str (str): The string representation of the spectrum, e.g., "[(mz, int),...]".
        min_peaks (int): Minimum number of peaks required.
        max_peaks (int): Maximum number of peaks to keep.

    Returns:
        np.ndarray: A processed 2D NumPy array of [m/z, intensity] or None if filtered out.
    """
    try:
        # Safely parse the spectrum string from the CSV using ast.literal_eval
        spec_list = ast.literal_eval(spectrum_str)
        if not isinstance(spec_list, list) or not spec_list:
            return None
        
        spectrum = np.array(spec_list, dtype=np.float32)
    except (SyntaxError, TypeError, ValueError):
        return None

    # 1. Filter by minimum number of peaks
    if len(spectrum) < min_peaks:
        return None

    # 2. Filter by maximum number of peaks (keep most intense)
    if len(spectrum) > max_peaks:
        spectrum = spectrum[spectrum[:, 1].argsort()[::-1]][:max_peaks]

    # 3. Normalize intensities
    max_intensity = np.max(spectrum[:, 1])
    if max_intensity > 0:
        spectrum[:, 1] /= max_intensity
    else:
        # Avoid division by zero if all intensities are 0 for some reason
        return None
        
    # 4. Sort by m/z (column 0) in ascending order (standard format)
    spectrum = spectrum[spectrum[:, 0].argsort()]
    
    return spectrum

def main(input_dir, output_dir):
    """
    Processes all chunked spectra files into individual MGF files
    by creating Spectrum objects first to ensure correct formatting.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all the final chunk files produced by the pipeline
    chunk_files = glob.glob(os.path.join(input_dir, 'chunk_*.csv'))
    
    if not chunk_files:
        print(f"Error: No processed chunk files found in '{input_dir}'.")
        print("Please ensure the master pipeline has completed and produced files.")
        return

    print(f"Found {len(chunk_files)} chunk files to process in '{input_dir}'.")

    # Process each chunk file individually
    for chunk_path in tqdm(chunk_files, desc="Processing Chunks"):
        df = pd.read_csv(chunk_path)
        
        # Apply the processing function to each spectrum string
        df['processed_peaks'] = df['peaks'].apply(process_spectrum)

        # Remove rows where the spectrum was filtered out
        df.dropna(subset=['processed_peaks'], inplace=True)
        
        if df.empty:
            print(f"Skipping {os.path.basename(chunk_path)}: No valid spectra after filtering.")
            continue

        # --- Create a list of matchms.Spectrum objects ---
        spectra_objects = []
        for _, row in df.iterrows():
            # Create a metadata dictionary. Keys are standardized by matchms.
            metadata = {
                'title': row['mol_id'],
                'precursor_mz': row['precursor_mz'],
                'smiles': row['smiles'],
                'adduct': row['prec_type']
            }
            
            # Create the Spectrum object
            spectrum = Spectrum(
                mz=row['processed_peaks'][:, 0],
                intensities=row['processed_peaks'][:, 1],
                metadata=metadata
            )
            spectra_objects.append(spectrum)

        # --- Write the list of Spectrum objects to an MGF file ---
        # Create a corresponding.mgf filename
        base_name = os.path.basename(chunk_path)
        # FIX: os.path.splitext returns a tuple (root, ext). We only need the root.
        mgf_name = os.path.splitext(base_name)[0] + '.mgf'
        output_path = os.path.join(output_dir, mgf_name)
        
        save_as_mgf(spectra_objects, output_path)
    
    print(f"\nDone. All chunks have been processed and saved as individual.mgf files in '{output_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Consolidate and preprocess MassFormer spectra chunks into individual MGF files using matchms.")
    parser.add_argument(
        "-i", "--input_dir", 
        default="final_data_chunks",
        help="Directory containing the final, processed CSV chunks from the pipeline."
    )
    parser.add_argument(
        "-o", "--output_dir", 
        default="preprocessed_mgf_chunks",
        help="Directory to save the output.mgf files (one for each input chunk)."
    )
    args = parser.parse_args()
    
    # You may need to install matchms: pip install matchms
    main(args.input_dir, args.output_dir)