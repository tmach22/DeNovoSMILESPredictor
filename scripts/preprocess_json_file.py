import json
import numpy as np
from matchms import Spectrum
from matchms.exporting import save_as_mgf
import argparse
import sys
from tqdm import tqdm

def preprocess_and_save_to_mgf(json_path, mgf_path):
    """
    Loads spectra from a JSON file, applies pre-processing filters, and saves
    the valid spectra to a DreaMS-compatible MGF file.

    Preprocessing steps:
    1. Normalizes peak intensities to the range.
    2. Removes noise peaks with intensity < 0.1% of the base peak.
    3. If a spectrum has > 100 peaks, it's trimmed to the 100 most intense peaks.
    4. Filters out spectra with fewer than 5 peaks.

    Args:
        json_path (str): Path to the input JSON file.
        mgf_path (str): Path for the output MGF file.
    """
    print(f"Loading spectra from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            spectra_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(spectra_data)} spectra. Starting preprocessing...")
    
    processed_spectra = []
    
    for spec_info in tqdm(spectra_data, desc="Processing Spectra"):
        peaks_data = spec_info.get('peaks',)
        
        # --- Filter 1: Ensure peak data is valid and has a minimum number of peaks ---
        if not peaks_data or not isinstance(peaks_data, list) or len(peaks_data) <= 5:
            continue
            
        peaks = np.array(peaks_data, dtype=np.float64)

        # --- Preprocessing Step 1: Normalize Intensities ---
        max_intensity = np.max(peaks[:, 1])
        if max_intensity == 0:
            continue # Skip spectra with no intensity
        
        processed_peaks = peaks.copy()
        processed_peaks[:, 1] /= max_intensity

        # --- Preprocessing Step 2: Remove Noise Peaks ---
        # Remove peaks with normalized intensity < 0.001 (0.1%)
        processed_peaks = processed_peaks[processed_peaks[:, 1] >= 0.001]

        # --- Preprocessing Step 3: Enforce Peak Count Limits ---
        # If more than 100 peaks, select the top 100 by intensity
        if len(processed_peaks) > 100:
            # Sort by intensity (column 1) in descending order
            sorted_by_intensity = processed_peaks[processed_peaks[:, 1].argsort()[::-1]]
            # Select the top 100
            processed_peaks = sorted_by_intensity[:100, :]

        # --- Filter 2: After processing, ensure we still have enough peaks ---
        if len(processed_peaks) <= 5:
            continue

        # --- Prepare for MGF export ---
        # Sort final peaks by m/z, which is required by matchms
        processed_peaks = processed_peaks[processed_peaks[:, 0].argsort()]
        
        # Map JSON keys to MGF-compatible metadata keys
        metadata = {
            'PEPMASS': spec_info.get('precursor_mz'),
            'SMILES': spec_info.get('SMILES'),
            'ADDUCT': spec_info.get('adduct'),
            'FEATURE_ID': spec_info.get('identifier'),
            'SCANS': str(spec_info.get('scans', -1)),
            'Collision energy': spec_info.get('energy'),
            'SPLIT': str(spec_info.get('split'))
        }
        
        # Remove None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}

        spectrum_obj = Spectrum(
            mz=processed_peaks[:, 0],
            intensities=processed_peaks[:, 1],
            metadata=metadata
        )
        processed_spectra.append(spectrum_obj)

    if not processed_spectra:
        print("No valid spectra remained after preprocessing. No MGF file was created.")
        return

    print(f"Preprocessing complete. {len(processed_spectra)} spectra passed all filters.")
    
    # --- Save the processed spectra to an MGF file ---
    print(f"Saving processed spectra to {mgf_path}...")
    save_as_mgf(processed_spectra, mgf_path)
    print("MGF file created successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess spectra from a JSON file and save to a DreaMS-compatible MGF file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input_json",
        required=True,
        help="Path to the input JSON file containing the spectra."
    )
    parser.add_argument(
        "--output_mgf",
        required=True,
        help="Path for the output MGF file that will be created."
    )

    args = parser.parse_args()
    preprocess_and_save_to_mgf(args.input_json, args.output_mgf)