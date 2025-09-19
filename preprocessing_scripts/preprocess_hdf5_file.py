import h5py
import numpy as np
import torch
import argparse
from tqdm import tqdm
import sys
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# --- Peak Processing Logic (from before) ---
def process_peaks_batch(mz_batch, intensity_batch, max_peaks=100, max_mz=500.0):
    batch_peak_bins = []
    for mz_array, intensity_array in zip(mz_batch, intensity_batch):
        if mz_array is None or len(mz_array) == 0 or not isinstance(mz_array, np.ndarray):
            batch_peak_bins.append(torch.zeros(max_peaks, dtype=torch.long))
            continue
        peaks = np.array([p for p in zip(mz_array, intensity_array) if p[0] <= max_mz])
        if len(peaks) == 0:
            batch_peak_bins.append(torch.zeros(max_peaks, dtype=torch.long))
            continue
        peaks = peaks[peaks[:, 1].argsort()][::-1][:max_peaks]
        peaks = peaks[peaks[:, 0].argsort()]
        mz_values = peaks[:, 0]
        padded_mz = np.zeros(max_peaks)
        padded_mz[:len(mz_values)] = mz_values
        peak_bins = np.round(padded_mz * 100).astype(int)
        batch_peak_bins.append(torch.tensor(peak_bins, dtype=torch.long))
    return torch.stack(batch_peak_bins, dim=0)

# --- NEW: Formula and Scaffold Processing Logic ---
def get_formula_from_smiles(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return ""
    return rdMolDescriptors.CalcMolFormula(mol)

def get_murcko_scaffold(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        return ""
    except: return ""

def main(args):
    file_path = args.file_path
    batch_size = args.batch_size

    try:
        with h5py.File(file_path, 'r+') as f:
            print(f"--- Processing file: {file_path} ---")
            
            datasets_to_create = ['peak_bins', 'formula_strings', 'scaffold_smiles']
            for dset_name in datasets_to_create:
                if dset_name in f:
                    del f[dset_name]
                    print(f"Existing '{dset_name}' dataset removed.")

            total_records = f['embeddings'].shape
            print(f"Found {total_records} records to process.")

            # Create new datasets
            if 'peak_bins' in datasets_to_create:
                f.create_dataset('peak_bins', shape=(total_records[0], 100), dtype=np.int64)
            if 'formula_strings' in datasets_to_create:
                f.create_dataset('formula_strings', shape=(total_records[0],), dtype=h5py.string_dtype(encoding='utf-8'))
            if 'scaffold_smiles' in datasets_to_create:
                f.create_dataset('scaffold_smiles', shape=(total_records[0],), dtype=h5py.string_dtype(encoding='utf-8'))

            print(f"Processing in batches of {batch_size}...")
            num_batches = int(np.ceil(total_records[0] / batch_size))

            print(f"Number of batches: {num_batches}")

            for i in tqdm(range(num_batches), desc="Preprocessing Data"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_records[0])
                
                # Process Peaks
                if 'peak_bins' in datasets_to_create:
                    mz_batch = f['peaks_mz'][start_idx:end_idx]
                    intensity_batch = f['peaks_intensities'][start_idx:end_idx]
                    processed_peaks = process_peaks_batch(mz_batch, intensity_batch)
                    f['peak_bins'][start_idx:end_idx] = processed_peaks.numpy()

                # Process Formulas and Scaffolds
                if 'formula_strings' in datasets_to_create or 'scaffold_smiles' in datasets_to_create:
                    smiles_batch = [s.decode('utf-8') for s in f['smiles'][start_idx:end_idx]]
                    if 'formula_strings' in datasets_to_create:
                        formulas = [get_formula_from_smiles(s) for s in smiles_batch]
                        f['formula_strings'][start_idx:end_idx] = formulas
                    if 'scaffold_smiles' in datasets_to_create:
                        scaffolds = [get_murcko_scaffold(s) for s in smiles_batch]
                        f['scaffold_smiles'][start_idx:end_idx] = scaffolds

        print("\n--- Preprocessing Complete ---")
        print(f"Successfully added pre-computed datasets to {file_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess peaks, formulas, and scaffolds in an HDF5 file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--file_path', required=True, help="Path to the HDF5 file to process.")
    parser.add_argument('--batch_size', type=int, default=10000, help="Batch size for processing.")
    args = parser.parse_args()
    main(args)