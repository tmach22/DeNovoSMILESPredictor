import h5py
import numpy as np
import argparse
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def main(scaffold_library_path, output_hdf5_path):
    """
    Pre-computes the full Tanimoto similarity matrix for a given scaffold library
    and saves it to an HDF5 file for efficient slicing during training.
    """
    print("--- Step 1: Loading and validating scaffold library ---")
    with open(scaffold_library_path, 'r') as f:
        raw_library = [line.strip() for line in f if line.strip()]

    valid_scaffolds = []
    valid_mols = []
    for smi in tqdm(raw_library, desc="Validating Scaffolds"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_scaffolds.append(smi)
            valid_mols.append(mol)
    
    n_scaffolds = len(valid_scaffolds)
    print(f"Found {n_scaffolds} valid scaffolds.")

    print("\n--- Step 2: Generating Morgan fingerprints ---")
    scaffold_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in tqdm(valid_mols, desc="Generating Fingerprints")]

    print(f"\n--- Step 3: Creating HDF5 file and dataset at {output_hdf5_path} ---")
    # Create the HDF5 file and the dataset with the final shape.
    # We will fill it row by row.
    with h5py.File(output_hdf5_path, 'w') as f:
        sim_matrix_dset = f.create_dataset(
            'similarity_matrix', 
            shape=(n_scaffolds, n_scaffolds), 
            dtype='float32', 
            chunks=(1, n_scaffolds) # Chunking by row for efficient reading
        )

        print("\n--- Step 4: Calculating and saving similarity matrix (this will take time) ---")
        # Calculate similarities row by row to keep memory usage low
        for i in tqdm(range(n_scaffolds), desc="Calculating Matrix Row-by-Row"):
            query_fp = scaffold_fps[i]
            # Use BulkTanimotoSimilarity for efficient calculation of one row
            sim_vector = DataStructs.BulkTanimotoSimilarity(query_fp, scaffold_fps)
            # Save the calculated row directly into the HDF5 file
            sim_matrix_dset[i, :] = sim_vector

    print(f"\nSUCCESS: Similarity matrix saved to {output_hdf5_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-compute Tanimoto similarity matrix for the scaffold library.")
    parser.add_argument("--scaffold_library_path", type=str, required=True, help="Path to the scaffold_library.txt file.")
    parser.add_argument("--output_hdf5_path", type=str, default="./similarity_matrix.hdf5", help="Path to save the output HDF5 matrix file.")
    args = parser.parse_args()
    main(args.scaffold_library_path, args.output_hdf5_path)