import h5py
import argparse
import os
from tqdm import tqdm
from generate_scaffold_library import get_scaffold # Re-use our existing function

def process_file(hdf5_path, scaffold_set):
    """Scans an HDF5 file and adds all unique scaffolds to a set."""
    print(f"Processing file: {hdf5_path}...")
    with h5py.File(hdf5_path, 'r') as f:
        smiles_list = f['smiles']
        for i in tqdm(range(len(smiles_list)), desc=f"Scanning {os.path.basename(hdf5_path)}"):
            try:
                smi = smiles_list[i].decode('utf-8')
                scaffold = get_scaffold(smi)
                if scaffold: # Ensure the scaffold is not empty
                    scaffold_set.add(scaffold)
            except Exception:
                # Ignore SMILES that might cause errors in RDKit
                continue

def main(args):
    """
    Generates a single master scaffold library from both training and validation sets.
    """
    print("--- Creating a Unified Master Scaffold Library ---")
    
    # Using a set automatically handles uniqueness
    master_scaffold_set = set()

    # Process both files
    process_file(args.train_path, master_scaffold_set)
    process_file(args.val_path, master_scaffold_set)

    # Convert set to a sorted list for deterministic order
    sorted_scaffolds = sorted(list(master_scaffold_set))

    # Write the final master library to the output file
    with open(args.output_path, 'w') as f:
        for scaffold in sorted_scaffolds:
            f.write(f"{scaffold}\n")

    print(f"\nSUCCESS: Created master library with {len(sorted_scaffolds)} unique scaffolds.")
    print(f"Master library saved to: {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a master scaffold library from train and val sets.")
    parser.add_argument("--train_path", required=True, help="Path to the training HDF5 file.")
    parser.add_argument("--val_path", required=True, help="Path to the validation HDF5 file.")
    parser.add_argument("--output_path", default="./master_scaffold_library.txt", help="Path for the output master library file.")
    args = parser.parse_args()
    main(args)