import h5py
import argparse
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

def get_scaffold(smiles_string: str) -> str:
    """
    Computes the Murcko scaffold from a SMILES string.

    The Murcko scaffold is the core ring system of a molecule, representing its
    fundamental topology. This is a standard and robust method in cheminformatics.

    Args:
        smiles_string: The input SMILES string of a molecule.

    Returns:
        The SMILES string of the Murcko scaffold, or an empty string if the
        input SMILES is invalid or scaffold extraction fails.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return ""  # Handle invalid SMILES input
    try:
        # GetScaffoldForMol returns a molecule object representing the scaffold
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold_mol)
    except Exception as e:
        # Catch rare RDKit errors during scaffold generation
        print(f"Warning: Could not generate scaffold for SMILES '{smiles_string}'. Error: {e}")
        return ""

def main(args):
    """
    Main function to read SMILES from an HDF5 file, compute all unique
    Murcko scaffolds, and save them to a text file.
    """
    print(f"--- Step 1: Reading SMILES from '{args.hdf5_filepath}' ---")
    
    all_scaffolds = []
    with h5py.File(args.hdf5_filepath, 'r') as f:
        # Ensure the 'smiles' dataset exists
        if 'smiles' not in f:
            print(f"Error: 'smiles' dataset not found in {args.hdf5_filepath}")
            return
            
        smiles_dataset = f['smiles']
        total_smiles = len(smiles_dataset)
        print(f"Found {total_smiles} total SMILES strings to process.")

        # Use tqdm for a progress bar as this can be a long process
        for smiles_bytes in tqdm(smiles_dataset, desc="Generating Scaffolds"):
            smiles_str = smiles_bytes.decode('utf-8')
            scaffold = get_scaffold(smiles_str)
            if scaffold:  # Only add non-empty scaffolds
                all_scaffolds.append(scaffold)

    print("\n--- Step 2: Identifying Unique Scaffolds ---")
    # Using a set is the most efficient way to find unique items
    unique_scaffolds = sorted(list(set(all_scaffolds)))
    
    print(f"Processed {total_smiles} molecules.")
    print(f"Found {len(unique_scaffolds)} unique scaffolds.")

    print(f"\n--- Step 3: Saving Unique Scaffold Library to '{args.output_file}' ---")
    with open(args.output_file, 'w') as f:
        for scaffold in unique_scaffolds:
            f.write(scaffold + '\n')
            
    print("Scaffold library created successfully.")
    print(f"\nExample scaffolds from the library:")
    for i, scaffold in enumerate(unique_scaffolds[:5]):
        print(f"  {i+1}: {scaffold}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a unique Murcko scaffold library from the SMILES in a training HDF5 file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--hdf5_filepath",
        type=str,
        required=True,
        help="Path to the input HDF5 file containing the training data (e.g., train.hdf5)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path for the output text file where the unique scaffold library will be saved (e.g., scaffold_library.txt)."
    )
    args = parser.parse_args()
    print(f"Arguments: {args}")
    main(args)