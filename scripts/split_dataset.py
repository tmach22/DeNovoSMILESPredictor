import h5py
import os
import argparse
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import dreams.utils.data as du # Import for MSData operations and splitting
from multiprocessing import Pool, cpu_count # Import for parallelization
import traceback # For detailed error logging
from rdkit.Chem.AllChem import MolFromSmiles, MolToSmiles

# --- Configuration ---
DREAMS_EMBEDDING_DIM = 1024 # Consistent with embedding generation script

# --- Utility Functions ---
def canonicalize_smiles(smiles_list):
    """Canonicalizes a SMILES string using RDKit."""
    try:
        # 1. Convert SMILES string to an RDKit molecule object
        mol = MolFromSmiles(smiles_list)
        
        # 2. Check if the conversion was successful
        if mol:
            # 3. Convert the RDKit molecule object back to a canonical SMILES string
            canonical_smiles = MolToSmiles(mol, canonical=True)
            return canonical_smiles
        return None # Return None if mol object is invalid (e.g., malformed SMILES)
    except Exception:
        # 4. Handle any exceptions during the process (e.g., severely malformed SMILES)
        return None # Return None if an error occurs

def compute_murcko_scaffold_single(smiles: str) -> str:
    """
    Computes the Murcko scaffold SMILES for a given SMILES string using RDKit.
    Returns an empty string if RDKit fails or no scaffold is found.
    This function is designed to be run in a multiprocessing pool.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, isomericSmiles=True, canonical=True)
    except Exception as e:
        print(f"Exception during computing murcko scaffold...{e}")
        pass
    return ""

def main():
    parser = argparse.ArgumentParser(description="Perform Murcko Histogram Disjoint Splitting on HDF5 dataset.")
    parser.add_argument("--input_hdf5_path", type=str, required=True,
                        help="Path to the input HDF5 dataset (in DreaMS MSData format).")
    parser.add_argument("--output_hdf5_path", type=str, required=True,
                        help="Path to the output HDF5 file with 'fold' information (in DreaMS MSData format).")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Proportion of data for the training set.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Proportion of data for the validation set.")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Proportion of data for the test set.")
    parser.add_argument("--smiles_col_name", type=str, default="smiles", # Column name for SMILES
                        help="Name of the column/dataset containing SMILES strings.")
    parser.add_argument("--spec_id_col_name", type=str, default="spectrum_id", # Column name for spectrum ID
                        help="Name of the column/dataset containing unique spectrum IDs.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of CPU workers for parallel scaffold computation. Defaults to all available CPUs.")
    args = parser.parse_args()

    # Normalize ratios to sum to 1.0
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        print(f"Warning: Train, validation, and test ratios ({total_ratio:.2f}) do not sum to 1.0. Adjusting.")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
        print(f"Adjusted ratios: Train={args.train_ratio:.2f}, Val={args.val_ratio:.2f}, Test={args.test_ratio:.2f}")

    np.random.seed(args.seed)
    
    os.makedirs(os.path.dirname(args.output_hdf5_path), exist_ok=True)

    print(f"--- Performing Murcko Histogram Disjoint Splitting on {os.path.basename(args.input_hdf5_path)} ---")

    # Load the dataset using dreams.utils.data.MSData
    msdata_obj = None
    try:
        msdata_obj = du.MSData.load(args.input_hdf5_path, mode='a')
        print(f"Loaded {len(msdata_obj)} spectra from {os.path.basename(args.input_hdf5_path)}.")
    except Exception as e:
        print(f"Error loading MSData from {args.input_hdf5_path}: {e}")
        traceback.print_exc()
        exit()

    # --- Collect SMILES and their corresponding spectrum IDs ---
    # Access SMILES and spectrum_id columns directly from the MSData object
    # Use [()] to read the entire dataset into memory as a NumPy array
    try:
        all_smiles_raw = msdata_obj[args.smiles_col_name]
        all_spec_ids_raw = msdata_obj[args.spec_id_col_name]
    except KeyError as e:
        print(f"Error: Required column '{e}' not found in the HDF5 dataset. Please check --smiles_col_name and --spec_id_col_name.")
        msdata_obj.close()
        exit()
    except Exception as e:
        print(f"Error accessing data from MSData object: {e}")
        traceback.print_exc()
        msdata_obj.close()
        exit()

    # Filter out invalid SMILES and corresponding IDs
    # Ensure SMILES are decoded if stored as bytes (common in HDF5 for strings)
    all_smiles_decoded = [s.decode('utf-8') if isinstance(s, bytes) else s for s in all_smiles_raw]
    all_spec_ids_decoded = [s.decode('utf-8') if isinstance(s, bytes) else s for s in all_spec_ids_raw]

    # Filter for valid SMILES and canonicalize them
    valid_indices = [i for i, s in enumerate(all_smiles_decoded) if s and canonicalize_smiles(s)]
    filtered_smiles = [canonicalize_smiles(all_smiles_decoded[i]) for i in valid_indices]
    filtered_spec_ids = [all_spec_ids_decoded[i] for i in valid_indices]

    if not filtered_smiles:
        print("No valid SMILES found in the dataset for splitting. Aborting.")
        msdata_obj.close()
        exit()

    print(f"Found {len(filtered_smiles)} valid SMILES for Murcko splitting.")

    # --- Compute Murcko Scaffolds in Parallel ---
    print("Computing Murcko scaffolds in parallel...")
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print(f"Using {num_workers} CPU workers.")
    
    with Pool(processes=num_workers) as pool:
        scaffolds = list(tqdm(pool.imap_unordered(compute_murcko_scaffold_single, filtered_smiles),
                              total=len(filtered_smiles),
                              desc="Computing Murcko scaffolds"))
    
    # Group spectrum IDs by scaffold
    scaffold_to_spec_ids = {}
    for i, scaffold in enumerate(scaffolds):
        if scaffold: # Only keep if scaffold was successfully computed
            if scaffold not in scaffold_to_spec_ids:
                scaffold_to_spec_ids[scaffold] = [] # Initialize an empty list
            scaffold_to_spec_ids[scaffold].append(filtered_spec_ids[i])

    if not scaffold_to_spec_ids:
        print("No valid scaffolds found after computation. Aborting.")
        msdata_obj.close()
        exit()
    
    unique_scaffolds = list(scaffold_to_spec_ids.keys())
    np.random.shuffle(unique_scaffolds) # Shuffle scaffolds for random assignment

    # --- Assign scaffolds to train/val/test sets (Improved Logic) ---
    total_unique_scaffolds = len(unique_scaffolds)
    num_train_scaffolds = int(total_unique_scaffolds * args.train_ratio)
    num_val_scaffolds = int(total_unique_scaffolds * args.val_ratio)
    # The rest go to test to ensure all are assigned
    num_test_scaffolds = total_unique_scaffolds - num_train_scaffolds - num_val_scaffolds

    # Assign based on shuffled order
    train_scaffolds = unique_scaffolds[:num_train_scaffolds]
    val_scaffolds = unique_scaffolds[num_train_scaffolds : num_train_scaffolds + num_val_scaffolds]
    test_scaffolds = unique_scaffolds[num_train_scaffolds + num_val_scaffolds :]

    # Calculate actual counts of spectra in each split for printing
    current_train_count = sum(len(scaffold_to_spec_ids[s]) for s in train_scaffolds)
    current_val_count = sum(len(scaffold_to_spec_ids[s]) for s in val_scaffolds)
    current_test_count = sum(len(scaffold_to_spec_ids[s]) for s in test_scaffolds)

    print(f"Split results (spectra counts): Train={current_train_count}, Val={current_val_count}, Test={current_test_count}")
    print(f"Split results (scaffold counts): Train={len(train_scaffolds)}, Val={len(val_scaffolds)}, Test={len(test_scaffolds)}")
    
    # Create a mapping from spectrum ID to fold type for all original spectra
    # Initialize all to 'unassigned' first
    fold_data_map = {sid: 'unassigned' for sid in all_spec_ids_decoded}
    for scaffold in train_scaffolds:
        for spec_id in scaffold_to_spec_ids[scaffold]:
            fold_data_map[spec_id] = 'train'
    for scaffold in val_scaffolds:
        for spec_id in scaffold_to_spec_ids[scaffold]:
            fold_data_map[spec_id] = 'val'
    for scaffold in test_scaffolds:
        for spec_id in scaffold_to_spec_ids[scaffold]:
            fold_data_map[spec_id] = 'test'

    # --- Add the new 'fold' column to the MSData object and save ---
    print(f"Adding 'fold' column to MSData object and saving to {os.path.basename(args.output_hdf5_path)}...")
    
    # Create a list of fold assignments in the original order of msdata_obj
    # This ensures the new 'fold' column aligns correctly with existing data.
    fold_column_data = [fold_data_map.get(spec_id, 'unassigned') for spec_id in all_spec_ids_decoded]
    
    # Assign the new column to the MSData object
    # MSData handles saving this new column as a dataset in its HDF5 file
    # Store as byte string ('S') for HDF5 compatibility with variable-length strings
    msdata_obj.add_column('fold', np.array(fold_column_data, dtype='S'), remove_old_if_exists=True)

    # Save the modified MSData object to a new HDF5 file
    # This will create a new HDF5 file with all original data plus the new 'fold' column
    try:
        if os.path.exists(args.output_hdf5_path):
            os.remove(args.output_hdf5_path)
            print(f"Removed existing output HDF5 file: {os.path.basename(args.output_hdf5_path)}")

        msdata_obj.close()
        print(f"Dataset with Murcko splits saved to: {os.path.basename(args.output_hdf5_path)}")
    except Exception as e:
        print(f"Error saving MSData object with fold information: {e}")
        traceback.print_exc()
    finally:
        # Ensure the MSData object's file handle is closed in all cases
        if msdata_obj:
            msdata_obj.close()

    print("--- Splitting Complete ---")

if __name__ == "__main__":
    main()