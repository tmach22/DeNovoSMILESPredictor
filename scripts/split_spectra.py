import h5py
import numpy as np
import argparse
import os
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

def get_scaffold(smiles_string):
    """
    Computes the Murcko scaffold for a given SMILES string.
    
    Args:
        smiles_string (str): A SMILES representation of a molecule.

    Returns:
        str: The SMILES representation of the molecule's scaffold.
             Returns the original SMILES if a scaffold cannot be generated.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
    except:
        # Return original SMILES on failure to process
        return smiles_string
    return smiles_string

def create_scaffold_split(hdf5_path, output_dir, split_ratios=(0.8, 0.1, 0.1)):
    """
    Splits an HDF5 dataset into train, validation, and test sets based on molecular scaffolds.

    Args:
        hdf5_path (str): Path to the input consolidated HDF5 file.
        output_dir (str): Directory to save the split HDF5 files.
        split_ratios (tuple): A tuple containing the ratios for (train, val, test).
    """
    if not os.path.exists(hdf5_path):
        print(f"Error: Input file not found at {hdf5_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print("Reading SMILES and grouping by scaffold...")
    
    # --- Step 1: Group indices by scaffold ---
    scaffold_to_indices = defaultdict(list)
    with h5py.File(hdf5_path, 'r') as hf:
        total_spectra = len(hf['smiles'])
        # Read SMILES in chunks for memory efficiency if needed, though for a small file this is fine.
        all_smiles = [s.decode('utf-8') for s in hf['smiles'][:]]

        for i, smiles in tqdm(enumerate(all_smiles), total=total_spectra, desc="Calculating Scaffolds"):
            scaffold = get_scaffold(smiles)
            scaffold_to_indices[scaffold].append(i)

    # --- Step 2: Split scaffolds into train, val, and test sets ---
    print("\nSplitting scaffolds into train, validation, and test sets...")
    scaffolds = list(scaffold_to_indices.keys())
    np.random.shuffle(scaffolds) # Randomize the order of scaffolds

    train_cutoff = int(len(scaffolds) * split_ratios[0])
    val_cutoff = int(len(scaffolds) * (split_ratios[0] + split_ratios[1]))

    train_scaffolds = scaffolds[:train_cutoff]
    val_scaffolds = scaffolds[train_cutoff:val_cutoff]
    test_scaffolds = scaffolds[val_cutoff:]

    # Collect all indices for each set
    train_indices = [i for scaffold in train_scaffolds for i in scaffold_to_indices[scaffold]]
    val_indices = [i for scaffold in val_scaffolds for i in scaffold_to_indices[scaffold]]
    test_indices = [i for scaffold in test_scaffolds for i in scaffold_to_indices[scaffold]]

    # Sort indices to optimize HDF5 reading (sequential access is faster)
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    print(f"\nTotal spectra: {total_spectra}")
    print(f"Train set: {len(train_indices)} spectra ({len(train_scaffolds)} scaffolds)")
    print(f"Validation set: {len(val_indices)} spectra ({len(val_scaffolds)} scaffolds)")
    print(f"Test set: {len(test_indices)} spectra ({len(test_scaffolds)} scaffolds)")

    # --- Step 3: Write the splits to new HDF5 files ---
    split_data = {
        'train_set.hdf5': train_indices,
        'val_set.hdf5': val_indices,
        'test_set.hdf5': test_indices
    }

    with h5py.File(hdf5_path, 'r') as hf_in:
        # Get dataset names and shapes from the source file
        dataset_info = {name: {'shape': hf_in[name].shape, 'dtype': hf_in[name].dtype} for name in hf_in.keys()}

        for filename, indices in split_data.items():
            if not indices:
                print(f"Skipping {filename} as it has no data.")
                continue
                
            output_path = os.path.join(output_dir, filename)
            print(f"\nWriting {len(indices)} records to {output_path}...")
            
            with h5py.File(output_path, 'w') as hf_out:
                # Create datasets in the new file
                out_datasets = {}
                for name, info in dataset_info.items():
                    # For resizable datasets, we need to handle shape differently
                    if len(info['shape']) > 1:
                        maxshape = (None,) + info['shape'][1:]
                    else:
                        maxshape = (None,)
                    
                    out_datasets[name] = hf_out.create_dataset(
                        name, 
                        shape=(0,) + info['shape'][1:], 
                        maxshape=maxshape, 
                        dtype=info['dtype'], 
                        chunks=True
                    )

                # Write data in chunks to be memory efficient
                chunk_size = 1000 
                for i in tqdm(range(0, len(indices), chunk_size), desc=f"Writing {filename}"):
                    batch_indices = indices[i:i+chunk_size]
                    
                    for name, dset in out_datasets.items():
                        current_size = dset.shape
                        data_batch = hf_in[name][batch_indices]
                        
                        # Resize and append
                        new_row_count = current_size[0] + len(data_batch)
                        dset.resize(new_row_count, axis=0)
                        dset[current_size[0]:] = data_batch

    print(f"\nSplitting complete. Files are saved in '{output_dir}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a consolidated HDF5 file into train, validation, and test sets based on molecular scaffolds.")
    parser.add_argument(
        "-i", "--input_file",
        default="dreams_dataset.hdf5",
        help="Path to the consolidated HDF5 dataset."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="split_dataset",
        help="Directory to save the output train/val/test HDF5 files."
    )
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.2, 0.1],
        help="A list of three floats for train, validation, and test split ratios (e.g., 0.8 0.1 0.1)."
    )
    args = parser.parse_args()

    if not np.isclose(sum(args.split_ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    create_scaffold_split(args.input_file, args.output_dir, tuple(args.split_ratios))