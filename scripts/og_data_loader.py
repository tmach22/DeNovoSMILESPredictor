import torch
import h5py
import os
import sys
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
cwd = Path.cwd()

script_dir = os.path.join(cwd, 'scripts')
# Add the parent directory to the Python path
sys.path.insert(0, script_dir)

import numpy as np
from torch.utils.data import Dataset, DataLoader
from tokenizer import TokenizerWrapper

VOCAB_FILE = "/bigdata/jianglab/shared/ExploreData/vocab/smiles_vocab.json"

class SpectraDataset(Dataset):
    """
    Custom PyTorch Dataset for loading spectra data from an HDF5 file,
    with built-in support for train/validation/test splits based on a 'fold' column.
    """
    
    def __init__(self, hdf5_path, fold='train'):
        """
        Initializes the Dataset object for a specific data fold.
        
        Args:
            hdf5_path (string): The path to the HDF5 file.
            fold (string): The fold to load. Must be one of 'train', 'val', or 'test'.
        """
        if fold not in ['train', 'val', 'test']:
            raise ValueError("Fold must be one of 'train', 'val', or 'test'.")
            
        self.hdf5_path = hdf5_path
        self.fold = fold
        self._file = None  # File handle managed internally for multiprocessing
        tokenizer = TokenizerWrapper()
        tokenizer.load_vocab(VOCAB_FILE)
        self.tokenizer = tokenizer
        # Store the padding token ID
        self.pad_id = self.tokenizer.special_tokens['<pad>']
        
        # Open the file to read the 'fold' column and determine the indices for this split.
        with h5py.File(self.hdf5_path, 'r') as f:
            all_folds = f['fold'][:]  # Load all fold labels into memory
            # Decode byte strings to regular strings for comparison
            decoded_folds = [label.decode('utf-8') for label in all_folds]
            
            # Find the numerical indices corresponding to the desired fold
            self.indices = np.where(np.array(decoded_folds) == self.fold)
            
        self.dataset_len = len(self.indices[0])

    def __len__(self):
        """
        Returns the number of samples in the specific fold ('train', 'val', or 'test').
        """
        return self.dataset_len

    def __getitem__(self, idx):
        """
        Fetches the sample at a given index *within the current fold*.
        
        Args:
            idx (int): The index within the current fold's subset of data.
            
        Returns:
            tuple: A tuple containing the DreaMS embedding, precursor m/z, and SMILES string.
        """
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')
            
        # Map the relative index (idx) to the absolute index in the HDF5 file
        actual_idx = self.indices[0][idx]

        embedding = self._file['dreams_embedding'][actual_idx]
        smiles_bytes = self._file['smiles'][actual_idx]

        # The user's tokenizer is assumed to take a string and return a list of integers
        smiles_string = smiles_bytes.decode('utf-8')
        
        # Convert embedding to a PyTorch tensor
        return torch.from_numpy(embedding), smiles_string
    
# --- Custom Collate Function for Padding ---
def collate_fn_with_padding(batch, pad_id, tokenizer):
    """
    Pads variable-length SMILES sequences to the maximum length in a batch.
    
    Args:
        batch (list): A list of tuples, where each tuple contains (embedding, smiles_tokens).
                      smiles_tokens is a list of integers.

    Returns:
        tuple: A tuple containing a batch of embeddings and a batch of padded SMILES tensors.
    """
    # Separate the embeddings and the SMILES tokens from the batch
    embeddings = [item[0] for item in batch]
    smiles_strings = [item[1] for item in batch]

    smiles_tokens = [torch.tensor(tokenizer.encode(s)) for s in smiles_strings]
    padded_smiles = pad_sequence(smiles_tokens, batch_first=True, padding_value=pad_id)
    
    # Convert the lists of tensors into single PyTorch tensors
    embeddings_tensor = torch.stack(embeddings)
    
    return embeddings_tensor, padded_smiles

# --- Example Usage Block ---
if __name__ == '__main__':
    HDF5_FILE_PATH = '/bigdata/jianglab/shared/ExploreData/extracted_data_json/for_tejas/hdf5_files/all_dreams_embeddings_with_smiles.hdf5'

    # Create a dummy HDF5 file for demonstration if it doesn't exist.
    if not os.path.exists(HDF5_FILE_PATH):
        print(f"'{HDF5_FILE_PATH}' not found. Creating a dummy file for demonstration.")
        exit(0)

    # --- Step 1: Instantiate a separate Dataset for each fold ---
    print("\n--- Initializing Datasets for each fold ---")
    train_dataset = SpectraDataset(hdf5_path=HDF5_FILE_PATH, fold='train')
    val_dataset = SpectraDataset(hdf5_path=HDF5_FILE_PATH, fold='val')
    test_dataset = SpectraDataset(hdf5_path=HDF5_FILE_PATH, fold='test')

    print(f"Number of samples in TRAIN set: {len(train_dataset)}")
    print(f"Number of samples in VALIDATION set: {len(val_dataset)}")
    print(f"Number of samples in TEST set: {len(test_dataset)}")

    # --- Step 2: Create a DataLoader for each Dataset ---
    NUM_WORKERS = 0
    tokenizer = TokenizerWrapper()
    tokenizer.load_vocab(VOCAB_FILE)
    pad_id = tokenizer.special_tokens['<pad>']
    print("\n--- Initializing DataLoaders for each fold ---")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True, # Shuffle is important for the training set
        num_workers=NUM_WORKERS,
        collate_fn=lambda x: collate_fn_with_padding(x, train_dataset.pad_id, tokenizer)
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False, # No need to shuffle validation or test data
        num_workers=NUM_WORKERS,
        collate_fn=lambda x: collate_fn_with_padding(x, train_dataset.pad_id, tokenizer)
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda x: collate_fn_with_padding(x, train_dataset.pad_id, tokenizer)
    )
    print("DataLoaders created successfully.")

    try:
        # --- Step 3: Inspect a batch from the training loader ---
        print("\n--- Inspecting a batch from the TRAIN loader ---")
        # next(iter(...)) is a convenient way to grab just one batch
        embeddings, smiles_strings = next(iter(train_loader))
        
        print(f"Embeddings batch shape: {embeddings.shape}")
        print(f"SMILES tokens batch: {smiles_strings.shape}")
        print(f"Number of SMILES in batch: {smiles_strings.dtype}")
        print("-----------------")
    except Exception as e:
        print(f"Error occurred while inspecting data loaders: {e}")