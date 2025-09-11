import json
import h5py
import os
from collections import Counter
import argparse

# DeepChem is now a required dependency for this script.
# Install it via: pip install deepchem
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer

class TokenizerWrapper:
    """
    A wrapper class that uses DeepChem's BasicSmilesTokenizer for the core
    tokenization logic and builds a vocabulary management system around it.
    """
    
    def __init__(self):
        """
        Initializes the wrapper, creating an instance of the DeepChem tokenizer
        and setting up special tokens.
        """
        # Use the BasicSmilesTokenizer directly from the DeepChem library
        self.tokenizer = BasicSmilesTokenizer()
        
        # Special tokens are essential for sequence modeling
        self.special_tokens = {
            '<pad>': 0,  # Padding token
            '<unk>': 1,  # Unknown token
            '<sos>': 2,  # Start of Sequence token
            '<eos>': 3   # End of Sequence token
        }
        
        self.vocab = {}
        self.inv_vocab = {}

    def build_vocab(self, smiles_list, min_frequency=1):
        """
        Builds the vocabulary from a list of SMILES strings using the DeepChem tokenizer.
        
        Args:
            smiles_list (list): A list of SMILES strings from the training dataset.
            min_frequency (int): The minimum frequency for a token to be included.
        """
        # Tokenize all SMILES using the deepchem tokenizer and count token frequencies
        token_counts = Counter(
            token for smiles in smiles_list for token in self.tokenizer.tokenize(smiles)
        )
        
        # Start vocabulary with special tokens
        self.vocab = self.special_tokens.copy()
        
        # Add tokens that meet the minimum frequency threshold
        for token, count in token_counts.items():
            if count >= min_frequency:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    
        # Create the inverse vocabulary for decoding
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        print(f"Vocabulary built with {len(self.vocab)} unique tokens.")

    def encode(self, smiles_string):
        """
        Encodes a SMILES string into a list of integer token IDs.
        
        Args:
            smiles_string (str): The SMILES string to encode.
            
        Returns:
            list: A list of integer token IDs, including start and end tokens.
        """
        # Use the deepchem tokenizer to split the string
        tokens = self.tokenizer.tokenize(smiles_string)
        
        encoded = [self.special_tokens['<sos>']]
        encoded.extend([self.vocab.get(token, self.special_tokens['<unk>']) for token in tokens])
        encoded.append(self.special_tokens['<eos>'])
        return encoded

    def decode(self, token_ids):
        """
        Decodes a list of integer token IDs back into a SMILES string.
        
        Args:
            token_ids (list): A list of integer token IDs.
            
        Returns:
            str: The reconstructed SMILES string.
        """
        smiles_list = []
        for token_id in token_ids:
            # Stop decoding if end token is encountered
            if token_id == self.special_tokens['<eos>']:
                break
            # Ignore start and padding tokens in the output string
            if token_id not in [self.special_tokens['<sos>'], self.special_tokens['<pad>']]:
                smiles_list.append(self.inv_vocab.get(token_id, ''))
        return "".join(smiles_list)

    def save_vocab(self, file_path):
        """Saves the vocabulary to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.vocab, f, indent=4)
        print(f"Vocabulary saved to {file_path}")

    def load_vocab(self, file_path):
        """Loads the vocabulary from a JSON file."""
        with open(file_path, 'r') as f:
            self.vocab = json.load(f)
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        print(f"Vocabulary loaded from {file_path}. Total tokens: {len(self.vocab)}")

def get_smiles_from_hdf5(hdf5_path):
    """Helper function to extract SMILES strings for a specific fold from the HDF5 file."""
    smiles_strings = []
    with h5py.File(hdf5_path, 'r') as f:
        all_smiles = [s.decode('utf-8') for s in f['smiles'][:]]
        
        for i, string in enumerate(all_smiles):
            smiles_strings.append(all_smiles[i])
    return smiles_strings

# --- Example Usage Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess spectra from a JSON file and save to a DreaMS-compatible MGF file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--hdf5_filepath",
        required=True,
        help="Path to the input JSON file containing the spectra."
    )
    parser.add_argument(
        "--vocab_file",
        required=True,
        help="Path for the output MGF file that will be created."
    )
    args = parser.parse_args()

    # (Dummy file creation logic can be added here if needed for testing)
    
    # 1. Initialize the tokenizer wrapper
    print("--- Step 1: Initializing Tokenizer ---")
    tokenizer_wrapper = TokenizerWrapper()
    print("DeepChem's BasicSmilesTokenizer has been instantiated.")

    # 2. Extract SMILES from the 'train' fold of your dataset
    print(f"\n--- Step 2: Extracting SMILES from '{args.hdf5_filepath}' ---")
    train_smiles = get_smiles_from_hdf5(args.hdf5_filepath)
    print(f"Found {len(train_smiles)} SMILES strings in the training set.")

    # 3. Build the vocabulary
    print("\n--- Step 3: Building Vocabulary ---")
    tokenizer_wrapper.build_vocab(train_smiles)
    
    # 4. Encode a SMILES string
    print("\n--- Step 4: Encoding a SMILES string ---")
    original_smiles = "CC(C)C(=O)O"
    print(f"Original SMILES: {original_smiles}")
    print(f"Tokenized with DeepChem: {tokenizer_wrapper.tokenizer.tokenize(original_smiles)}")
    encoded_ids = tokenizer_wrapper.encode(original_smiles)
    print(f"Encoded IDs: {encoded_ids}")

    # 5. Decode the token IDs back to a SMILES string
    print("\n--- Step 5: Decoding the IDs ---")
    decoded_smiles = tokenizer_wrapper.decode(encoded_ids)
    print(f"Decoded SMILES: {decoded_smiles}")
    print(f"Reconstruction successful: {original_smiles == decoded_smiles}")
    
    # 6. Save and load the vocabulary
    print(f"\n--- Step 6: Saving and Loading Vocabulary to/from '{args.vocab_file}' ---")
    tokenizer_wrapper.save_vocab(args.vocab_file)
    
    new_tokenizer_wrapper = TokenizerWrapper()
    new_tokenizer_wrapper.load_vocab(args.vocab_file)