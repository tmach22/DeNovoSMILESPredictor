import json
import h5py
import os
from collections import Counter
import argparse
from tqdm import tqdm

# This script requires RDKit and DeepChem.
# Install them via: pip install rdkit deepchem
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer

# --- [Your TokenizerWrapper Class - Unchanged] ---
# This class is used exactly as you provided it. It's a robust wrapper
# around the DeepChem tokenizer.
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
        self.tokenizer = BasicSmilesTokenizer()
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3
        }
        self.vocab = {}
        self.inv_vocab = {}

    def build_vocab(self, smiles_list, min_frequency=1):
        """
        Builds the vocabulary from a list of SMILES strings using the DeepChem tokenizer.
        """
        token_counts = Counter(
            token for smiles in smiles_list for token in self.tokenizer.tokenize(smiles)
        )
        self.vocab = self.special_tokens.copy()
        for token, count in token_counts.items():
            print(f"Token: {token}, Count: {count}")
            if count >= min_frequency:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        print(f"Vocabulary built with {len(self.vocab)} unique tokens.")

    def encode(self, smiles_string):
        """Encodes a SMILES string into a list of integer token IDs."""
        tokens = self.tokenizer.tokenize(smiles_string)
        encoded = [self.special_tokens['<sos>']]
        encoded.extend([self.vocab.get(token, self.special_tokens['<unk>']) for token in tokens])
        encoded.append(self.special_tokens['<eos>'])
        return encoded

    def decode(self, token_ids):
        """Decodes a list of integer token IDs back into a SMILES string."""
        smiles_list = []
        for token_id in token_ids:
            if token_id == self.special_tokens['<eos>']:
                break
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


# --- [NEW: Helper Functions for Scaffold Generation] ---

def get_murcko_scaffold(smiles_string: str) -> str:
    """
    Computes the Murcko scaffold SMILES from a full SMILES string using RDKit.
    Returns an empty string if the molecule has no scaffold or if SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        return ""
    except:
        return ""

def get_scaffolds_from_hdf5(hdf5_path: str) -> list:
    """
    Extracts all SMILES from the HDF5 file, converts them to Murcko scaffolds,
    and returns a list of scaffold SMILES strings. This list is then used to build the vocabulary.
    """
    scaffold_strings = []
    print("Reading full SMILES from HDF5 and generating scaffolds...")
    with h5py.File(hdf5_path, 'r') as f:
        # Ensure the 'smiles' dataset exists
        if 'smiles' not in f:
            raise KeyError("Dataset 'smiles' not found in the HDF5 file.")
        all_smiles = [s.decode('utf-8') for s in f['smiles'][:]]
        
        for smiles in tqdm(all_smiles, desc="Generating Scaffolds"):
            scaffold = get_murcko_scaffold(smiles)
            # Only include non-empty scaffolds in our vocabulary data
            if len(scaffold) > 0:
                scaffold_strings.append(scaffold)
    return scaffold_strings


# --- [Main Execution Block] ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build a Murcko Scaffold vocabulary from the training set using a DeepChem tokenizer.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--train_path",
        required=True,
        help="Path to the training HDF5 file. This is the ONLY data that will be used."
    )
    parser.add_argument(
        "--scaffold_vocab_out",
        required=True,
        help="Path for the output JSON vocabulary file that will be created."
    )
    args = parser.parse_args()

    # 1. Initialize the tokenizer wrapper
    print("--- Step 1: Initializing Tokenizer ---")
    tokenizer = TokenizerWrapper()
    print("Tokenizer wrapper with DeepChem's BasicSmilesTokenizer is ready.")

    # 2. Extract scaffolds from the 'train' dataset
    print(f"\n--- Step 2: Extracting Scaffolds from '{args.train_path}' ---")
    # This is the key change: we generate scaffolds first.
    train_scaffolds = get_scaffolds_from_hdf5(args.train_path)
    print(f"Successfully generated {len(train_scaffolds)} scaffold SMILES strings.")

    # 3. Build the vocabulary using only the scaffold strings
    print("\n--- Step 3: Building Vocabulary from Scaffolds ---")
    tokenizer.build_vocab(train_scaffolds)
    
    # 4. Save the vocabulary to the specified file
    print(f"\n--- Step 4: Saving Vocabulary to '{args.scaffold_vocab_out}' ---")
    tokenizer.save_vocab(args.scaffold_vocab_out)

    print("\nScaffold vocabulary building process complete.")