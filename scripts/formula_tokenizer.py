import json
import h5py
import os
from collections import Counter
import argparse

# RDKit is required to calculate the molecular formula from SMILES.
# Install it via: pip install rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

class FormulaTokenizer:
    """
    A character-level tokenizer specifically designed for chemical formulas.
    It provides methods for building a vocabulary, encoding formulas to token IDs,
    decoding IDs back to formulas, and saving/loading the vocabulary.
    """
    
    def __init__(self):
        """
        Initializes the tokenizer and sets up the essential special tokens.
        """
        # Special tokens are essential for sequence modeling
        self.special_tokens = {
            '<pad>': 0,  # Padding token
            '<unk>': 1,  # Unknown token
            '<sos>': 2,  # Start of Sequence token
            '<eos>': 3   # End of Sequence token
        }
        
        self.vocab = {}
        self.inv_vocab = {}

    def tokenize(self, formula_string):
        """
        Performs character-level tokenization on a formula string.
        
        Args:
            formula_string (str): The chemical formula to tokenize.
            
        Returns:
            list: A list of single-character tokens.
        """
        if not isinstance(formula_string, str):
            return
        return list(formula_string)

    def build_vocab(self, formula_list, min_frequency=1):
        """
        Builds the vocabulary from a list of formula strings.
        
        Args:
            formula_list (list): A list of formula strings from the training dataset.
            min_frequency (int): The minimum frequency for a token to be included.
        """
        # Count frequencies of each character (token) across all formulas
        token_counts = Counter(
            token for formula in formula_list for token in self.tokenize(formula)
        )
        
        # Start vocabulary with special tokens
        self.vocab = self.special_tokens.copy()
        
        # Add characters that meet the minimum frequency threshold
        for token, count in token_counts.items():
            if count >= min_frequency:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    
        # Create the inverse vocabulary for decoding
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        print(f"Formula vocabulary built with {len(self.vocab)} unique tokens.")

    def encode(self, formula_string):
        """
        Encodes a formula string into a list of integer token IDs.
        
        Args:
            formula_string (str): The formula string to encode.
            
        Returns:
            list: A list of integer token IDs, including start and end tokens.
        """
        tokens = self.tokenize(formula_string)
        
        encoded = [self.special_tokens['<sos>']]
        encoded.extend([self.vocab.get(token, self.special_tokens['<unk>']) for token in tokens])
        encoded.append(self.special_tokens['<eos>'])
        return encoded

    def decode(self, token_ids):
        """
        Decodes a list of integer token IDs back into a formula string.
        
        Args:
            token_ids (list): A list of integer token IDs.
            
        Returns:
            str: The reconstructed formula string.
        """
        formula_list = []
        for token_id in token_ids:
            # Stop decoding if end token is encountered
            if token_id == self.special_tokens['<eos>']:
                break
            # Ignore start and padding tokens in the output string
            if token_id not in [self.special_tokens['<sos>'], self.special_tokens['<pad>']]:
                formula_list.append(self.inv_vocab.get(token_id, ''))
        return "".join(formula_list)

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

def get_formulas_from_hdf5(hdf5_path):
    """Helper function to extract SMILES, then compute and return molecular formulas."""
    formulas = []
    with h5py.File(hdf5_path, 'r') as f:
        all_smiles = [s.decode('utf-8') for s in f['smiles'][:]]
        
        for smiles in tqdm(all_smiles, desc="Calculating Formulas"):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                formula = rdMolDescriptors.CalcMolFormula(mol)
                formulas.append(formula)
    return formulas

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build and save a character-level vocabulary for chemical formulas.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--hdf5_filepath",
        required=True,
        help="Path to the input HDF5 file containing the training data (e.g., train.hdf5)."
    )
    parser.add_argument(
        "--vocab_file",
        required=True,
        help="Path for the output JSON vocabulary file that will be created."
    )
    args = parser.parse_args()

    # 1. Initialize the tokenizer
    print("--- Step 1: Initializing FormulaTokenizer ---")
    formula_tokenizer = FormulaTokenizer()

    # 2. Extract formulas from the training dataset
    print(f"\n--- Step 2: Extracting Formulas from '{args.hdf5_filepath}' ---")
    train_formulas = get_formulas_from_hdf5(args.hdf5_filepath)
    print(f"Successfully calculated {len(train_formulas)} formulas.")

    # 3. Build the vocabulary
    print("\n--- Step 3: Building Vocabulary ---")
    formula_tokenizer.build_vocab(train_formulas)
    
    # 4. Encode a sample formula
    print("\n--- Step 4: Encoding a sample formula ---")
    original_formula = "C10H12N2O"
    print(f"Original Formula: {original_formula}")
    print(f"Tokenized: {formula_tokenizer.tokenize(original_formula)}")
    encoded_ids = formula_tokenizer.encode(original_formula)
    print(f"Encoded IDs: {encoded_ids}")

    # 5. Decode the token IDs back to a formula string
    print("\n--- Step 5: Decoding the IDs ---")
    decoded_formula = formula_tokenizer.decode(encoded_ids)
    print(f"Decoded Formula: {decoded_formula}")
    print(f"Reconstruction successful: {original_formula == decoded_formula}")
    
    # 6. Save the vocabulary
    print(f"\n--- Step 6: Saving Vocabulary to '{args.vocab_file}' ---")
    formula_tokenizer.save_vocab(args.vocab_file)

    #7. Load the vocabulary back and verify
    print(f"\n--- Step 7: Loading Vocabulary from '{args.vocab_file}' ---")
    new_tokenizer = FormulaTokenizer()
    new_tokenizer.load_vocab(args.vocab_file)
    encoded_ids = new_tokenizer.encode(original_formula)
    re_decoded_formula = new_tokenizer.decode(encoded_ids)
    print(f"Re-decoded Formula: {re_decoded_formula}")
    print(f"Reconstruction successful after loading vocab: {original_formula == re_decoded_formula}")