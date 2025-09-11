# evaluate.py
import torch
import numpy as np
import os
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit import RDLogger
from torch.utils.data import DataLoader

# Import the components we've built
from data_loader import get_data_loader, TokenizerWrapper
from og_data_loader import SpectraDataset, collate_fn_with_padding
from model import SmilesDecoder

# Suppress RDKit console error messages (e.g., for invalid SMILES)
RDLogger.DisableLog('rdApp.*')

# --- Configuration ---
# Make sure to update these paths to match your training script
HDF5_PATH = "/bigdata/jianglab/shared/ExploreData/extracted_data_json/for_tejas/hdf5_files/all_dreams_embeddings_with_smiles.hdf5"
VOCAB_PATH = "/bigdata/jianglab/shared/ExploreData/vocab/smiles_vocab.json"
MODEL_PATH = "/bigdata/jianglab/shared/ExploreData/models/best_model.pth" # Path to your saved model

# Model Hyperparameters (must match the trained model)
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
COND_DIM = 1024

# Evaluation and Generation Parameters
BATCH_SIZE = 128 # Can use a larger batch size for inference
MAX_SMILES_LEN = 150 # Maximum length of the generated SMILES string

def generate_smiles(model, embedding, tokenizer, device, max_len=MAX_SMILES_LEN):
    """
    Generates a SMILES string from a DreaMS embedding using greedy decoding.
    """
    model.eval()
    sos_token_id = tokenizer.vocab['<sos>']
    eos_token_id = tokenizer.vocab['<eos>']
    
    input_tokens = torch.tensor([sos_token_id], dtype=torch.long).unsqueeze(0).to(device)
    embedding = embedding.unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            output_logits = model(embedding, input_tokens)
            last_token_logits = output_logits[:, -1, :]
            predicted_token_id = torch.argmax(last_token_logits, dim=-1)
            
            input_tokens = torch.cat([input_tokens, predicted_token_id.unsqueeze(0)], dim=1)
            
            if predicted_token_id.item() == eos_token_id:
                break
                
    generated_ids = input_tokens.squeeze(0).cpu().numpy().tolist()
    return tokenizer.decode(generated_ids)

def calculate_tanimoto(smiles1, smiles2):
    """Calculates Tanimoto similarity between two SMILES strings."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return TanimotoSimilarity(fp1, fp2)

def main():
    """Main function to load the model and evaluate it on the entire test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize tokenizer and load vocabulary
    tokenizer = TokenizerWrapper()
    tokenizer.load_vocab(VOCAB_PATH)
    vocab_size = len(tokenizer.vocab)

    # 2. Initialize and load the trained model
    model = SmilesDecoder(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        cond_dim=COND_DIM
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model loaded from {MODEL_PATH}")

    # 3. Load the test dataset
    # print("Loading test data...")
    # test_loader = get_data_loader(HDF5_PATH, tokenizer, split='test', batch_size=BATCH_SIZE)
    test_dataset = SpectraDataset(hdf5_path=HDF5_PATH, fold='test')
    print(f"Number of samples in TRAIN set: {len(test_dataset)}")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=True, # Shuffle is important for the training set
        num_workers=2,
        collate_fn=lambda x: collate_fn_with_padding(x, test_dataset.pad_id, tokenizer)
    )
    if not test_loader:
        print("Could not load test data. Exiting.")
        return
    
    # 4. Full Evaluation Loop
    print("\n--- Starting Full Evaluation on Test Set ---")
    total_samples = 0
    valid_smiles_count = 0
    exact_matches = 0
    tanimoto_scores = []

    for embeddings, ground_truth_padded in tqdm(test_loader, desc="Evaluating"):
        for i in range(embeddings.size(0)):
            total_samples += 1
            embedding_sample = embeddings[i]
            
            # Generate the predicted SMILES
            predicted_smiles = generate_smiles(model, embedding_sample, tokenizer, device)
            
            # --- Metric 1: Chemical Validity ---
            mol = Chem.MolFromSmiles(predicted_smiles)
            if mol is not None:
                valid_smiles_count += 1
                
                # Canonicalize predicted SMILES for a fair comparison
                canonical_predicted_smiles = Chem.MolToSmiles(mol)
                
                # Decode the ground truth SMILES
                ground_truth_ids = ground_truth_padded[i].cpu().numpy().tolist()
                ground_truth_smiles = tokenizer.decode(ground_truth_ids)
                
                # --- Metric 2: Exact Match Accuracy ---
                if canonical_predicted_smiles == ground_truth_smiles:
                    exact_matches += 1
                
                # --- Metric 3: Tanimoto Similarity ---
                similarity = calculate_tanimoto(ground_truth_smiles, canonical_predicted_smiles)
                tanimoto_scores.append(similarity)
            else:
                # If the SMILES is invalid, its Tanimoto similarity is 0
                tanimoto_scores.append(0.0)

    # 5. Calculate and Print Final Results
    validity_rate = (valid_smiles_count / total_samples) * 100
    exact_match_accuracy = (exact_matches / total_samples) * 100
    avg_tanimoto = np.mean(tanimoto_scores) if tanimoto_scores else 0.0
    
    print("\n--- Evaluation Complete ---")
    print(f"Total Test Samples: {total_samples}")
    print("-" * 30)
    print(f"Chemical Validity Rate: {validity_rate:.2f}%")
    print(f"Exact Match Accuracy:   {exact_match_accuracy:.2f}%")
    print(f"Average Tanimoto Sim.:  {avg_tanimoto:.4f}")
    print("-" * 30)

if __name__ == '__main__':
    main()