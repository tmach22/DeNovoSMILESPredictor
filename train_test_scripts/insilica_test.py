import torch
import h5py
import numpy as np
import argparse
import os
from tqdm import tqdm

# Import RDKit for chemical validation and similarity calculation
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem

from pathlib import Path
import os
import sys
cwd = Path.cwd()

script_dir = os.path.join(cwd, 'scripts')
print(f"Adding {script_dir} to sys.path")
# Add the parent directory to the Python path
sys.path.insert(0, script_dir)

data_loader_dir = os.path.join(cwd, 'data_loaders')
print(f"Adding {data_loader_dir} to sys.path")
sys.path.insert(0, data_loader_dir)

model_dir = os.path.join(cwd, 'model_scripts')
print(f"Adding {model_dir} to sys.path")
sys.path.insert(0, model_dir)

# Import your custom model and tokenizer classes
from model import SmilesRecyclingDecoder
from data_loader_fast import smiles_to_graph_data
from tokenizer import TokenizerWrapper

# --- Model Hyperparameters (Must match the trained model) ---
D_MODEL = 256
NHEAD = 8
# UPDATED: Renamed NUM_LAYERS to be more specific and added GNN_LAYERS
GNN_LAYERS = 3
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 1024
DREAMS_DIM = 1024
FORMULA_EMB_DIM = 64
SCAFFOLD_EMB_DIM = 128
GNN_HIDDEN_DIM = 256
# NEW: Add drop_edge_rate to match the model's __init__ signature
DROP_EDGE_RATE = 0.2 # This is ignored during eval, but needed for initialization
NUM_RECYCLING_ITERS = 3

def calculate_tanimoto(smiles1, smiles2):
    """Calculates the Tanimoto similarity between two SMILES strings."""
    try:
        mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None: return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return TanimotoSimilarity(fp1, fp2)
    except: return 0.0

def generate_smiles(model, dreams_emb, formula_tokens, scaffold_graph, smiles_tokenizer, device, max_len=256):
    """
    Generates a SMILES string autoregressively using the GNN-enhanced model.
    """
    model.eval()
    sos_token, eos_token = smiles_tokenizer.vocab['<sos>'], smiles_tokenizer.vocab['<eos>']
    generated_seq = torch.tensor([[sos_token]], dtype=torch.long, device=device)

    scaffold_graph_list = [scaffold_graph]

    with torch.no_grad():
        for _ in range(max_len):
            output_logits = model(dreams_emb, formula_tokens, scaffold_graph_list, generated_seq)
            last_token_logits = output_logits[:, -1, :]
            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)
            generated_seq = torch.cat([generated_seq, next_token], dim=1)
            if next_token.item() == eos_token: break
            
    return smiles_tokenizer.decode(generated_seq.squeeze(0).cpu().tolist())

def main(args):
    if(torch.cuda.is_available()):
        device = args.device
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print("--- Loading Vocabularies ---")
    smiles_tokenizer, formula_tokenizer = TokenizerWrapper(), TokenizerWrapper()
    smiles_tokenizer.load_vocab(args.smiles_vocab_path)
    formula_tokenizer.load_vocab(args.formula_vocab_path)
    smiles_vocab_size = len(smiles_tokenizer.vocab)
    formula_vocab_size = len(formula_tokenizer.vocab)

    print("\n--- Initializing GNN-Enhanced Model Architecture ---")
    # --- FIXED: Updated the model initialization to match the new signature ---
    # - 'num_layers' is now 'num_decoder_layers'
    # - Added 'num_gnn_layers' and 'drop_edge_rate' arguments
    model = SmilesRecyclingDecoder(
        smiles_vocab_size=smiles_vocab_size, d_model=D_MODEL, nhead=NHEAD,
        num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dreams_dim=DREAMS_DIM,
        formula_vocab_size=formula_vocab_size, formula_emb_dim=FORMULA_EMB_DIM,
        scaffold_emb_dim=SCAFFOLD_EMB_DIM, gnn_hidden_dim=GNN_HIDDEN_DIM,
        num_gnn_layers=GNN_LAYERS,
        drop_edge_rate=DROP_EDGE_RATE,
        num_recycling_iters=NUM_RECYCLING_ITERS
    ).to(device)

    print(f"Loading trained model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print(f"Loading test data from: {args.test_path}")
    with h5py.File(args.test_path, 'r') as f:
        test_embeddings = f['embeddings'][:]
        test_smiles = [s.decode('utf-8') for s in f['smiles'][:]]
        test_formulas = [f.decode('utf-8') for f in f['formula_strings'][:]]
        test_scaffolds = [s.decode('utf-8') for s in f['scaffold_smiles'][:]]
    if args.num_samples: test_embeddings, test_smiles, test_formulas, test_scaffolds = test_embeddings[:args.num_samples], test_smiles[:args.num_samples], test_formulas[:args.num_samples], test_scaffolds[:args.num_samples]

    print(f"\n--- Running Inference on {len(test_smiles)} Samples ---")
    predictions, tanimoto_scores, valid_count = [],[], 0

    for i in tqdm(range(len(test_smiles)), desc="Testing"):
        true_smiles = test_smiles[i]
        dreams_emb = torch.tensor(test_embeddings[i], dtype=torch.float32).unsqueeze(0).to(device)
        
        formula_str = test_formulas[i]
        scaffold_str = test_scaffolds[i]
        
        formula_tokens = torch.tensor(formula_tokenizer.encode(formula_str), dtype=torch.long).unsqueeze(0).to(device)
        
        scaffold_graph = smiles_to_graph_data(scaffold_str)
        
        predicted_smiles = generate_smiles(model, dreams_emb, formula_tokens, scaffold_graph, smiles_tokenizer, device)
        predictions.append(predicted_smiles)
        
        if Chem.MolFromSmiles(predicted_smiles) is not None: valid_count += 1
        tanimoto_scores.append(calculate_tanimoto(true_smiles, predicted_smiles))

    avg_tanimoto = np.mean(tanimoto_scores); validity_percent = (valid_count / len(test_smiles)) * 100
    exact_matches = sum(1 for true, pred in zip(test_smiles, predictions) if true == pred)
    exact_match_percent = (exact_matches / len(test_smiles)) * 100

    print("\n--- Evaluation Complete ---")
    print(f"Total Samples:          {len(test_smiles)}")
    print(f"SMILES Validity:          {validity_percent:.2f}%")
    print(f"Exact Match Accuracy:   {exact_match_percent:.2f}%")
    print(f"Average Tanimoto Score: {avg_tanimoto:.4f}")
    
    print("\n--- Example Predictions ---")
    for i in range(min(5, len(test_smiles))):
        print(f"\nSample {i+1}:\n  Ground Truth: {test_smiles[i]}\n  Prediction:   {predictions[i]}\n  Tanimoto:     {tanimoto_scores[i]:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the GNN-enhanced SMILES generator.")
    parser.add_argument("--device", type=str, default="cpu", required=True, help="Device to use for inference")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test HDF5 file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained best_model_gnn.pth checkpoint.")
    parser.add_argument("--smiles_vocab_path", type=str, required=True, help="Path to SMILES vocabulary.")
    parser.add_argument("--formula_vocab_path", type=str, required=True, help="Path to formula vocabulary.")
    parser.add_argument("--num_samples", type=int, default=None, help="Optional: Number of samples for a quick evaluation.")
    args = parser.parse_args()
    main(args)