import torch
import torch.nn as nn
import torch.optim as optim
# --- Add the torch.multiprocessing import ---
import torch.multiprocessing as mp
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np

from retriever_model import ContrastiveRetriever
from retriever_dataloader import ShardedDataset
from scaffold_processor import smiles_to_graph

# --- Configuration ---
SHARED_EMBEDDING_DIM = 256
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 5

# The train_epoch and validate_and_test functions remain unchanged.
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        anchor_proj, pos_proj, neg_proj = model(batch.anchor_embedding, batch.positive_graph, batch.negative_graph)
        loss = criterion(anchor_proj, pos_proj, neg_proj)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_and_test(model, val_loader, scaffold_library, device, criterion):
    model.eval()
    total_loss = 0
    
    scaffold_graphs = [smiles_to_graph(smi) for smi in scaffold_library]
    scaffold_loader = PyGDataLoader(scaffold_graphs, batch_size=BATCH_SIZE * 2)
    scaffold_index = []
    with torch.no_grad():
        for batch in tqdm(scaffold_loader, desc="Building scaffold index"):
            scaffold_embs = model.scaffold_encoder(batch.to(device))
            scaffold_index.append(scaffold_embs.cpu())
    scaffold_index = torch.cat(scaffold_index, dim=0).to(device)
    scaffold_index = nn.functional.normalize(scaffold_index, p=2, dim=1)

    top1, top5, top10, total = 0, 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = batch.to(device, non_blocking=True)
            anchor_proj, pos_proj, neg_proj = model(batch.anchor_embedding, batch.positive_graph, batch.negative_graph)
            loss = criterion(anchor_proj, pos_proj, neg_proj)
            total_loss += loss.item()

            query_embs = model.dreams_projector(batch.anchor_embedding)
            query_embs = nn.functional.normalize(query_embs, p=2, dim=1)
            
            similarity_scores = torch.matmul(query_embs, scaffold_index.T)
            _, top_indices = torch.topk(similarity_scores, k=10, dim=1)

            gt_scaffolds = [g.smiles for g in batch.positive_graph.to_data_list()]
            
            for i, gt_smi in enumerate(gt_scaffolds):
                total += 1
                preds = [scaffold_library[j] for j in top_indices[i].cpu().tolist()]
                if gt_smi == preds[0]: top1 += 1
                if gt_smi in preds[:5]: top5 += 1
                if gt_smi in preds: top10 += 1

    val_loss = total_loss / len(val_loader)
    if total == 0: return val_loss, 0, 0, 0
    return val_loss, (top1/total)*100, (top5/total)*100, (top10/total)*100


def main(args):
    # --- THE FIX IS HERE ---
    # Set the sharing strategy BEFORE any multiprocessing starts (like the DataLoader).
    try:
        mp.set_sharing_strategy('file_system')
        print("Set multiprocessing sharing strategy to 'file_system'")
    except RuntimeError:
        print("Sharing strategy already set. Continuing.")
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.scaffold_library_path, 'r') as f:
        scaffold_library = [line.strip() for line in f if line.strip()]

    train_dataset = ShardedDataset(args.train_processed_dir)
    val_dataset = ShardedDataset(args.val_processed_dir)
    
    train_loader = PyGDataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = PyGDataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    sample_data = smiles_to_graph(scaffold_library[0])
    node_feature_dim = sample_data.num_node_features
    
    model = ContrastiveRetriever(1024, node_feature_dim, SHARED_EMBEDDING_DIM).to(device)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, top1, top5, top10 = validate_and_test(model, val_loader, scaffold_library, device, criterion)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print(f"Val Accuracy | Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | Top-10: {top10:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_retriever.pth'))
            print("Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{PATIENCE}")
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered."); break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Final, high-performance training script.")
    parser.add_argument("--train_processed_dir", required=True, help="Directory of pre-processed training .pt files.")
    parser.add_argument("--val_processed_dir", required=True, help="Directory of pre-processed validation .pt files.")
    parser.add_argument("--scaffold_library_path", required=True, help="Path to the MASTER scaffold library file.")
    parser.add_argument("--save_dir", default="./retriever_models")
    args = parser.parse_args()
    main(args)