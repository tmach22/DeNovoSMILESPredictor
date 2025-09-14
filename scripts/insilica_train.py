import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import os
import argparse

from data_loader import get_graph_data_loader
from tokenizer import TokenizerWrapper 
from model import SmilesRecyclingDecoder

# --- Model Hyperparameters ---
D_MODEL = 256
NHEAD = 8
GNN_LAYERS = 2 
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 1024
DREAMS_DIM = 1024
FORMULA_EMB_DIM = 64
SCAFFOLD_EMB_DIM = 128
GNN_HIDDEN_DIM = 128
NUM_RECYCLING_ITERS = 3

# --- Training Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 3e-4 
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
DROP_EDGE_RATE = 0.2
CLIP_GRAD = 1.0
PATIENCE = 10

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        if batch is None: continue
        embeddings = batch["embedding"].to(device)
        formulas = batch["formula"].to(device)
        smiles_padded = batch["smiles"].to(device)
        scaffold_graph_list = batch["scaffold_graph_list"] 
        tgt_input = smiles_padded[:, :-1]
        tgt_output = smiles_padded[:, 1:]
        optimizer.zero_grad()
        output_logits = model(embeddings, formulas, scaffold_graph_list, tgt_input)
        loss = criterion(output_logits.reshape(-1, output_logits.shape[-1]), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            if batch is None: continue
            embeddings = batch["embedding"].to(device)
            formulas = batch["formula"].to(device)
            smiles_padded = batch["smiles"].to(device)
            scaffold_graph_list = batch["scaffold_graph_list"]
            tgt_input = smiles_padded[:, :-1]
            tgt_output = smiles_padded[:, 1:]
            output_logits = model(embeddings, formulas, scaffold_graph_list, tgt_input)
            loss = criterion(output_logits.reshape(-1, output_logits.shape[-1]), tgt_output.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    print("--- Initializing Tokenizers and Loading Vocabularies ---")
    smiles_tokenizer = TokenizerWrapper()
    formula_tokenizer = TokenizerWrapper()
    smiles_tokenizer.load_vocab(args.smiles_vocab_path)
    formula_tokenizer.load_vocab(args.formula_vocab_path)
    smiles_pad_idx = smiles_tokenizer.vocab['<pad>']
    smiles_vocab_size = len(smiles_tokenizer.vocab)
    formula_vocab_size = len(formula_tokenizer.vocab)

    print("\n--- Creating Graph DataLoaders ---")
    train_loader = get_graph_data_loader(
        args.train_path, smiles_tokenizer, formula_tokenizer, 'train', 
        args.batch_size, num_workers=args.num_workers
    )
    val_loader = get_graph_data_loader(
        args.val_path, smiles_tokenizer, formula_tokenizer, 'val', 
        args.batch_size, num_workers=args.num_workers
    )

    print("\n--- Initializing GNN-Enhanced Model ---")
    model = SmilesRecyclingDecoder(
        smiles_vocab_size=smiles_vocab_size, d_model=D_MODEL, nhead=NHEAD,
        num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dreams_dim=DREAMS_DIM,
        formula_vocab_size=formula_vocab_size, formula_emb_dim=FORMULA_EMB_DIM,
        scaffold_emb_dim=SCAFFOLD_EMB_DIM, gnn_hidden_dim=GNN_HIDDEN_DIM,
        num_gnn_layers=GNN_LAYERS,
        drop_edge_rate=args.drop_edge_rate,
        num_recycling_iters=NUM_RECYCLING_ITERS
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    criterion = nn.CrossEntropyLoss(ignore_index=smiles_pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print("\n--- Setting up Learning Rate Scheduler ---")
    warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    main_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])
    print(f"Using CosineAnnealingLR with a {args.warmup_epochs}-epoch linear warmup.")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        # --- FIXED: Access the learning rate from the first parameter group ---
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(args.save_dir, 'best_model_gnn.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Validation loss improved. Model saved to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{args.patience}")
        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break
            
    print("\nTraining complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the GNN-enhanced SMILES generator.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--smiles_vocab_path", type=str, required=True)
    parser.add_argument("--formula_vocab_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./models_gnn")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--drop_edge_rate", type=float, default=DROP_EDGE_RATE)
    args = parser.parse_args()
    main(args)