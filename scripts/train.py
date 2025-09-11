# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

# Import the components we've built
from data_loader import get_data_loader, TokenizerWrapper
from og_data_loader import SpectraDataset, collate_fn_with_padding
from torch.utils.data import DataLoader
from model import SmilesDecoder

# --- Configuration and Hyperparameters ---
# Make sure to update these paths
HDF5_PATH = "/bigdata/jianglab/shared/ExploreData/extracted_data_json/for_tejas/hdf5_files/all_dreams_embeddings_with_smiles.hdf5"
VOCAB_PATH = "/bigdata/jianglab/shared/ExploreData/vocab/smiles_vocab.json"
SAVE_DIR = "/bigdata/jianglab/shared/ExploreData/models"  # Directory to save model checkpoints

# Model Hyperparameters (tuned for a good balance of performance and speed)
D_MODEL = 256           # The dimensionality of the model's hidden states
NHEAD = 8               # Number of attention heads in each multi-head attention layer
NUM_LAYERS = 6          # Number of FiLMDecoderLayer blocks in the decoder
DIM_FEEDFORWARD = 1024  # Dimensionality of the feed-forward network
COND_DIM = 1024         # Dimensionality of the DreaMS embedding (should be 1024)

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
CLIP_GRAD = 1.0         # Gradient clipping to prevent exploding gradients

# --- Early Stopping Parameters ---
PATIENCE = 5  # Number of epochs to wait for improvement before stopping

def train_epoch(model, loader, optimizer, criterion, device):
    """Runs one full epoch of training."""
    model.train()  # Set the model to training mode
    total_loss = 0
    
    for embeddings, smiles_padded in tqdm(loader, desc="Training"):
        # Move data to the configured device (GPU or CPU)
        embeddings = embeddings.to(device)
        smiles_padded = smiles_padded.to(device)

        # Prepare inputs and targets for "teacher forcing"
        # The model receives the sequence up to the second-to-last token
        tgt_input = smiles_padded[:, :-1]
        # The model's target is to predict the sequence from the second token onwards
        tgt_output = smiles_padded[:, 1:]

        optimizer.zero_grad()
        
        # Forward pass: get model's predictions (logits)
        output_logits = model(embeddings, tgt_input)
        
        # Reshape for loss calculation: (Batch * SeqLen, VocabSize)
        # The loss function expects a 2D tensor of logits and a 1D tensor of targets
        loss = criterion(output_logits.reshape(-1, output_logits.shape[-1]), tgt_output.reshape(-1))
        
        loss.backward()
        
        # Clip gradients to prevent them from exploding, a common issue in RNNs/Transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    """Runs one full epoch of validation."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    
    with torch.no_grad():  # Disable gradient calculations for efficiency
        for embeddings, smiles_padded in tqdm(loader, desc="Validating"):
            embeddings = embeddings.to(device)
            smiles_padded = smiles_padded.to(device)

            tgt_input = smiles_padded[:, :-1]
            tgt_output = smiles_padded[:, 1:]

            output_logits = model(embeddings, tgt_input)
            
            loss = criterion(output_logits.reshape(-1, output_logits.shape[-1]), tgt_output.reshape(-1))
            
            total_loss += loss.item()
            
    return total_loss / len(loader)

def main():
    """Main function to orchestrate the training process."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure the save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Initialize tokenizer and load vocabulary
    tokenizer = TokenizerWrapper()
    tokenizer.load_vocab(VOCAB_PATH)
    pad_idx = tokenizer.vocab['<pad>']
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")

    # 2. Create DataLoaders
    print("Loading data...")
    # train_loader = get_data_loader(HDF5_PATH, tokenizer, split='train', batch_size=BATCH_SIZE)
    # val_loader = get_data_loader(HDF5_PATH, tokenizer, split='val', batch_size=BATCH_SIZE)
    train_dataset = SpectraDataset(hdf5_path=HDF5_PATH, fold='train')
    val_dataset = SpectraDataset(hdf5_path=HDF5_PATH, fold='val')
    print(f"Number of samples in TRAIN set: {len(train_dataset)}")
    print(f"Number of samples in VALIDATION set: {len(val_dataset)}")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True, # Shuffle is important for the training set
        num_workers=2,
        collate_fn=lambda x: collate_fn_with_padding(x, train_dataset.pad_id, tokenizer)
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False, # No need to shuffle validation or test data
        num_workers=2,
        collate_fn=lambda x: collate_fn_with_padding(x, train_dataset.pad_id, tokenizer)
    )
    print("Data loading complete.")

    # 3. Initialize model
    model = SmilesDecoder(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        cond_dim=COND_DIM
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # 4. Setup loss function and optimizer
    # We ignore the padding token when calculating loss
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0  # --- Early Stopping: Initialize counter ---

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save the model checkpoint if it has the best validation loss so far
        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset counter
            checkpoint_path = os.path.join(SAVE_DIR, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Validation loss improved. Model saved to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{PATIENCE}")

        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break  # Exit the training loop

if __name__ == '__main__':
    main()