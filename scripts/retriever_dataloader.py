import torch
import os
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from glob import glob

class ShardedDataset(Dataset):
    """
    Reads data from sharded .pt files. This is robust against file systems
    that are slow with many small files. Includes a simple cache for performance.
    """
    def __init__(self, processed_dir: str):
        self.processed_dir = processed_dir
        
        # Find all shard files regardless of worker ID and sort them.
        shard_files = sorted(glob(os.path.join(processed_dir, 'worker-*_shard-*.pt')))
        shard_files.extend(sorted(glob(os.path.join(processed_dir, 'shard_*.pt'))))

        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {processed_dir}")
            
        self.shard_paths = sorted(list(set(shard_files)))
        
        # Dynamically calculate the total number of samples
        print(f"Calculating total number of samples from {len(self.shard_paths)} shards...")
        self._lengths = [len(torch.load(p, weights_only=False)) for p in self.shard_paths]
        self.num_samples = sum(self._lengths)
        self._cumulative_lengths = [0] + list(np.cumsum(self._lengths))
        
        print(f"Found {self.num_samples} total samples in {processed_dir}.")

        self._cache = {}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Find which shard this index belongs to
        shard_idx = np.searchsorted(self._cumulative_lengths, idx, side='right') - 1
        idx_in_shard = idx - self._cumulative_lengths[shard_idx]

        if self._cache.get("shard_idx") == shard_idx:
            shard_data = self._cache["data"]
        else:
            shard_path = self.shard_paths[shard_idx]
            shard_data = torch.load(shard_path, weights_only=False)
            self._cache["shard_idx"] = shard_idx
            self._cache["data"] = shard_data
            
        return shard_data[idx_in_shard]

# --- NEW: Integrated Testing Code Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test the ShardedDataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--processed_dir", required=True, help="Path to the directory of pre-processed .pt files.")
    args = parser.parse_args()

    print("--- Verification Step 1: Initializing ShardedDataset ---")
    try:
        dataset = ShardedDataset(args.processed_dir)
        print(f"Dataset created successfully with {len(dataset)} samples.")
        assert len(dataset) > 0, "Dataset should not be empty."
    except Exception as e:
        print(f"FAILURE: Error creating dataset: {e}")
        exit()

    print("\n--- Verification Step 2: Inspecting a Single Sample ---")
    try:
        sample = dataset[0]
        print(f"Successfully retrieved one sample of type: {type(sample)}")
        
        print("\n[Verifying Sample Attributes]")
        assert hasattr(sample, 'anchor_embedding'), "Sample missing 'anchor_embedding'"
        assert hasattr(sample, 'positive_graph'), "Sample missing 'positive_graph'"
        assert hasattr(sample, 'negative_graph'), "Sample missing 'negative_graph'"
        print("  - All required attributes are present.")

        assert sample.anchor_embedding.shape == torch.Size([1, 1024]), f"Incorrect anchor shape: {sample.anchor_embedding.shape}"
        print(f"  - Anchor embedding shape is correct: {sample.anchor_embedding.shape}")

    except Exception as e:
        print(f"FAILURE: Error retrieving or inspecting a single sample: {e}")
        exit()
        
    print("\n--- Verification Step 3: Testing with PyG DataLoader ---")
    try:
        # Use the same DataLoader that the training script will use
        test_loader = PyGDataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(test_loader))
        
        print(f"Successfully retrieved one batch of type: {type(batch)}")
        print(f"Number of graphs in batch: {batch.num_graphs}")
        
        assert batch.num_graphs == 4, f"Expected 8 graphs (4 pos + 4 neg), but got {batch.num_graphs}"
        print("  - Correct number of graphs in the batch.")
        
        assert batch.anchor_embedding.shape == torch.Size([4, 1024]), f"Incorrect batched anchor shape: {batch.anchor_embedding.shape}"
        print(f"  - Batched anchor embedding shape is correct: {batch.anchor_embedding.shape}")

    except Exception as e:
        print(f"FAILURE: Error creating DataLoader or fetching a batch: {e}")
        exit()

    print("\n\033[92mSUCCESS: The sharded dataloader is working correctly!\033[0m")