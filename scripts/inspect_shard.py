import torch
import argparse
from torch_geometric.data import Data

def main(args):
    """
    Loads a single shard file (.pt) and prints a detailed inspection of its
    structure and the contents of the first sample within it.
    """
    print(f"--- Inspecting Shard File: {args.shard_path} ---")

    try:
        # --- THE FIX IS HERE: Set weights_only=False ---
        # This tells PyTorch to trust the file and load the complex
        # torch_geometric.data.Data objects contained within.
        shard_data = torch.load(args.shard_path, weights_only=False)

    except FileNotFoundError:
        print(f"ERROR: File not found at '{args.shard_path}'. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    # --- 1. Top-Level Verification ---
    print("\n[Top-Level Structure]")
    print(f"  - Type of loaded object: {type(shard_data)}")
    if isinstance(shard_data, list):
        print(f"  - Number of samples in this shard: {len(shard_data)}")
    else:
        print("  - WARNING: Expected a list of samples, but got a different type.")
        return

    if not shard_data:
        print("  - WARNING: The shard is empty.")
        return

    # --- 2. Single Sample Verification ---
    first_sample = shard_data[0]
    print("\n[Inspecting the First Sample]")
    print(f"  - Type of a single sample: {type(first_sample)}")

    if not isinstance(first_sample, Data):
        print("  - WARNING: Expected a torch_geometric.data.Data object, but got a different type.")
        return
        
    # --- 3. Attribute Verification ---
    print("\n[Verifying Sample Attributes]")
    
    if hasattr(first_sample, 'anchor_embedding'):
        anchor = first_sample.anchor_embedding
        print(f"  - `anchor_embedding` found:")
        print(f"    - Type: {type(anchor)}")
        print(f"    - Dtype: {anchor.dtype}")
        print(f"    - Shape: {anchor.shape}  <-- Should be [1, 1024]")
    else:
        print("  - `anchor_embedding`: NOT FOUND!")

    if hasattr(first_sample, 'positive_graph'):
        pos_graph = first_sample.positive_graph
        print(f"  - `positive_graph` found:")
        print(f"    - Type: {type(pos_graph)}")
        print(f"    - Node features (x) shape: {pos_graph.x.shape if hasattr(pos_graph, 'x') else 'N/A'}")
        print(f"    - Edge index (edge_index) shape: {pos_graph.edge_index.shape if hasattr(pos_graph, 'edge_index') else 'N/A'}")
    else:
        print("  - `positive_graph`: NOT FOUND!")

    if hasattr(first_sample, 'negative_graph'):
        neg_graph = first_sample.negative_graph
        print(f"  - `negative_graph` found:")
        print(f"    - Type: {type(neg_graph)}")
        print(f"    - Node features (x) shape: {neg_graph.x.shape if hasattr(neg_graph, 'x') else 'N/A'}")
        print(f"    - Edge index (edge_index) shape: {neg_graph.edge_index.shape if hasattr(neg_graph, 'edge_index') else 'N/A'}")
    else:
        print("  - `negative_graph`: NOT FOUND!")
        
    print("\n--- Inspection Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect a pre-processed .pt shard file.")
    parser.add_argument("--shard_path", required=True, help="Path to the shard file (e.g., 'processed_data/train/worker-0_shard-0.pt').")
    args = parser.parse_args()
    main(args)