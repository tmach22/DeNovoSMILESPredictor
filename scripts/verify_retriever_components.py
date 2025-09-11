# verify_retriever_components.py
import torch
import h5py
import numpy as np
from torch_geometric.loader import DataLoader as GraphDataLoader

# Import the components we have just built
from scaffold_processor import smiles_to_graph, get_atom_features
from retriever_model import ContrastiveRetriever

# --- Configuration ---
# IMPORTANT: Update these paths to your actual file locations
TRAIN_HDF5_PATH = "/bigdata/jianglab/shared/ExploreData/data_splits/train_set.hdf5"
SCAFFOLD_LIBRARY_PATH = "/bigdata/jianglab/shared/ExploreData/scaffold_library/scaffold_library.txt"

def main():
    """
    This script verifies that the scaffold processor and retriever model
    are working together correctly before we build the full training pipeline.
    """
    print("--- Verification Step 1: Loading Sample Data ---")
    
    # Load a few DreaMS embeddings from the training file
    with h5py.File(TRAIN_HDF5_PATH, 'r') as f:
        sample_embeddings = torch.tensor(f['embeddings'][:4], dtype=torch.float32)
    print(f"Loaded {sample_embeddings.shape} sample DreaMS embeddings.")

    # Load a few scaffold SMILES from our generated library
    with open(SCAFFOLD_LIBRARY_PATH, 'r') as f:
        sample_scaffolds_smi = [next(f).strip() for _ in range(4)]
    print("Loaded sample scaffold SMILES:", sample_scaffolds_smi)

    print("\n--- Verification Step 2: Processing Scaffolds into Graphs ---")
    
    # Convert the scaffold SMILES into PyTorch Geometric Data objects
    scaffold_graphs = [smiles_to_graph(smi) for smi in sample_scaffolds_smi]
    
    # Check if conversion was successful
    if any(g is None for g in scaffold_graphs):
        print("Error: Failed to convert one or more scaffold SMILES to graphs.")
        return
        
    print("Successfully converted SMILES to PyG graph objects.")
    print("Example graph object:", scaffold_graphs)

    print("\n--- Verification Step 3: Batching Graphs with PyG DataLoader ---")
    
    # PyTorch Geometric has its own DataLoader that knows how to correctly
    # collate individual graph objects into a single large "batch" graph.
    graph_loader = GraphDataLoader(scaffold_graphs, batch_size=4, shuffle=False)
    batched_graphs = next(iter(graph_loader))
    
    print("Successfully created a batch of graphs.")
    print("Batched graph object:", batched_graphs)
    print(f"Batch contains {batched_graphs.num_graphs} individual graphs.")

    print("\n--- Verification Step 4: Model Forward Pass ---")
    
    # Determine the node feature dimension from the first graph object
    node_feature_dim = scaffold_graphs[0].num_node_features
    print(f"Detected node feature dimension: {node_feature_dim}")

    # Instantiate the main retriever model
    model = ContrastiveRetriever(
        dreams_dim=1024,
        node_feature_dim=node_feature_dim,
        shared_embedding_dim=256
    )
    print("ContrastiveRetriever model instantiated successfully.")

    # Perform a single forward pass. We use the same batch of graphs for both
    # the "positive" and "negative" inputs for this simple test.
    try:
        anchor_proj, positive_proj, negative_proj = model(
            dreams_embedding=sample_embeddings,
            positive_scaffold_graph=batched_graphs,
            negative_scaffold_graph=batched_graphs
        )
        print("Model forward pass executed without errors.")
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        return

    print("\n--- Verification Step 5: Checking Output Shapes ---")
    
    expected_shape = torch.Size([4, 256]) # [batch_size, shared_embedding_dim]
    
    print(f"Anchor projection shape: {anchor_proj.shape} (Expected: {expected_shape})")
    print(f"Positive projection shape: {positive_proj.shape} (Expected: {expected_shape})")
    print(f"Negative projection shape: {negative_proj.shape} (Expected: {expected_shape})")

    if anchor_proj.shape == expected_shape and \
       positive_proj.shape == expected_shape and \
       negative_proj.shape == expected_shape:
        print("\nSUCCESS: All components are working together correctly!")
    else:
        print("\nFAILURE: Output shapes are incorrect. Please check model dimensions.")

if __name__ == '__main__':
    main()