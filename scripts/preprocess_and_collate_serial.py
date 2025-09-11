import torch
import h5py
import numpy as np
import random
import argparse
import os
from torch_geometric.data import Data
from tqdm import tqdm

from scaffold_processor import smiles_to_graph
from generate_scaffold_library import get_scaffold

def main(args):
    """
    Processes the entire dataset sequentially and saves the output in sharded .pt files.
    """
    print("--- Starting SERIAL sharded pre-processing ---")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Load shared data ONCE ---
    with open(args.scaffold_library_path, 'r') as f:
        scaffold_library = [line.strip() for line in f if line.strip()]
    scaffold_map = {smi: i for i, smi in enumerate(scaffold_library)}

    print("Pre-creating graph cache (this may take a moment)...")
    graph_cache = {smi: smiles_to_graph(smi) for smi in tqdm(scaffold_library)}
    print("Graph cache created.")

    # --- Step 2: Open HDF5 files and process sequentially ---
    with h5py.File(args.hdf5_path, 'r') as h5_data, \
         h5py.File(args.similarity_matrix_path, 'r') as h5_sim:

        print("Successfully opened HDF5 data and similarity matrix files.")
        sim_matrix = h5_sim['similarity_matrix']
        embeddings = h5_data['embeddings']
        smiles_list = h5_data['smiles']
        num_total_samples = len(smiles_list)
        print(f"Found {num_total_samples} total samples to process.")

        shard_num = 0
        samples_in_shard = []
        total_valid_processed = 0

        # --- The main sequential loop over all samples ---
        for i in tqdm(range(num_total_samples), desc="Processing Samples"):
            original_smi = smiles_list[i].decode('utf-8')
            positive_scaffold_smi = get_scaffold(original_smi)
            print(f"Original SMILES: {original_smi} --> Scaffold: {positive_scaffold_smi}")

            # Filter out spectra that don't produce a valid scaffold
            if positive_scaffold_smi not in scaffold_map:
                continue
            
            # If we reach here, the sample is valid
            total_valid_processed += 1
            anchor_embedding = torch.from_numpy(embeddings[i])
            pos_scaffold_idx = scaffold_map.get(positive_scaffold_smi)

            # Perform biased negative sampling
            sim_vector = sim_matrix[pos_scaffold_idx, :].copy()
            sim_vector[pos_scaffold_idx] = 0
            
            if np.sum(sim_vector) > 0:
                prob_vector = np.exp(sim_vector / 0.1)
                prob_vector /= np.sum(prob_vector)
                negative_scaffold_smi = np.random.choice(scaffold_library, p=prob_vector)
            else:
                negative_scaffold_smi = random.choice(scaffold_library)
            
            while negative_scaffold_smi == positive_scaffold_smi:
                negative_scaffold_smi = random.choice(scaffold_library)

            # Create the final PyG Data object
            data_obj = Data(
                anchor_embedding=anchor_embedding.unsqueeze(0),
                positive_graph=graph_cache[positive_scaffold_smi],
                negative_graph=graph_cache[negative_scaffold_smi]
            )
            samples_in_shard.append(data_obj)

            # If the shard is full, save it to a file
            if len(samples_in_shard) == args.shard_size:
                shard_filename = f'shard_{shard_num}.pt' # Filename is simpler now
                torch.save(samples_in_shard, os.path.join(args.output_dir, shard_filename))
                shard_num += 1
                samples_in_shard = [] # Reset for the next shard

        # Save any remaining samples in the final shard
        if samples_in_shard:
            shard_filename = f'shard_{shard_num}.pt'
            torch.save(samples_in_shard, os.path.join(args.output_dir, shard_filename))

    print(f"\nSUCCESS: Serial pre-processing complete. Processed {total_valid_processed} valid samples.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-process the dataset serially into sharded .pt files.")
    parser.add_argument("--hdf5_path", required=True)
    parser.add_argument("--scaffold_library_path", required=True)
    parser.add_argument("--similarity_matrix_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard_size", type=int, default=50000)
    # The --num_workers argument is no longer needed
    args = parser.parse_args()
    main(args)