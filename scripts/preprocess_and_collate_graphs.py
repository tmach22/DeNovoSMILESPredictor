import torch
import h5py
import numpy as np
import random
import argparse
import os
import torch.multiprocessing as mp
from torch_geometric.data import Data
from tqdm import tqdm
from functools import partial

from scaffold_processor import smiles_to_graph
from generate_scaffold_library import get_scaffold

# --- This is the function that each worker process will execute ---
def process_chunk(indices, worker_id, args, scaffold_map, scaffold_library):
    """
    Processes a specific chunk of the dataset. This function is executed
    by each parallel worker.
    """
    start_idx, end_idx = indices

    # --- NEW: Each worker now builds its own graph_cache ---
    # This avoids the massive inter-process communication that causes memory map errors.
    # The list of SMILES is small enough to be passed without issue.
    graph_cache = {smi: smiles_to_graph(smi) for smi in scaffold_library}
    
    # Each worker must open its own handle to the HDF5 files for parallel access
    with h5py.File(args.hdf5_path, 'r') as h5_data, \
         h5py.File(args.similarity_matrix_path, 'r') as h5_sim:
        
        sim_matrix = h5_sim['similarity_matrix']
        embeddings = h5_data['embeddings']
        smiles_list = h5_data['smiles']

        shard_num = 0
        samples_in_shard = []

        # The worker's main loop over its assigned indices
        for i in range(start_idx, end_idx):
            anchor_embedding = torch.from_numpy(embeddings[i])
            original_smi = smiles_list[i].decode('utf-8')
            
            positive_scaffold_smi = get_scaffold(original_smi)
            if positive_scaffold_smi not in scaffold_map:
                continue

            pos_scaffold_idx = scaffold_map.get(positive_scaffold_smi)
            sim_vector = sim_matrix[pos_scaffold_idx, :]
            sim_vector[pos_scaffold_idx] = 0
            
            if np.sum(sim_vector) > 0:
                prob_vector = np.exp(sim_vector / 0.1)
                prob_vector /= np.sum(prob_vector)
                negative_scaffold_smi = np.random.choice(scaffold_library, p=prob_vector)
            else:
                negative_scaffold_smi = random.choice(scaffold_library)
            
            while negative_scaffold_smi == positive_scaffold_smi:
                negative_scaffold_smi = random.choice(scaffold_library)

            data_obj = Data(
                anchor_embedding=anchor_embedding.unsqueeze(0),
                positive_graph=graph_cache[positive_scaffold_smi],
                negative_graph=graph_cache[negative_scaffold_smi]
            )
            samples_in_shard.append(data_obj)

            if len(samples_in_shard) == args.shard_size:
                shard_filename = f'worker-{worker_id}_shard-{shard_num}.pt'
                torch.save(samples_in_shard, os.path.join(args.output_dir, shard_filename))
                shard_num += 1
                samples_in_shard = []

        if samples_in_shard:
            shard_filename = f'worker-{worker_id}_shard-{shard_num}.pt'
            torch.save(samples_in_shard, os.path.join(args.output_dir, shard_filename))
            print(f"Worker {worker_id} saved final shard with {len(samples_in_shard)} samples.")

        print(f"Worker {worker_id} processed samples {start_idx} to {end_idx}")
            
    return end_idx - start_idx

def main(args):
    print("--- Starting PARALLEL pre-processing ---")
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        mp.set_sharing_strategy('file_system')
        mp.get_context("fork")
        print("Set multiprocessing sharing strategy to 'file_system'")
    except RuntimeError:
        print("Sharing strategy already set or not supported. Continuing.")
        pass

    # --- The main process now only loads small, easily passable data ---
    with open(args.scaffold_library_path, 'r') as f:
        scaffold_library = [line.strip() for line in f if line.strip()]
    scaffold_map = {smi: i for i, smi in enumerate(scaffold_library)}
    
    # The large graph_cache is NO LONGER created here.

    with h5py.File(args.hdf5_path, 'r') as f:
        num_total_samples = len(f['smiles'])
    print(f"Found {num_total_samples} total samples to process.")

    chunk_size = int(np.ceil(num_total_samples / args.num_workers))
    chunks = [(i, min(i + chunk_size, num_total_samples)) for i in range(0, num_total_samples, chunk_size)]

    worker_func = partial(
        process_chunk, 
        args=args, 
        # graph_cache is no longer passed
        scaffold_map=scaffold_map, 
        scaffold_library=scaffold_library
    )
    
    print(f"Launching {args.num_workers} worker processes...")
    with mp.Pool(args.num_workers) as pool:
        results = list(tqdm(pool.starmap(worker_func, zip(chunks, range(args.num_workers))), total=len(chunks), desc="Processing Chunks"))

    total_processed = sum(results)
    print(f"\nSUCCESS: Parallel pre-processing complete. Processed {total_processed} samples.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-process the dataset in parallel into sharded .pt files.")
    parser.add_argument("--hdf5_path", required=True)
    parser.add_argument("--scaffold_library_path", required=True)
    parser.add_argument("--similarity_matrix_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    main(args)