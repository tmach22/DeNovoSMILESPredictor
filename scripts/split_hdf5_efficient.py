import h5py
import numpy as np
import argparse
from tqdm import tqdm
import sys

def sample_hdf5_multi_dataset(source_path, dest_path, sample_size, batch_size=10000):
    """
    Randomly samples entries from a large HDF5 file with multiple datasets,
    ensuring consistent sampling across all datasets.

    This function uses a sorted-index batching strategy to transform the I/O
    pattern from random to sequential, dramatically improving performance and
    maintaining a low memory footprint.

    Args:
        source_path (str): Path to the source HDF5 file.
        dest_path (str): Path to the destination HDF5 file.
        sample_size (int): The number of samples to extract.
        batch_size (int): The number of samples to process in each batch.
    """
    try:
        # === Stage 1: Index Generation and Metadata Collection ===
        print(f"Opening source file to gather metadata: {source_path}")
        with h5py.File(source_path, 'r') as f_source:
            dataset_keys = list(f_source.keys())
            if not dataset_keys:
                print("Error: No datasets found in the source file.")
                sys.exit(1)
            
            print(f"Found datasets: {dataset_keys}")

            # --- CORRECTED LINE: Use the first dataset key (a string) as a reference ---
            ref_dset_name = dataset_keys[0]
            # --- CORRECTED LINE: Get the number of records (the first dimension) ---
            total_records = f_source[ref_dset_name].shape[0]

            if sample_size > total_records:
                print(f"Error: Sample size ({sample_size}) is larger than the total number of records ({total_records}).")
                sys.exit(1)

            # Store metadata (shape, dtype) for each dataset
            metadata = {}
            for key in dataset_keys:
                dset = f_source[key]
                # Check if the number of records matches the reference
                if dset.shape[0]!= total_records:
                    print(f"Warning: Dataset '{key}' has a different length ({dset.shape}) than reference dataset '{ref_dset_name}' ({total_records}).")
                
                record_shape = dset.shape[1:] if len(dset.shape) > 1 else ()
                metadata[key] = {'shape': record_shape, 'dtype': dset.dtype}

            print(f"\nGenerating {sample_size} unique random indices...")
            # Generate unique random indices without replacement
            indices = np.random.choice(total_records, size=sample_size, replace=False)
            
            print("Sorting indices for sequential-like reading...")
            # Sort indices to make disk access pattern sequential
            indices.sort()

        # === Stage 2: Batched Data Transfer ===
        print(f"Creating destination file: {dest_path}")
        with h5py.File(source_path, 'r') as f_source, h5py.File(dest_path, 'w') as f_dest:
            # Create all destination datasets upfront, making them resizable
            dest_datasets = {}
            for key, meta in metadata.items():
                dest_datasets[key] = f_dest.create_dataset(
                    key,
                    shape=(0,) + meta['shape'],
                    maxshape=(None,) + meta['shape'],
                    dtype=meta['dtype'],
                    chunks=True
                )
            
            print(f"Starting data transfer with batch size {batch_size}...")
            num_batches = int(np.ceil(sample_size / batch_size))
            
            # The main loop iterates through batches of indices
            with tqdm(total=sample_size, unit='records', desc="Sampling Progress") as pbar:
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, sample_size)
                    batch_indices = indices[start_idx:end_idx]
                    
                    # For each batch, copy data for all datasets
                    for key in dataset_keys:
                        source_dset = f_source[key]
                        dest_dset = dest_datasets[key]
                        
                        # Use fancy indexing to read the batch of data efficiently
                        batch_data = source_dset[batch_indices]
                        
                        # Get current size, resize, and append the new data
                        current_size = dest_dset.shape[0]
                        dest_dset.resize(current_size + len(batch_data), axis=0)
                        dest_dset[current_size:] = batch_data
                    
                    pbar.update(len(batch_indices))

        print("\nSampling complete.")
        print(f"Successfully saved {sample_size} records to {dest_path}")

    except FileNotFoundError:
        print(f"Error: Source file not found at {source_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Memory-efficient random sampling of a large HDF5 file with multiple datasets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--source_file',
        required=True,
        help="Path to the source HDF5 file."
    )
    parser.add_argument(
        '--dest_file',
        required=True,
        help="Path for the new output HDF5 file."
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        required=True,
        help="The number of records to randomly sample."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10000,
        help="Number of records to process in each batch (default: 10000)."
    )

    args = parser.parse_args()
    
    sample_hdf5_multi_dataset(
        args.source_file,
        args.dest_file,
        args.sample_size,
        args.batch_size
    )