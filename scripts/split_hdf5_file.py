import h5py
import os
import math
from tqdm import tqdm

# --- Configuration ---
# Path to your large HDF5 file
INPUT_HDF5_FILE = '/bigdata/jianglab/shared/ExploreData/large_data_split/train_set.hdf5'

# Directory where the smaller chunk files will be saved
OUTPUT_DIR = '/bigdata/jianglab/shared/ExploreData/raw_ms_data/'

# Approximate target size for each chunk in Gigabytes.
# Note: The script splits by rows, so the final file sizes will be an approximation.
# This is due to the variable-length string datasets.
TARGET_CHUNK_SIZE_GB = 15

def split_hdf5_file(input_file, output_dir, target_chunk_size_gb):
    """
    Splits a large HDF5 file into smaller, row-based chunks.

    Args:
        input_file (str): Path to the input HDF5 file.
        output_dir (str): Directory to save the chunk files.
        target_chunk_size_gb (float): Target size of each chunk file in GB.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Opening file: {input_file}")
    
    with h5py.File(input_file, 'r') as original_file:
        
        # Get the names of all datasets in the root group
        dataset_names = list(original_file.keys())
        
        if not dataset_names:
            print("Error: No datasets found in the HDF5 file.")
            return

        # Get the total number of entries from the first dataset
        total_entries = original_file[dataset_names[0]].shape[0]
        
        # --- Estimate row chunk size based on file size ---
        original_file_size_gb = os.path.getsize(input_file) / (1024**3)
        num_chunks = math.ceil(original_file_size_gb / target_chunk_size_gb)
        
        # Calculate the number of rows per chunk
        rows_per_chunk = math.ceil(total_entries / num_chunks)

        print(f"File size: {original_file_size_gb:.2f} GB")
        print(f"Total entries: {total_entries}")
        print(f"Estimated number of chunks: {num_chunks}")
        print(f"Rows per chunk: {rows_per_chunk}")

        chunk_counter = 0
        for i in tqdm(range(0, total_entries, rows_per_chunk), desc="Splitting HDF5 file"):
            start_row = i
            end_row = min(i + rows_per_chunk, total_entries)
            
            chunk_filename = f"{os.path.basename(input_file).split('.')[0]}_chunk_{chunk_counter}.hdf5"
            output_path = os.path.join(output_dir, chunk_filename)
            
            with h5py.File(output_path, 'w') as chunk_file:
                
                # Copy attributes from the original root group
                for attr_name, attr_value in original_file.attrs.items():
                    chunk_file.attrs[attr_name] = attr_value

                # Add custom metadata for the chunk
                chunk_file.attrs['source_file'] = os.path.basename(input_file)
                chunk_file.attrs['start_row'] = start_row
                chunk_file.attrs['end_row'] = end_row
                
                for ds_name in dataset_names:
                    # Read a slice of the dataset
                    data_slice = original_file[ds_name][start_row:end_row]
                    
                    # Create a new dataset in the chunk file and write the data
                    chunk_file.create_dataset(ds_name, data=data_slice, compression="gzip", compression_opts=9)
            
            chunk_counter += 1

    print(f"\nSplitting complete. Created {chunk_counter} files in '{output_dir}'.")

if __name__ == '__main__':
    # --- Uncomment the following lines and set your file path to run ---
    # input_file_path = "/path/to/your/large/file.hdf5"
    # output_directory = "./hdf5_chunks"
    # split_hdf5_file(input_file_path, output_directory, TARGET_CHUNK_SIZE_GB)

    # Example call with placeholder paths
    split_hdf5_file(INPUT_HDF5_FILE, OUTPUT_DIR, TARGET_CHUNK_SIZE_GB)