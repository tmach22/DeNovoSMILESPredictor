import os
import sys
import numpy as np
from pyteomics import mgf
from tqdm import tqdm

def split_mgf_file(input_file, output_prefix, max_size_mb):
    """
    Splits an MGF file into multiple smaller files based on a maximum size.
    
    Args:
        input_file (str): Path to the large .mgf file.
        output_prefix (str): Prefix for the output filenames (e.g., 'chunk_').
        max_size_mb (int): Maximum size of each output file in megabytes.
    """
    if not os.path.exists(input_file):
        print(f"Error: The input file '{input_file}' does not exist.")
        print("Please provide a valid file or run without arguments to create a dummy file.")
        sys.exit(1)

    max_size_bytes = max_size_mb * 1024 * 1024
    
    chunk_index = 0
    output_file = None
    writer = None
    current_chunk = []
    current_size = 0
    
    print(f"--- Splitting '{os.path.basename(input_file)}' into chunks <= {max_size_mb}MB ---")

    try:
        with mgf.read(input_file) as reader:
            for spectrum in tqdm(reader, desc="Splitting file"):
                # Estimate size of current spectrum (a rough estimate)
                spectrum_string = str(spectrum)
                spectrum_size = len(spectrum_string.encode('utf-8'))

                # Check if adding the next spectrum would exceed the size limit
                if current_size + spectrum_size > max_size_bytes and current_chunk:
                    # Write the current chunk to a new file
                    output_filename = f"/bigdata/jianglab/shared/ExploreData/mgf_chunks/{output_prefix}{chunk_index}.mgf"
                    print(f"\nWriting new chunk file: {output_filename}")
                    mgf.write(current_chunk, output_filename)
                    
                    # Reset for the new chunk
                    chunk_index += 1
                    current_chunk = []
                    current_size = 0
                
                # Add the spectrum to the current chunk
                current_chunk.append(spectrum)
                current_size += spectrum_size
        
        # Write any remaining spectra in the last chunk
        if current_chunk:
            output_filename = f"/bigdata/jianglab/shared/ExploreData/mgf_chunks/{output_prefix}{chunk_index}.mgf"
            print(f"\nWriting final chunk file: {output_filename}")
            mgf.write(current_chunk, output_filename)

    except Exception as e:
        print(f"\nAn error occurred during splitting: {e}")
    
    print("\n--- Splitting complete! ---")
    print(f"Created {chunk_index} chunk(s) from the input file.")

if __name__ == '__main__':
    # Default parameters
    default_input_file = "/bigdata/jianglab/shared/ExploreData/data_for_dreams_in_silica/chunk_sd.mgf"
    default_output_prefix = "chunk_sd_"
    default_max_size_mb = 100

    input_file_path = default_input_file
    output_prefix = default_output_prefix
    max_size = default_max_size_mb

    split_mgf_file(input_file_path, output_prefix, max_size)