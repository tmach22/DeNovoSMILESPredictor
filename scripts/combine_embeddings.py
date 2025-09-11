import h5py
import os
import argparse
import glob
from tqdm import tqdm

def combine_hdf5_files(input_dir, output_path):
    """
    Combines multiple HDF5 files with identical structures into a single HDF5 file.

    Args:
        input_dir (str): Directory containing the HDF5 files to be merged.
        output_path (str): Path for the final combined HDF5 file.
    """
    # Use glob to find all HDF5 files in the input directory
    hdf5_files = sorted(glob.glob(os.path.join(input_dir, '*.hdf5')))
    
    if not hdf5_files:
        print(f"Error: No HDF5 files found in directory '{input_dir}'.")
        return

    print(f"Found {len(hdf5_files)} HDF5 files to combine.")

    # Use the first file to define the structure of the combined file
    first_file_path = hdf5_files[0]
    
    with h5py.File(output_path, 'w') as f_out:
        print(f"Creating combined file structure based on '{os.path.basename(first_file_path)}'...")
        
        with h5py.File(first_file_path, 'r') as f_first:
            dataset_keys = list(f_first.keys())

            print(f"Datasets to be combined: {dataset_keys}")
            
            # Initialize empty, resizable datasets in the output file
            for key in tqdm(dataset_keys, desc="Initializing datasets"):
                source_dset = f_first[key]

                # --- CORRECTED SECTION ---
                source_shape = source_dset.shape
                
                # Start with an empty shape along the first axis
                initial_shape = (0,) + source_shape[1:]
                
                # Define maxshape for resizability (unlimited along the first axis) [1]
                maxshape = (None,) + source_shape[1:]
                
                # Create the dataset with chunking enabled (required for resizing)
                f_out.create_dataset(
                    name=key,
                    shape=tuple(initial_shape),
                    maxshape=tuple(maxshape),
                    dtype=source_dset.dtype,
                    chunks=True 
                )

        # Append data from each file into the combined file
        print("\nAppending data from individual files...")
        for file_path in tqdm(hdf5_files, desc="Combining files"):
            with h5py.File(file_path, 'r') as f_in:
                for key in dataset_keys:
                    dset_in = f_in[key]
                    dset_out = f_out[key]
                    
                    # Get current and new sizes
                    current_rows = dset_out.shape[0]
                    new_rows = dset_in.shape[0]

                    # 2. Calculate the new total number of rows
                    new_total_rows = current_rows + new_rows

                    # 3. Create a list from the current shape
                    new_shape_list = list(dset_out.shape)

                    # 4. Update the row count in the list
                    new_shape_list[0] = new_total_rows
                    
                    # Resize the output dataset to accommodate the new data [4]
                    # 5. Convert the list back to a tuple
                    new_shape = tuple(new_shape_list)
                    
                    dset_out.resize(new_shape)
                    
                    # Append the new data
                    dset_out[current_rows:] = dset_in[:]
                    
    print(f"\nSuccessfully combined all files into '{output_path}'")
    with h5py.File(output_path, 'r') as f_final:
        print("\nFinal combined dataset info:")
        total_spectra = 0
        for key in f_final.keys():
            print(f"  - Dataset '{key}': shape {f_final[key].shape}, dtype {f_final[key].dtype}")
            if total_spectra == 0:
                total_spectra = f_final[key].shape
        print(f"\nTotal spectra combined: {total_spectra}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combine multiple DreaMS HDF5 files into a single file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing the individual HDF5 files to be merged."
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path for the final, combined HDF5 file."
    )

    args = parser.parse_args()
    combine_hdf5_files(args.input_dir, args.output_file)