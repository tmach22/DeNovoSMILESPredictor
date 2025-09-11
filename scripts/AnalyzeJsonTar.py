import tarfile
import os
import pandas as pd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import numpy as np

def extract_tar_file(tar_file_path, extract_path):
    """
    Extracts a tar archive to a specified directory.
    
    Args:
        tar_file_path (str): The path to the input .tar file.
        extract_path (str): The directory where contents will be extracted.
    """
    print(f"Starting extraction of '{tar_file_path}' to '{extract_path}'...")
    try:
        # Check if the tar file exists
        if not os.path.exists(tar_file_path):
            raise FileNotFoundError(f"Error: The file '{tar_file_path}' does not exist.")
            
        # Open and extract the tar file
        with tarfile.open(tar_file_path, 'r') as tar:
            tar.extractall(path=extract_path)
        print("Extraction complete.")
        
    except tarfile.ReadError as e:
        print(f"Error reading tar file: {e}")
        return False
    except FileNotFoundError as e:
        print(e)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return False
    return True


def analyze_feather_files(data_directory):
    """
    Reads all Feather files from a directory and performs basic analysis.
    
    Args:
        data_directory (str): The directory containing the Feather files.
    """
    print(f"Starting analysis of Feather files in '{data_directory}'...")
    
    # Find all feather files in the directory
    feather_files = [f for f in os.listdir(data_directory) if f.endswith('.feather')]
    
    if not feather_files:
        print("No .feather files found in the specified directory. Analysis aborted.")
        return

    scan_counter = 1
    output_data_directory = os.path.join(data_directory, 'json_outputs')

    if not os.path.exists(output_data_directory):
        os.makedirs(output_data_directory)

    output_dataframe = pd.DataFrame()

    for file_name in feather_files:
        file_path = os.path.join(data_directory, file_name)
        try:
            # Read the feather file into a pandas DataFrame
            df = pd.read_feather(file_path)
            # all_data.append(df)

            print(f"\n--- Analyzing '{file_name}' ---")

            # Create a list of [mz, intensity] pairs for the 'peaks' key
            # We use zip to efficiently iterate through both columns at once
            peaks_list = df.apply(
                lambda row: [list(pair) for pair in zip(row['fragment_mz'], row['fragment_intensities'])],
                axis=1
            )

            split_name = file_name.split('_')[0]

            temp_df = pd.DataFrame()

            temp_df['scan'] = np.arange(scan_counter, scan_counter + len(df))
            temp_df['identifier'] = df['spectrum_id']
            temp_df['precursor_mz'] = df['precursor_mz']
            temp_df['adduct'] = df['adduct']
            temp_df['energy'] = df['collision_energy']
            temp_df['SMILES'] = df['smiles']
            temp_df['peaks'] = peaks_list
            temp_df['split'] = split_name

            scan_counter += len(df)
            output_dataframe = pd.concat([output_dataframe, temp_df], ignore_index=True)

        except Exception as e:
            print(f"Error processing file '{file_name}': {e}")

        # Define the output file path for the JSON file
        output_file_name = "combined_json_file" + '.json'
        output_file_path = os.path.join(output_data_directory, output_file_name)

        # Save the DataFrame to a JSON file
        output_dataframe.to_json(output_file_path, indent=4, orient='records')

        print(f"Successfully processed '{file_name}' and saved to '{output_file_path}'.")

    print("\nAnalysis complete.")

if __name__ == "__main__":
    # Define file and directory paths
    tar_file_name = '/bigdata/jianglab/shared/ExploreData/for_tejas.tar'
    extract_folder_name = '/bigdata/jianglab/shared/ExploreData/extracted_data_json'

    # Extract the tar file
    if extract_tar_file(tar_file_name, extract_folder_name):
        # The tar file contains a directory, so we need to point to that.
        data_path = os.path.join(extract_folder_name, 'for_tejas')

        print(f"Reading JSON files from '{data_path}'...")

        # Analyze the extracted feather files
        analyze_feather_files(data_path)