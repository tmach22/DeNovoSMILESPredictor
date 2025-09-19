import h5py
import os
import numpy as np
import traceback

# --- Configuration ---
# Path to your HDF5 file
# Replace this with the actual path to your combined HDF5 file or a chunk file
HDF5_FILE_PATH = '/data/nas-gpu/wang/tmach007/DeNovoSMILESPredictor/test/test_set.hdf5'
# --- Inspection Functions ---

def inspect_hdf5_structure(file_path):
    """
    Inspects and prints the hierarchical structure of an HDF5 file.
    Shows groups, datasets, and their paths.
    """
    print(f"\n--- Inspecting HDF5 File Structure: {os.path.basename(file_path)} ---")
    try:
        with h5py.File(file_path, 'r') as f:
            print("Root Group Contents:")
            # Iterate through top-level items (groups and datasets)
            for name, item in f.items():
                print(f"{name}: {item}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while inspecting structure: {e}")
        traceback.print_exc()

def inspect_fold_column(file_path, column_name='fold'):
    """
    Inspects the unique values and their counts in a specified column (dataset)
    within an HDF5 file, specifically handling byte strings.
    """
    print(f"\n--- Inspecting '{column_name}' column in {os.path.basename(file_path)} ---")
    try:
        with h5py.File(file_path, 'r') as f:
            if column_name in f:
                fold_dataset = f[column_name]
                print(f"Dataset '{column_name}' found. Shape: {fold_dataset.shape}, Dtype: {fold_dataset.dtype}")

                # Read the data from the 'fold' dataset
                # Since the dtype is '|S10' (fixed-length byte string), we need to decode it.
                # Dataset.asstr() is suitable for this.[1, 2]
                fold_values_bytes = fold_dataset[:]
                
                # Decode byte strings to Python strings for readability
                # Using.astype(str) or a list comprehension with.decode()
                # For NumPy >= 2.0, astype('T') is recommended for variable-width strings.[1]
                # For general compatibility, a simple decode is often sufficient for fixed-length byte strings.
                if fold_values_bytes.dtype.kind == 'S': # Check if it's a byte string array
                    fold_values_decoded = np.array([s.decode('utf-8') for s in fold_values_bytes])
                else:
                    fold_values_decoded = fold_values_bytes # Assume it's already in a readable format

                # Get unique values and their counts [3, 4]
                unique_folds, counts = np.unique(fold_values_decoded, return_counts=True)

                print(f"\nUnique values in '{column_name}' column and their counts:")
                for fold, count in zip(unique_folds, counts):
                    print(f"- Fold: '{fold}', Count: {count}")

                # Optionally, print the total number of unique folds (K)
                print(f"\nTotal number of unique folds (K): {len(unique_folds)}")
                if len(unique_folds) > 0:
                    print("This indicates the dataset is likely prepared for K-fold cross-validation.[5]")

                return unique_folds, counts

            else:
                print(f"Error: Column '{column_name}' not found in the HDF5 file.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while inspecting the '{column_name}' column: {e}")


def plot_fold_distribution(folds, counts, output_filename='/rhome/tmach007/bigdata/jianglab/tmach007/DreaMS_Project_TejasMachkar/final_analysis_plots/fold_distribution.png'):
    """
    Creates and saves a bar chart of the fold distribution.
    """
    if folds is None or counts is None:
        print("No data to plot.")
        return

    print(f"\n--- Generating Fold Distribution Plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(folds, counts, color=['#2a9d8f', '#e9c46a', '#f4a261'], edgecolor='black')
    
    ax.set_title('Distribution of Data Across Folds', fontsize=16, fontweight='bold')
    ax.set_xlabel('Fold Name', fontsize=12)
    ax.set_ylabel('Number of Spectra (Frequency)', fontsize=12)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    # Add counts on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{int(yval):,}', va='bottom', ha='center', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved successfully as '{output_filename}'")
    # plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the HDF5 file exists before attempting to inspect
    if not os.path.exists(HDF5_FILE_PATH):
        print(f"The specified HDF5 file does not exist: {HDF5_FILE_PATH}")
        print("Please ensure your embedding generation script has run successfully and created this file.")
        exit()

    # Step 1: Inspect the overall structure of the HDF5 file
    inspect_hdf5_structure(HDF5_FILE_PATH)

    # # Step 2: Inspect the values of the 'fold' column
    # folds, counts = inspect_fold_column(HDF5_FILE_PATH, 'fold')

    # # Step 3: Create and save the bar graph
    # if folds is not None:
    #     plot_fold_distribution(folds, counts)