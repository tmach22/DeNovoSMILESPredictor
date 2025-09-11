import os
import csv
import re
from tqdm import tqdm

# --- Configuration ---
# Path to your input .smi file
INPUT_SMI_FILE = '/bigdata/jianglab/shared/ExploreData/raw_ms_data/BABA.smi' 

# Path for the output .csv file
OUTPUT_CSV_FILE = '/bigdata/jianglab/shared/ExploreData/raw_ms_data/vsmall_test.csv'

def smi_to_csv(input_file, output_file):
    """
    Converts a .smi file with SMILES and ZINC IDs into a .csv file.

    Args:
        input_file (str): The path to the input .smi file.
        output_file (str): The path to the output .csv file.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Converting '{input_file}' to '{output_file}'...")
    
    # Use a regular expression to handle one or more spaces/tabs as a separator
    # This is more robust than using a simple .split()
    whitespace_splitter = re.compile(r'\s+')

    with open(input_file, 'r') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        csv_writer = csv.writer(outfile)
        
        # Write the header row
        csv_writer.writerow(['smiles', 'mol_id'])

        line_count = 0
        for line in tqdm(infile, desc="Processing lines", unit="lines"):
            line = line.strip()
            if not line:
                continue

            parts = whitespace_splitter.split(line, maxsplit=1)
            
            # The script assumes there are always two parts
            if len(parts) == 2:
                smiles, zinc_id = parts
                csv_writer.writerow([smiles, zinc_id])
                line_count += 1
            else:
                print(f"Warning: Skipping malformed line: {line}")
        
    print(f"Conversion complete. Wrote {line_count} entries to '{output_file}'.")

if __name__ == '__main__':
    # Make sure to replace this with your actual file paths
    smi_to_csv(INPUT_SMI_FILE, OUTPUT_CSV_FILE)