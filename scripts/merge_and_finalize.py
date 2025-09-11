import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import argparse
import sys

# Mass of a proton for [M+H]+ calculation
PROTON_MASS = 1.007276

def calculate_precursor_mz(smiles, adduct_type):
    """Calculates the precursor m/z for a given SMILES and adduct type."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    try:
        exact_mass = Descriptors.ExactMolWt(mol)
        
        if adduct_type == "[M+H]+":
            precursor_mz = exact_mass + PROTON_MASS
        # Add other adduct calculations here if needed
        else:
            # Return None for unsupported adduct types
            return None
            
        return precursor_mz
    except Exception:
        # Catch any other potential calculation errors from RDKit
        return None

def main(smiles_file, spectra_file, output_file):
    """
    Merges original SMILES with MassFormer output, adds precursor_mz, cleans, and saves.
    """
    print(f"Reading original SMILES input from: {smiles_file}")
    try:
        # Assuming the input file has headers 'mol_id' and 'smiles'
        smiles_df = pd.read_csv(smiles_file)
    except FileNotFoundError:
        print(f"Error: Original SMILES file not found at '{smiles_file}'.")
        sys.exit(1)

    print(f"Reading MassFormer output from: {spectra_file}")
    try:
        spectra_df = pd.read_csv(spectra_file)
    except FileNotFoundError:
        print(f"Error: MassFormer output file not found at '{spectra_file}'.")
        sys.exit(1)

    print(f"Found {len(smiles_df)} original molecules and {len(spectra_df)} predicted spectra.")
    
    # --- Step 1: Merge the two dataframes using 'mol_id' as the key ---
    # This ensures the original, trusted SMILES is used.
    print("Merging datasets on 'mol_id'...")
    merged_df = pd.merge(spectra_df, smiles_df, on='mol_id', how='left', suffixes=('_pred', '_original'))
    
    # Use the original SMILES for calculations, handling potential column name conflicts
    if 'smiles_original' in merged_df.columns:
        merged_df['smiles'] = merged_df['smiles_original']
    
    print("Calculating precursor m/z values...")
    merged_df['precursor_mz'] = merged_df.apply(
        lambda row: calculate_precursor_mz(row['smiles'], row['prec_type']),
        axis=1
    )
    
    # --- Step 2: Clean the data ---
    rows_before_drop = len(merged_df)
    merged_df.dropna(subset=['precursor_mz'], inplace=True)
    rows_after_drop = len(merged_df)
    
    num_discarded = rows_before_drop - rows_after_drop
    if num_discarded > 0:
        print(f"Discarded {num_discarded} rows due to invalid SMILES or calculation errors.")
    
    # --- Step 3: Format the final output ---
    final_columns = [
        'spec_id',
        'mol_id', 
        'group_id', 
        'prec_type', 
        'prec_mz', 
        'nce', 
        'inst_type', 
        'frag_mode', 
        'spec_type', 
        'ion_mode', 
        'peaks', 
        'smiles', 
        'precursor_mz'
    ]

    
    # Check if all necessary columns exist in the merged dataframe
    if all(col in merged_df.columns for col in final_columns):
        final_df = merged_df[final_columns]
        
        print(f"Saving {len(final_df)} cleaned rows to: {output_file}")
        final_df.to_csv(output_file, index=False)
        
        print("\nProcessing complete.")
        print(f"Final dataset saved to '{output_file}'")
        print("\nPreview of the final data structure:")
        print(final_df.head())
    else:
        print("Error: Could not create the final dataframe. One or more required columns are missing.")
        print(f"Expected columns: {final_columns}")
        print(f"Available columns after merge: {merged_df.columns.tolist()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge MassFormer output with original SMILES and add precursor m/z.")
    parser.add_argument(
        "-s", "--smiles_input", 
        required=True,
        help="Path to the original CSV file that was input to MassFormer (e.g., zinc_for_prediction.csv)."
    )
    parser.add_argument(
        "-p", "--predictions_input", 
        required=True,
        help="Path to the output CSV file from MassFormer (e.g., predicted_spectra.csv)."
    )
    parser.add_argument(
        "-o", "--output", 
        required=True,
        help="Path for the new, final cleaned and merged CSV file."
    )
    args = parser.parse_args()
    
    main(args.smiles_input, args.predictions_input, args.output)