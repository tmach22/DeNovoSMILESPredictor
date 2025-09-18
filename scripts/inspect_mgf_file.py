import os
import sys
import numpy as np
from pyteomics import mgf

def create_dummy_mgf(file_path):
    """
    Creates a small, dummy .mgf file for demonstration purposes.
    This allows the script to be run immediately.
    """
    mgf_content = """
BEGIN IONS
TITLE=Spectrum 1
PEPMASS=500.25 1000.0
CHARGE=2+
RTINSECONDS=10.5
200.1 500
300.2 750
400.3 1200
END IONS

BEGIN IONS
TITLE=Spectrum 2
PEPMASS=650.5 500.0
CHARGE=3+
RTINSECONDS=20.0
150.0 300
250.0 600
350.0 900
450.0 1100
550.0 800
END IONS
"""
    try:
        with open(file_path, "w") as f:
            f.write(mgf_content)
        print(f"Created dummy .mgf file at: {file_path}\n")
    except IOError as e:
        print(f"Error creating dummy file: {e}")
        sys.exit(1)


def analyze_mgf_file(file_path):
    """
    Analyzes an .mgf file and provides key statistics.

    Args:
        file_path (str): The path to the .mgf file.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        # If the file doesn't exist, create a dummy file for the user to try again
        create_dummy_mgf(file_path)
        print("A dummy .mgf file has been created. Please run the script again.")
        sys.exit(1)

    print(f"--- Analyzing MGF file: {os.path.basename(file_path)} ---")
    
    # Use lists to collect data for analysis
    pepmass_list = []
    charge_list = []
    num_peaks_list = []
    
    first_spectrum = None
    spectrum_count = 0
    
    try:
        # Use pyteomics to efficiently read the MGF file
        # The MGF parser returns a generator of dictionaries,
        # where each dictionary is one spectrum.
        for spectrum in mgf.read(file_path):
            spectrum_count += 1
            
            # Save the first spectrum for plotting later
            if first_spectrum is None:
                first_spectrum = spectrum

            # Extract data from the spectrum's metadata
            pepmass = spectrum['params'].get('pepmass')
            charge = spectrum['params'].get('charge')
            
            if pepmass:
                # The 'pepmass' can be a list or a single float
                if isinstance(pepmass, list) and len(pepmass) > 0:
                    pepmass_list.append(pepmass[0])
                elif isinstance(pepmass, (float, int)):
                    pepmass_list.append(pepmass)
            
            if charge:
                # The 'charge' can be a list or a single float
                if isinstance(charge, list) and len(charge) > 0:
                    charge_list.append(charge[0])
                elif isinstance(charge, (float, int)):
                    charge_list.append(charge)

            # Get the number of peaks (m/z-intensity pairs)
            num_peaks = len(spectrum['m/z array'])
            num_peaks_list.append(num_peaks)

    except Exception as e:
        print(f"An error occurred while reading the MGF file: {e}")
        return

    # --- Displaying Analysis Results ---
    print("\n--- Summary Statistics ---")
    print(f"Total number of spectra: {spectrum_count:,}")

    if pepmass_list:
        print(f"Average precursor m/z: {np.mean(pepmass_list):.4f}")
        print(f"Median precursor m/z: {np.median(pepmass_list):.4f}")
    
    if num_peaks_list:
        print(f"Average peaks per spectrum: {np.mean(num_peaks_list):.2f}")
        print(f"Min peaks per spectrum: {np.min(num_peaks_list)}")
        print(f"Max peaks per spectrum: {np.max(num_peaks_list)}")
        
    if charge_list:
        # We can use a Counter to show the distribution of charge states
        from collections import Counter
        charge_counts = Counter(charge_list)
        print("\nCharge state distribution:")
        for charge, count in sorted(charge_counts.items()):
            print(f"  Charge {charge}: {count} spectra")
    
    # --- Visualization ---
    if first_spectrum:
        print("\n--- Listing parameters for the first spectrum ---")
        print("All columns/parameters found:")
        for key, value in first_spectrum['params'].items():
            print(f"  {key}: {value}")

if __name__ == '__main__':
    # Default file path
    default_file_path = "/bigdata/jianglab/shared/ExploreData/data_for_dreams_in_silica/chunk_aj.mgf"
    
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        file_path_to_analyze = sys.argv[1]
    else:
        file_path_to_analyze = default_file_path

    # Run the analysis
    analyze_mgf_file(file_path_to_analyze)