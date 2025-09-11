import matplotlib.pyplot as plt
import umap
from dreams.api import dreams_embeddings
import os

# Define the path to the example.mgf file
# This file should be located in your DreaMS installation directory,
# typically under 'data/examples/'
# You might need to adjust this path based on where you cloned the DreaMS repository
# For instance, if you are running this script from the root of the DreaMS repo:
example_mgf_path = 'data/examples/example_5_spectra.mgf'

# --- Step 1: Compute DreaMS Embeddings ---
print(f"Loading and computing DreaMS embeddings for {example_mgf_path}...")
try:
    # The dreams_embeddings function will load the pre-trained model and process the spectra
    # The output 'embs' will be a NumPy array where each row is a 1024-dimensional embedding
    embs = dreams_embeddings(example_mgf_path)
    print(f"Successfully computed {embs.shape} embeddings, each {embs.shape[1]}-dimensional.")
except FileNotFoundError:
    print(f"Error: The file '{example_mgf_path}' was not found.")
    print("Please ensure you have cloned the DreaMS repository and the path to the example.mgf file is correct.")
    print("You can download the DreaMS data, including example files, from their Hugging Face Hub repository.")
    exit()
except Exception as e:
    print(f"An error occurred while computing embeddings: {e}")
    exit()

# --- Step 2: Perform Dimensionality Reduction using UMAP ---
print("Performing dimensionality reduction with UMAP...")
# Initialize UMAP. The DreaMS paper uses cosine similarity for UMAP.
# We'll reduce to 2 dimensions for a 2D plot.
reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)

# Fit UMAP to the high-dimensional DreaMS embeddings and transform them
# This will project the 1024-dimensional embeddings into a 2-dimensional space
embedding_2d = reducer.fit_transform(embs)
print("UMAP reduction complete.")

# # --- Step 3: Visualize the Embeddings ---
print("Generating UMAP visualization...")
plt.figure(figsize=(8, 6))
# Create a scatter plot of the 2D embeddings
# For a more meaningful visualization, you would typically color-code points
# based on known molecular properties (e.g., molecular formula, compound class)
# if that information was available for your example spectra.
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=10, alpha=0.8)

plt.title('DreaMS Embeddings Visualized with UMAP')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('dreams_embeddings_umap.png')

print("Visualization saved to dreams_embeddings_umap.png")