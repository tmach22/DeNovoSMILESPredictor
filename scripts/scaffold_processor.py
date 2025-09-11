import torch
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem.rdchem import HybridizationType

# --- Define a feature mapping ---
# This is a simple example; it can be made much more complex.
allowable_features = {
    'atomic_num': list(range(1, 119)),
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'hybridization': [HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP2D, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2, HybridizationType.UNSPECIFIED, HybridizationType.OTHER],
    'is_aromatic': [False, True],
}

def one_hot_encode(val, allowable_set):
    """Helper function for one-hot encoding."""
    if val not in allowable_set:
        val = allowable_set[-1] # Default to last element if not found
    return list(map(lambda s: val == s, allowable_set))

def get_atom_features(atom):
    """Computes a feature vector for a single atom."""
    features = one_hot_encode(atom.GetAtomicNum(), allowable_features['atomic_num']) + \
               one_hot_encode(atom.GetFormalCharge(), allowable_features['formal_charge']) + \
               one_hot_encode(atom.GetHybridization(), allowable_features['hybridization']) + \
               one_hot_encode(atom.GetIsAromatic(), allowable_features['is_aromatic'])
    return torch.tensor(features, dtype=torch.float)

def smiles_to_graph(smiles_string: str) -> Data:
    """
    Converts a SMILES string into a PyTorch Geometric Data object.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    # Get atom (node) features
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_features)

    # Get bond (edge) features
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]]) # Add edges in both directions for undirected graph

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)