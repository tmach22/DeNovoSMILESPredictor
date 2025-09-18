import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import argparse

from tokenizer import TokenizerWrapper 

atom_permitted_list = ['Ac','Ag','Al','Am','Ar','As','At','Au','B','Ba','Be','Bh','Bi','Bk','Br','C','Ca','Cd','Ce','Cf','Cl','Cm','Cn','Co','Cr','Cs','Cu','Db','Ds','Dy','Er','Es','Eu','F','Fe','Fl','Fm','Fr','Ga','Gd','Ge','H','He','Hf','Hg','Ho','Hs','I','In','Ir','K','Kr','La','Li','Lr','Lu','Lv','Mc','Md','Mg','Mn','Mo','Mt','N','Na','Nb','Nd','Ne','Nh','Ni','No','Np','O','Og','Os','P','Pa','Pb','Pd','Pm','Po','Pr','Pt','Pu','Ra','Rb','Re','Rf','Rg','Rh','Rn','Ru','S','Sb','Sc','Se','Sg','Si','Sm','Sn','Sr','Ta','Tb','Tc','Te','Th','Ti','Tl','Tm','Ts','U','V','W','Xe','Y','Yb','Zn','Zr']
hybridization_permitted_list = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED', 'OTHER']
bond_permitted_list = ['AROMATIC', 'DATIVE', 'DATIVEL', 'DATIVEONE', 'DATIVER', 'DOUBLE', 'FIVEANDAHALF', 'FOURANDAHALF', 'HEXTUPLE', 'HYDROGEN', 'IONIC', 'ONEANDAHALF', 'OTHER', 'QUADRUPLE', 'QUINTUPLE', 'SINGLE', 'THREEANDAHALF', 'THREECENTER', 'TRIPLE', 'TWOANDAHALF']

# --- [Helper functions remain unchanged] ---
def get_formula_from_smiles(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return ""
    return rdMolDescriptors.CalcMolFormula(mol)

def get_murcko_scaffold(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        return ""
    except: return ""

def one_hot_encode(value, permitted_list):
    if value not in permitted_list: value = permitted_list[-1]
    return [int(value == s) for s in permitted_list]

def get_atom_features(atom: Chem.Atom) -> list:
    features = []
    features += one_hot_encode(atom.GetSymbol(), atom_permitted_list)
    features += one_hot_encode(atom.GetDegree(), list(range(11)))
    features += one_hot_encode(atom.GetTotalNumHs(), list(range(9)))
    features += one_hot_encode(str(atom.GetHybridization()), hybridization_permitted_list)
    features.append(atom.GetIsAromatic())
    features.append(atom.IsInRing())
    return features

def get_bond_features(bond: Chem.Bond) -> list:
    features = []; bt = bond.GetBondType()
    features += one_hot_encode(bt, bond_permitted_list)
    features.append(bond.GetIsConjugated())
    return features

def create_dummy_graph():
    node_features = torch.zeros(1, 148, dtype=torch.float)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_attr = torch.empty(0, 21, dtype=torch.float)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

def smiles_to_graph_data(smiles: str) -> Data:
    if not smiles:
        return create_dummy_graph()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return create_dummy_graph()
    atom_features_list = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features_list, dtype=torch.float)
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); bond_feats = get_bond_features(bond)
        edge_indices.extend([(i, j), (j, i)]); edge_attrs.extend([bond_feats, bond_feats])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# UPDATED: The dataset no longer needs a transform argument.
class SmilesGraphDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            self.embeddings = f['embeddings'][:]; self.smiles = [s.decode('utf-8') for s in f['smiles'][:]]

    def __len__(self): return len(self.smiles)

    def __getitem__(self, idx):
        smiles_string = self.smiles[idx]
        formula = get_formula_from_smiles(smiles_string)
        scaffold_smiles = get_murcko_scaffold(smiles_string)
        scaffold_graph = smiles_to_graph_data(scaffold_smiles)
        
        return {"embedding": self.embeddings[idx], "formula": formula, "scaffold_graph": scaffold_graph, "smiles": smiles_string}

# UPDATED: The function no longer needs a drop_edge_rate argument.
def get_graph_data_loader(hdf5_path, smiles_tokenizer, formula_tokenizer, split, batch_size=32, num_workers=0):
    dataset = SmilesGraphDataset(hdf5_path)
    smiles_pad_idx = smiles_tokenizer.vocab['<pad>']
    formula_pad_idx = formula_tokenizer.vocab['<pad>']

    def collate_graphs(data_list):
        embeddings = [item['embedding'] for item in data_list]
        formulas = [item['formula'] for item in data_list]
        scaffold_graphs = [item['scaffold_graph'] for item in data_list]
        smiles_strings = [item['smiles'] for item in data_list]
        
        embeddings_tensor = torch.tensor(np.stack(embeddings), dtype=torch.float32)
        formula_tokens = [torch.tensor(formula_tokenizer.encode(f)) for f in formulas]
        formulas_padded = pad_sequence(formula_tokens, batch_first=True, padding_value=formula_pad_idx)
        smiles_tokens = [torch.tensor(smiles_tokenizer.encode(s)) for s in smiles_strings]
        smiles_padded = pad_sequence(smiles_tokens, batch_first=True, padding_value=smiles_pad_idx)
        
        return {"embedding": embeddings_tensor, "formula": formulas_padded, "scaffold_graph_list": scaffold_graphs, "smiles": smiles_padded}

    should_shuffle = True if split == 'train' else False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle, collate_fn=collate_graphs, num_workers=num_workers)
    return loader

# --- [UPDATED Main block for testing the data loader] ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the GNN data loader with rich features.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training HDF5 file.")
    parser.add_argument("--smiles_vocab_path", type=str, required=True, help="Path to the pre-built SMILES vocabulary.")
    parser.add_argument("--formula_vocab_path", type=str, required=True, help="Path to the pre-built formula vocabulary.")
    args = parser.parse_args()

    print("--- Step 1: Initializing Tokenizers and Loading Vocabularies ---")
    smiles_tokenizer = TokenizerWrapper()
    formula_tokenizer = TokenizerWrapper()
    smiles_tokenizer.load_vocab(args.smiles_vocab_path)
    formula_tokenizer.load_vocab(args.formula_vocab_path)
    
    print("\n--- Step 2: Creating Data Loader ---")
    # Using a small batch size for easy inspection
    test_loader = get_graph_data_loader(
        args.train_path, smiles_tokenizer, formula_tokenizer, 
        split='train', batch_size=64, num_workers=0
    )
    print("Data loader created successfully.")

    print("\n--- Step 3: Fetching and Inspecting a Single Batch ---")
    if test_loader:
        first_batch = next(iter(test_loader))
        second_batch = next(iter(test_loader))
        first_batch = second_batch
        
        if first_batch:
            print("Batch Contents:")
            print(f"  DreaMS Embeddings Shape: {first_batch['embedding'].shape}")
            print(f"  Formula Tokens Shape:    {first_batch['formula'].shape}")
            print(f"  SMILES Tokens Shape:     {first_batch['smiles'].shape}")
            
            scaffold_list = first_batch['scaffold_graph_list']
            print(f"  Scaffold Graph List:     Contains {len(scaffold_list)} graph objects.")

            # Inspect the very first graph in the batch
            if scaffold_list:
                first_graph = scaffold_list[0]
                print("\n--- Inspecting the First Graph Object in the Batch ---")
                print(f"  Graph Object Type:       {type(first_graph)}")
                
                # These shapes are CRITICAL for debugging your GNN model input dimensions
                print(f"  Node Features (x) Shape: {first_graph.x.shape}")
                print(f"     -> Interpretation: [{first_graph.x.shape[0]} atoms, {first_graph.x.shape[1]} features/atom]")
                
                print(f"  Edge Index Shape:          {first_graph.edge_index.shape}")
                print(f"     -> Interpretation: [2 (source/target), {first_graph.edge_index.shape[1]} directed edges]")
                
                print(f"  Edge Attributes Shape:     {first_graph.edge_attr.shape}")
                print(f"     -> Interpretation: [{first_graph.edge_attr.shape[0]} edges, {first_graph.edge_attr.shape[1]} features/bond]")

                print("\n--- Context for the First Graph ---")
                decoded_smiles = smiles_tokenizer.decode(first_batch['smiles'][0].tolist())
                decoded_formula = formula_tokenizer.decode(first_batch['formula'][0].tolist())
                print(f"  Original Full SMILES: {decoded_smiles}")
                print(f"  Original Formula:     {decoded_formula}")
        else:
            print("  Could not retrieve a valid batch. Check data filtering in collate_fn.")
    else:
        print("  Data loader creation failed. Check file paths.")

    print("\nData loader inspection complete.")