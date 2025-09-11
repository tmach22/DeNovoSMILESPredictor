import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class DreamsProjector(nn.Module):
    """Tower 1: Projects DreaMS embeddings (remains the same)."""
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, dreams_embedding):
        return self.projector(dreams_embedding)

class ScaffoldGNNEncoder(nn.Module):
    """
    Tower 2 (NEW): A GNN-based encoder for scaffold graphs.
    """
    def __init__(self, node_feature_dim, hidden_dim=128, output_dim=256, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        # data is a PyG Data object or a Batch object
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            
        # Apply global mean pooling to get a single graph-level embedding
        x = global_mean_pool(x, batch)
        
        return self.fc(x)

class ContrastiveRetriever(nn.Module):
    """
    The main model wrapper, now using the GNN encoder.
    """
    def __init__(self, dreams_dim, node_feature_dim, shared_embedding_dim=256):
        super().__init__()
        self.dreams_projector = DreamsProjector(input_dim=dreams_dim, output_dim=shared_embedding_dim)
        self.scaffold_encoder = ScaffoldGNNEncoder(node_feature_dim=node_feature_dim, output_dim=shared_embedding_dim)

    def forward(self, dreams_embedding, positive_scaffold_graph, negative_scaffold_graph):
        anchor_proj = self.dreams_projector(dreams_embedding)
        positive_proj = self.scaffold_encoder(positive_scaffold_graph)
        negative_proj = self.scaffold_encoder(negative_scaffold_graph)
        
        return anchor_proj, positive_proj, negative_proj