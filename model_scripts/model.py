import torch
import torch.nn as nn
import math

# UPDATED: Import GINEConv, global_mean_pool, GraphNorm as before
from torch_geometric.nn import GINEConv, global_mean_pool, GraphNorm
from torch_geometric.data import Batch
# NEW: Import the dropout_edge utility function for structural regularization
from torch_geometric.utils import dropout_edge

# --- ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]; return self.dropout(x)

class FiLMGenerator(nn.Module):
    def __init__(self, cond_dim: int, num_layers: int, d_model: int):
        super().__init__()
        output_dim = num_layers * 4 * d_model
        self.generator = nn.Sequential(nn.Linear(cond_dim, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, output_dim))
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.generator(condition)

class FiLMDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1, self.linear2 = nn.Linear(d_model, dim_feedforward), nn.Linear(dim_feedforward, d_model)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout, self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self.activation = nn.ReLU()
    def forward(self, tgt: torch.Tensor, film_params: tuple, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        gamma1, beta1, gamma2, beta2 = film_params
        x = tgt; attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, need_weights=False)
        x = x + self.dropout1(attn_output); x = self.norm1(x)
        x = gamma1.unsqueeze(1) * x + beta1.unsqueeze(1)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output); x = self.norm2(x)
        x = gamma2.unsqueeze(1) * x + beta2.unsqueeze(1)
        return x

class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(); self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
    def forward(self, tokens):
        embedded = self.embedding(tokens); _, hidden = self.rnn(embedded); return hidden[-1]

# --- UPDATED: GNN Encoder with integrated DropEdge and flexible depth ---
class ScaffoldGINEEncoder(nn.Module):
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, node_dropout_rate=0.2, drop_edge_rate=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # NEW: Store the drop_edge_rate for use in the forward pass
        self.drop_edge_rate = drop_edge_rate

        for i in range(num_layers):
            in_dim = node_feature_dim if i == 0 else hidden_dim
            nn_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINEConv(nn_mlp, edge_dim=edge_feature_dim))
            self.norms.append(GraphNorm(hidden_dim))

        self.lin = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(node_dropout_rate)

    def forward(self, graph_batch: Batch) -> torch.Tensor:
        x, edge_index, edge_attr, batch = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch

        # --- NEW: Apply DropEdge directly within the model's forward pass ---
        # This correctly applies dropout only during training (`self.training` is True)
        # and is automatically disabled during evaluation.
        if self.training and self.drop_edge_rate > 0:
            edge_index, edge_mask = dropout_edge(
                edge_index, 
                p=self.drop_edge_rate, 
                training=self.training
            )
            # If the graph has edge attributes, we need to filter them with the mask
            if edge_attr is not None:
                edge_attr = edge_attr[edge_mask]

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = self.relu(x)
            x = self.dropout(x)

        graph_embedding = global_mean_pool(x, batch)
        return self.lin(graph_embedding)

# --- Main Model ---
class SmilesRecyclingDecoder(nn.Module):
    def __init__(self, smiles_vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, 
                 dreams_dim, formula_vocab_size, formula_emb_dim, 
                 scaffold_emb_dim, gnn_hidden_dim=128, num_gnn_layers=2, 
                 drop_edge_rate=0.2, num_recycling_iters=3): # UPDATED: Added drop_edge_rate
        super().__init__()
        self.d_model = d_model
        self.num_recycling_iters = num_recycling_iters

        NODE_FEATURE_DIM = 148
        EDGE_FEATURE_DIM = 21

        self.formula_encoder = SequenceEncoder(formula_vocab_size, formula_emb_dim, formula_emb_dim, num_layers=2)
        
        # UPDATED: Pass num_gnn_layers and drop_edge_rate to the GNN encoder
        self.scaffold_encoder = ScaffoldGINEEncoder(
            node_feature_dim=NODE_FEATURE_DIM,
            edge_feature_dim=EDGE_FEATURE_DIM,
            hidden_dim=gnn_hidden_dim,
            output_dim=scaffold_emb_dim,
            num_layers=num_gnn_layers,
            drop_edge_rate=drop_edge_rate
        )
        
        combined_cond_dim = dreams_dim + formula_emb_dim + scaffold_emb_dim
        self.film_generator = FiLMGenerator(combined_cond_dim, num_decoder_layers, d_model)
        
        self.smiles_embedding = nn.Embedding(smiles_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = FiLMDecoderLayer(d_model, nhead, dim_feedforward)
        self.recycling_block = nn.ModuleList([decoder_layer for _ in range(num_decoder_layers)])
        self.fc_out = nn.Linear(d_model, smiles_vocab_size)

    def forward(self, dreams_embedding, formula_tokens, scaffold_graph_list, tgt_tokens):
        device = tgt_tokens.device
        
        scaffold_batch = Batch.from_data_list(scaffold_graph_list).to(device)

        formula_emb = self.formula_encoder(formula_tokens)
        scaffold_emb = self.scaffold_encoder(scaffold_batch)

        combined_cond = torch.cat([dreams_embedding, formula_emb, scaffold_emb], dim=1)
        film_params_flat = self.film_generator(combined_cond)
        film_params = film_params_flat.view(dreams_embedding.size(0), len(self.recycling_block), 4, self.d_model)

        tgt_emb = self.smiles_embedding(tgt_tokens) * math.sqrt(self.d_model)
        refined_emb = self.pos_encoder(tgt_emb)
        
        for cycle in range(self.num_recycling_iters):
            is_last_cycle = (cycle == self.num_recycling_iters - 1)
            with torch.set_grad_enabled(is_last_cycle):
                current_input = refined_emb.detach() if not is_last_cycle else refined_emb
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).to(device)
                
                for i, layer in enumerate(self.recycling_block):
                    layer_params = (film_params[:, i, 0, :], film_params[:, i, 1, :], film_params[:, i, 2, :], film_params[:, i, 3, :])
                    current_input = layer(current_input, film_params=layer_params, tgt_mask=tgt_mask)
                refined_emb = current_input
                
        return self.fc_out(refined_emb)