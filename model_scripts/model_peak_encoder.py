import torch
import torch.nn as nn
import math

# GINEConv is needed for the scaffold encoder
from torch_geometric.nn import GINEConv, global_mean_pool, GraphNorm
from torch_geometric.data import Batch

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
        # Scale the embedding before adding positional encoding
        x = x * math.sqrt(x.size(-1))
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class FiLMGenerator(nn.Module):
    def __init__(self, cond_dim: int, num_layers: int, d_model: int):
        super().__init__()
        # FiLM needs to generate 2 parameters (gamma, beta) per layer
        output_dim = num_layers * 2 * d_model
        self.generator = nn.Sequential(nn.Linear(cond_dim, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, output_dim))
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.generator(condition)

class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers, batch_first=True)
    def forward(self, tokens):
        embedded = self.embedding(tokens)
        _, hidden = self.rnn(embedded)
        return hidden[-1]

class ScaffoldGINEEncoder(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, output_dim, num_layers=3, dropout_rate=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_feature_dim if i == 0 else hidden_dim
            nn_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINEConv(nn_mlp, edge_dim=edge_feature_dim))
            self.norms.append(GraphNorm(hidden_dim))
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, graph_batch: Batch) -> torch.Tensor:
        x, edge_index, edge_attr, batch = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = self.relu(x)
            x = self.dropout(x)
        return self.lin(global_mean_pool(x, batch))

# --- NEW: Transformer Encoder for processing peak sequences ---
class PeakEncoder(nn.Module):
    def __init__(self, peak_vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(peak_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    def forward(self, peak_bins, src_key_padding_mask=None):
        peak_emb = self.embedding(peak_bins)
        peak_emb = self.pos_encoder(peak_emb)
        return self.transformer_encoder(peak_emb, src_key_padding_mask=src_key_padding_mask)

# --- NEW: Hybrid Decoder Layer combining Self-Attention, FiLM, and Cross-Attention ---
class HybridDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout); self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, film_params, tgt_mask=None, memory_key_padding_mask=None):
        # 1. Masked Self-Attention on SMILES tokens
        x = tgt
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, need_weights=False)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 2. FiLM Augmentation using global context
        gamma, beta = film_params
        x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)
        
        # 3. Cross-Attention to encoded peaks
        attn_output, _ = self.cross_attn(query=x, key=memory, value=memory, key_padding_mask=memory_key_padding_mask, need_weights=False)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        
        # 4. Feed-Forward Network
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(ff_output)
        x = self.norm3(x)
        return x

# --- NEW: Main Model integrating all components ---
class HybridSMILESDecoder(nn.Module):
    def __init__(self, smiles_vocab_size, peak_vocab_size, d_model, nhead, num_decoder_layers, num_encoder_layers, 
                 dim_feedforward, dreams_dim, formula_vocab_size, formula_emb_dim, 
                 scaffold_emb_dim, gnn_hidden_dim, num_gnn_layers, num_recycling_iters=3):
        super().__init__()
        self.d_model = d_model
        self.num_recycling_iters = num_recycling_iters

        # Peak Encoder for local context
        self.peak_encoder = PeakEncoder(peak_vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward)
        
        # Encoders for global context
        self.formula_encoder = SequenceEncoder(formula_vocab_size, formula_emb_dim, formula_emb_dim, 2)
        self.scaffold_encoder = ScaffoldGINEEncoder(
            node_feature_dim=148, # From your new data loader
            edge_feature_dim=21,  # From your new data loader
            hidden_dim=gnn_hidden_dim, 
            output_dim=scaffold_emb_dim, 
            num_layers=num_gnn_layers
        )
        
        # FiLM Generator for global context
        combined_cond_dim = dreams_dim + formula_emb_dim + scaffold_emb_dim
        self.film_generator = FiLMGenerator(combined_cond_dim, num_decoder_layers, d_model)
        
        # SMILES Decoder components
        self.smiles_embedding = nn.Embedding(smiles_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = HybridDecoderLayer(d_model, nhead, dim_feedforward)
        self.recycling_block = nn.ModuleList([decoder_layer for _ in range(num_decoder_layers)])
        
        self.fc_out = nn.Linear(d_model, smiles_vocab_size)

    def forward(self, dreams_embedding, formula_tokens, scaffold_graph_list, peak_bins, tgt_tokens):
        device = tgt_tokens.device
        
        # 1. Encode local context (peaks)
        # Create a padding mask for peaks (0 is the padding token)
        peak_padding_mask = (peak_bins == 0)
        peak_memory = self.peak_encoder(peak_bins, src_key_padding_mask=peak_padding_mask)
        
        # 2. Encode global context (DreaMS, formula, scaffold)
        scaffold_batch = Batch.from_data_list(scaffold_graph_list).to(device)
        formula_emb = self.formula_encoder(formula_tokens)
        scaffold_emb = self.scaffold_encoder(scaffold_batch)
        
        # 3. Generate FiLM parameters from global context
        combined_cond = torch.cat([dreams_embedding, formula_emb, scaffold_emb], dim=1)
        film_params_flat = self.film_generator(combined_cond)
        film_params = film_params_flat.view(dreams_embedding.size(0), len(self.recycling_block), 2, self.d_model)

        # 4. Prepare SMILES input for the decoder
        tgt_emb = self.smiles_embedding(tgt_tokens)
        refined_emb = self.pos_encoder(tgt_emb)
        
        # 5. Run the decoder with recycling
        for cycle in range(self.num_recycling_iters):
            is_last_cycle = (cycle == self.num_recycling_iters - 1)
            with torch.set_grad_enabled(is_last_cycle):
                current_input = refined_emb.detach() if not is_last_cycle else refined_emb
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).to(device)
                
                for i, layer in enumerate(self.recycling_block):
                    layer_params = (film_params[:, i, 0, :], film_params[:, i, 1, :])
                    current_input = layer(current_input, peak_memory, film_params=layer_params, tgt_mask=tgt_mask, memory_key_padding_mask=peak_padding_mask)
                refined_emb = current_input
                
        return self.fc_out(refined_emb)