import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe) # store without making it a learnable param

    def forward(self, x):
        # x: (B, S, D)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)

class TransformerAutoencoder(nn.Module):
    def __init__(self,
                 input_dim,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=512,
                 dropout=0.1,
                 max_seq_len=512,
                 latent_dim=None,
                 use_latent_token=False):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Option: compress to a latent vector (optional)
        self.use_latent_token = use_latent_token
        if use_latent_token:
            # Learnable latent token that attends to encoder outputs
            self.latent_token = nn.Parameter(torch.randn(1, 1, d_model))
            # Decoder will receive replicated latent as target start; we still decode sequence length
        else:
            # Use pooling then expand to seq length for decoder input
            self.latent_fc = nn.Linear(d_model, d_model) if latent_dim is None else nn.Linear(d_model, latent_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation='gelu')
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # For autoregressive decoding or direct sequence reconstruction:
        # We'll feed the encoder's compressed representation expanded to seq_len as the decoder memory,
        # and use a simple zero tensor (or shifted target) as decoder input embeddings.
        self.out_proj = nn.Linear(d_model, input_dim)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # x: (batch, seq_len, input_dim)
        b, seq_len, _ = x.shape
        src = self.input_proj(x)             # (b, seq_len, d_model)
        src = self.pos_enc(src)              # add positional encodings
        # Transformer expects (seq_len, batch, d_model)
        src_t = src.transpose(0, 1)          # (seq_len, b, d_model)
        enc_out = self.encoder(src_t,
                               mask=src_mask,
                               src_key_padding_mask=src_key_padding_mask)  # (seq_len, b, d_model)

        # produce decoder memory
        # Option A: use latent token
        if self.use_latent_token:
            # Apply attention from latent token to encoder outputs via a small cross-attention step:
            # Simplest: take mean pooling to create latent (could substitute learned token)
            latent = enc_out.mean(dim=0, keepdim=True)  # (1, b, d_model)
            # replicate latent to seq_len as "memory" or use latent as memory and decode from zeros.
            memory = latent.repeat(seq_len, 1, 1)  # (seq_len, b, d_model)
        else:
            # Option B: mean-pool encoder outputs and project back to d_model then expand
            pooled = enc_out.mean(dim=0)  # (b, d_model)
            expanded = pooled.unsqueeze(1).repeat(1, seq_len, 1)  # (b, seq_len, d_model)
            memory = expanded.transpose(0, 1)  # (seq_len, b, d_model)

        # Prepare decoder input: can be zeros or the input embeddings shifted
        # We'll use zeros with positional encodings to reconstruct entire sequence in parallel
        tgt = torch.zeros_like(src_t)  # (seq_len, b, d_model)
        # add positional enc for tgt (transpose back to (b,seq,d) to use pos_enc)
        tgt_pos = self.pos_enc(tgt.transpose(0,1)).transpose(0,1)

        dec_out = self.decoder(tgt_pos, memory,
                               tgt_mask=None,
                               memory_mask=None,
                               tgt_key_padding_mask=None,
                               memory_key_padding_mask=None)  # (seq_len, b, d_model)

        dec_out = dec_out.transpose(0, 1)  # (b, seq_len, d_model)
        out = self.out_proj(dec_out)       # (b, seq_len, input_dim)
        return out