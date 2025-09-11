import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

class LstmAutoencoder(nn.Module):
    def __init__(self, model_params):#input_size, hidden_size, latent_dim):
        super(LstmAutoencoder, self).__init__()

        # Extract model params from dictionary
        input_size = model_params['input_size']
        hidden_size = model_params['hidden_size']
        latent_dim = model_params['latent_dim']
        # Encoder LSTM with Bidirectional
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        # Latent space (bottleneck)
        self.latent = nn.Linear(hidden_size * 2, latent_dim)  # 2 * hidden_size for bidirectional
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_size, batch_first=True)
        # Final linear layer to reconstruct the input sequence
        self.decoder_out = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Encoder (Bidirectional LSTM)
        enc_out, (h_n, c_n) = self.encoder(x)
        # Take the last hidden state (concatenating the forward and backward hidden states)
        # Shape: [batch_size, hidden_size*2] due to bidirectionality
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        # Latent space (bottleneck)
        latent = self.latent(h_n)  # Shape: [batch_size, latent_dim]
        # Decoder LSTM input will be the latent vector
        decoder_input = latent.unsqueeze(1).repeat(1, x.size(1), 1)  # Repeat latent for each time step
        decoder_out, _ = self.decoder_lstm(decoder_input)
        # Reconstruct the original input
        decoded = self.decoder_out(decoder_out)

        return decoded

class LstmAutoencoderPk(nn.Module):
    def __init__(self, model_params):#input_size, hidden_size, latent_dim):
        super(LstmAutoencoderPk, self).__init__()

        # Extract model params from dictionary
        input_size = model_params['input_size']
        hidden_size = model_params['hidden_size']
        latent_dim = model_params['latent_dim']
        # Encoder LSTM with Bidirectional
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        # Latent space (bottleneck)
        self.latent = nn.Linear(hidden_size * 2, latent_dim)  # 2 * hidden_size for bidirectional
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_size, batch_first=True)
        # Final linear layer to reconstruct the input sequence
        self.decoder_out = nn.Linear(hidden_size, input_size)

    def forward(self, packed_input: PackedSequence, padded_input: torch.Tensor, lengths):
        # Encoder (Bidirectional LSTM)
        enc_out, (h_n, c_n) = self.encoder(packed_input)
        # Take the last hidden state (concatenating the forward and backward hidden states)
        # Shape: [batch_size, hidden_size*2] due to bidirectionality
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        # Latent space (bottleneck)
        latent = self.latent(h_n)  # Shape: [batch_size, latent_dim]
        # Decoder LSTM input will be the latent vector
        batch_size, max_len, _ = padded_input.shape
        decoder_input = latent.unsqueeze(1).repeat(1, max_len, 1)  # Repeat latent for each time step
        decoder_out, _ = self.decoder_lstm(decoder_input)
        # Reconstruct the original input
        decoded = self.decoder_out(decoder_out)

        return decoded