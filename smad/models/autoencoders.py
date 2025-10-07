import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import random

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
        # Set start token for potential use in training
        start_token = nn.Parameter(torch.randn(1,1,input_size))
        self.register_parameter('start_token',start_token)
        # Encoder LSTM with Bidirectional
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        # Latent space (bottleneck)
        self.latent = nn.Linear(hidden_size * 2, latent_dim)  # 2 * hidden_size for bidirectional
        # Decoder LSTM
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        # Final linear layer to reconstruct the input sequence
        self.decoder_out = nn.Linear(hidden_size, input_size)

    def forward(self, packed_input: PackedSequence, padded_input: torch.Tensor, lengths, learned_start_token = False, teacher_forcing=True, teacher_forcing_ratio = 1.0, noise_std = 0.0):
        batch_size, max_len, feat_dim = padded_input.shape

        # Encoder (Bidirectional LSTM)
        enc_out, (h_n, c_n) = self.encoder(packed_input)
        # Take the last hidden state (concatenating the forward and backward hidden states)
        # Shape: [batch_size, hidden_size*2] due to bidirectionality
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        # Latent space (bottleneck)
        latent = self.latent(h_n)  # Shape: [batch_size, latent_dim]
        # Decoder LSTM input will be the latent vector
        hidden = self.latent_to_hidden(latent).unsqueeze(0)
        c0 = torch.zeros_like(hidden)

        outputs = []
        if learned_start_token:
            # trains on start token - more generalizable
            decoder_input = self.start_token.expand(batch_size, 1, -1)
        else:
            # trains on initial input
            decoder_input = padded_input[:, 0, :].unsqueeze(1) # always start with first input, even though this is not an SOS token
        for t in range(1, max_len):
            out, (hidden, c0) = self.decoder_lstm(decoder_input, (hidden, c0))
            pred = self.decoder_out(out)
            outputs.append(pred)
            if teacher_forcing and random.random() < teacher_forcing_ratio:
                decoder_input = padded_input[:,t,:].unsqueeze(1)
            else:
                decoder_input = pred
            if noise_std > 0.0:
                noise = torch.randn_like(decoder_input)*noise_std # creates random gaussian noise
                decoder_input = decoder_input+noise # adds random gaussian noise to input
        #decoder_out, _ = self.decoder_lstm(decoder_input)
        # Reconstruct the original input
        #decoded = self.decoder_out(decoder_out)
        decoded = torch.cat(outputs, dim=1)
        return decoded