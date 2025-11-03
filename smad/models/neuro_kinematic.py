import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

class NeuroKinematicDecoder(nn.Module):
    """
    LSTM neural decoder for kinematic data
    """
    def __init__(self, model_params):
        super(NeuroKinematicDecoder, self).__init__()

        # Extract model params from dictionary
        input_size = model_params['input_size']
        hidden_size = model_params['hidden_size']
        latent_dim = model_params['latent_dim']
        output_size = model_params['output_size']

        # Encoder
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        # Latent space
        self.latent = nn.Linear(hidden_size * 2, latent_dim)

        # Output layer
        self.output_layer = nn.Linear(latent_dim, output_size)

    def forward(self, x, bound: bool = False):
        x, _ = self.encoder(x)
        if bound:
            # bound latent space
            x = torch.tanh(self.latent(x))
        else:
            # let model learn on unbounded output
            x = self.latent(x)
        out = self.output_layer(x)
        return out


