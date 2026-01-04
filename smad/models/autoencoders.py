import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


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

"""
MAIN CLASS
"""

class LstmAutoencoderPk(nn.Module):
    def __init__(self, model_params):#input_size, hidden_size, latent_dim):
        super(LstmAutoencoderPk, self).__init__()

        # Extract model params from dictionary
        input_size = model_params['input_size']
        hidden_size = model_params['hidden_size']
        latent_dim = model_params['latent_dim']
        self.pooling = model_params['pooling']
        self.num_layers_enc = model_params['num_encoder_layers']
        self.num_layers_dec = model_params['num_decoder_layers']
        dropout = model_params['dropout'] if hidden_size > 1 else 0.0
        use_skip = model_params['skip_connections']
        
        # LSTM dropout only works when num_layers > 1
        enc_dropout = dropout if self.num_layers_enc > 1 else 0.0
        dec_dropout = dropout if self.num_layers_dec > 1 else 0.0

        # Set up skip connections
        self.use_skip = use_skip
        if self.use_skip:
            self.skip_alpha  = nn.Parameter(torch.tensor(0.5))

        # Set start token for potential use in training
        start_token = nn.Parameter(torch.randn(1,1,input_size))
        self.register_parameter('start_token',start_token)
        
        # Encoder LSTM with Bidirectional
        self.encoder = nn.LSTM(input_size=input_size, num_layers=self.num_layers_enc,hidden_size=hidden_size, batch_first=True, bidirectional=True, dropout=enc_dropout)
        
        # Latent space (bottleneck)
        if self.pooling:
            self.latent = nn.Linear(hidden_size * 4, latent_dim) # account for mean pooling
        else:
            self.latent = nn.Linear(hidden_size * 2, latent_dim)  # 2 * hidden_size for bidirectional
        
        # Decoder LSTM
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(input_size=input_size, num_layers=self.num_layers_dec, hidden_size=hidden_size, batch_first=True, dropout=dec_dropout)
        
        # Final linear layer to reconstruct the input sequence
        self.decoder_out = nn.Linear(hidden_size, input_size)
        
        # Dropout layers
        self.input_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def encode(self, packed_input: PackedSequence):
        """
        Extract latent space for pass
        
        :param self: 
        :param packed_input: Packed input for model
        :type packed_input: PackedSequence
        """

        self.eval()
        _, (h_n, _) = self.encoder(packed_input)
        # top layer forward/backward for bidirectional encoder
        h_top_fwd = h_n[-2]   # (B, hidden)
        h_top_bwd = h_n[-1]   # (B, hidden)
        h_cat = torch.cat((h_top_fwd, h_top_bwd), dim=1)  # (B, 2*hidden)

        z = self.latent(h_cat)  # (B, latent_dim)
        return z
    
    @torch.no_grad()
    def decode(self, z: torch.tensor, max_len: int):
        """
        Decode from latent vector - evaluation
        
        :param z: latent input vector
        :type z: torch.tensor
        :param max_len: Description
        :type max_len: int
        """
        self.eval()
        B = z.shape[0]
        device = z.device

        # initialize hidden state from latent
        h0 = self.latent_to_hidden(z)
        h = h0.unsqueeze(0).repeat(self.num_layers_dec, 1, 1)
        c = torch.zeros_like(h)

        # start token
        x = self.start_token.expand(B, 1, -1).to(device)

        outs = []
        for _ in range(1, max_len):
            y, (h, c) = self.decoder_lstm(x, (h,c))
            x = self.decoder_out(y)
            outs.append(x)

        return torch.cat(outs, dim=1)

    def forward(self, packed_input: PackedSequence, padded_input: torch.Tensor, lengths, learned_start_token = False, teacher_forcing=True, teacher_forcing_ratio = 1.0, noise_std = 0.0):
        
        batch_size, max_len, feat_dim = padded_input.shape

        # Encoder (Bidirectional LSTM)
        enc_out, (h_n, c_n) = self.encoder(packed_input)
        # Take the last hidden state (concatenating the forward and backward hidden states)
        # Shape: [batch_size, hidden_size*2] due to bidirectionality
        h_top_fwd = h_n[-2]
        h_top_bwd = h_n[-1]
        
        #h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        h_n = torch.cat((h_top_fwd, h_top_bwd), dim=1) # gives last hidden
        latent_input = h_n

        # Mean pooling
        if self.pooling:
            enc_out_padded, _ = pad_packed_sequence(enc_out, batch_first=True) # pad out packed sequence
            B, T, _ = enc_out_padded.shape

            lengths = lengths.to(enc_out_padded.device)
            mask = (torch.arange(T, device=enc_out_padded.device)[None, :] < lengths[:, None]).unsqueeze(-1).float()
            mean_hidden = (enc_out_padded*mask).sum(dim=1) / lengths.unsqueeze(-1).float()

            latent_input = torch.cat([h_n, mean_hidden], dim=1)
            latent = self.latent(latent_input)
            
        # Latent space (bottleneck)
        latent = self.latent(latent_input)  # Shape: [batch_size, latent_dim]
        
        # Decoder LSTM input will be the latent vector
        #hidden = self.latent_to_hidden(latent).unsqueeze(0)
        h0_single = self.latent_to_hidden(latent)
        hidden = h0_single.unsqueeze(0).repeat(self.num_layers_dec, 1, 1)
        c0 = torch.zeros_like(hidden)

        outputs = []

        if learned_start_token:
            # trains on start token - more generalizable
            decoder_input = self.start_token.expand(batch_size, 1, -1)
        else:
            # trains on initial input
            decoder_input = padded_input[:, 0, :].unsqueeze(1) # always start with first input, even though this is not an SOS token
        for t in range(1, max_len):
            decoder_input = self.input_dropout(decoder_input) # dropout
            out, (hidden, c0) = self.decoder_lstm(decoder_input, (hidden, c0))
            out = self.output_dropout(out) # dropout
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

        if self.use_skip:
            alpha = torch.sigmoid(self.skip_alpha) # constrain between 0-1
            decoded = alpha * decoded + (1-alpha)*padded_input[:,1:,:]

        return decoded