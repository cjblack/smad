import torch
import torch.nn as nn

class CallosalLSTM(nn.Module):
    def __init__(self, model_params, enc_layers = 3):
        super().__init__()

        # Extract model params from dictionary
        self.input_size = model_params['input_size']
        hidden_size = model_params['hidden_size']
        latent_dim = model_params['latent_dim']
        self.pooling = model_params['pooling']
        self.num_layers_enc = model_params['num_encoder_layers']
        self.num_layers_dec = model_params['num_decoder_layers']
        dropout = model_params['dropout'] if hidden_size > 1 else 0.0
        use_skip = model_params['skip_connections']
        self.num_modalities = len(self.input_size)

        # Set lists for functions
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        self.output_linears = nn.ModuleDict()
        self.start_token_params = nn.ParameterDict()

        for key, val in self.input_size.items():
            self.encoders[key] = nn.ModuleList() # create a list instead of each LSTM having multiple layers to gain access to each layer individually
            
            input_dim = val
            for i in range(enc_layers):
                self.encoders[key].append(nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True))
                input_dim = hidden_size # change input dimension for subsequent layers in stack
            
