import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def train_test_split(data, train: float =0.8):
    data_len = int(np.floor(len(data)/10)*10) # make sure you have even splits, divisible by 10
    train_data = data[:int(train*data_len)] # get training data
    test_data = data[int(train*data_len):] # get testing data
    return train_data, test_data

def collate_fn(batch):
    # pad and pack data batches during training, simpler for complex models
    lengths = [len(seq) for seq in batch]
    padded = pad_sequence(batch, batch_first=True)  # (batch, max_len, 4)
    packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
    return packed, padded, lengths
