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

def teacher_forcing_inverse_sigmoid(epoch, initial, final, total_epochs, decay, k=5.0):
    decay_epochs = max(1, int(round(decay*total_epochs)))
    t = min(max(epoch / decay_epochs, 0.0), 1.0)
    inv_sig = 1.0 / (1.0 + np.exp(k*(t-0.5)))
    return final + (initial - final) * inv_sig #k / (k+np.exp(epoch / k))

def teacher_forcing_linear(epoch, initial, final, total_epochs=200, decay=0.6):
    decay_epochs = int(round(decay*total_epochs))
    return max(final, initial * (1 - epoch / decay_epochs))

def get_teacher_forcing_ratio(tf_function, epoch, total_epochs=200, k=5.0, initial=1.0, final=0.2, decay=0.6):
    if tf_function == 'inverse_sigmoid':
        tf_ratio = teacher_forcing_inverse_sigmoid(epoch, initial, final, total_epochs, decay, k)
    elif tf_function == 'linear':
        tf_ratio = teacher_forcing_linear(epoch, initial, final, total_epochs, decay)
    elif tf_function == 'off':
        tf_ratio=0.0
    return tf_ratio