"""
Utils for data
"""

from pathlib import Path
import pickle
import torch
import json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.signal import savgol_filter

DATA_TEST_SETS = Path(__file__).resolve().parent / 'test_sets'

def create_data_loader(data, batch_size: int, shuffle: bool =True, collate_fn = None):
    """
    Creates data loader object
    """
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn = collate_fn)
    return data_loader

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize between 0 and 1
    # Scale data - transpose data as it requires n_samples x n_features
    data = scaler.fit_transform(data.T)  
    return data.T, scaler

def pad_tensor_list(tensor_list: list):
    # pad data with zeros
    tensor_list_pad = torch.nn.utils.rnn.pad_sequence(tensor_list,batch_first=True) 
    return tensor_list_pad

def load_data(fname):
    data = []
    if str(fname).split('.')[-1] == 'pt':
        # pt format - ok for full models
        data = torch_load_data(fname)
    elif str(fname).split('.')[-1] == 'pth':
        # pth format - better for model states
        data = torch_load_data(fname)

    if data:
        return data
    
def split_train_val(data: list, val_size=0.2, train_size=0.8):
    data_len = len(data)
    num_val = int(round(val_size*data_len))
    train_data = data[:data_len - num_val]
    val_data = data[data_len - num_val:]
    return train_data, val_data

def torch_load_data(fname):
    data = torch.load(fname)
    return data

def pickle_save_data(fname, data):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def pickle_load_data(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

def json_save_data(fname, data):
    with open(fname, 'w') as f:
        f.write(data)

