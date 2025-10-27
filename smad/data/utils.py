"""
Utils for data
"""

from pathlib import Path
import pickle
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pyspark.sql import SparkSession
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
        data = torch_load_data(fname)

    if data:
        return data

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
