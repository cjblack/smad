import torch
from sklearn.preprocessing import MinMaxScalar
from scipy.signal import savgol_filter

def create_data_loader(data, batch_size: int, shuffle: bool =True):
    data_loader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=shuffle)
    return data_loader

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize between 0 and 1
    data = scaler.fit_transform(data.T)  # Scale data - transpose data as it requires n_samples x n_features
    return data.T, scaler

def pad_tensor_list(tensor_list: list):
    tensor_list_pad = torch.nn.utils.rnn.pad_sequence(tensor_list,batch_first=True) # pad data with zeros
    return tensor_list_pad