import numpy as np

def train_test_split(data, train: float =0.8):
    data_len = int(np.floor(len(data)/10)*10) # make sure you have even splits, divisible by 10
    train_data = data[:int(train*data_len)] # get training data
    test_data = data[int(train*data_len):] # get testing data
    return train_data, test_data
