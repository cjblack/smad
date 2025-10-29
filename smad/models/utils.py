import os
import glob
import importlib
import torch
import pickle
from smad.data.utils import pickle_load_data, torch_load_data

def create_model(cfg: dict):
    # Set up model from config
    model_name = cfg['model_name']  # get model name
    module = cfg['module']  # get module for model
    model_params = cfg['params']['model']  # get params from dictionary
    model_class = getattr(importlib.import_module(module), model_name)  # get model class
    model = model_class(model_params)  # create model class with selected parameters
    return model

def save_model(model, model_info, directory):
    # using pickle for model_info but may change to .h5
    model_name = model_info['cfg']['model_name']
    file_path = directory+f'/{model_name}_model.pth'
    model_info_path = directory+f'/{model_name}_info.pkl'
    torch.save(model.state_dict(), file_path)
    with open(model_info_path,'wb') as f:
        pickle.dump(model_info,f)

def load_model_package(directory, load_model=False):
    model_package = dict()
    files_ = glob.glob(directory + '/*')
    for f in files_:
        fname = f.split('\\')[-1].split('.')[0]
        ftype = f.split('.')[-1]
        if ftype == 'pkl':
            model_package[fname] = pickle_load_data(f)
        elif ftype == 'pth':
            model_package['state_dict'] = torch_load_data(f)
            model_package['model_name'] = fname
    if load_model:
        cfg = model_package['training_info']['cfg']
        model = create_model(cfg)
        model = model.load_state_dict(model_package['state_dict'])
        model_package['model'] = model
    return model_package