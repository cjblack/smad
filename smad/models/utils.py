import importlib
import torch
import pickle

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
    file_path = directory+f'/{model_name}_model.pt'
    model_info_path = directory+f'/{model_name}_info.pkl'
    torch.save(model, file_path)
    with open(model_info_path,'wb') as f:
        pickle.dump(model_info,f)