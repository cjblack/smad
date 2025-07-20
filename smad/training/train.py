#import matplotlib.pyplot as plt
#import numpy as np

from smad.utils import get_config, save_config
from smad.models import *
#from models.trajectory_autoencoder import TrajectoryAutoencoder
import importlib
#from Analysis.Examples.PawTrajectoryAnalysis import PawTrajAnalysis
#from utils import autoencoder_utils as aeu
#import torch
#import torch.nn as nn
#import os
#import glob
#import time

def train_model(model_params: str | dict):#input_size=1, hidden_size=96, latent_dim = 32):
    if isinstance(model_params, str):
        cfg = get_config(model_params)
    elif isinstance(model_params, dict):
        save_config(model_params)
        cfg = model_params
    else:
        raise TypeError("Input must be either a valid string or a dictionary")
    model_name = cfg['model_name']
    module = cfg['module']
    model_params = cfg['params']['model']
    model_class = getattr(importlib.import_module(module),model_name)
    model = model_class(model_params)
    return model

