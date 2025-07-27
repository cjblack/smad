import yaml
from pathlib import Path

cfg_path = Path(__file__).resolve().parent / 'config' # create path variable for config files

def check_and_return_config(model_params: str | dict):
    # check if loading param file
    if isinstance(model_params, str):
        cfg = get_config(model_params) # get param file
    elif isinstance(model_params, dict):
        save_config(model_params) # save params to param file
        cfg = model_params # set cfg to dictionary params
    else:
        raise TypeError("Input must be either a valid string or a dictionary")

    return cfg

def get_config(config_file: str) -> dict:
    '''
    Loads specified config file
    :param config_file:
    :return: dictionary of config file
    '''

    # check if the config_file name was entered with or without the extension
    if (cfg_path / config_file ).is_file():
        cfg_file = cfg_path / config_file
        with open(cfg_file, "r") as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        return cfg
    else:
        print('No config file found.')
        return -1

def save_config(config_dict: dict):
    '''
    Saves config file using dictionary
    :param config_dict:
    :return:
    '''
    config_file = config_dict['cfg_name']+'.yml'
    cfg_file = cfg_path / config_file
    with open(cfg_file, "w") as f:
        yaml.dump(config_dict, f)
