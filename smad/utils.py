import yaml
from pathlib import Path

cfg_path = Path(__file__).resolve().parent / 'config' # create path variable for config files

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
