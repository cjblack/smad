import yaml
from pathlib import Path
import os
import argparse
from datetime import datetime
import uuid

CFG_PATH = Path(__file__).resolve().parent / 'config' # create path variable for config files

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
    if (CFG_PATH / config_file ).is_file():
        cfg_file = CFG_PATH / config_file
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
    cfg_file = CFG_PATH / config_file
    with open(cfg_file, "w") as f:
        yaml.dump(config_dict, f)

def get_output_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", help='Folder for saving model outputs')
    args = parser.parse_args()

    # create output directory that is HPC and local friendly
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.environ.get("SMAD_OUTPUT_DIR","./outputs")

    # Create job id package
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        output_dir = os.path.join(output_dir, f"job_{job_id}")
    else:
        # if running locally, create a folder with unique id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]
        output_dir = os.path.join(output_dir, f"run_{timestamp}_{uid}")

    os.makedirs(output_dir, exist_ok=True)

    return output_dir

def get_data_dir(data_dir=None):

    if data_dir:
        return data_dir
    
    environ_data_dir = os.environ.get("DATA_DIR")
    
    if environ_data_dir:
        return environ_data_dir

    raise ValueError("No data directory provided and DATA_DIR is not set.")