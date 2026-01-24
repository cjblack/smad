import os
import argparse
from smad.training.train import train_model_packed, auto_regressive_fine_tuning
from smad.training.utils import collate_fn
from smad.evaluation.diagnostics import evaluate
from smad.data.utils import load_data, pickle_save_data, create_data_loader, json_save_data, split_train_val
from smad.data.data_types import SeqDataSet
from smad.utils import get_output_dir, get_data_dir
from smad.plotting.training_plots import plot_reconstruction, plot_training_error
from smad.plotting.evaluation_plots import plot_corr_coef
from smad.models.utils import save_model
from pathlib import Path


if __name__ == "__main__":
    # this is just for testing at the moment...
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help='Folder where data is located')
    args = parser.parse_args()

    config_file = 'smad_coord_vp_autoencoder.yml'
    sub_data_dir = 'climbing_velprofiles_C57'
    train_set = 'climbing_C57_coord_vp_train.pt'
    test_set = 'climbing_C57_coord_vp_test.pt'
    val_set = 'climbing_C57_coord_vp_val.pt'
    
    data_path = Path(get_data_dir(args.data_dir)) / sub_data_dir

    #data_path = Path(__file__).resolve().parent.parent / 'smad/data/test_sets/climbing_velprofiles_C57'
    data_train = load_data(data_path / train_set)
    data_test = load_data(data_path / test_set)
    data_val = load_data(data_path / val_set)
    dataset_train = SeqDataSet(data_train)
    dataset_val = SeqDataSet(data_val)
    dataset_test = SeqDataSet(data_test)
    batch_size = 32
    train_loader = create_data_loader(data=dataset_train, batch_size=batch_size, collate_fn = collate_fn)
    val_loader = create_data_loader(data=dataset_val, shuffle=False, batch_size=batch_size, collate_fn = collate_fn)
    test_loader = create_data_loader(data=dataset_test, shuffle=False, batch_size=batch_size, collate_fn = collate_fn)
    print('Starting training...')
    model, training_info, criterion, optimizer, device = train_model_packed(config_file,train_loader, val_loader, noise=0.005)
    #model, fine_tuning_info = auto_regressive_fine_tuning(model,train_loader, training_info, criterion, optimizer, device)
    output_dir = get_output_dir()

    # Update training info 
    training_info['train_set_name'] = train_set
    training_info['training_file'] = os.path.basename(__file__)
    
    save_model(model,training_info,output_dir)
    pickle_save_data(output_dir+'/training_info.pkl',training_info)
    #pickle_save_data(output_dir + '/autoregressive_rt_info.pkl', fine_tuning_info)
    #json_save_data(output_dir+'/model_cfg.json', training_info['cfg'])

    plot_reconstruction(model, dataset_train, output_dir)
    plot_training_error(training_info, output_dir) # not sure why this is causing a failure on HPC
    
    all_output_test, all_target_test, all_latent = evaluate(model,test_loader) # will incorporate analysis in future
    plot_corr_coef(all_output_test, all_target_test, output_dir) # use for cross correlation...probably create a script for analyses

    print('Finished training...')