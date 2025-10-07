from smad.training.train import train_model_packed
from smad.training.utils import collate_fn
from smad.evaluation.diagnostics import evaluate
from smad.data.utils import load_data, pickle_save_data, create_data_loader
from smad.data.data_types import SeqDataSet
from smad.utils import get_output_dir
from smad.plotting.training_plots import plot_reconstruction
from smad.models.utils import save_model
from pathlib import Path


if __name__ == "__main__":
    # this is just for testing at the moment...
    config_file = 'smad_multi_limb_autoencoder.yml'
    train_set = 'climbing_C57_train.pt'
    test_set = 'climbing_C57_test.pt'
    data_path = Path(__file__).resolve().parent.parent / 'smad/data/test_sets/climbing_multilimb_C57'
    data_train = load_data(data_path / train_set)
    data_test = load_data(data_path / test_set)
    dataset_train = SeqDataSet(data_train)
    dataset_test = SeqDataSet(data_test)
    batch_size = 32
    train_loader = create_data_loader(data=dataset_train, batch_size=batch_size, collate_fn = collate_fn)
    test_loader = create_data_loader(data=dataset_test, batch_size=batch_size, collate_fn = collate_fn)
    print('Starting training...')
    model, training_info = train_model_packed(config_file,train_loader, noise=0.05)
    output_dir = get_output_dir()
    save_model(model,training_info,output_dir)
    pickle_save_data(output_dir+'/training_info.pkl',training_info)
    plot_reconstruction(model, dataset_train, output_dir)
    all_output_test, all_target_test = evaluate(model,test_loader) # will incorporate analysis in future
    print('Finished training...')