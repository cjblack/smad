from smad.training.train import train_model
from smad.data.utils import pickle_load_data, create_data_loader
from pathlib import Path


if __name__ == "__main__":
    # this is just for testing at the moment...
    config_file = 'srad_autoencoder.yml'
    test_set = 'forepaw_training_example_set.pkl'
    data_path = Path(__file__).resolve().parent.parent / 'smad/data/test_sets'
    data = pickle_load_data(data_path / test_set)
    batch_size = 32
    train_loader = create_data_loader(data=data, batch_size=batch_size)
    print('Starting training...')
    model, training_info = train_model(config_file,train_loader)
    print(training_info)
    print('Finished training...')