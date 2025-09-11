from smad.training.train import train_model_packed
from smad.training.utils import collate_fn
from smad.data.utils import load_data, pickle_save_data, create_data_loader
from smad.data.data_types import SeqDataSet
from smad.utils import get_output_dir
from smad.models.utils import save_model
from pathlib import Path


if __name__ == "__main__":
    # this is just for testing at the moment...
    config_file = 'smad_multi_limb_autoencoder.yml'
    test_set = '4x4_climb_velocity_profiles_packed.pt'
    data_path = Path(__file__).resolve().parent.parent / 'smad/data/test_sets'
    data = load_data(data_path / test_set)
    dataset = SeqDataSet(data)
    batch_size = 32
    train_loader = create_data_loader(data=dataset, batch_size=batch_size, collate_fn = collate_fn)
    print('Starting training...')
    model, training_info = train_model_packed(config_file,train_loader)
    output_dir = get_output_dir()
    save_model(model,training_info,output_dir)
    pickle_save_data(output_dir+'/training_info.pkl',training_info)
    print('Finished training...')