from pathlib import Path
from smad.training.train import train_model_packed, auto_regressive_fine_tuning
from smad.training.utils import collate_fn
from smad.data.utils import load_data, create_data_loader
from smad.data.data_types import SeqDataSet

def test_training():
    """
    Test single epoch training
    """
    config_file = 'test_smad_multi_limb_autoencoder.yml'
    train_set = 'climbing_C57_train.pt'
    data_path = Path(__file__).resolve().parent.parent / 'smad/data/test_sets/climbing_multilimb_C57'
    data_train = load_data(data_path / train_set)
    dataset_train = SeqDataSet(data_train)
    dataset_train = dataset_train[:50] # reduce size of input data for testing
    batch_size=32
    train_loader = create_data_loader(data=dataset_train, batch_size=batch_size, collate_fn=collate_fn)
    print('Autoencoder Training Test V1.')
    model, training_info, criterion, optimizer, device = train_model_packed(config_file, train_loader, noise=0.05)
    model, fine_tuning_info = auto_regressive_fine_tuning(model, train_loader, training_info, criterion, optimizer, device)
    training_mse = training_info['epoch_mse'][0]
    fine_tuning_mse = fine_tuning_info['total_loss'][0]

    assert (training_mse == 0) & (fine_tuning_mse == 0)