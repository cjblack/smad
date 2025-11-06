[![Tests](https://github.com/cjblack/smad/actions/workflows/unit_training_test.yml/badge.svg)](https://github.com/cjblack/smad/actions/workflows/unit_training_test.yml)

# skilled movement anomaly detection
WIP: code is being refactored from another repository to make this application more generalizable. Main purpose is to train models and perform anomaly detection on kinematic data extracted from markerless pose estimation.

## To-Do
- [ ] overhaul data functionality
- [ ] expand model utility
  - [ ] saving
  - [ ] loading
  - [ ] latent space extraction
- [ ] plotting

## Usage
### Local training

1. Train autoencoder model

```python

from smad.training.train import train_model_packed, auto_regressive_fine_tuning
from smad.training.utils import collate_fn
from smad.data.utils import load_data, pickle_save_data, create_data_loader
from smad.data.data_types import SeqDataSet

# Change the file names accordingly
config_file = 'config_file.yml'
train_set = 'train_set.pt'
batch_size = 32

# load data
data_train = load_data(train_set)

# create sequenced datasets
dataset_train = SeqDataSet(data_train)

# create data loaders
train_loader = create_data_loader(data=dataset_train, batch_size=batch_size, collate_fn=collate_fn)

# run training
model, training_info, criterion, optimizer, device = train_model_packed(config_file, train_loader, noise=0.05)

```

2. Save model output

```python

# create output directory
output_dir = get_output_dir()

# save training info
save_model(model, training_info, output_dir)

```

3. Plot and save reconstruction

```python
from smad.plotting.training_plots import plot_reconstruction

plot_reconstruction(model, dataset_train, output_dir)

```

### Training on HPC with slurm

1. Before you can run any scripts, you'll need to make sure you have created the smad conda environment on your preferred system.
2. You'll need to open your `.bashrc` profile and store the following: `export SMAD_OUTPUT_DIR=/path/to/output/directory`. Whenever you run training by using slurm, the data will be saved to this location in a folder with a unique identifier.
3. Modify slurm job template in `smad/smad/hpc_scripts` with relevant information (see comments in template file).
4. Run sbatch command:
```shell
sbatch smad_slurm_template_C57_climb_training.slurm
```
