from smad.utils import check_and_return_config
from smad.models.utils import *
import torch
import time
import numpy as np
import subprocess
#import matplotlib.pyplot as plt


def train_model(model_params: str | dict, train_loader: torch.utils.data.DataLoader):
    # Make sure model runs on cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # check if loading param file
    cfg = check_and_return_config(model_params)
    model = create_model(cfg).to(device) # change this to either create a new model or train a pre-made model
    training_params = cfg['params']['training'] # get training params from dictionary

    # Set vars
    training_info = {'total_loss': np.empty(training_params['epochs'],dtype=np.float32),
                     'epoch_time': np.empty(training_params['epochs'], dtype=np.float32),
                     'cfg':cfg}  # preallocate empty arrays for epoch loss and time

    # Set up training
    criterion = getattr(torch.nn,training_params['criterion'])() # create criterion
    optimizer = getattr(torch.optim, training_params['optimizer']) # create optimizer
    optimizer = optimizer(model.parameters(), lr=training_params['learning_rate']) # set optimizer parameters

    # run basic training
    start = time.process_time() # variable for runtime start
    epochs = training_params['epochs']
    for epoch in range(epochs):
        model.train() # set to train
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad() # zero out gradient

            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch)

            # Backwards pass & optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
            print(torch.cuda.memory_summary())
        training_info['total_loss'][epoch] = loss.item() # store epoch's total loss
        training_info['epoch_time'][epoch] = time.process_time() - start # store epoch time
        start = time.process_time()

    return model, training_info


def train_model_packed(model_params: str | dict, train_loader: torch.utils.data.DataLoader, teacher_forcing = True, teacher_forcing_ratios: list = [1.0,0.0], noise: float = 0.0):
    # Make sure model runs on cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # check if loading param file
    cfg = check_and_return_config(model_params)
    model = create_model(cfg).to(device) # change this to either create a new model or train a pre-made model
    training_params = cfg['params']['training'] # get training params from dictionary

    # Set vars
    training_info = {'total_loss': np.empty(training_params['epochs'],dtype=np.float32),
                     'epoch_time': np.empty(training_params['epochs'], dtype=np.float32),
                     'noise': noise,
                     'teacher_forcing': teacher_forcing,
                     'cfg':cfg}  # preallocate empty arrays for epoch loss and time

    # Set up training
    criterion = getattr(torch.nn,training_params['criterion'])(reduction='none') # create criterion
    optimizer = getattr(torch.optim, training_params['optimizer']) # create optimizer
    optimizer = optimizer(model.parameters(), lr=training_params['learning_rate']) # set optimizer parameters

    # run basic training
    start = time.process_time() # variable for runtime start
    epochs = training_params['epochs']

    # teacher forcing decay
    initial_ratio = teacher_forcing_ratios[0] # starting ratio
    final_ratio = teacher_forcing_ratios[1] # final ratio
    decay_epochs = int(round(0.6*epochs)) # set decay epochs to 60% of total epochs

    for epoch in range(epochs):
        model.train() # set to train
        running_loss = 0.0
        total_loss = 0.0
        total_count = 0.0
        ratio = max(final_ratio, initial_ratio * (1-epoch/decay_epochs)) # decay ratio over training - this will inevitably yield a value of 0
        noise_std = min(noise, (epoch/epochs)*noise) # gradually increase gaussian noise to set level -> if set level = 0.0, no noise will be added in model (see model class)
        for packed, padded, lengths in train_loader:

            packed = packed.to(device)
            padded = padded.to(device)
            lengths = torch.tensor(lengths, device=device)

            optimizer.zero_grad() # zero out gradient

            # Forward pass
            outputs = model(packed, padded, lengths, teacher_forcing=teacher_forcing, teacher_forcing_ratio=ratio, noise_std = noise_std)

            # Masked loss
            target = padded[:, 1:, :]
            max_len_minus1 = target.size(1)

            mask = torch.arange(max_len_minus1, device=device)[None, :] < (lengths-1)[:, None]
            mask = mask.unsqueeze(-1)

            per_timestep_loss = (criterion(outputs, target) * mask)
            total_loss += per_timestep_loss.sum().item()
            total_count += mask.sum().item()
            loss = per_timestep_loss.sum() / mask.sum()

            # Backwards pass & optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0) # might remove
            optimizer.step()

            running_loss += loss.item()

        # epoch mse
        epoch_mse = total_loss/total_count
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, MSE: {epoch_mse:.4f}")
        training_info['epoch_mse'][epoch] = total_loss/total_count#loss.item() # store epoch's total loss
        training_info['epoch_time'][epoch] = time.process_time() - start # store epoch time
        start = time.process_time()

    return model, training_info

