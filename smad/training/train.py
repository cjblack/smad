from smad.utils import check_and_return_config
from smad.models.utils import *
import torch
import time
import numpy as np
#import matplotlib.pyplot as plt


def train_model(model_params: str | dict, train_loader: torch.utils.data.DataLoader):

    # check if loading param file
    cfg = check_and_return_config(model_params)
    model = create_model(cfg) # change this to either create a new model or train a pre-made model
    training_params = cfg['params']['training'] # get training params from dictionary

    # Set vars
    training_info = {'total_loss': np.empty(training_params['epochs'],dtype=np.float32),
                     'epoch_time': np.empty(training_params['epochs'], dtype=np.float32)}  # preallocate empty arrays for epoch loss and time

    # Set up training
    criterion = getattr(torch.nn,training_params['criterion'])() # create criterion
    optimizer = getattr(torch.optim, training_params['optimizer']) # create optimizer
    optimizer = optimizer(model.parameters(), lr=training_params['learning_rate']) # set optimizer parameters

    # run basic training
    start = time.process_time() # variable for runtime start
    for epoch in range(training_params['epochs']):
        model.train() # set to train
        running_loss = 0.0

        for batch in train_data:
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

        training_info['total_loss'][epoch] = loss.item() # store epoch's total loss
        training_info['epoch_time'][epoch] = time.process_time() - start # store epoch time
        start = time.process_time()

    return model, training_info

