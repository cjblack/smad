from smad.utils import check_and_return_config
from smad.models.utils import *
from smad.training.utils import *
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


def train_model_packed(model_params: str | dict, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, teacher_forcing_function: str = 'inverse_sigmoid', noise: float = 0.0):
    # Make sure model runs on cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # check if loading param file
    cfg = check_and_return_config(model_params)
    model = create_model(cfg).to(device) # change this to either create a new model or train a pre-made model
    training_params = cfg['params']['training'] # get training params from dictionary

    # Handle teacher forcing
    teacher_forcing = True # set this to true by default
    if teacher_forcing_function == 'off':
        teacher_forcing = False

    # Set vars
    training_info = {'epoch_mse_train': np.empty(training_params['epochs'],dtype=np.float32),
                     'epoch_mse_val_ar': np.empty(training_params['epochs'], dtype=np.float32),
                     'epoch_mse_val_tf': np.empty(training_params['epochs'], dtype=np.float32),
                     'epoch_time': np.empty(training_params['epochs'], dtype=np.float32),
                     'noise': noise,
                     'teacher_forcing': teacher_forcing,
                     'cfg':cfg}  # preallocate empty arrays for epoch loss and time

    # Set up training
    criterion = getattr(torch.nn,training_params['criterion'])(reduction='none') # create criterion
    optimizer = getattr(torch.optim, training_params['optimizer']) # create optimizer
    optimizer = optimizer(model.parameters(), lr=training_params['learning_rate']) # set optimizer parameters

    # run basic training
    #start = time.process_time() # variable for runtime start
    epochs = training_params['epochs']

    # teacher forcing decay
    #initial_ratio = teacher_forcing_ratios[0] # starting ratio
    #final_ratio = teacher_forcing_ratios[1] # final ratio
    #decay_epochs = int(round(0.6*epochs)) # set decay epochs to 60% of total epochs


    for epoch in range(epochs):

        start = time.perf_counter() # variable for runtime start
        model.train() # set to train
        running_loss = 0.0
        total_loss = 0.0
        total_count = 0.0
        #tf_ratio = max(final_ratio, initial_ratio * (1-epoch/decay_epochs)) # decay ratio over training - this will inevitably yield a value of 0
        tf_ratio = get_teacher_forcing_ratio(teacher_forcing_function, epoch, total_epochs=epochs) # this is a basic implementation
        noise_std = min(noise, (epoch/epochs)*noise) # gradually increase gaussian noise to set level -> if set level = 0.0, no noise will be added in model (see model class)
        for packed, padded, lengths in train_loader:

            packed = packed.to(device)
            padded = padded.to(device)
            lengths = torch.tensor(lengths, device=device)

            optimizer.zero_grad() # zero out gradient

            # Forward pass
            outputs = model(packed, padded, lengths, teacher_forcing=teacher_forcing, teacher_forcing_ratio=tf_ratio, noise_std = noise_std)

            # Masked loss
            target = padded[:, 1:, :]
            max_len_minus1 = target.size(1)

            mask = torch.arange(max_len_minus1, device=device)[None, :] < (lengths-1)[:, None]
            mask = mask.unsqueeze(-1)

            per_timestep_loss = (criterion(outputs, target)) # * mask)
            masked_loss = per_timestep_loss * mask

            batch_loss_sum = masked_loss.sum()


            #total_loss += per_timestep_loss.sum().item()
            denom_recon = mask.sum()*outputs.size(-1) + 1e-8
            loss = batch_loss_sum / denom_recon
            total_loss += batch_loss_sum.item()
            total_count += denom_recon.item()#mask.sum().item()
            #loss = per_timestep_loss.sum() / denom_recon#mask.sum() # stabilize backpropagation against longer samples

            # Backwards pass & optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0) # might remove
            optimizer.step()

            running_loss += loss.item()

        # epoch mse
        epoch_mse = total_loss/(total_count + 1e-8)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, MSE: {epoch_mse:.4f}")
        training_info['epoch_mse_train'][epoch] = total_loss/total_count#loss.item() # store epoch's total loss
        #start = time.process_time()

        """VALIDATION"""
        epoch_mse_val_ar, epoch_mse_val_tf = validation_eval(model, val_loader, criterion, device, tf_ratio)

        training_info['epoch_mse_val_ar'][epoch] = epoch_mse_val_ar
        training_info['epoch_mse_val_tf'][epoch] = epoch_mse_val_tf
        training_info['epoch_time'][epoch] = time.perf_counter() - start # store epoch time


    return model, training_info, criterion, optimizer, device

def train_model_kinematics_inspired(model_params: str | dict, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, lambda_smooth = 0.1, dt = 1/200., teacher_forcing_function: str = 'inverse_sigmoid', noise: float = 0.0):
    # Make sure model runs on cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # check if loading param file
    cfg = check_and_return_config(model_params)
    model = create_model(cfg).to(device) # change this to either create a new model or train a pre-made model
    training_params = cfg['params']['training'] # get training params from dictionary

    # Handle teacher forcing
    teacher_forcing = True # set this to true by default
    if teacher_forcing_function == 'off':
        teacher_forcing = False

    # Set vars
    training_info = {'epoch_mse_train': np.empty(training_params['epochs'],dtype=np.float32),
                     'epoch_mse_val_ar': np.empty(training_params['epochs'], dtype=np.float32),
                     'epoch_mse_val_tf': np.empty(training_params['epochs'], dtype=np.float32),
                     'epoch_time': np.empty(training_params['epochs'], dtype=np.float32),
                     'noise': noise,
                     'teacher_forcing': teacher_forcing,
                     'cfg':cfg}  # preallocate empty arrays for epoch loss and time

    # Set up training
    criterion = getattr(torch.nn,training_params['criterion'])(reduction='none') # create criterion
    optimizer = getattr(torch.optim, training_params['optimizer']) # create optimizer
    optimizer = optimizer(model.parameters(), lr=training_params['learning_rate']) # set optimizer parameters

    # run basic training
    #start = time.process_time() # variable for runtime start
    epochs = training_params['epochs']


    for epoch in range(epochs):

        start = time.perf_counter() # variable for runtime start
        model.train() # set to train
        running_loss = 0.0
        total_loss = 0.0
        total_count = 0.0
        tf_ratio = get_teacher_forcing_ratio(teacher_forcing_function, epoch, total_epochs=epochs) # this is a basic implementation
        noise_std = min(noise, (epoch/epochs)*noise) # gradually increase gaussian noise to set level -> if set level = 0.0, no noise will be added in model (see model class)
        for packed, padded, lengths in train_loader:

            packed = packed.to(device)
            padded = padded.to(device)
            lengths = torch.tensor(lengths, device=device)

            optimizer.zero_grad() # zero out gradient

            # Forward pass
            outputs = model(packed, padded, lengths, teacher_forcing=teacher_forcing, teacher_forcing_ratio=tf_ratio, noise_std = noise_std)

            # Masked loss
            target = padded[:, 1:, :]
            max_len_minus1 = target.size(1)

            mask = torch.arange(max_len_minus1, device=device)[None, :] < (lengths-1)[:, None]
            mask = mask.unsqueeze(-1)

            # Criterion Loss
            per_timestep_loss = (criterion(outputs, target)) #* mask)
            masked_loss = per_timestep_loss * mask
            batch_loss_sum = masked_loss.sum()

            denom_recon = mask.sum()*outputs.size(-1) + 1e-8
            recon_loss = batch_loss_sum / denom_recon
            total_loss += batch_loss_sum.item()
            total_count += denom_recon.item()#mask.sum().item()
            #loss = per_timestep_loss.sum() / denom_recon#mask.sum() # stabilize backpropagation against longer samples

            # Smoothness loss
            dv = (outputs[:, 1:, :] - outputs[:, :-1, :]) / dt
            mask_dv = (mask[:, 1:, :] * mask[:, :-1, :]).float()
            denom_smooth = mask_dv.sum() * outputs.size(-1) + 1e-8
            smooth_loss = ((dv ** 2) * mask_dv).sum() / denom_smooth
            
            # Total loss
            loss = recon_loss + (smooth_loss * lambda_smooth)

            # Backwards pass & optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0) # might remove
            optimizer.step()

            running_loss += loss.item()

        # epoch mse
        epoch_mse = total_loss/total_count
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, MSE: {epoch_mse:.4f}")
        training_info['epoch_mse_train'][epoch] = total_loss/total_count#loss.item() # store epoch's total loss
        #start = time.process_time()

        """VALIDATION"""
        epoch_mse_val_ar, epoch_mse_val_tf = validation_eval(model, val_loader, criterion, device, tf_ratio)
        
        training_info['epoch_mse_val_ar'][epoch] = epoch_mse_val_ar
        training_info['epoch_mse_val_tf'][epoch] = epoch_mse_val_tf
        training_info['epoch_time'][epoch] = time.perf_counter() - start # store epoch time


    return model, training_info, criterion, optimizer, device

def validation_eval(model, val_loader, criterion, device, tf_ratio):
    
    # ***ADD KINEMATICS FOR EVAL***
    val_loss_ar = 0.0
    val_loss_tf = 0.0
    val_count_ar = 0.0
    val_count_tf = 0.0
    model.eval()
    with torch.no_grad():
        for packed, padded, lengths in val_loader:
            packed = packed.to(device)
            padded = padded.to(device)
            lengths = torch.tensor(lengths, device=device)

            out_ar = model(packed, padded, lengths, teacher_forcing=False, noise_std=0.0)
            out_tf = model(packed, padded, lengths, teacher_forcing=True, teacher_forcing_ratio=tf_ratio, noise_std=0.0)

            target = padded[:, 1:, :]
            max_len_minus1 = target.size(1)

            mask = torch.arange(max_len_minus1, device=device)[None, :] < (lengths-1)[:, None]
            mask = mask.unsqueeze(-1)

            per_timestep_loss_ar = (criterion(out_ar, target) * mask)
            per_timestep_loss_tf = (criterion(out_tf, target) * mask)

            val_loss_ar += per_timestep_loss_ar.sum().item()
            val_loss_tf += per_timestep_loss_tf.sum().item()
            denom_ar = mask.sum()*out_ar.size(-1) + 1e-8
            denom_tf = mask.sum()*out_tf.size(-1) + 1e-8

            val_count_ar += denom_ar.item()#mask.sum().item()
            val_count_tf += denom_tf.item()
    epoch_mse_val_ar = val_loss_ar / val_count_ar
    epoch_mse_val_tf = val_loss_tf / val_count_tf

    return epoch_mse_val_ar, epoch_mse_val_tf


            
def auto_regressive_fine_tuning(model, train_loader: torch.utils.data.DataLoader, training_info: dict, criterion, optimizer, device, freeze_encoder = False):
    fine_tune_epochs = training_info['cfg']['params']['training']['autoregressive_ft']['epochs']
    fine_tune_lr = training_info['cfg']['params']['training']['autoregressive_ft']['learning_rate']

    # Control encoder freezing
    if freeze_encoder and hasattr(model,"encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
        print('Encoder frozen for autoregressive fine-tuning.')
    else:
        print('Full model fine-tuning (encoder & decoder).')

    # Control adjusted learning rate
    if fine_tune_lr is not None:
        for g in optimizer.param_groups:
            g['lr'] = fine_tune_lr # change learning rate
        print(f'Fine-tuning LR set to {fine_tune_lr}')

    # Create training info output
    fine_tune_info = {
        'total_loss': np.empty(fine_tune_epochs, dtype = np.float32),
        'epoch_time': np.empty(fine_tune_epochs, dtype = np.float32)
    }

    # Run training loop
    start = time.process_time()
    model.use_skip = False # set this to remove skip connections during AR-FT
    model.train()
    for epoch in range(fine_tune_epochs):
        running_loss = 0.0
        running_count = 0.0
        for packed, padded, lengths in train_loader:
            packed, padded = packed.to(device), padded.to(device)
            lengths = torch.tensor(lengths, device=device)
            optimizer.zero_grad()
            # auto regress without teacher forcing
            outputs = model(packed, padded, lengths, teacher_forcing=False, teacher_forcing_ratio=0.0)

            target = padded[:,1:,:]
            max_len_minus1 = target.size(1)
            mask = torch.arange(max_len_minus1, device=device)[None, :] < (lengths - 1)[:, None]
            mask = mask.unsqueeze(-1)

            masked_loss = (criterion(outputs, target)*mask)
            running_loss += masked_loss.sum().item()
            running_count += mask.sum().item()
            loss = (criterion(outputs, target) * mask).sum() / mask.sum() # stabilize backpropagation against longer samples
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_loss = running_loss / running_count
        fine_tune_info['total_loss'][epoch] = avg_loss
        fine_tune_info['epoch_time'][epoch] = time.process_time() - start
        start = time.process_time()
    return model, fine_tune_info