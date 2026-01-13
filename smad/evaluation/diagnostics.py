import numpy as np
import matplotlib.pyplot as plt
import torch

'''
This is for evaluating model with test data set
'''

def evaluate(model, dataloader, teacher_forcing=False):
    model.eval()
    all_outputs, all_targets, all_latent = [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for packed, padded, lengths in dataloader:
            padded = padded.to(device)
            packed = packed.to(device)
            lengths = torch.tensor(lengths, device=device)
            #if isinstance(lengths, torch.Tensor):
            #    lengths = lengths.cpu()

            decoded = model(packed, padded, lengths, teacher_forcing=teacher_forcing)
            z = model.encode(packed, lengths)
            all_latent.append(z)
            # Loop over batch to collect outputs (variable lengths)
            for i, seq_len in enumerate(lengths):
                pred = decoded[i, :seq_len].cpu()
                target = padded[i, :seq_len].cpu()
                all_outputs.append(pred)
                all_targets.append(target)
    all_latent = torch.cat(all_latent,dim=0)
    return all_outputs, all_targets, all_latent

def evaluate_masked(model, dataloader, teacher_forcing=False):
    """Evaluation maintaining mask for easier GPU computation"""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_outputs, all_targets, all_latent, all_mask = [], [], [], []
    with torch.no_grad():
        for packed, padded, lengths in dataloader:
            padded = padded.to(device)
            packed = packed.to(device)
            lengths = torch.tensor(lengths,device=device)
            max_len = padded.shape[1]
            mask = torch.arange(max_len)[None, :].to(device) < lengths[:, None] # (B, S)
            #mask = mask.unsqueeze(2).repeat(1,1,2)
            decoded = model(packed, padded, lengths, teacher_forcing=teacher_forcing)
            z = model.encode(packed, lengths)
            all_latent.append(z)
            pred = decoded
            target = padded
            all_outputs.append(pred)
            all_targets.append(target)
            all_mask.append(mask)
    all_latent = torch.cat(all_latent, dim=0)
    return all_outputs, all_targets, all_latent, all_mask

def per_timestep_mse(all_outputs, all_targets, output_dir = None):
    # all_outputs and all_targets are lists of (seq_len, feat_dim) tensors (CPU)
    max_len = max(o.shape[0] for o in all_targets)
    sums = np.zeros(max_len)
    counts = np.zeros(max_len)

    for out, tgt in zip(all_outputs, all_targets):
        l = tgt.shape[0]
        l_o = out.shape[0]
        if l == l_o:
            err = ((out.numpy() - tgt.numpy())**2).mean(axis=1)  # per timestep MSE across features
            sums[:l] += err
            counts[:l] += 1

    mean_per_t = sums / np.maximum(counts, 1.0)
    fig, ax = plt.subplots()
    ax.plot(mean_per_t)
    ax.set_ylabel('Mean MSE')
    ax.set_xlabel('timestep')
    if output_dir:
        plt.savefig(output_dir+'/mean_error_across_timestamps.pdf')
        plt.savefig(output_dir+'/mean_error_across_timestamps.png')

    return mean_per_t