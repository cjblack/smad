import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_corr_coef(out_, tar_, output_dir = None):
    no_samples = len(out_) # get number of samples in evaluation
    input_size = out_[0].shape[1] # lazy implementation
    cross_corr_array = np.zeros((no_samples,input_size))
    for i in range(no_samples):
        min_seq_len = min([out_[i].shape[0], tar_[i].shape[0]]) # this compensates for having reconstructed n-1 seq length...not sure why this is happening
        for ii in range(input_size):
            data_cat = torch.vstack([out_[i][:min_seq_len,ii], tar_[i][:min_seq_len,ii]])
            cross_corr = torch.corrcoef(data_cat)
            cross_corr = cross_corr[0][1]
            cross_corr_array[i,ii] = cross_corr.detach().numpy()

    fig, axes = plt.subplots(ncols = input_size, figsize=(12,3))
    for i in range(input_size):
        mean_cross_corr = np.mean(cross_corr_array[:,i])
        axes[i].hist(cross_corr_array[:,i], density=True)
        axes[i].axvline(mean_cross_corr,linestyle='--',color='black')
        axes[i].set_xlim([0,1])
        axes[i].set_title(f'Node {i} mean: {np.round(mean_cross_corr,3)}')
        axes[i].set_xlabel('Corr Coef')
        axes[i].set_ylabel('Density')
    fig.suptitle('Reconstruction-Original Cross Correlation')
    fig.tight_layout()
    if output_dir:
        plt.savefig(output_dir+'/cross_correlation_eval.pdf')
        plt.savefig(output_dir+'/cross_correlation_eval.png')
        np.save(output_dir+'/cross_correlation_eval.npy', cross_corr_array)
    plt.show()