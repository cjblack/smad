import torch
import matplotlib.pyplot as plt

def plot_reconstruction(model, dataset, output_dir = None, device = "cuda", dindex = 0):
    model.eval()
    S, F = dataset[dindex].shape
    fig, axes = plt.subplots(F, 1, figsize=(10,3*F))
    if F == 1:
        axes = [axes]

    with torch.no_grad():
        #for i in range(n_examples):
        seq = dataset[dindex].to(device).unsqueeze(0)
        length = [seq.size(1)]
        
        # pad and pack
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, length, batch_first=True,enforce_sorted=False)
        #packed.to(device)
        #seq.to(device)
        # for some reason when making packed sequence, length has to be on cpu, but obviously needs to be on GPU for model?
        length = torch.tensor(length,device=device)
        # forward pass

        reconstruction = model(packed, seq, length)
        reconstruction = reconstruction.squeeze(0).cpu().numpy()
        original = seq.squeeze(0).cpu().numpy()

        # plot
        for i in range(F):
            axes[i].plot(original[:,i], label="Original (ft 0)")
            axes[i].plot(reconstruction[:,i], label="Reconstruction (ft 0)", linestyle='--')
            axes[i].set_title(f"Sequence {i} reconstruction")

        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir+'/training_reconstruction_example.pdf')
            plt.savefig(output_dir+'/training_reconstruction_example.png')
        plt.show()

def plot_training_error(training_info: dict, output_dir = None):
    """
    Plots training and validation error
    """
    training_loss = training_info['epoch_mse_train']
    val_ar_loss = training_info['epoch_mse_val_ar']
    val_tf_loss = training_info['epoch_mse_val_tf']
    fig, ax = plt.subplots()
    ax.plot(training_loss, color='black', linewidth=2, label='Training Loss')
    ax.plot(val_ar_loss, color='green', linewidth=2, linestyle='--', label='Val AR Loss')
    ax.plot(val_tf_loss, color='blue', linewidth=2, linestyle='--', label='Val TF Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    loss_name = training_info['cfg']['params']['training']['criterion']
    ax.set_ylabel(f"{loss_name}")
    cfg_name = training_info['cfg']['cfg_name']
    ax.set_title(f'Training Loss - {cfg_name}')
    if output_dir:
        plt.savefig(output_dir+'/training_error.pdf')
        plt.savefig(output_dir+'/training_error.png')
    plt.show()