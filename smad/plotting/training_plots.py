import torch
import matplotlib.pyplot as plt

def plot_reconstruction(model, dataset, output_dir = None, device = "cuda", n_examples = 3):
    model.eval()
    fig, axes = plt.subplots(n_examples, 1, figsize=(10,3*n_examples))
    if n_examples == 1:
        axes = [axes]

    with torch.no_grad():
        for i in range(n_examples):
            seq = dataset[i].to('cpu').unsqueeze(0)
            length = [seq.size(1)]
            
            # pad and pack
            packed = torch.nn.utils.rnn.pack_padded_sequence(seq, length, batch_first=True,enforce_sorted=False)
            packed.to(device)
            seq.to(device)
            length = torch.tensor(length,device=device)
            # forward pass

            reconstruction = model(packed, seq, length)
            reconstruction = reconstruction.squeeze(0).cpu().numpy()
            original = seq.squeeze(0).cpu().numpy()

            # plot
            axes[i].plot(original[:,0], label="Original (ft 0)")
            axes[i].plot(reconstruction[:,0], label="Reconstruction (ft 0)", linestyle='--')
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
    plt.plot(training_loss, color='black', linewidth=2, label='Training Loss')
    plt.plot(val_ar_loss, color='green', linewidth=2, linestyle='--', label='Val AR Loss')
    plt.plot(val_tf_loss, color='blue', linewidth=2, linestyle='--', label='Val TF Loss')
    plt.xlabel('Epoch')
    plt.legend()
    loss_name = training_info['cfg']['params']['training']['criterion']
    plt.ylabel(f"{loss_name}")
    cfg_name = training_info['cfg']['cfg_name']
    plt.title(f'Training Loss - {cfg_name}')
    if output_dir:
        plt.savefig(output_dir+'/training_error.pdf')
        plt.savefig(output_dir+'/training_error.png')
    plt.show()