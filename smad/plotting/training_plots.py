import torch
import matplotlib.pyplot as plt

def plot_reconstruction(model, dataset, device = "cuda", n_examples = 3):
    model.eval()
    fig, axes = plt.subplots(n_examples, 1, figsize=(10,3*n_examples))
    if n_examples == 1:
        axes = [axes]

    with torch.no_grad():
        for i in range(n_examples):
            seq = dataset[i].to(device).unsqueeze(0)
            length = [seq.size(1)]

            # pad and pack
            packed = torch.nn.utils.rnn.pack_padded_sequence(seq, length, batch_first=True,enforce_sorted=False)

            # forward pass
            reconstruction = model(packed, seq, length)
            reconstruction = reconstruction.squeeze(0).cpu().numpy()
            original = seq.squeeze(0).cpu().numpy()

            # plot
            axes[i].plot(original[:,0], label="Original (ft 0)")
            axes[i].plot(reconstruction[:,0], label="Reconstruction (ft 0)", linestyle='--')
            axes[i].set_title(f"Sequence {i} reconstruction")

        plt.tight_layout()
        plt.show()
