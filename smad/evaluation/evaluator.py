import torch
'''
This is for evaluating model with test data set
'''

def evaluate(model, dataloader, teacher_forcing=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_outputs, all_targets = [], []

    with torch.no_grad():
        for packed, padded, lengths in dataloader:
            padded = padded.to(device)
            packed = packed.to(device)
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.cpu()

            decoded = model(packed, padded, lengths, teacher_forcing=teacher_forcing)

            # Loop over batch to collect outputs (variable lengths)
            for i, seq_len in enumerate(lengths):
                pred = decoded[i, :seq_len].cpu()
                target = padded[i, :seq_len].cpu()
                all_outputs.append(pred)
                all_targets.append(target)

    return all_outputs, all_targets
