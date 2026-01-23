import torch
from smad.models.utils import load_trained_model
from smad.data.data_types import SeqDataSet
from smad.data.utils import load_data, create_data_loader
from smad.training.utils import collate_fn


"""
THIS IS A CARRY OVER FROM THE EVALUATION MODULE...
"""

def reconstruct(model, dataset: torch.utils.data.DataLoader | str, device=None):
    
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if type(dataset) == str:
        dataset = load_data(dataset)
        dataset = SeqDataSet(dataset)
        dataset = create_data_loader(data=dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    

    model.to(device)
    #dataset.to(device)


    all_outputs, all_targets, all_latent, all_mask = [], [], [], []

    with torch.no_grad():
        for packed, padded, lengths in dataset:
            padded = padded.to(device)
            packed = packed.to(device)
            lengths = torch.tensor(lengths,device=device)
            max_len = padded.shape[1]
            mask = torch.arange(max_len)[None, :].to(device) < lengths[:, None] # (B, S)
            decoded = model(packed, padded, lengths, teacher_forcing=False)
            z = model.encode(packed, lengths)
            all_latent.append(z)
            pred = decoded
            target = padded
            all_outputs.append(pred)
            all_targets.append(target)
            all_mask.append(mask)
    all_latent = torch.cat(all_latent, dim=0)

    return all_outputs, all_targets, all_latent, all_mask