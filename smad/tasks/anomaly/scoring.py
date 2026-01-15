import torch
import torch.nn.functional as F

def masked_mse(x, x_hat, mask):
    
    se = (x-x_hat)**2  # (B,T,F)
    se = se.mean(dim=-1)
    se = se * mask
    per_sample_mse = se.sum(dim=1) / mask.sum(dim=1)

    return per_sample_mse