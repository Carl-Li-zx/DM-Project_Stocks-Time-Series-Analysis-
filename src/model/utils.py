import numpy as np
import torch


def triangular_causal_mask(B, L, device="cpu"):
    mask_shape = [B, 1, L, L]
    with torch.no_grad():
        mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    return mask


