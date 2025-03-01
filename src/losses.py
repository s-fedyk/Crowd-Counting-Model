import torch
import torch.nn as nn
import torch.nn.functional as F
# Code modified from https://github.com/cvlab-stonybrook/DM-Count/blob/master/losses/bregman_pytorch.py
import torch
from geomloss import SamplesLoss

class CrowdCountingLoss(nn.Module):
    def __init__(self, alpha=0.1, sinkhorn_blur=0.05):
        super().__init__()
        self.alpha = alpha
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn", 
            p=2, 
            # Specify spatial dimensions for [B, H, W] tensors
            backend="multiscale",  # Crucial for image data
            diameter=224.0,        # Match your image size
            scaling=0.8,
            blur=sinkhorn_blur
        )        

    def forward(self, pred_map, gt_map):
        pred_count = pred_map.sum(dim=[1,2,3])
        gt_count = gt_map.sum(dim=[1,2,3])

        count_loss = F.mse_loss(pred_count, gt_count)

        pred_map = pred_map.squeeze(1)
        gt_map = gt_map.squeeze(1)

        spatial_loss = torch.mean(self.sinkhorn(pred_map, gt_map))

        return self.alpha * spatial_loss + (1-self.alpha) * count_loss
