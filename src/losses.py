import torch
import torch.nn as nn
import torch.nn.functional as F
# Code modified from https://github.com/cvlab-stonybrook/DM-Count/blob/master/losses/bregman_pytorch.py
import torch
from geomloss import SamplesLoss

class CrowdCountingLoss(nn.Module):
    def __init__(self, alpha=1, sinkhorn_blur=0.2, density_scale=10):  # Increased blur
        super().__init__()
        self.alpha = alpha
        self.density_scale = density_scale
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn", 
            p=2,
            backend="multiscale",
            blur=sinkhorn_blur,  
            scaling=0.9,
            reach=0.1  
        )

    def forward(self, pred_map, gt_map, gt_blur_map):
        # Add density map reconstruction loss
        density_loss = F.mse_loss(pred_map, gt_blur_map)
        
        # Scale counts appropriately
        pred_count = pred_map.sum(dim=[1,2,3])
        gt_count = gt_map.sum(dim=[1,2,3])
        
        count_loss = F.mse_loss(pred_count, gt_count)
        
        pred_map = pred_map.squeeze(1)
        gt_map = gt_map.squeeze(1)

        density_loss = F.mse_loss(pred_map,gt_map)
        spatial_loss = torch.mean(self.sinkhorn(pred_map, gt_map))
        
        return density_loss + count_loss + self.alpha * spatial_loss
