import torch
import torch.nn as nn
import torch.nn.functional as F
# Code modified from https://github.com/cvlab-stonybrook/DM-Count/blob/master/losses/bregman_pytorch.py
import torch
from geomloss import SamplesLoss




class CrowdCountingLoss(nn.Module):
    def __init__(self, alpha=1, sinkhorn_blur=0.05, density_scale=10):  # Increased blur
        super().__init__()
        self.alpha = alpha
        self.density_scale = density_scale
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn", 
            p=2,
            backend="auto",
            blur=sinkhorn_blur,  
            scaling=0.1,
            reach=0.5  
        )

    def forward(self, pred_map, gt_map, gt_blur_map):
        # Add density map reconstruction loss
        gt_map = gt_map.squeeze(0).squeeze(0)
        gt_blur_map = gt_blur_map.squeeze(0).squeeze(0)

        density_loss = F.mse_loss(pred_map, gt_blur_map)

        # Scale counts appropriately
        pred_count = pred_map.sum(dim=[0,1])
        gt_count = gt_map.sum(dim=[0,1])
        
        count_loss = F.l1_loss(pred_count, gt_count)
        density_loss = F.mse_loss(pred_map, gt_blur_map)

        spatial_loss = self.sinkhorn(pred_map, gt_map)
        return density_loss + count_loss + self.alpha * spatial_loss
