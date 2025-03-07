import torch
import torch.nn as nn
import torch.nn.functional as F
# Code modified from https://github.com/cvlab-stonybrook/DM-Count/blob/master/losses/bregman_pytorch.py
import torch
from geomloss import SamplesLoss




class CrowdCountingLoss(nn.Module):
    def __init__(self, alpha=0.1, sinkhorn_blur=0.2, density_scale=10):  # Increased blur
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

    def sinkhorn_divergence_from_maps(self, pred_map, gt_map):
        """
        Computes the Sinkhorn divergence between two 2D density maps using the tensorized backend.
        
        Args:
            pred_map (torch.Tensor): Predicted density map of shape [H, W].
            gt_map (torch.Tensor): Ground truth density map of shape [H, W].
        
        Returns:
            torch.Tensor: The computed Sinkhorn divergence (a scalar tensor).
        """
        # Get height and width.
        H, W = pred_map.shape
        N = H * W

        # Flatten the maps into 1D measures with shape [1, N].
        # Here, we assume a batch size of 1.
        a = pred_map.view(1, -1)  # measure for prediction
        b = gt_map.view(1, -1)    # measure for ground truth

        # Create a grid of coordinates (pixel positions) for the support.
        # The grid will have shape [H*W, 2]. We use (x, y) where x corresponds to columns.
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=pred_map.device),
            torch.arange(W, dtype=torch.float32, device=pred_map.device),
            indexing='ij'
        )
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # shape [N, 2]

        # Unsqueeze to add a batch dimension: now shape [1, N, 2].
        x_support = grid.unsqueeze(0)
        y_support = x_support  # Assuming both measures share the same support

        # Now call sinkhorn_tensorized.
        divergence = self.sinkhorn(a,x_support,b, y_support)
        return divergence

    def forward(self, pred_map, gt_map, gt_blur_map):
        # Add density map reconstruction loss
        gt_map = gt_map.squeeze(0).squeeze(0)
        gt_blur_map = gt_blur_map.squeeze(0).squeeze(0)

        density_loss = F.mse_loss(pred_map, gt_blur_map)

        # Scale counts appropriately
        pred_count = pred_map.sum(dim=[0,1])
        gt_count = gt_map.sum(dim=[0,1])
        
        count_loss = F.l1_loss(pred_count, gt_count)
        density_loss = F.mse_loss(pred_map,gt_map)

        spatial_loss = self.sinkhorn_divergence_from_maps(pred_map, gt_map)
        
        return density_loss + count_loss + self.alpha * spatial_loss


class PatchReconstructionLoss(nn.Module):
    def __init__(self):
        return

    def forward(self, pred_map, gt_map, gt_blur_map):
        return


