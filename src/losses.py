import torch
import torch.nn as nn

class CrowdCountingLoss(nn.Module):
    def __init__(self, count_loss_weight=1.0):
        super(CrowdCountingLoss, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.count_loss_weight = count_loss_weight

    def forward(self, pred_points, gt_points):
        pred_count = pred_points.sum(dim=[1, 2, 3])
        gt_count = gt_points.sum(dim=[1, 2, 3])
        count_loss = torch.mean((pred_count - gt_count) ** 2)

        return count_loss
