import torch
import torch.nn as nn
import torch.nn.functional as F


class CrowdCountingLoss(nn.Module):
    def __init__(self, count_loss_weight=1.0):
        super(CrowdCountingLoss, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.count_loss_weight = count_loss_weight

    def forward(self, pred_points, gt_points):

        mse = self.mse_loss(pred_points, gt_points)
        pred_count = pred_points.sum(dim=[1, 2, 3])
        gt_count = gt_points.sum(dim=[1, 2, 3])
        count_loss = torch.mean((pred_count - gt_count) ** 2)

        total_loss = mse + self.count_loss_weight * count_loss

        return total_loss


class HeadPointRegressor(nn.Module):
    def __init__(self, in_channels, mid_channels=64, upsample_scale=32):
        super(HeadPointRegressor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.upsample_scale = upsample_scale

    def forward(self, visual_features):
        visual_features = self.conv1(visual_features)
        visual_features = self.relu(visual_features)

        point_map = self.conv2(visual_features)
        if self.upsample_scale is not None:
            point_map = F.interpolate(point_map, scale_factor=self.upsample_scale,
                                      mode='bilinear', align_corners=False)

        return point_map


def reshape_tokens_to_grid(tokens):
    B, N, D = tokens.shape
    grid_size = int(N ** 0.5)

    assert grid_size * grid_size == N, "Expected a square grid of patches"
    grid = tokens.view(B, grid_size, grid_size, D)
    grid = grid.permute(0, 3, 1, 2)

    return grid


class CLIPGCC(nn.Module):
    def __init__(self, clip_model, regressor_channels=64):
        super(CLIPGCC, self).__init__()
        self.clip_model = clip_model

        self.feature_dim = 768

        self.regressor = HeadPointRegressor(
            in_channels=768,
            mid_channels=regressor_channels,
            upsample_scale=32
        )

    def forward(self, x):
        grid_features = self.get_grid_features(x)

        return self.regressor(grid_features)

    def get_grid_features(self, x):
        _, tokens = self.clip_model.visual(x)

        grid_of_patch_tokens = reshape_tokens_to_grid(tokens)

        return grid_of_patch_tokens
