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
    """This network takes in features, outputs a grid of where it thinks heads are"""

    def __init__(self, in_channels, mid_channels=64, upsample_scale=32, threshold = 0.5):
        super(HeadPointRegressor, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1),
            nn.ReLU()
        )  # Repeat 5x for 32x upsampling

    def forward(self, x):
        x = self.decoder(x)
        for _ in range(5):  # 2^5=32 upsampling
            x = self.upsampler(x)
        return torch.relu(x) + 1e-7


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
